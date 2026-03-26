import numpy as np
from mpi4py import MPI
import dolfinx as dfx
from dolfinx import mesh, fem, plot, io, default_scalar_type

import dolfinx.fem.petsc as dfpet

if dfx.__version__ == '0.9.0':
    import dolfinx.io as gmshio

import basix
import adios4dolfinx 

import ufl

import scipy.sparse as scp
import numpy as np
# import scipy.sparse.linalg
import scipy.sparse.linalg as ssla
import time



import discrete_divergence as discrete_divergence
import spaces_def as s_def
import generate_mesh as genmesh
petsc_options_prefix = "basic_linear_problem"
# set this globally

from petsc4py import PETSc
from slepc4py import SLEPc


######################################################################################
# scalar assemble fun
######################################################################################

def my_scalar_assemble(inner, comm, jit_ops = None): # müssen noch comm übergeben von mesh bzw nach lesen von mesh, z.B. mesh.comm
    local_scalar = dfx.fem.assemble_scalar(dfx.fem.form(inner, jit_options = jit_ops))
    return comm.allreduce(local_scalar, op=MPI.SUM)

######################################################################################
# norms
######################################################################################

def H1kappa_norm_ord(u1,u2,kappa,comm):   
    n1 = my_scalar_assemble( ( 1/kappa**2 * ufl.inner(  ufl.grad(u1), ufl.grad(u1) ) + ufl.inner(u1, u1)   ) *ufl.dx , comm) 
    n2 = my_scalar_assemble( ( 1/kappa**2 * ufl.inner(  ufl.grad(u2), ufl.grad(u2) ) + ufl.inner(u2, u2)   ) *ufl.dx , comm) 
    
    return np.sqrt( n1 + n2)


def H1_norm(A1,comm):
    return np.sqrt( my_scalar_assemble(   ( ufl.inner( ufl.grad( A1() ) , ufl.grad( A1() ) )  +  ufl.inner(  A1() ,  A1()  )  ) *ufl.dx ) , comm ) 


def H1_curl_div(A,comm):

    curl_norm =   ufl.inner( ufl.rot( A ) , ufl.rot( A ) ) * ufl.dx   
    div_norm =   ufl.inner( ufl.div( A ) , ufl.div( A ) ) * ufl.dx 

    
    return np.sqrt( my_scalar_assemble( curl_norm + div_norm  , comm) )
######################################################################################
# E(u,A)
######################################################################################

def compute_grad_bil(u_real,phi_real, MagPot,kappa):
    
    return  1/kappa**(2) * ufl.inner(ufl.grad(u_real), ufl.grad(phi_real)) * ufl.dx

def compute_A_L2_bil(u_real,phi_real, MagPot,kappa):
    
    return   ufl.inner(MagPot, MagPot)*u_real*phi_real *ufl.dx


def compute_u_A_nabla_phi(u,phi, MagPot,kappa):
    
    return 1 / kappa  *    u*ufl.dot(MagPot, ufl.grad(phi))*ufl.dx



def compute_inner_curl(MagPot1,MagPot2):
    return ufl.inner( ufl.rot(MagPot1) , ufl.rot(MagPot2)) *ufl.dx


def compute_inner_div(MagPot1,MagPot2):
    return ufl.inner( ufl.div(MagPot1) , ufl.div(MagPot2)) *ufl.dx

def compute_magn_energy(MagPot,H):
    return ufl.inner( ufl.rot(MagPot) - H , ufl.rot(MagPot) - H)  *ufl.dx



# (i/kappa nabla u + Au)(B phi)^*
def compute_i_nabla_pA_times_B_phi(u_real,u_imag, phi_real, phi_imag, MagPot, A_proj, kappa):
    return 1/kappa * ufl.inner(ufl.grad(u_real), A_proj) * phi_imag * ufl.dx \
            - 1/kappa * ufl.inner(ufl.grad(u_imag), A_proj) * phi_real * ufl.dx  \
                + ufl.inner(MagPot, A_proj) * ( u_real * phi_real + u_imag * phi_imag ) *ufl.dx


# (i/kappa nabla u)(B phi)^*
def compute_i_phi_star_nabla_times_B_phi(u_real,u_imag, phi_real, phi_imag, A_proj,kappa):
    return 1/kappa * ufl.inner(ufl.grad(u_real), A_proj) * phi_imag * ufl.dx \
            - 1/kappa * ufl.inner(ufl.grad(u_imag), A_proj) * phi_real * ufl.dx  \





#####################################
    

def compute_bila_ord(u_real, u_imag , phi_real ,phi_imag,  MagPot,kappa):
    
    bila =       compute_grad_bil(u_real,phi_real, MagPot,kappa)   \
                + compute_grad_bil(u_imag,phi_imag, MagPot,kappa) \
                \
                + compute_A_L2_bil(u_real,phi_real, MagPot,kappa) \
                + compute_A_L2_bil(u_imag,phi_imag, MagPot,kappa) \
                 \
                + compute_u_A_nabla_phi(phi_imag, u_real, MagPot,kappa)  \
                - compute_u_A_nabla_phi(u_real,phi_imag, MagPot,kappa)   \
                +  compute_u_A_nabla_phi(u_imag,phi_real, MagPot,kappa) \
                -  compute_u_A_nabla_phi(phi_real , u_imag, MagPot,kappa)

    return bila

'''
computes a_A(u,phi) + stab (u,phi_L^2s)
'''
def compute_bila_ord_stab(u_real, u_imag , phi_real ,phi_imag,  MagPot,kappa,StabPar):
    
    if StabPar == 0.0:
        bila =   compute_bila_ord(u_real, u_imag , phi_real ,phi_imag,  MagPot,kappa) 
    else:
        bila =   compute_bila_ord(u_real, u_imag , phi_real ,phi_imag,  MagPot,kappa) \
           + StabPar**2 * (u_real*phi_real + u_imag * phi_imag ) *ufl.dx

    return bila



def compute_linear_energy_ord(u_real, u_imag ,MagPot,kappa,comm):

    return 0.5*my_scalar_assemble(compute_bila_ord(u_real, u_imag , u_real ,u_imag,  MagPot,kappa), comm)




def compute_energy_ord(u_real, u_imag ,MagPot,kappa,comm):

    energy1 =   0.25* (1 - u_real**2 -u_imag**2)**2  *ufl.dx
    
    energy11 = my_scalar_assemble(energy1,comm)
    
    
    energy2 = compute_linear_energy_ord(u_real, u_imag ,MagPot,kappa,comm)

    return energy11 + energy2 


def compute_energy_magn_part(MagPot,H,inc_div,comm):
    if inc_div:
        energy1 =  0.5 * compute_magn_energy(MagPot,H) + 0.5* compute_inner_div(MagPot, MagPot)
    else:
        energy1 =  0.5 * compute_magn_energy(MagPot,H) 

    return my_scalar_assemble(energy1,comm)

def compute_energy_full(u_real, u_imag ,MagPot,H,kappa,inc_div,comm):

    return compute_energy_magn_part(MagPot,H,inc_div,comm) + compute_energy_ord(u_real, u_imag ,MagPot,kappa,comm) 




######################################################################################
# E'(u,A)
######################################################################################


def compute_energy_prime_ord(u_real, u_imag , phi_real ,phi_imag,  MagPot,kappa):
    
    comp1 = compute_bila_ord(u_real, u_imag , phi_real ,phi_imag,  MagPot,kappa)
    
    comp2 = (u_real*u_real + u_imag*u_imag - 1) * (u_real*phi_real + u_imag*phi_imag) *ufl.dx
    
    return comp1 + comp2

def compute_energy_prime_ord_with_split(u_full , phi_real ,phi_imag,  MagPot,kappa):
    
    u_real, u_imag   = u_full.split()

    comp1 = compute_bila_ord(u_real, u_imag , phi_real ,phi_imag,  MagPot,kappa)
    
    comp2 = (u_real*u_real + u_imag*u_imag - 1) * (u_real*phi_real + u_imag*phi_imag) *ufl.dx
    
    return comp1 + comp2



def compute_energy_prime_mag(u_real, u_imag , MagPot, H, B, kappa,inc_div):
    
    comp1 =( (u_real*u_real + u_imag*u_imag) * ufl.inner(MagPot, B)  + ufl.inner(ufl.rot(MagPot) - H, ufl.rot(B))   ) *ufl.dx 
    
    if inc_div:
        comp1 = comp1 +  ufl.inner(ufl.div(MagPot), ufl.div(B)) *ufl.dx
    
    comp2 = 1/kappa *  ( u_imag * ufl.inner( ufl.grad(u_real), B )  - u_real * ufl.inner( ufl.grad(u_imag), B )  ) *ufl.dx
    
    return comp1 + comp2




def compute_energy_prime(u_real, u_imag , phi_real ,phi_imag,  MagPot, H , B , kappa,inc_div):
    
    comp1 = compute_energy_prime_ord(u_real, u_imag , phi_real ,phi_imag,  MagPot,kappa)
    
    comp2 = compute_energy_prime_mag(u_real, u_imag , MagPot, H, B, kappa,inc_div)
    
    return comp1 + comp2






def compute_energy_prime_components(u_real, u_imag, MagPot,H,problem_dict, spaces_config_dict, FEM_spaces_dict):
    
    
    kappa = problem_dict['kappa']
    only_ord = problem_dict['only_ord']

    inc_div = spaces_config_dict['inc_div']
    Fem_type_mag = spaces_config_dict['Fem_type_mag']

    V_ord = FEM_spaces_dict['V_ord']
    V_mag_col = FEM_spaces_dict['V_mag_col']
    bc_V_mag = FEM_spaces_dict['bc_V_mag']



    tau = 1.0

    
    (u1,u2) = ufl.TrialFunctions(V_ord)
    (phi_real,phi_imag) = ufl.TestFunctions(V_ord)

    #   I + E'(...)
    
    energy_prime_ord = u1*phi_real*ufl.dx + u2*phi_imag*ufl.dx  \
        + tau * (u_real*u_real + u_imag*u_imag - 1) * (u1*phi_real + u2*phi_imag) *ufl.dx \
        + tau * compute_bila_ord(u1, u2 , phi_real ,phi_imag,  MagPot,kappa)
    
    rhs_ord = u_real*phi_real*ufl.dx + u_imag*phi_imag*ufl.dx


    #set uo and solve var problem for u
    var_problem_ord = dfpet.LinearProblem(energy_prime_ord, rhs_ord)
    u_new = var_problem_ord.solve()
    (u_real_new, u_imag_new) = u_new.split()           




    ##############################
    if not only_ord:
        A = ufl.TrialFunction(V_mag_col)
        B = ufl.TestFunction(V_mag_col)
        
        if inc_div:
            energy_prime_mag = (1.0 + tau * (u_real*u_real + u_imag*u_imag) )* ufl.inner(A, B) * ufl.dx \
                                + tau * ufl.inner(ufl.rot(A), ufl.rot(B)) *ufl.dx  \
                                + tau * ufl.inner(ufl.div(A), ufl.div(B)) *ufl.dx
        else:
            energy_prime_mag = (1.0 + tau * (u_real*u_real + u_imag*u_imag) )* ufl.inner(A, B) * ufl.dx \
                                + tau * ufl.inner(ufl.rot(A), ufl.rot(B)) *ufl.dx
                                
    
    
        rhs_mag = ufl.inner(MagPot, B) *ufl.dx + tau * ufl.inner(H, ufl.rot(B)) *ufl.dx  \
                        - (tau/kappa) * (u_imag * ufl.inner(ufl.grad(u_real), B) - u_real * ufl.inner(ufl.grad(u_imag), B) ) *ufl.dx
    
    
        if Fem_type_mag == 'Lagrange':
            var_problem_mag = dfpet.LinearProblem(energy_prime_mag, rhs_mag, bcs = bc_V_mag)
        else:
            var_problem_mag = dfpet.LinearProblem(energy_prime_mag, rhs_mag)
    
        A_new = var_problem_mag.solve()
    else:
        A_new = MagPot


    return u_real_new.collapse() , u_imag_new.collapse()  , A_new, u_real_new - u_real , u_imag_new - u_imag, A_new - MagPot






def get_a_proj_u(u1, u2 , phi_real ,phi_imag,  MagPot,kappa,stabPar,u_real, u_imag, bila_type):
    '''
    computes a_proj = a_{A,stab}(u,phi) + (|u_n|^2-1) *(u*phi^*)  /  u = u1 + i*u2, u_n =  u_real + i*u_imag
    without the -1 (v1) / nonlinearity (pure_linear)
    used for \nabla_X E by solving for 8u1,u2) where u_real, u_imag given from last iteration
    '''
    if bila_type == 'pure_linear':
        a_proj =  compute_bila_ord_stab(u1, u2 , phi_real ,phi_imag,  MagPot,kappa,stabPar)
    
    if bila_type == 'nonlinear_u_v1':
        a_proj =  compute_bila_ord_stab(u1, u2 , phi_real ,phi_imag,  MagPot,kappa,stabPar) \
                    + (u_real*u_real + u_imag*u_imag )*  (u1*phi_real + u2*phi_imag)  * ufl.dx
  
    if bila_type == 'nonlinear_u_v2':
        a_proj =  compute_bila_ord_stab(u1, u2 , phi_real ,phi_imag,  MagPot,kappa,stabPar) \
                    + (u_real*u_real + u_imag*u_imag - 1.0 )*  (u1*phi_real + u2*phi_imag)  * ufl.dx
        
    if bila_type == 'nonlinear_u_v3':
        delta = 0.5
        a_proj =  compute_bila_ord_stab(u1, u2 , phi_real ,phi_imag,  MagPot,kappa,stabPar) \
                    + (u_real*u_real + u_imag*u_imag + delta +  ufl.inner(MagPot,MagPot) )*  (u1*phi_real + u2*phi_imag)  * ufl.dx

    return a_proj

def get_a_proj_A(u_real, u_imag , A ,B, inc_div,bila_type):
    # 1. a(nabla E(u)),phi)
    if bila_type == 'pure_linear':
        if inc_div:
            a_proj_A = ( u_real * u_real + u_imag * u_imag)*ufl.inner(A, B) *ufl.dx \
                        + ufl.inner(ufl.rot(A), ufl.rot(B)) *ufl.dx \
                        + ufl.inner(ufl.div(A), ufl.div(B)) *ufl.dx
        else:
            a_proj_A = ( u_real * u_real + u_imag * u_imag)*ufl.inner(A, B) *ufl.dx \
                        + ufl.inner(ufl.rot(A), ufl.rot(B)) *ufl.dx
    # 
        return a_proj_A

'''
\nabla_X \partial_u E needed for the Sobolev gradient flow
'''
def compute_Ritz_for_nabla_E_ord(u_real, u_imag,MagPot,problem_dict,FEM_spaces_dict, stabPar,solver_setup):
    
    kappa = problem_dict['kappa']

    V_ord = FEM_spaces_dict['V_ord']


    (u1,u2) = ufl.TrialFunctions(V_ord) 
    (phi_real,phi_imag) = ufl.TestFunctions(V_ord) 

    bila_type = 'pure_linear'
    # bila_type = 'nonlinear_u_v1'
    bila_type = 'nonlinear_u_v2'
    bila_type = 'nonlinear_u_v3'




    # compute \nabla_X 
    
    # 1. a(nabla E(u)),phi)
    a_proj = get_a_proj_u(u1, u2 , phi_real ,phi_imag,  MagPot,kappa,stabPar,u_real, u_imag, bila_type)

    # (E'(u),phi)
    rhs_proj = compute_energy_prime_ord(u_real, u_imag , phi_real ,phi_imag,  MagPot,kappa)


    if dfx.__version__ == '0.10.0':
        key_args = {'petsc_options_prefix':petsc_options_prefix, 'petsc_options':solver_setup}
    elif dfx.__version__ == '0.9.0':
        key_args = {'petsc_options':solver_setup}

    var_problem_proj = dfpet.LinearProblem(a_proj, rhs_proj,**key_args)

    
    
    u_proj = var_problem_proj.solve()
    (u_proj_real, u_proj_imag) = u_proj.split() 
  


    return u_proj_real ,u_proj_imag, bila_type


'''
\nabla_X \partial_A E needed for the Sobolev gradient flow
'''
def compute_Ritz_for_nabla_E_mag(u_real, u_imag,MagPot,H, problem_dict,FEM_spaces_dict,spaces_config_dict,solver_setup):

    kappa = problem_dict['kappa']
    V_ord = FEM_spaces_dict['V_ord']
    V_mag_col = FEM_spaces_dict['V_mag_col']
    bc_V_mag = FEM_spaces_dict['bc_V_mag']
    inc_div = spaces_config_dict['inc_div']

    Fem_type_mag = spaces_config_dict['Fem_type_mag']

    A = ufl.TrialFunction(V_mag_col)
    B = ufl.TestFunction(V_mag_col)

    
    (u1,u2) = ufl.TrialFunctions(V_ord)
    (phi_real,phi_imag) = ufl.TestFunctions(V_ord)

    bila_type = 'pure_linear'
    # bila_type = 'nonlinear_u_v1'
    # bila_type = 'nonlinear_u_v2'



    # compute \nabla_X 
    bila_type = 'pure_linear'

    a_proj = get_a_proj_A(u_real, u_imag , A ,B, inc_div, bila_type )

                        
    # (E'(A),B)
    rhs_proj =   compute_energy_prime_mag(u_real, u_imag , MagPot, H, B, kappa,inc_div)


    if dfx.__version__ == '0.10.0':
        key_args = {'petsc_options_prefix':petsc_options_prefix, 'petsc_options':solver_setup}
    elif dfx.__version__ == '0.9.0':
        key_args = {'petsc_options':solver_setup}

    if Fem_type_mag == 'Lagrange':
        var_problem_proj = dfpet.LinearProblem(a_proj, rhs_proj, bcs = bc_V_mag, **key_args)
    else:
        var_problem_proj = dfpet.LinearProblem(a_proj, rhs_proj, **key_args)

    A_proj = var_problem_proj.solve()
    
    return A_proj 



def get_line_search_coeff_ord(u_real, u_imag, u_proj_real, u_proj_imag, MagPot,kappa):
    
    a0_form = compute_bila_ord(u_real, u_imag, u_proj_real,u_proj_imag,MagPot,kappa)  \
                + ( u_real * u_real + u_imag * u_imag - 1.0  ) *( u_real * u_proj_real + u_imag * u_proj_imag   ) *ufl.dx
    
    
    a1_form = 1.0 * ( u_real * u_real + u_imag * u_imag - 1.0  ) * ( u_proj_real * u_proj_real + u_proj_imag * u_proj_imag ) * ufl.dx \
                + 2.0 * ( u_real * u_proj_real + u_imag * u_proj_imag   )**2 * ufl.dx \
                + 1.0 * compute_bila_ord(u_proj_real,u_proj_imag ,u_proj_real,u_proj_imag ,  MagPot,kappa) 
    
    a2_form = 3.0 * ( u_proj_real * u_proj_real + u_proj_imag * u_proj_imag) *(u_real * u_proj_real + u_imag * u_proj_imag     )  *ufl.dx
    
    a3_form = 1.0 * ( u_proj_real * u_proj_real + u_proj_imag * u_proj_imag)**2 *ufl.dx
    
    
    return a0_form , a1_form, a2_form , a3_form 



def get_line_search_coeff_full_1(u_real, u_imag, u_proj_real, u_proj_imag, MagPot,A_proj,kappa):
    
   
    b1_form = compute_i_nabla_pA_times_B_phi(u_real, u_imag, u_proj_real, u_proj_imag, MagPot,A_proj,kappa) \
                + compute_i_nabla_pA_times_B_phi(u_proj_real, u_proj_imag, u_real, u_imag, MagPot,A_proj,kappa)
    
    b2_form = ufl.inner(A_proj,A_proj) * ( u_real * u_proj_real + u_imag * u_proj_imag )  * ufl.dx \
                +   2.0* compute_i_nabla_pA_times_B_phi(u_proj_real, u_proj_imag, u_proj_real, u_proj_imag, MagPot,A_proj,kappa)

    
    b3_form = 1.0 * ufl.inner(A_proj,A_proj)* ( u_proj_real * u_proj_real + u_proj_imag * u_proj_imag) *ufl.dx
    
    
    return b1_form, b2_form , b3_form 


def get_line_search_coeff_full_2(u_real, u_imag, u_proj_real, u_proj_imag, MagPot,A_proj,H,kappa,inc_div):
    
    c0_form =  (u_real * u_real + u_imag * u_imag)* ufl.inner(MagPot,A_proj) *ufl.dx \
                + ufl.inner(ufl.rot(MagPot), ufl.rot(A_proj)) * ufl.dx  - ufl.inner(H, ufl.rot(A_proj)) *ufl.dx \
                +  compute_i_phi_star_nabla_times_B_phi(u_real, u_imag, u_real, u_imag, A_proj,kappa) 
    if inc_div:
        c0_form = c0_form + ufl.inner(ufl.div(MagPot), ufl.div(A_proj)) *ufl.dx
    
    c1_form = (u_real * u_real + u_imag * u_imag)* ufl.inner(A_proj,A_proj) *ufl.dx  \
        + 2.0 * (u_real*u_proj_real+u_imag*u_proj_imag) * ufl.inner(MagPot,A_proj) * ufl.dx   \
        +   compute_i_phi_star_nabla_times_B_phi(u_real, u_imag, u_proj_real, u_proj_imag, A_proj,kappa) \
                + compute_i_phi_star_nabla_times_B_phi(u_proj_real, u_proj_imag, u_real, u_imag, A_proj,kappa) \
                + ufl.inner(ufl.rot(A_proj),ufl.rot(A_proj)) *ufl.dx
    if inc_div:
        c1_form = c1_form +   ufl.inner(ufl.div(A_proj),ufl.div(A_proj)) *ufl.dx
        
        
    c2_form = ufl.inner(MagPot,A_proj) * ( u_proj_real * u_proj_real + u_proj_imag * u_proj_imag )  * ufl.dx \
                + 2.0 * (u_real*u_proj_real+u_imag*u_proj_imag) * ufl.inner(A_proj,A_proj) * ufl.dx \
                +   compute_i_phi_star_nabla_times_B_phi(u_proj_real, u_proj_imag, u_proj_real, u_proj_imag, A_proj,kappa)

    
    c3_form = 1.0 * ufl.inner(A_proj,A_proj)* ( u_proj_real * u_proj_real + u_proj_imag * u_proj_imag) *ufl.dx
    
    
    return c0_form, c1_form, c2_form , c3_form 



def compute_Sobolev_grad_flow(u_real, u_imag, MagPot,H,problem_dict, FEM_spaces_dict,spaces_config_dict,minimizer_dict,stabPar,\
                              u_real_old = None, u_imag_old = None, u_proj_real_old =None, u_proj_imag_old=None, A_proj_old= None, comm = None, logger = None):    
     

    gradient_too_small = False


    only_ord = problem_dict['only_ord']
    kappa = problem_dict['kappa']

    inc_div = spaces_config_dict['inc_div']

    tau = minimizer_dict['tau']
    line_search = minimizer_dict['line_search']
    conjugate = minimizer_dict['conjugate']
    V_ord_real_col  =FEM_spaces_dict['V_ord_real_col']
    comm = V_ord_real_col.mesh.comm
    rank = comm.Get_rank()

    #  solver_setup={"ksp_type": "cg", "pc_type": "none", "ksp_rtol": 1e-6, "ksp_atol": 1e-10, "ksp_max_it": 1000} 
    # solver_setup={"ksp_type": "cg", "pc_type": "gamg", "ksp_rtol": 1e-6, "ksp_atol": 1e-10, "ksp_max_it": 1000} 
    solver_setup={"ksp_type": "preonly", "pc_type": "lu"}#, "ksp_rtol": 1e-6, "ksp_atol": 1e-10, "ksp_max_it": 1000}

    u_proj_real, u_proj_imag, bila_type  = compute_Ritz_for_nabla_E_ord(u_real, u_imag,MagPot,problem_dict,FEM_spaces_dict, stabPar,solver_setup)


    norm_u_proj_real_grad = np.sqrt(my_scalar_assemble(dfx.fem.form(ufl.inner(u_proj_real,u_proj_real)*ufl.dx),comm))
    norm_u_proj_imag_grad = np.sqrt(my_scalar_assemble(dfx.fem.form(ufl.inner(u_proj_imag,u_proj_imag)*ufl.dx),comm))   
    norm_u_proj_grad = np.sqrt(norm_u_proj_real_grad**2 + norm_u_proj_imag_grad**2 )


    if rank == 0:
        logger.info(f'norm_u_proj_real_grad =  {norm_u_proj_real_grad}')        
        logger.info(f'norm_u_proj_imag_grad =  {norm_u_proj_imag_grad}')      
        logger.info(f'norm_u_proj_grad =  {norm_u_proj_grad}')        
        logger.info(f'norm_u_proj_grad**4 =  {norm_u_proj_grad**4}')  


    if conjugate and u_proj_real_old:

        #compute Polak-Ribiere parameter beta_PR
        beta_PR  = my_scalar_assemble(get_a_proj_u(u_proj_real, u_proj_imag , u_proj_real - u_proj_real_old ,u_proj_imag - u_proj_imag_old,  MagPot,kappa,stabPar,u_real, u_imag, bila_type),comm)
        beta_PR2 = my_scalar_assemble(get_a_proj_u(u_proj_real_old, u_proj_imag_old , u_proj_real_old , u_proj_imag_old,  MagPot,kappa,stabPar,u_real_old, u_imag_old, bila_type),comm)
        # print(f'beta_PR = {beta_PR}')
        # print(f'beta_PR2 = {beta_PR2}')
        beta_PR_ord_saved = beta_PR
        beta_PR2_ord_saved = beta_PR2

        # beta_PR += 1
        # logger.info(f'beta_PR = {beta_PR}')
        # logger.info(f'beta_PR_ord_saved = {beta_PR_ord_saved}')
        # logger.info(f'beta_PR = {beta_PR}')
        # logger.info(f'beta_PR2 = {beta_PR2}')


        if beta_PR < 0: # or beta_PR2 < 10e-10:
            beta_PR = 0
        else:   
            beta_PR = beta_PR / beta_PR2
        if rank == 0:
            logger.info(f'beta_PR_u_used = {beta_PR}')
            
        if only_ord:
            u_proj_real.x.array[:] +=  beta_PR * u_proj_real_old.x.array[:]
            u_proj_imag.x.array[:] +=  beta_PR * u_proj_imag_old.x.array[:]


    if not only_ord:
        A_proj = compute_Ritz_for_nabla_E_mag(u_real, u_imag,MagPot,H, problem_dict, FEM_spaces_dict,spaces_config_dict,solver_setup)
        norm_A_proj_grad = np.sqrt(my_scalar_assemble(dfx.fem.form(ufl.inner(A_proj ,A_proj )*ufl.dx),comm))
        if rank == 0:    
            logger.info(f'norm_A_proj_grad = {norm_A_proj_grad}'+'\n')   





        if conjugate and A_proj_old:
            logger.debug(f'are inside "if conjugate and A_proj_old" ')   
            bila_type_A = 'pure_linear'
           
            beta_PR_A  = beta_PR_ord_saved + my_scalar_assemble(get_a_proj_A(u_real,u_imag, A_proj ,A_proj - A_proj_old, inc_div, bila_type_A  ) ,comm )
            if rank == 0:
                logger.info(f'beta_PR_A = {beta_PR_A}')
           
            beta_PR2_A  = beta_PR2_ord_saved + my_scalar_assemble(get_a_proj_A(u_real_old, u_imag_old, A_proj_old ,A_proj_old, inc_div, bila_type_A  ) ,comm )
            if rank == 0:
                logger.info(f'beta_PR2_A = {beta_PR2_A}')
           
            if beta_PR_A < 0: #or beta_PR2_A < 10e-10:  
                beta_PR_A = 0
            else:   
                beta_PR_A = beta_PR_A / beta_PR2_A
                if rank == 0:
                    logger.info(f'beta_PR_A after division = {beta_PR_A}')

            if rank == 0:
                logger.info(f'beta_PR_A_used = {beta_PR_A}'+'\n')

            A_proj.x.array[:] +=  beta_PR_A * A_proj_old.x.array[:]
            u_proj_real.x.array[:] +=  beta_PR_A * u_proj_real_old.x.array[:]
            u_proj_imag.x.array[:] +=  beta_PR_A * u_proj_imag_old.x.array[:]


 
        norm_A_proj  = np.sqrt(my_scalar_assemble(dfx.fem.form(ufl.inner(A_proj ,A_proj )*ufl.dx),comm))
        if rank == 0:    
            logger.info(f'norm_A_proj = search direction before div cor = {norm_A_proj}')  

        _,_,A_proj = discrete_divergence.compute_divergence_free_cor(u_proj_real, u_proj_imag, V_ord_real_col.mesh, A_proj,problem_dict,spaces_config_dict, FEM_spaces_dict,cor_all=False,logger = logger)
        

        norm_A_proj  = np.sqrt(my_scalar_assemble(dfx.fem.form(ufl.inner(A_proj ,A_proj )*ufl.dx),comm))
        if rank == 0:    
            logger.info(f'norm_A_proj = search direction after div correction = {norm_A_proj}')     
            
             

    norm_u_proj_real = np.sqrt(my_scalar_assemble(dfx.fem.form(ufl.inner(u_proj_real,u_proj_real)*ufl.dx),comm))
    norm_u_proj_imag = np.sqrt(my_scalar_assemble(dfx.fem.form(ufl.inner(u_proj_imag,u_proj_imag)*ufl.dx),comm))       
    norm_u_proj = np.sqrt(norm_u_proj_real**2 + norm_u_proj_imag**2)    

    if norm_u_proj**4 < 1e-16:
        gradient_too_small = True

    if rank == 0:
        logger.info(f'norm_u_proj_real = search direction = {norm_u_proj_real}')        
        logger.info(f'norm_u_proj_imag = search direction = {norm_u_proj_imag}')        
        logger.info(f'norm_u_proj = search direction = {norm_u_proj}')        
        logger.info(f'norm_u_proj**4 = search direction = {norm_u_proj**4}'+'\n')        
        if not only_ord:
            logger.info(f'norm_A_proj**2 * norm_u_proj**2 = search direction = {norm_A_proj**2 * norm_u_proj**2}'+'\n')        



    if only_ord and line_search:   
        t0 = time.time()
        # u_n+1 = u_n - tau_n u_proj
        a0_form , a1_form, a2_form ,a3_form  = get_line_search_coeff_ord(u_real, u_imag, u_proj_real, u_proj_imag, MagPot,kappa)

    
        a0 = my_scalar_assemble(a0_form,comm)
        # print(f'a0')  

        a1 = my_scalar_assemble(a1_form,comm)
        # print(f'a1')  

        a2 = my_scalar_assemble(a2_form,comm)
        # print(f'a2')  

        a3 = my_scalar_assemble(a3_form,comm)
        # print(f'a3')  

        coeff = [-1.0*a3 ,a2 ,-1.0*a1 ,a0]
        if rank == 0:
            logger.info(f'coeff={coeff}')
        roots = np.roots(coeff) 
        
        imags = abs(roots.imag)
        reals = roots.real
        # print(f' roots = {roots}')
        r0_real , r1_real, r2_real = False, False, False
        
        # logger.info(f'imags = {imags}')

        if len(roots) == 0:
            tau = 0.0
        else:
            r0_real , r1_real, r2_real = False, False, False
                
            # print(f' imags = {imags}')
            
            
            
            if imags[0] < 10**-5:
                r0_real = True
            if imags[1] < 10**-5:
                r1_real = True
            if imags[2] < 10**-5:
                r2_real = True
                
            if r0_real and r1_real and r2_real:
                tau = max(reals[0],reals[1],reals[2])
    
            elif r0_real:
                tau = reals[0]
                
            elif r1_real:
                tau = reals[1]
                
            elif r2_real:
                tau = reals[2]

        if rank == 0:
            logger.info(f'tau = {tau}')
            #print(f'tau = {tau}')
        
        t1 = time.time()
       
        


    if not only_ord and line_search:   
  
        a0_form , a1_form, a2_form ,a3_form  = get_line_search_coeff_ord(u_real, u_imag, u_proj_real, u_proj_imag, MagPot,kappa)
        b1_form, b2_form ,b3_form  = get_line_search_coeff_full_1(u_real, u_imag, u_proj_real, u_proj_imag, MagPot,A_proj,kappa)
        c0_form , c1_form, c2_form ,c3_form  = get_line_search_coeff_full_2(u_real, u_imag, u_proj_real, u_proj_imag,\
                                                                            MagPot,A_proj,H,kappa,inc_div)

        a0 = my_scalar_assemble(a0_form + c0_form,comm)
        # print(f'a0')  

        a1 = my_scalar_assemble(a1_form + b1_form + c1_form,comm)
        # print(f'a1')  

        a2 = my_scalar_assemble(a2_form + b2_form + c2_form,comm)
      # print(f'a2')  

        a3 = my_scalar_assemble(a3_form+ b3_form + c3_form,comm)

        coeff = [-1.0*a3 ,a2 ,-1.0*a1 ,a0]
        if rank == 0:
            logger.info(f'coeff={coeff}')
            
        roots = np.roots(coeff) 
        
        imags = abs(roots.imag)
        reals = roots.real
        # print(f' roots = {roots}')
        r0_real , r1_real, r2_real = False, False, False
        
        # print(f' imags = {imags}')
        
        
        
        if imags[0] < 10**-5:
            r0_real = True
        if imags[1] < 10**-5:
            r1_real = True
        if imags[2] < 10**-5:
            r2_real = True
            
        if r0_real and r1_real and r2_real:
            tau = max(reals[0],reals[1],reals[2])

        elif r0_real:
            tau = reals[0]
            
        elif r1_real:
            tau = reals[1]
            
        elif r2_real:
            tau = reals[2]
        
        # t1 = time.time()
        # print(f'compt tau = {t1-t0}s')  
     # no line search, given tau
    if rank == 0:
        logger.info(f'tau = {tau}')
    


    u_real_old.x.array[:] = u_real.x.array[:]
    u_imag_old.x.array[:] = u_imag.x.array[:]
    if not only_ord:
        MagPot.x.array[:] = MagPot.x.array[:]


    u_real.x.array[:]  = u_real.x.array[:] - tau * u_proj_real.collapse().x.array[:]
    u_imag.x.array[:]  = u_imag.x.array[:] - tau * u_proj_imag.collapse().x.array[:]



    if only_ord:
         A_new = MagPot
    else:        
         MagPot.x.array[:] = MagPot.x.array[:] - tau * A_proj.x.array[:]
         
    if only_ord:  
        return  u_real , u_imag, u_proj_real, u_proj_imag, MagPot, u_real_old, u_imag_old, None

    else:
        return  u_real , u_imag, u_proj_real, u_proj_imag, MagPot, u_real_old, u_imag_old, A_proj





'''
Newtons method
'''

def compute_energy_Newton(u_real, u_imag, MagPot,H , problem_dict, FEM_spaces_dict,spaces_config_dict,minimizer_dict,comm = None, logger = None):
 
    only_ord = problem_dict['only_ord']
    kappa = problem_dict['kappa']

    V_ext = FEM_spaces_dict['V_ext']
    V = FEM_spaces_dict['V']
    V_ord = FEM_spaces_dict['V_ord']
    bc_V = FEM_spaces_dict['bc_V']

    comm = V.mesh.comm
    rank = comm.Get_rank()

    Fem_type_mag = spaces_config_dict['Fem_type_mag']
    inc_div = spaces_config_dict['inc_div']

    tau = minimizer_dict['tau']
    line_search = minimizer_dict['line_search']
    extended = minimizer_dict['Newton_extended']

    line_search = True
    # line_search = False

    tau = 1.0
    solver_type = 'default'
    solver_type = 'mod'

    # solver_setup={"ksp_type": "preonly", "pc_type": "lu", "ksp_rtol": 1e-6, "ksp_atol": 1e-10, "ksp_max_it": 1000}
    # solver_setup={"ksp_type": "cgls","pc_type": "none",  "ksp_rtol": 1e-6, "ksp_atol": 1e-10, "ksp_max_it": 1000}
    solver_setup={"ksp_type": "cg", "pc_type": "none", "ksp_rtol": 1e-6, "ksp_atol": 1e-10, "ksp_max_it": 1000} 
    
    # solver_setup={"ksp_type": "lsqr","pc_type": "none",  "ksp_rtol": 1e-6, "ksp_atol": 1e-10, "ksp_max_it": 1000}

    
    
    if only_ord:
        (phi_real,phi_imag) = ufl.TrialFunctions(V_ord)
        (h_real,h_imag) = ufl.TestFunctions(V_ord)

        a = compute_duu_energy(u_real, u_imag , phi_real ,phi_imag, h_real,h_imag, MagPot,kappa)                       
        L = -1.0 * compute_energy_prime_ord(u_real, u_imag , h_real , h_imag,  MagPot, kappa) 



        A = dfx.fem.assemble_matrix(dfx.fem.form(a))
        mat_A = A.to_scipy()
        
        s = int( V_ord.dofmap.index_map.size_local * V_ord.dofmap.index_map_bs  /2 )
        
        
        
        #########################################
        # V1: set Re <phi ,i u > = 0
        #########################################
        V_ones = V_ord
        test_real,test_imag = ufl.TestFunction(V_ones)
        ones =  dfx.fem.assemble_vector(dfx.fem.form((test_imag*u_real-test_real*u_imag)*ufl.dx)).array
        
        mat_A = scp.hstack((mat_A,ones[:,None]))

        z2 = np.zeros(2*s+1)
        z2[:2*s] =  ones

        mat_A = scp.vstack((mat_A,z2[None,:]))
        # print(mat_A.get_shape())
        #########################################   
       

        l = dfx.fem.assemble_vector(dfx.fem.form(L)).array
        l= np.append(l,np.zeros(1)) 
        
        # cg - method
        sol_ext, exit_code = ssla.cg(mat_A,l,atol=1e-10)
           

        if exit_code == 0:
            logger.info('    sucesfully converged Newton step')
        else:
            logger.info(f'    no convergence, code: {exit_code}')

        
        sol_ext_real = sol_ext[:s]
        sol_ext_imag = sol_ext[s:-1]
        
# 
        sol = fem.Function(V_ord)
        sol.x.array[:s]  = sol_ext_real
        sol.x.array[s:]  = sol_ext_imag

        (delta_u_real, delta_u_imag) = sol.split() 


        
        eig_check = True
        eig_check = False

        if eig_check:


            mass = dfx.fem.form( (phi_real * h_real + phi_imag * h_imag) *ufl.dx )

            M = dfx.fem.assemble_matrix(mass)
            mat_M = M.to_scipy()
            
            s = mat_M.get_shape()[0]
        
            z1 = np.zeros(s)
            mat_M = scp.hstack((mat_M,z1[:,None]))
    
            z2 = np.zeros(s+1)
            z2[-1] = mat_M.data[0]#1.0
            mat_M = scp.vstack((mat_M,z2[None,:]))
            print(f'm_00 = {mat_M.data[0]}')
            
            
            eigenvalues  = ssla.eigsh(mat_A, M= mat_M, sigma = 0.0 ,return_eigenvectors=False, k = 10 )
            print(f'eigenvalues = {eigenvalues}')
        

        if line_search:
            a0_form , a1_form, a2_form ,a3_form  = get_line_search_coeff_ord(u_real, u_imag, delta_u_real, delta_u_imag, MagPot,kappa)
    
        
            a0 = my_scalar_assemble(a0_form, comm)
            # print(f'a0 = {a0}')  
    
            a1 = my_scalar_assemble(a1_form, comm)
            # print(f'a1 = {a1}')  
    
            a2 = my_scalar_assemble(a2_form, comm)
            # print(f'a2= {a2}')  
    
            a3 = my_scalar_assemble(a3_form, comm)
            # print(f'a3= {a3}')  
    
            coeff = [a3 ,a2 ,a1 ,a0]
            roots = np.roots(coeff) 
            
            imags = abs(roots.imag)
            reals = roots.real
            # print(f' roots = {roots}')
            # print(f' reals = {reals}')

            if len(roots) == 0:
                tau = 0.0
            else:
                r0_real , r1_real, r2_real = False, False, False
                
                # print(f' imags = {imags}')
                
                
                
                if imags[0] < 10**-5:
                    r0_real = True
                if imags[1] < 10**-5:
                    r1_real = True
                if imags[2] < 10**-5:
                    r2_real = True
                    
                if r0_real and r1_real and r2_real:
                    tau = max(reals[0],reals[1],reals[2])
        
                elif r0_real:
                    tau = reals[0]
                    
                elif r1_real:
                    tau = reals[1]
                    
                elif r2_real:
                    tau = reals[2]
            
        if rank == 0:
            logger.info(f'tau = {tau}')
            #print(f'tau = {tau}')


        u_real.x.array[:]  = u_real.x.array[:] + tau *delta_u_real.collapse().x.array[:]
        u_imag.x.array[:]  = u_imag.x.array[:] + tau *delta_u_imag.collapse().x.array[:]
        # return  u_real , u_imag  , MagPot



    else:
        if not extended:
            (phi_real,phi_imag,B)= ufl.TrialFunctions(V)
    
            
            (h_real,h_imag,C)= ufl.TestFunctions(V)
            
        
    
        
            a = compute_duu_energy(u_real, u_imag , phi_real ,phi_imag, h_real,h_imag, MagPot,kappa) \
                                 + compute_dAA_energy(u_real, u_imag, MagPot, B , C, kappa,inc_div) \
                                 +compute_duA_energy(u_real, u_imag, phi_real ,phi_imag, MagPot,C, kappa ) \
                                 +compute_duA_energy(u_real, u_imag, h_real ,h_imag, MagPot,B, kappa )
                            
    
    
            L = -1.0 * compute_energy_prime(u_real, u_imag , h_real , h_imag,  MagPot, H, C , kappa,inc_div) 
    
    

            if dfx.__version__ == '0.10.0':
                key_args = {'petsc_options_prefix':petsc_options_prefix, 'petsc_options':solver_setup}
            elif dfx.__version__ == '0.9.0':
                key_args = {'petsc_options':solver_setup}
    
            
            if Fem_type_mag == 'Lagrange':
                var_problem_Newton = dfpet.LinearProblem( a, L, bcs = bc_V)
    
            else:
               if solver_type == 'default':
                   var_problem_Newton = dfpet.LinearProblem( a, L)
               else:
                   var_problem_Newton = dfpet.LinearProblem( a, L, **key_args) 

            eig_check = True
            eig_check = False

            if eig_check:
                   A = dfx.fem.assemble_matrix(dfx.fem.form(a))
                   mat_A = A.to_scipy()

                   mass = dfx.fem.form( (phi_real * h_real + phi_imag * h_imag + ufl.inner(B,C)  ) *ufl.dx )

                   M = dfx.fem.assemble_matrix(mass)
                   mat_M = M.to_scipy()
                   eigenvalues  = ssla.eigsh(mat_A, M= mat_M, sigma = 0.0 ,return_eigenvectors=False, k = 10 )
                   print(f'eigenvalues = {eigenvalues}')
            
            sol = var_problem_Newton.solve()   
            (delta_u_real, delta_u_imag,delta_A) = sol.split()
  
    
  
        else:            
            (phi_real,phi_imag,B,psi_trial)= ufl.TrialFunctions(V_ext)
    
           
            (h_real,h_imag,C,psi_test)= ufl.TestFunctions(V_ext)
            

    
    
        
            a = compute_duu_energy(u_real, u_imag , phi_real ,phi_imag, h_real,h_imag, MagPot,kappa) \
                                 + compute_dAA_energy(u_real, u_imag, MagPot, B , C, kappa,inc_div) \
                                 +compute_duA_energy(u_real, u_imag, phi_real ,phi_imag, MagPot,C, kappa ) \
                                 +compute_duA_energy(u_real, u_imag, h_real ,h_imag, MagPot,B, kappa ) \
                                     + ufl.inner( B,  ufl.grad(psi_test)  ) *ufl.dx \
                                     + ufl.inner( C, ufl.grad(psi_trial)  ) *ufl.dx     
                            
    
            eig_check = True
            eig_check = False

            if eig_check:
                   A = dfx.fem.assemble_matrix(dfx.fem.form(a))
                   mat_A = A.to_scipy()

                   mass = dfx.fem.form( (phi_real * h_real + phi_imag * h_imag + ufl.inner(B,C) + psi_test * psi_trial ) *ufl.dx )

                   M = dfx.fem.assemble_matrix(mass)
                   mat_M = M.to_scipy()
                   eigenvalues  = ssla.eigsh(mat_A, M= mat_M, sigma = 0.0 ,return_eigenvectors=False, k = 10 )
                   print(f'eigenvalues = {eigenvalues}')

    
            L = -1.0 * compute_energy_prime(u_real, u_imag , h_real , h_imag,  MagPot, H, C , kappa,inc_div) 
    
            
            if Fem_type_mag == 'Lagrange':
                var_problem_Newton = dfpet.LinearProblem( a, L, bcs = bc_V)
    
            else:
                print('    extended mode')  
            
            
            
                A = dfx.fem.assemble_matrix(dfx.fem.form(a))
                mat_A = A.to_scipy()
                
                s = int( V_ord.dofmap.index_map.size_local * V_ord.dofmap.index_map_bs  /2 )
                s2 = int(mat_A.get_shape()[0])
                
                
                
                
                #########################################
                # V1: set Re <phi ,i u > = 0
                #########################################
                V_ones = V_ord
                test_real,test_imag = ufl.TestFunction(V_ones)
                ones =  dfx.fem.assemble_vector(dfx.fem.form((test_imag*u_real-test_real*u_imag)*ufl.dx)).array
                
                
                z1 = np.zeros(s2)
                z1[0:2*s] = ones# np.ones(s)
                mat_A = scp.hstack((mat_A,z1[:,None]))
                
        
                z2 = np.zeros(s2+1)
                z2[:2*s] =  ones
        
                mat_A = scp.vstack((mat_A,z2[None,:]))
              

        
                l = dfx.fem.assemble_vector(dfx.fem.form(L)).array
                l= np.append(l,np.zeros(1)) 
              
                #direct solver
                sol_ext = ssla.spsolve(mat_A,l)
                
                
                sol_ext_real = sol_ext[:s]
                sol_ext_imag = sol_ext[s:-1]
                
              
                sol = fem.Function(V_ext)
                sol.x.array[:]  = sol_ext[:-1]
            
           
            (delta_u_real, delta_u_imag,delta_A,psi) = sol.split()



        if line_search:
            a0_form , a1_form, a2_form ,a3_form  = get_line_search_coeff_ord(u_real, u_imag, delta_u_real, delta_u_imag, MagPot,kappa)
    
            b1_form, b2_form ,b3_form  = get_line_search_coeff_full_1(u_real, u_imag, delta_u_real, delta_u_imag, MagPot,delta_A,kappa)
            c0_form , c1_form, c2_form ,c3_form  = get_line_search_coeff_full_2(u_real, u_imag, delta_u_real, delta_u_imag,\
                                                                                MagPot,delta_A,H,kappa,inc_div)
    
            a0 = my_scalar_assemble(a0_form + c0_form, comm)
            # print(f'a0')  
    
            a1 = my_scalar_assemble(a1_form + b1_form + c1_form, comm)
            # print(f'a1')  
    
            a2 = my_scalar_assemble(a2_form + b2_form + c2_form, comm)
          # print(f'a2')  
    
            a3 = my_scalar_assemble(a3_form+ b3_form + c3_form, comm)
    
            coeff = [a3 ,a2 , a1 ,a0]
            # print(f' coeff = {coeff}')

            
            roots = np.roots(coeff) 
            
            imags = abs(roots.imag)
            reals = roots.real
            # print(f' roots = {roots}')
            if len(roots) == 0:
                tau = 0.0
            else:
                r0_real , r1_real, r2_real = False, False, False
                
                # print(f' imags = {imags}')
                
                
                
                if imags[0] < 10**-5:
                    r0_real = True
                if imags[1] < 10**-5:
                    r1_real = True
                if imags[2] < 10**-5:
                    r2_real = True
                    
                if r0_real and r1_real and r2_real:
                    tau = max(reals[0],reals[1],reals[2])
        
                elif r0_real:
                    tau = reals[0]
                    
                elif r1_real:
                    tau = reals[1]
                    
                elif r2_real:
                    tau = reals[2]


        if rank == 0:
            logger.info(f'tau = {tau}')
            #print(f'tau = {tau}')
        u_real.x.array[:]  = u_real.x.array[:] + tau * delta_u_real.collapse().x.array[:]
        u_imag.x.array[:]  = u_imag.x.array[:] + tau* delta_u_imag.collapse().x.array[:]
        
        MagPot.x.array[:] = MagPot.x.array[:] + tau * delta_A.collapse().x.array[:]
       
      
    
       
    return  u_real , u_imag , MagPot



def compute_energy_Newton_nullspace(u_real, u_imag, MagPot,H , problem_dict, FEM_spaces_dict,spaces_config_dict,minimizer_dict,comm = None, logger = None):
 
    only_ord = problem_dict['only_ord']
    kappa = problem_dict['kappa']

    V_ext = FEM_spaces_dict['V_ext']
    V = FEM_spaces_dict['V']
    V_ord = FEM_spaces_dict['V_ord']
    V_ord_real = FEM_spaces_dict['V_ord_real']
    V_ord_real_col = FEM_spaces_dict['V_ord_real_col']

    bc_V = FEM_spaces_dict['bc_V']

    comm = V.mesh.comm
    rank = comm.Get_rank()

    Fem_type_mag = spaces_config_dict['Fem_type_mag']
    inc_div = spaces_config_dict['inc_div']

    tau = minimizer_dict['tau']
    line_search = minimizer_dict['line_search']
    extended = minimizer_dict['Newton_extended']

    line_search = True
    # line_search = False

    tau = 1.0
    
    petsc_options = {
            "ksp_error_if_not_converged": True,
            "ksp_type": "preonly",
            # "pc_type": "cholesky", # "lu" possible
            "pc_type": "lu", # "cholesky" possible
            "pc_factor_mat_solver_type": "mumps",
            "ksp_monitor": None,
        }
    
    # petsc_options = {
    #     "ksp_error_if_not_converged": True,
    #     "ksp_type": "cg",              # iterative solver
    #     "pc_type": "none",              # incomplete Cholesky preconditioner (for SPD)
    #     "ksp_rtol": 1e-4,              # relative tolerance
    #     "ksp_atol": 1e-4,              # absolute tolerance
    #     "ksp_monitor": None,           # optional: print convergence info
    #     }   

    ksp = PETSc.KSP().create(comm)
    ksp.setOptionsPrefix("singular_direct")
    opts = PETSc.Options()
    opts.prefixPush(ksp.getOptionsPrefix())
    for key, value in petsc_options.items():
        opts[key] = value
    ksp.setFromOptions()
    for key, value in petsc_options.items():
        del opts[key]
    opts.prefixPop()
    
    if only_ord:

        (phi_real,phi_imag) = ufl.TrialFunctions(V_ord)
        (h_real,h_imag) = ufl.TestFunctions(V_ord)

        a = compute_duu_energy(u_real, u_imag , phi_real ,phi_imag, h_real,h_imag, MagPot,kappa)                       
        L = -1.0 * compute_energy_prime_ord(u_real, u_imag , h_real , h_imag,  MagPot, kappa) 

        A = dfx.fem.petsc.assemble_matrix(dfx.fem.form(a))
        A.assemble()
        b = dfx.fem.petsc.assemble_vector(dfx.fem.form(L))
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        ksp.setOperators(A)


        null_vec = dfx.fem.Function(V_ord)
        index = null_vec.x.array[:].shape[0] // 2 
        null_vec.x.array[:index] = - u_imag.x.array[:]
        null_vec.x.array[index:] = u_real.x.array[:]

        null_vec.x.scatter_forward()
        dfx.la.orthonormalize([null_vec.x])

        nullspace = PETSc.NullSpace().create(vectors=[null_vec.x.petsc_vec])

        # assert nullspace.test(A)
        if rank == 0:
            logger.info(f'nullspace.test(A) = {nullspace.test(A)}')
        A.setNullSpace(nullspace)

        sol = dfx.fem.Function(V_ord)
        ksp.solve(b, sol.x.petsc_vec)
        sol.x.scatter_forward()
        ksp.destroy()
        
        # logger.info(f'sol.x.array = {sol.x.array}')
        (delta_u_real, delta_u_imag) = sol.split() 


        if line_search:
            a0_form , a1_form, a2_form ,a3_form  = get_line_search_coeff_ord(u_real, u_imag, delta_u_real, delta_u_imag, MagPot,kappa)
    
        
            a0 = my_scalar_assemble(a0_form, comm)
            # print(f'a0 = {a0}')  
    
            a1 = my_scalar_assemble(a1_form, comm)
            # print(f'a1 = {a1}')  
    
            a2 = my_scalar_assemble(a2_form, comm)
            # print(f'a2= {a2}')  
    
            a3 = my_scalar_assemble(a3_form, comm)
            # print(f'a3= {a3}')  
    
            coeff = [a3 ,a2 ,a1 ,a0]
            roots = np.roots(coeff) 
            
            imags = abs(roots.imag)
            reals = roots.real
            # print(f' roots = {roots}')
            # print(f' reals = {reals}')

            if len(roots) == 0:
                tau = 0.0
            else:
                r0_real , r1_real, r2_real = False, False, False
                
                # print(f' imags = {imags}')
                
                
                
                if imags[0] < 10**-5:
                    r0_real = True
                if imags[1] < 10**-5:
                    r1_real = True
                if imags[2] < 10**-5:
                    r2_real = True
                    
                if r0_real and r1_real and r2_real:
                    tau = max(reals[0],reals[1],reals[2])
        
                elif r0_real:
                    tau = reals[0]
                    
                elif r1_real:
                    tau = reals[1]
                    
                elif r2_real:
                    tau = reals[2]
            
        if rank == 0:
            logger.info(f'tau = {tau}')


        u_real.x.array[:]  += tau *delta_u_real.collapse().x.array[:]
        u_imag.x.array[:]  += tau *delta_u_imag.collapse().x.array[:]
   

    else:
        (phi_real,phi_imag,B,psi_trial)= ufl.TrialFunctions(V_ext)         
        (h_real,h_imag,C,psi_test)= ufl.TestFunctions(V_ext)

        a = compute_duu_energy(u_real, u_imag , phi_real ,phi_imag, h_real,h_imag, MagPot,kappa) \
                                 + compute_dAA_energy(u_real, u_imag, MagPot, B , C, kappa,inc_div) \
                                 +compute_duA_energy(u_real, u_imag, phi_real ,phi_imag, MagPot,C, kappa ) \
                                 +compute_duA_energy(u_real, u_imag, h_real ,h_imag, MagPot,B, kappa ) \
                                     + ufl.inner( B,  ufl.grad(psi_test)  ) *ufl.dx \
                                     + ufl.inner( C, ufl.grad(psi_trial)  ) *ufl.dx   
        L = -1.0 * compute_energy_prime(u_real, u_imag , h_real , h_imag,  MagPot, H, C , kappa,inc_div) 

        A = dfx.fem.petsc.assemble_matrix(dfx.fem.form(a))
        A.assemble()
        b = dfx.fem.petsc.assemble_vector(dfx.fem.form(L))
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        ksp.setOperators(A)


        null_vec = dfx.fem.Function(V_ext)
        test_ord = dfx.fem.Function(V_ord)
        index = test_ord.x.array[:].shape[0] // 2 


        null_vec.x.array[:index] = - u_imag.x.array[:]
        null_vec.x.array[index:2*index] = u_real.x.array[:]
 
       
        null_vec.x.scatter_forward()
        dfx.la.orthonormalize([null_vec.x])

        nullspace = PETSc.NullSpace().create(vectors=[null_vec.x.petsc_vec])

        # assert nullspace.test(A)
        A.setNullSpace(nullspace)

        sol = dfx.fem.Function(V_ext)
        ksp.solve(b, sol.x.petsc_vec)
        sol.x.scatter_forward()
        ksp.destroy()


        (delta_u_real, delta_u_imag,delta_A,psi) = sol.split()

        
        if line_search:
            a0_form , a1_form, a2_form ,a3_form  = get_line_search_coeff_ord(u_real, u_imag, delta_u_real, delta_u_imag, MagPot,kappa)
    
            b1_form, b2_form ,b3_form  = get_line_search_coeff_full_1(u_real, u_imag, delta_u_real, delta_u_imag, MagPot,delta_A,kappa)
            c0_form , c1_form, c2_form ,c3_form  = get_line_search_coeff_full_2(u_real, u_imag, delta_u_real, delta_u_imag,\
                                                                                MagPot,delta_A,H,kappa,inc_div)
    
            a0 = my_scalar_assemble(a0_form + c0_form, comm)
            # print(f'a0')  
    
            a1 = my_scalar_assemble(a1_form + b1_form + c1_form, comm)
            # print(f'a1')  
    
            a2 = my_scalar_assemble(a2_form + b2_form + c2_form, comm)
          # print(f'a2')  
    
            a3 = my_scalar_assemble(a3_form+ b3_form + c3_form, comm)
    
            coeff = [a3 ,a2 , a1 ,a0]
            # print(f' coeff = {coeff}')

            
            roots = np.roots(coeff) 
            
            imags = abs(roots.imag)
            reals = roots.real
            # print(f' roots = {roots}')
            if len(roots) == 0:
                tau = 0.0
            else:
                r0_real , r1_real, r2_real = False, False, False
                
                # print(f' imags = {imags}')
                
                
                
                if imags[0] < 10**-5:
                    r0_real = True
                if imags[1] < 10**-5:
                    r1_real = True
                if imags[2] < 10**-5:
                    r2_real = True
                    
                if r0_real and r1_real and r2_real:
                    tau = max(reals[0],reals[1],reals[2])
        
                elif r0_real:
                    tau = reals[0]
                    
                elif r1_real:
                    tau = reals[1]
                    
                elif r2_real:
                    tau = reals[2]

        if rank == 0:
            logger.info(f'tau = {tau}')
        
        u_real.x.array[:]  += tau * delta_u_real.collapse().x.array[:]
        u_imag.x.array[:]  += tau * delta_u_imag.collapse().x.array[:]
        MagPot.x.array[:]  += tau * delta_A.collapse().x.array[:]



    return  u_real , u_imag , MagPot, tau




 # leider nicht mögliche wegn singular matrix
def compute_energy_Newton_petsc(u_real, u_imag, MagPot,H , problem_dict, FEM_spaces_dict,spaces_config_dict,minimizer_dict,comm = None, logger = None):
 
    only_ord = problem_dict['only_ord']
    kappa = problem_dict['kappa']

    V_ext = FEM_spaces_dict['V_ext']
    V = FEM_spaces_dict['V']
    V_ord = FEM_spaces_dict['V_ord']
    V_ord_real = FEM_spaces_dict['V_ord_real']
    V_ord_real_col = FEM_spaces_dict['V_ord_real_col']

    bc_V = FEM_spaces_dict['bc_V']

    comm = V.mesh.comm
    rank = comm.Get_rank()

    Fem_type_mag = spaces_config_dict['Fem_type_mag']
    inc_div = spaces_config_dict['inc_div']

    tau = minimizer_dict['tau']
    line_search = minimizer_dict['line_search']
    extended = minimizer_dict['Newton_extended']

    line_search = True
    # line_search = False

    tau = 1.0
    

    if only_ord:

        (h_real,h_imag) = ufl.TestFunctions(V_ord)
        u_full = dfx.fem.Function(V_ord)
        index = u_full.x.array[:].shape[0] // 2 
        u_full.x.array[:index] = u_real.x.array[:]
        u_full.x.array[index:] = u_imag.x.array[:]
        F = compute_energy_prime_ord_with_split(u_full, h_real ,h_imag,  MagPot,kappa)

        petsc_options = {
            "snes_type": "newtonls",
            "snes_linesearch_type": "none",
            "snes_atol": 1e-1,
            "snes_rtol": 1e-1,
            "snes_monitor": None,
            "ksp_error_if_not_converged": True,
            "ksp_type": "gmres",
            "ksp_rtol": 1e-1,
            "ksp_monitor": None,
            # "pc_type": "hypre",
            # "pc_hypre_type": "boomeramg",
            # "pc_hypre_boomeramg_max_iter": 1,
            # "pc_hypre_boomeramg_cycle_type": "v",
        }

        nonlin_problem = dfx.fem.petsc.NonlinearProblem(
            F,
            u_full,
            petsc_options=petsc_options,
            petsc_options_prefix="energy_prime=0",
        )

        logger.info('bis hier gekommen')
        nonlin_problem.solve()
        logger.info('solver durchgelaufen')
        converged = nonlin_problem.solver.getConvergedReason()
        num_iter = nonlin_problem.solver.getIterationNumber()
        assert converged > 0, "Solver did not converge, got {converged}."
        print(
            f"Solver converged after {num_iter} iterations with converged reason {converged}."
        )

  
        (u_real, u_imag) = u_full.split() 



    return  u_real , u_imag , MagPot, tau




def compute_smallest_eigs(u_real, u_imag, MagPot,H , num_eigs, problem_dict, FEM_spaces_dict,spaces_config_dict,minimizer_dict, get_eigs = False, logger = None):

    only_ord = problem_dict['only_ord']
    kappa = problem_dict['kappa']

    V_ext = FEM_spaces_dict['V_ext']
    V = FEM_spaces_dict['V']
    V_ord = FEM_spaces_dict['V_ord']
    V_ord_real = FEM_spaces_dict['V_ord_real']
    V_ord_real_col = FEM_spaces_dict['V_ord_real_col']

    bc_V = FEM_spaces_dict['bc_V']

    comm = V.mesh.comm
    rank = comm.Get_rank()

    Fem_type_mag = spaces_config_dict['Fem_type_mag']
    inc_div = spaces_config_dict['inc_div']
    extended = minimizer_dict['Newton_extended']

    # if only_ord:

    (phi_real,phi_imag) = ufl.TrialFunctions(V_ord)
    (h_real,h_imag) = ufl.TestFunctions(V_ord)

    a = compute_duu_energy(u_real, u_imag , phi_real ,phi_imag, h_real,h_imag, MagPot,kappa)     
    m = dfx.fem.form( (phi_real * h_real + phi_imag * h_imag) *ufl.dx )

    A = dfx.fem.petsc.assemble_matrix(dfx.fem.form(a))
    A.assemble()
    A.setOption(PETSc.Mat.Option.HERMITIAN, True) 

    M = dfx.fem.petsc.assemble_matrix(dfx.fem.form(m))
    M.assemble()
    M.setOption(PETSc.Mat.Option.HERMITIAN, True) 


    eps = SLEPc.EPS().create(comm)
    eps.setOperators(A,M)
    eps.setProblemType(SLEPc.EPS.ProblemType.GHEP) # GHEP = gen. hermitian
    
    tol = 1e-5
    eps.setTolerances(tol=tol)
    eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    eps.setTarget(0.0) 
    eps.setDimensions(nev=num_eigs)

    # Add shift-and-invert
    st = eps.getST()
    st.setType(SLEPc.ST.Type.SINVERT)
    eps.solve()
    if get_eigs:
        u_eig = dfx.fem.Function(V_ord)
        eps.getEigenpair(1, u_eig.x.petsc_vec)
        eig_min = eps.getEigenvalue(1) 
        if rank == 0:
            logger.info(f'eig_min = {eig_min}')
            logger.info(f'u_eig = {u_eig.x.array}')

    # eps.view()
    # eps.getConverged()
    # eps.getIterationNumber()
    # eps.getConvergedReason()

    nconv = eps.getConverged()
    if nconv:
        if rank == 0 :
            logger.info(f'Number of converged eigenpairs: {nconv}')
            
        # eps.view()
        # eps.errorView()
        vals = [eps.getEigenvalue(i) for i in range(num_eigs)]
    
        if not get_eigs:
            return  vals
        else:
            return vals, u_eig
    else:
        logger.info(f'no convergence')
        if not get_eigs:
            return None
        else:
            return None, None



######################################################################################
# E''(u,A)[phi,h]
######################################################################################


def compute_duu_energy(u_real, u_imag , phi_real ,phi_imag, h_real,h_imag, MagPot,kappa):
    
    comp1 = compute_bila_ord(phi_real, phi_imag , h_real ,h_imag,  MagPot,kappa)
    
    comp2 = ( 2* u_real*u_real + 2* u_imag*u_imag - 1) * (phi_real * h_real + phi_imag * h_imag) *ufl.dx
    
    comp3 = ( u_real*u_real - u_imag*u_imag) * (phi_real * h_real - phi_imag * h_imag) *ufl.dx

    comp4 = 2* u_real * u_imag * ( phi_real * h_imag + phi_imag * h_real ) *ufl.dx
    
    return comp1 + comp2 + comp3 + comp4


def compute_dAA_energy(u_real, u_imag, MagPot, B , C, kappa,inc_div):
    
    comp1 =  (  u_real*u_real + u_imag*u_imag ) * ufl.inner(B, C) *ufl.dx 

    if inc_div:
        comp2 =  ufl.inner( ufl.rot(B) , ufl.rot(C) ) *ufl.dx  + ufl.inner( ufl.div(B) , ufl.div(C) ) *ufl.dx  
    else:
        comp2 =  ufl.inner( ufl.rot(B) , ufl.rot(C) ) *ufl.dx  

    return comp1 + comp2



def compute_duA_energy(u_real, u_imag, phi_real ,phi_imag, MagPot, B, kappa ):
    
    comp1 = 2* (  u_real*phi_real + u_imag*phi_imag ) * ufl.inner(MagPot, B) *ufl.dx
    
    comp2 = 1/kappa * (u_imag * ufl.inner(ufl.grad(phi_real), B) - u_real * ufl.inner(ufl.grad(phi_imag), B) ) * ufl.dx
    
    comp3 = 1/kappa * (phi_imag * ufl.inner(ufl.grad(u_real), B) - phi_real * ufl.inner(ufl.grad(u_imag), B) ) * ufl.dx
    
    
    return comp1 + comp2 + comp3 



def compute_d2_total_energy(u_real, u_imag , phi_real ,phi_imag, h_real,h_imag, MagPot , B , C , kappa,inc_div):

    
    comp1 =  compute_duu_energy(u_real, u_imag , phi_real ,phi_imag, h_real,h_imag, MagPot,kappa)
    
    comp2 =  compute_duA_energy(u_real, u_imag, phi_real ,phi_imag, MagPot, B, kappa )
    
    comp3 =  compute_duA_energy(u_real, u_imag, h_real ,h_imag, MagPot, C, kappa )
    
    comp4 =  compute_dAA_energy(u_real, u_imag, MagPot, B , C, kappa,inc_div)


    return comp1 + comp2 + comp3 + comp4




def compute_E_prime_prime_matrix(u_real, u_imag, MagPot, problem_dict,FEM_spaces_dict,spaces_config_dict):

    
    only_ord = problem_dict['only_ord']
    kappa = problem_dict['kappa']

    V = FEM_spaces_dict['V']
    V_ord = FEM_spaces_dict['V_ord']

    inc_div = spaces_config_dict['inc_div']


  
    mat_A_uu =None
    mat_A_AA=None
    mat_A_uA=None
    mat_M_uu =None
    mat_M_AA=None
    mat_A_rot=None  
  
    if not only_ord:
        
        (phi_real,phi_imag,B) = ufl.TrialFunctions(V)
        

        (h_real,h_imag,C)= ufl.TestFunctions(V)

        
        a = dfx.fem.form( \
                         compute_duu_energy(u_real, u_imag , phi_real ,phi_imag, h_real,h_imag, MagPot,kappa) \
                             + compute_dAA_energy(u_real, u_imag, MagPot, B , C, kappa,inc_div) \
                             +compute_duA_energy(u_real, u_imag, phi_real ,phi_imag, MagPot,C, kappa ) \
                             +compute_duA_energy(u_real, u_imag, h_real ,h_imag, MagPot,B, kappa )
                         )
    
        A = dfx.fem.assemble_matrix(a)
        mat_A = A.to_scipy()

        mass = dfx.fem.form( (phi_real * h_real + phi_imag * h_imag + ufl.inner(B,C) ) *ufl.dx )

        M = dfx.fem.assemble_matrix(mass)
        mat_M = M.to_scipy()

        return mat_A, mat_M   ,mat_A_uu, mat_A_AA, mat_A_uA, mat_M_uu ,  mat_M_AA, mat_A_rot
 
    
    else:
      
        phi_real,phi_imag = ufl.TrialFunction(V_ord)
        h_real,h_imag =  ufl.TestFunction(V_ord)

    
    
        a = dfx.fem.form( compute_duu_energy(u_real, u_imag , phi_real ,phi_imag, h_real,h_imag, MagPot,kappa))
    
        A = dfx.fem.assemble_matrix(a)
        mat_A = A.to_scipy()

        mass = dfx.fem.form( (phi_real * h_real + phi_imag * h_imag ) *ufl.dx )

        M = dfx.fem.assemble_matrix(mass)
        mat_M = M.to_scipy()

        return mat_A, mat_M   ,mat_A_uu, mat_A_AA, mat_A_uA, mat_M_uu ,  mat_M_AA, mat_A_rot
        
        
        
   