import numpy as np
from mpi4py import MPI
import dolfinx as dfx
from dolfinx import mesh, fem, plot, io, default_scalar_type

import dolfinx.fem.petsc as dfpet
# from dolfinx.io import gmshio
import basix
import adios4dolfinx 

import ufl

import scipy.sparse as scp
import numpy as np
import scipy.sparse.linalg as ssla



import spaces_def as s_def
import GL_FEM_energies as GL_energy
import generate_mesh as genmesh

petsc_options_prefix="basic_linear_problem"
from petsc4py import PETSc



def compute_discrete_divergence(msh, A,spaces_config_dict,logger=None):
    Fem_type_mag = spaces_config_dict['Fem_type_mag']
    degree_FEM_mag = spaces_config_dict['degree_FEM_mag']
    inc_bc_div_h = spaces_config_dict['inc_bc_div_h']
         
    
    if Fem_type_mag == 'Nedelec':
        V = fem.functionspace(msh, ("Lagrange", 1))
    else:
        V = fem.functionspace(msh, ("Lagrange", degree_FEM_mag+1))
        
    comm = msh.comm

    uD = fem.Function(V)
    uD.interpolate(lambda x: 0.0*x[0])
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(msh.topology)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(uD, boundary_dofs)



    b = ufl.TrialFunction(V)
    phi = ufl.TestFunction(V)

    a = b * phi *ufl.dx
    if Fem_type_mag == 'Nedelec':
        L = -1.0 * ufl.inner(A, ufl.grad(phi)) * ufl.dx 
    else:
        L = -1.0 * ufl.inner(A, ufl.grad(phi)) * ufl.dx 



    solver_setup={"ksp_type": "preonly", "pc_type": "lu"}

    if dfx.__version__ == '0.10.0':
        key_args = {'petsc_options_prefix':petsc_options_prefix, 'petsc_options':solver_setup}
    elif dfx.__version__ == '0.9.0':
        key_args = {'petsc_options':solver_setup}

    if inc_bc_div_h:
        var_problem = dfpet.LinearProblem(a, L, bcs  = [bc], **key_args)
    else:
        var_problem = dfpet.LinearProblem(a, L, **key_args)

    logger.debug('before solve')
    div_A = var_problem.solve() 
    logger.debug('after solve'+'\n')


    size_div_A =  GL_energy.my_scalar_assemble(dfx.fem.form(ufl.inner(div_A,div_A)*ufl.dx),comm) 


    return div_A, np.sqrt(size_div_A) 




def compute_divergence_free_cor(u_real,u_imag,msh,A,problem_dict,spaces_config_dict, FEM_spaces_dict,cor_all=True,logger = None):
    
    kappa = problem_dict['kappa']

    Fem_type_mag = spaces_config_dict['Fem_type_mag']
    degree_FEM_mag = spaces_config_dict['degree_FEM_mag']
    inc_bc_div_h = spaces_config_dict['inc_bc_div_h']

    V = FEM_spaces_dict['V']
    V_ord_real_col = FEM_spaces_dict['V_ord_real_col']
    V_ord_imag_col = FEM_spaces_dict['V_ord_imag_col']
    V_mag_col = FEM_spaces_dict['V_mag_col']
    V_div = FEM_spaces_dict['V_div']

    # comm = msh.comm
    comm = V_div.mesh.comm
    rank = comm.Get_rank()
        
    if inc_bc_div_h:
        uD = fem.Function(V_div)
        uD.interpolate(lambda x: 0.0*x[0])
        tdim = msh.topology.dim
        fdim = tdim - 1
        msh.topology.create_connectivity(fdim, tdim)
        boundary_facets = mesh.exterior_facet_indices(msh.topology)
        boundary_dofs = fem.locate_dofs_topological(V_div, fdim, boundary_facets)
        bc = fem.dirichletbc(uD, boundary_dofs)
    
    b = ufl.TrialFunction(V_div)
    phi = ufl.TestFunction(V_div)
    
    
    a =  ufl.inner(ufl.grad(b) , ufl.grad(phi)) * ufl.dx 

    A_bil = dfx.fem.petsc.assemble_matrix(dfx.fem.form(a))
    A_bil.assemble()
    if Fem_type_mag == 'Nedelec':
        L =  ufl.inner(A, ufl.grad(phi)) * ufl.dx 
        
    else:
        # L =  -1.0*ufl.inner(ufl.div(A), phi) * ufl.dx 
        L =  ufl.inner(A, ufl.grad(phi)) * ufl.dx 
    
    b = dfx.fem.petsc.assemble_vector(dfx.fem.form(L))
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)


    ksp_iterative = PETSc.KSP().create(comm)
    ksp_iterative.setOptionsPrefix("singular_iterative")
    petsc_options = {
        "ksp_error_if_not_converged": True,
        "ksp_monitor": None,
        "ksp_type": "gmres",
        "pc_type": "hypre",
        "pc_hypre_type": "boomeramg",
        "pc_hypre_boomeramg_max_iter": 1,
        "pc_hypre_boomeramg_cycle_type": "v",
    }

    

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
    ksp.setTolerances(rtol=1e-12, atol=1e-12)

    nullspace = PETSc.NullSpace().create(constant=True, comm=comm)
    assert nullspace.test(A_bil)
    A_bil.setNearNullSpace(nullspace)
    ksp.setOperators(A_bil)
    nullspace.remove(b)

    psi = dfx.fem.Function(V_div)
    ksp.solve(b, psi.x.petsc_vec)
    reason = ksp.getConvergedReason()

    if rank == 0:
        if reason < 0:
            print("Solver diverged:", reason)
        else:
            print("Solver converged:", reason)

    psi.x.scatter_forward()
    ksp.destroy()
    norm_psi = np.sqrt(GL_energy.my_scalar_assemble(dfx.fem.form(ufl.inner(psi,psi)*ufl.dx),comm))
    if rank == 0 and logger:
        logger.info(f'norm_psi =  {norm_psi}')        


    corrector_grad = fem.Function(V_mag_col)
    A_new = fem.Function(V_mag_col)


    if dfx.__version__ == '0.10.0':
        corrector_grad.interpolate(fem.Expression(ufl.grad(psi), V_mag_col.element.interpolation_points))
    elif dfx.__version__ == '0.9.0':
        corrector_grad.interpolate(fem.Expression(ufl.grad(psi), V_mag_col.element.interpolation_points()))
    
    norm_grad_psi = np.sqrt(GL_energy.my_scalar_assemble(dfx.fem.form(ufl.inner(corrector_grad,corrector_grad)*ufl.dx),comm))
    if rank == 0 and logger:
        logger.info(f'norm_grad_psi =  {norm_grad_psi}')  



    A_new.x.array[:] = A.x.array[:] - corrector_grad.x.array[:]
    norm_A_new = np.sqrt(GL_energy.my_scalar_assemble(dfx.fem.form(ufl.inner(A_new,A_new)*ufl.dx),comm))
    if rank == 0 and logger:
        logger.info(f'norm_A_new =  {norm_A_new}')  

 


    u_real_new =  fem.Function(V_ord_real_col)
    u_imag_new = fem.Function(V_ord_imag_col)

    if cor_all:

        corrector_cos = fem.Function(V_ord_real_col)
        corrector_sin = fem.Function(V_ord_real_col)



        if dfx.__version__ == '0.10.0':
            cor_exp_cos = fem.Expression(ufl.cos(kappa*psi), V_ord_real_col.element.interpolation_points)
        elif dfx.__version__ == '0.9.0':
            cor_exp_cos = fem.Expression(ufl.cos(kappa*psi), V_ord_real_col.element.interpolation_points())

        corrector_cos.interpolate(cor_exp_cos) 
        

        if dfx.__version__ == '0.10.0':
            cor_exp_sin = fem.Expression(ufl.sin(kappa*psi), V_ord_real_col.element.interpolation_points)
        elif dfx.__version__ == '0.9.0':
            cor_exp_sin = fem.Expression(ufl.sin(kappa*psi), V_ord_real_col.element.interpolation_points())
            
        corrector_sin.interpolate(cor_exp_sin) 
        
        u_real_new.x.array[:] =       u_real.x.array[:]* corrector_cos.x.array[:]  +  1.0* u_imag.x.array[:]* corrector_sin.x.array[:]  
        u_imag_new.x.array[:] = -1.0* u_real.x.array[:]* corrector_sin.x.array[:]  +       u_imag.x.array[:]* corrector_cos.x.array[:] 



    return u_real_new, u_imag_new, A_new


