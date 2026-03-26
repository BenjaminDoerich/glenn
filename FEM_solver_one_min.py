import numpy as np
from mpi4py import MPI
import dolfinx as dfx
from dolfinx import mesh, fem, plot, io, default_scalar_type

import dolfinx.fem.petsc as dfpet

# if dfx.__version__ == '0.9.0':
#     import dolfinx.io as gmshio

import basix

import ufl

import scipy.sparse as scp
import numpy as np
# import scipy.sparse.linalg
import scipy.sparse.linalg as ssla
from pathlib import Path
import adios4dolfinx
import time



import GL_FEM_energies as GL_energy
import spaces_def as s_def
import discrete_divergence as discrete_divergence
import GL_FEM_initial_values as inits


###########################################  
import os 

newpath = "sol_screenshots/"
if not os.path.exists(newpath):
   os.makedirs(newpath)  


# ###########################################
def read_function(filename: Path, timestamp: float, degree_FEM_ord: int, degree_FEM_mag: int,only_ord: bool,comm):
    in_mesh = adios4dolfinx.read_mesh(filename, comm=comm,)
    # in_mesh = adios4dolfinx.read_mesh(filename, MPI.COMM_WORLD)
    W_cg = dfx.fem.functionspace(in_mesh, ('Lagrange', degree_FEM_ord))
    
    if not only_ord:
        element_dG = basix.ufl.element("DG", in_mesh.basix_cell(), 1,shape = (1,) )
        W_dg = fem.functionspace(in_mesh,element_dG)

        element_curl = basix.ufl.element(basix.ElementFamily.N1E, basix.CellType.triangle, degree =  degree_FEM_mag)
        W_Ned = fem.functionspace(in_mesh,element_curl)

    u_real = fem.Function(W_cg)
    u_imag = fem.Function(W_cg)

    if not only_ord:
        rot_A = fem.Function(W_dg)
        MagPot = fem.Function(W_Ned)


    adios4dolfinx.read_function(filename, u_real, time=timestamp, name = "u1")
    adios4dolfinx.read_function(filename, u_imag, time=timestamp, name = "u2")

    if not only_ord:
        adios4dolfinx.read_function(filename, MagPot, time=timestamp, name = "MagPot")
        # adios4dolfinx.read_function(filename, rot_A,  time=timestamp, name = "rot_A")


    if only_ord:
            return W_cg , in_mesh, u_real, u_imag
    else:
        return W_cg ,W_dg, W_Ned, in_mesh, u_real, u_imag , MagPot#, rot_A
    ###########################################



def prepare_initial_data(sol_Re, sol_Im, A_in , H,model_config,training_config, loading_config,FEM_solver_config,experiment_config,cf,FEM_spaces_dict,use_rand=False, logger=None):
    problem_dict = cf.problem_dict
    spaces_config_dict = cf.spaces_config_dict
    minimizer_dict = cf.minimizer_dict

    kappa = problem_dict['kappa']
    only_ord = problem_dict['only_ord']
    geo = problem_dict['geo']
    circ_rad = problem_dict['circ_rad']
    omega_by_pi_times_ten = problem_dict['omega_by_pi_times_ten']
    u_type =  problem_dict['u_type']
    mag_scale =  problem_dict['mag_scale']

    use_ref = spaces_config_dict['use_ref']
    has = spaces_config_dict['has']
    h_ref = spaces_config_dict['h_ref']
    kappa_ref = spaces_config_dict['kappa_ref']

    Fem_type_mag = spaces_config_dict['Fem_type_mag']
    degree_FEM_ord = spaces_config_dict['degree_FEM_ord']
    degree_FEM_mag = spaces_config_dict['degree_FEM_mag']

    inc_div = spaces_config_dict['inc_div']
    corr_div = spaces_config_dict['corr_div']
    if corr_div:
        div_label = 'zero_div' 
    else:
        div_label = 'no_zero_div'  
    
    tol = minimizer_dict['tol']
    it_num = minimizer_dict['it_num']
    grad_type = minimizer_dict['grad_type']
    Newton = minimizer_dict['Newton']
    tol_Newton = minimizer_dict['tol_Newton']
    step_Newton_max = minimizer_dict['step_Newton_max']
    mag_scale = problem_dict['mag_scale'] 
    H_type = problem_dict['H_type'] 
    A_type = problem_dict['A_type'] 
    u_type = problem_dict['u_type'] 

    V = FEM_spaces_dict['V']
    V_ord = FEM_spaces_dict['V_ord']
    V_ord_real = FEM_spaces_dict['V_ord_real']
    V_ord_real_col = FEM_spaces_dict['V_ord_real_col']
    V_ord_imag_col = FEM_spaces_dict['V_ord_imag_col']
    V_mag_col = FEM_spaces_dict['V_mag_col']
    # h = spaces_config_dict['has'] 
    h_ref = spaces_config_dict['h_ref']   
    comm = V.mesh.comm
    rank = comm.Get_rank()
    # size = comm.Get_size()

    ##################################

    msh = V.mesh
    if only_ord:
        energy_init = GL_energy.compute_energy_ord(sol_Re,sol_Im,A_in, kappa,comm)
    else:
        energy_init = GL_energy.compute_energy_full(sol_Re,sol_Im,A_in,H, kappa,inc_div,comm)

      
    u1  = fem.Function(V_ord_real_col)
    u2  = fem.Function(V_ord_imag_col) 
   
    if dfx.__version__ == '0.10.0':
        u1.interpolate(fem.Expression(sol_Re, V_ord_real.element.interpolation_points))   
        u2.interpolate(fem.Expression(sol_Im, V_ord_real.element.interpolation_points))
    
    elif dfx.__version__ == '0.9.0':
        u1.interpolate(fem.Expression(sol_Re, V_ord_real.element.interpolation_points()))   
        u2.interpolate(fem.Expression(sol_Im, V_ord_real.element.interpolation_points()))
    
    if not only_ord:
        A = fem.Function(V_mag_col)
        A.interpolate(fem.Expression(A_in, V_mag_col.element.interpolation_points))
        _,_,A = discrete_divergence.compute_divergence_free_cor(u1, u2, V_ord_real_col.mesh, A,problem_dict,spaces_config_dict, FEM_spaces_dict,cor_all=False,logger=logger)
        _,size = discrete_divergence.compute_discrete_divergence( V_ord_real_col.mesh, A,spaces_config_dict,logger=logger)
        logger.info(f"L2 von div A = {size}")
        

    '''
    using reference solution as initial value
    '''
    if use_ref and not FEM_solver_config['use_NN_initial_guess'] and not use_rand:
        filename_ref = f'{FEM_solver_config["output_path"]}/ord_{grad_type}_{geo}_{circ_rad}_init_{u_type}_{A_type}_{H_type}_kappa_{kappa}_mag_scale_{mag_scale}_h_{h_ref}_tol_{tol}_{Fem_type_mag}_{div_label}'
        
        if geo == "circle_slice":
            filename_ref = f'{FEM_solver_config["output_path"]}/ord_{grad_type}_{geo}_{circ_rad}_{omega_by_pi_times_ten}_init_{u_type}_{A_type}_{H_type}_kappa_{kappa}_mag_scale_{mag_scale}_h_{h_ref}_tol_{tol}_{Fem_type_mag}_{div_label}'


        if FEM_solver_config['use_NN_initial_guess']:
            NN_string ='_NN_used'
            filename_ref +=NN_string

        filename_ref +='.bp'  
        filename_ref = Path(filename_ref) 



        if only_ord:
            W_cg_ref , _ , u_real_ref, u_imag_ref =\
                            read_function(filename_ref,timestamp=0.0, degree_FEM_ord= degree_FEM_ord, degree_FEM_mag =degree_FEM_mag,only_ord=only_ord,comm = comm)
        else:
            # W_cg_ref , _ , W_Ned_ref, _, u_real_ref, u_imag_ref, =\
            W_cg_ref , _ , W_Ned_ref, _, u_real_ref, u_imag_ref, A_ref =\
                            read_function(filename_ref,timestamp=0.0, degree_FEM_ord= degree_FEM_ord, degree_FEM_mag =degree_FEM_mag,only_ord=only_ord,comm =comm)
        ########################
        # setup of interpolate_nonmatching provided by Tim Buchholz
        ########################
        num_cells_cg = V_ord_real_col.dofmap.list.shape[0]
        cells_cg = np.arange(num_cells_cg,dtype=np.int32)

        interpolation_data_cg = (
            dfx.fem.create_interpolation_data(
                V_ord_real_col,
                W_cg_ref,
                cells_cg,
            )
        )

        u1  = fem.Function(V_ord_real_col)
        u2  = fem.Function(V_ord_imag_col) 

         
        u1.interpolate_nonmatching(u_real_ref, cells_cg,interpolation_data_cg)  
        u2.interpolate_nonmatching(u_imag_ref, cells_cg,interpolation_data_cg)  
        if rank == 0:
            logger.info(f'used ref solution for u')

        if not only_ord:
            logger.info('trying interpolation on A')
            A = fem.Function(V_mag_col)

            num_cells_curl = V_mag_col.dofmap.list.shape[0]
            cells_curl = np.arange(num_cells_curl,dtype=np.int32)

            interpolation_data_curl = (
                dfx.fem.create_interpolation_data(
                    V_mag_col,
                    W_Ned_ref,
                    cells_curl,
                )
            )
            A.interpolate_nonmatching(A_ref, cells_curl,interpolation_data_curl)  

            logger.info('managed interpolation on A'+'\n')
            _ ,div_size_2 =  discrete_divergence.compute_discrete_divergence(msh, A,spaces_config_dict,logger=logger)

            if rank == 0 :
                logger.info(f' div_h A at start = {div_size_2}')
    
    
    
    
    '''
    using NN solution as initial value
    '''
    if FEM_solver_config["use_NN_initial_guess"] and not use_ref and not use_rand:
        logger.info('start NN interpolation')
        
        filename_NN = f'{FEM_solver_config["output_path"]}/{FEM_solver_config["results_filename"]}_init_{u_type}_{A_type}_{H_type}_kappa_{kappa}_h_{h_ref}_tol_{tol}'

        if geo == "circle_slice":
            filename_NN = f'{FEM_solver_config["output_path"]}/{FEM_solver_config["results_filename"]}_init_{u_type}_{A_type}_{H_type}_kappa_{kappa}_h_{h_ref}_tol_{tol}'


        NN_string ='_NN_interp'
        filename_NN +=NN_string

        filename_NN +='.bp'  
        filename_NN = Path(filename_NN)
        
        comm_NN = comm



        
        msh_NN = adios4dolfinx.read_mesh(filename_NN, comm_NN,'BP4' )
        FEM_spaces_dict = s_def.def_get_all_V(msh_NN,problem_dict,spaces_config_dict)

        u1_NN = fem.Function(FEM_spaces_dict['V_ord_real_col'])
        u2_NN = fem.Function(FEM_spaces_dict['V_ord_real_col'])

        adios4dolfinx.read_function(filename_NN, u1_NN, name="u1_prep")
        adios4dolfinx.read_function(filename_NN, u2_NN, name="u2_prep")
        
        if not only_ord:
            A_NN = fem.Function(FEM_spaces_dict['V_mag_col'])
            adios4dolfinx.read_function(filename_NN, A_NN, name="A_prep")
        else:
            A_NN = inits.get_A1(A_type, 0.5*mag_scale, ufl.SpatialCoordinate(msh_NN),omega_by_pi_times_ten)
        
        num_cells_cg = V_ord_real_col.dofmap.list.shape[0]
        cells_cg = np.arange(num_cells_cg,dtype=np.int32)

        interpolation_data_cg = (
            dfx.fem.create_interpolation_data(
                V_ord_real_col,
                FEM_spaces_dict['V_ord_real_col'],
                cells_cg,
            )
        )

        u1  = fem.Function(V_ord_real_col)
        u2  = fem.Function(V_ord_imag_col) 
         
        u1.interpolate_nonmatching(u1_NN, cells_cg,interpolation_data_cg)  
        u2.interpolate_nonmatching(u2_NN, cells_cg,interpolation_data_cg)  



        if not only_ord:
            logger.info('trying interpolation on A')
            A = fem.Function(V_mag_col)

            num_cells_curl = V_mag_col.dofmap.list.shape[0]
            cells_curl = np.arange(num_cells_curl,dtype=np.int32)

            interpolation_data_curl = (
                dfx.fem.create_interpolation_data(
                    V_mag_col,
                    FEM_spaces_dict['V_mag_col'],
                    cells_curl,
                )
            )
            A.interpolate_nonmatching(A_NN, cells_curl,interpolation_data_curl)  

            logger.info('managed interpolation on A'+'\n')
            _ ,div_size_2 =  discrete_divergence.compute_discrete_divergence(msh, A,spaces_config_dict,logger=logger)

            if rank == 0 :
                logger.info(f' div_h A at start = {div_size_2}')

   

        logger.info('finished NN interpolation')


    '''
    using random solution as initial value
    '''    
    if use_rand and not use_ref and not FEM_solver_config['use_NN_initial_guess']:
        if rank == 0:
            logger.info('using rand now')
        x_shape = u1.x.array.shape[0]
        r1 = 2*np.random.rand(x_shape)-1
        r2 = 2*np.random.rand(x_shape)-1
        r_max = np.sqrt(np.max(r1**2 + r2**2))
        r1 ,r2  = r1 / r_max, r2 / r_max

   

    if only_ord:
        A = A_in    
        pot_qualy = GL_energy.my_scalar_assemble(GL_energy.compute_magn_energy(A , H),comm)
        if rank == 0:
            logger.info(f'L2 norm at start of curl A - H = {pot_qualy}')

        
        
    

    
    if only_ord:
        energy_init = GL_energy.compute_energy_ord(u1,u2,A, kappa,comm)
    else:
        energy_init = GL_energy.compute_energy_full(u1,u2,A,H, kappa,inc_div,comm)    
    
    
    logger.info(f'energy_init compouted within the prep step = {energy_init}')

    return u1, u2 , A , energy_init


def compute_minimum_grad_flow(u1, u2, A , H, cf,FEM_spaces_dict,results_dict,use_rand=False, logger=None):

    problem_dict = cf.problem_dict
    spaces_config_dict = cf.spaces_config_dict
    minimizer_dict = cf.minimizer_dict

    kappa = problem_dict['kappa']
    only_ord = problem_dict['only_ord']
    geo = problem_dict['geo']
    circ_rad = problem_dict['circ_rad']
    omega_by_pi_times_ten = problem_dict['omega_by_pi_times_ten']
    u_type =  problem_dict['u_type']
    mag_scale =  problem_dict['mag_scale']

    use_ref = spaces_config_dict['use_ref']
    has = spaces_config_dict['has']
    h_ref = spaces_config_dict['h_ref']
    kappa_ref = spaces_config_dict['kappa_ref']

    Fem_type_mag = spaces_config_dict['Fem_type_mag']
    degree_FEM_ord = spaces_config_dict['degree_FEM_ord']
    degree_FEM_mag = spaces_config_dict['degree_FEM_mag']

    inc_div = spaces_config_dict['inc_div']
    corr_div = spaces_config_dict['corr_div']
    if corr_div:
        div_label = 'zero_div' 
    else:
        div_label = 'no_zero_div'  
    
    tol = minimizer_dict['tol']
    it_num = minimizer_dict['it_num']
    grad_type = minimizer_dict['grad_type']
    Newton = minimizer_dict['Newton']
    tol_Newton = minimizer_dict['tol_Newton']
    step_Newton_max = minimizer_dict['step_Newton_max']
    

    V = FEM_spaces_dict['V']
    V_ord = FEM_spaces_dict['V_ord']
    V_ord_real = FEM_spaces_dict['V_ord_real']
    V_ord_real_col = FEM_spaces_dict['V_ord_real_col']
    V_ord_imag_col = FEM_spaces_dict['V_ord_imag_col']
    V_mag_col = FEM_spaces_dict['V_mag_col']
    comm = V.mesh.comm
    rank = comm.Get_rank()
    # size = comm.Get_size()

    ##################################

    msh = V.mesh
    
    if only_ord:
        energy_init = GL_energy.compute_energy_ord(u1,u2,A, kappa,comm)
    else:
        energy_init = GL_energy.compute_energy_full(u1,u2,A,H, kappa,inc_div,comm)
    energy_old = energy_init 
    delta_energy = 1.0  + tol
    
    if rank == 0:   
       logger.info(f'begin compute_minimum: energy_init = {energy_init} ')

    count = 0
    
    

    
    if only_ord:
        energy_init = GL_energy.compute_energy_ord(u1,u2,A, kappa,comm)
    else:
        energy_init = GL_energy.compute_energy_full(u1,u2,A,H, kappa,inc_div,comm)
    results_dict['energy_init'] = energy_init
    
    while(abs(delta_energy) > tol and count < it_num):
    
        count = count + 1 
        if rank == 0:
            logger.info(f'step {count} of max {it_num}')
    
        
        if grad_type == 'L2':
            u1,u2,A, _, _,  delta_A =  GL_energy.compute_energy_prime_components(u1, u2, A, H , problem_dict,spaces_config_dict, FEM_spaces_dict)

        if grad_type == 'Sobolev':
            if count == 1:
                  u_real_proj,u_imag_proj, A_proj  = None,None, None
                  u1_old = fem.Function(V_ord_real_col) 
                  u2_old = fem.Function(V_ord_imag_col) 

            # beta = 0.1
            beta = 1.0
            
            u1,u2,u_real_proj,u_imag_proj, A, u1_old, u2_old, A_proj \
                                =  GL_energy.compute_Sobolev_grad_flow(u1, u2, A, H , problem_dict, FEM_spaces_dict,spaces_config_dict,minimizer_dict,stabPar=beta,\
                                                                                   u_real_old = u1_old,u_imag_old = u2_old, u_proj_real_old = u_real_proj,u_proj_imag_old = u_imag_proj, A_proj_old = A_proj, comm = comm, logger = logger)


        if only_ord:
            energy = GL_energy.compute_energy_ord(u1,u2,A, kappa,comm)
        else:
            energy = GL_energy.compute_energy_full(u1,u2,A,H, kappa,inc_div,comm)

        delta_energy = energy - energy_old
        energy_old = energy
        if rank == 0:
            logger.info(f'kappa = {kappa}')
            logger.info(f'initial energy = {energy_init}')
            logger.info(f'energy = {energy}')
            logger.info(f'delta energy = {delta_energy}'+'\n')
            

        if not only_ord:
            _ ,div_size_2 =  discrete_divergence.compute_discrete_divergence(msh, A,spaces_config_dict,logger)
            if rank == 0:
                logger.info(f' div_h A outside = {div_size_2}'+'\n')   
    

    results_dict['energy'] = energy
    results_dict['count_gf'] = count

    str_gd = grad_type
    if minimizer_dict['conjugate'] and grad_type == 'Sobolev':
        str_gd += '_cg'

    
    pot_qualy = GL_energy.my_scalar_assemble(GL_energy.compute_magn_energy(A , H),comm)

    return u1,u2,A ,  energy, np.sqrt(1.), np.sqrt(pot_qualy) , results_dict





def compute_minimum_Newton(u1, u2, A , H, FEM_solver_config,cf,FEM_spaces_dict, results_dict, use_rand=False,energy_GF = 0.0, logger=None):
    problem_dict = cf.problem_dict
    spaces_config_dict = cf.spaces_config_dict
    minimizer_dict = cf.minimizer_dict

    kappa = problem_dict['kappa']
    only_ord = problem_dict['only_ord']
    geo = problem_dict['geo']
    circ_rad = problem_dict['circ_rad']
    omega_by_pi_times_ten = problem_dict['omega_by_pi_times_ten']
    u_type =  problem_dict['u_type']
    mag_scale =  problem_dict['mag_scale']

    use_ref = spaces_config_dict['use_ref']
    has = spaces_config_dict['has']
    h_ref = spaces_config_dict['h_ref']
    kappa_ref = spaces_config_dict['kappa_ref']

    Fem_type_mag = spaces_config_dict['Fem_type_mag']
    degree_FEM_ord = spaces_config_dict['degree_FEM_ord']
    degree_FEM_mag = spaces_config_dict['degree_FEM_mag']

    inc_div = spaces_config_dict['inc_div']
    corr_div = spaces_config_dict['corr_div']
    if corr_div:
        div_label = 'zero_div' 
    else:
        div_label = 'no_zero_div'  
    
    tol = minimizer_dict['tol']
    it_num = minimizer_dict['it_num']
    grad_type = minimizer_dict['grad_type']
    Newton = minimizer_dict['Newton']
    tol_Newton = minimizer_dict['tol_Newton']
    step_Newton_max = minimizer_dict['step_Newton_max']
    

    V = FEM_spaces_dict['V']
    V_ord = FEM_spaces_dict['V_ord']
    V_ord_real = FEM_spaces_dict['V_ord_real']
    V_ord_real_col = FEM_spaces_dict['V_ord_real_col']
    V_ord_imag_col = FEM_spaces_dict['V_ord_imag_col']
    V_mag_col = FEM_spaces_dict['V_mag_col']
    comm = V.mesh.comm
    rank = comm.Get_rank()
    # size = comm.Get_size()

    ##################################

    msh = V.mesh

    energy_bN = energy_GF
    count_Newton = 0
    
    norm_delta = tol_Newton**(0.5) + 1.0
    delta_energy = tol_Newton + 1.0


    while (abs(delta_energy) > tol_Newton and norm_delta > tol_Newton**(0.5)   and count_Newton < step_Newton_max):  

        count_Newton =  count_Newton +1
        if rank == 0 :
            logger.info(f'Newton step {count_Newton}:')
            logger.info(f'kappa = {kappa}')

        u1_old = u1.copy()
        u2_old = u2.copy()
        if not only_ord:
            A_old = A.copy()
        
        u1,u2,A,tau = GL_energy.compute_energy_Newton_nullspace(u1, u2, A,H, problem_dict, FEM_spaces_dict,spaces_config_dict,minimizer_dict,comm,logger)   
        


        if rank == 0 :
            logger.info(f'tau = {tau}')
    

        if only_ord:
            energy = GL_energy.compute_energy_ord(u1,u2,A, kappa,comm)
        else:
            energy = GL_energy.compute_energy_full(u1,u2,A,H, kappa,inc_div,comm)

        delta_energy = energy - energy_bN#energy_old
        energy_old = energy

        if rank == 0 :
            logger.info(f'energy before Newton = {energy_bN}')
            logger.info(f'energy after Newton = {energy}')
            logger.info(f'delta energy = {delta_energy}')

        _ ,div_size_2 =  discrete_divergence.compute_discrete_divergence(msh, A,spaces_config_dict,logger)

        if rank == 0 :
            logger.info(f'div_h A from Newton = {div_size_2}')
            


        if only_ord:
            energy = GL_energy.compute_energy_ord(u1,u2,A, kappa,comm)
        else:
            energy = GL_energy.compute_energy_full(u1,u2,A,H, kappa,inc_div,comm)

        delta_energy = energy - energy_bN#energy_old
        energy_old = energy

        if rank == 0 :
            logger.info(f'energy before Newton = {energy_bN}')
            logger.info(f'energy after Newton = {energy}')
            logger.info(f'delta energy = {delta_energy}')

        delta_u1 = u1_old - u1
        delta_u2 = u2_old - u2

        norm_delta  = GL_energy.H1kappa_norm_ord(delta_u1, delta_u2, kappa,comm) 

        if not only_ord:
            delta_A = A_old - A
            norm_delta  = GL_energy.H1_curl_div(delta_A,comm)  + norm_delta
        if rank == 0 :
            logger.info(f'delta norm = {norm_delta}'+'\n')
        
        

    if rank == 0 :
        logger.info(f'gain in energy by Newton = {energy - energy_bN}, negative = successful')
    
    pot_qualy = GL_energy.my_scalar_assemble(GL_energy.compute_magn_energy(A , H),comm)
    
    results_dict['count_Newton'] = count_Newton


    return u1,u2,A ,  energy, np.sqrt(1.), np.sqrt(pot_qualy) , results_dict


