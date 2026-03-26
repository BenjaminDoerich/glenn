import c0_GL_FEM_config as conf
import numpy as np
from mpi4py import MPI
import dolfinx as dfx
from dolfinx import mesh, fem, plot, io, default_scalar_type

import dolfinx.fem.petsc as dfpet

if dfx.__version__ == '0.9.0':
    import dolfinx.io as gmshio
elif dfx.__version__ == '0.10.0':
    import dolfinx.io.gmsh as gmshio

import basix


from mpi4py import MPI

import adios4dolfinx

import ufl
import time
from pathlib import Path

import scipy.sparse as scp
import numpy as np
# import scipy.sparse.linalg
#import scipy.sparse.linalg as ssla

# import GL_FEM_compute_one_minimizer as GL_min
import GL_FEM_energies as GL_energy
import spaces_def as s_def
import discrete_divergence as discrete_divergence
import generate_mesh as gen_msh

from Norms import error_norm_ref, norm_L2

import matplotlib.pyplot as plt
import os 

import logging 
import sys 


newpath = "conv_plots/"
if not os.path.exists(newpath):
   os.makedirs(newpath)  


###########################################  

newpath = "sol_screenshots/"
if not os.path.exists(newpath):
   os.makedirs(newpath)  
###########################################
import yaml 




def read_function(filename: Path, degree_FEM_ord: int, degree_FEM_mag: int,only_ord: bool,comm):
    in_mesh = adios4dolfinx.read_mesh(filename, comm=comm,)
    # in_mesh = adios4dolfinx.read_mesh(filename, MPI.COMM_WORLD)
    W_cg = dfx.fem.functionspace(in_mesh, ('Lagrange', degree_FEM_ord))
    
    if not only_ord:
        # element_dG = basix.ufl.element("DG", in_mesh.basix_cell(), 1,shape = (1,) )
        element_dG = basix.ufl.element("DG", in_mesh.basix_cell(), 0,shape = (1,) )

        W_dg = fem.functionspace(in_mesh,element_dG)

        element_curl = basix.ufl.element(basix.ElementFamily.N1E, basix.CellType.triangle, degree =  degree_FEM_mag)
        W_Ned = fem.functionspace(in_mesh,element_curl)

    u_real = fem.Function(W_cg)
    u_imag = fem.Function(W_cg)

    if not only_ord:
        rot_A = fem.Function(W_dg)
        MagPot = fem.Function(W_Ned)


    adios4dolfinx.read_function(filename, u_real,  name = "u1")
    adios4dolfinx.read_function(filename, u_imag,  name = "u2")

    # if False:
    if not only_ord:
        adios4dolfinx.read_function(filename, MagPot,  name = "MagPot")
        # adios4dolfinx.read_function(filename, rot_A,  time=timestamp, name = "rot_A")

    if only_ord:
        return W_cg , None, None,  in_mesh, u_real, u_imag, None
    else:
        return W_cg ,W_dg, W_Ned, in_mesh, u_real, u_imag , MagPot#, rot_A


def eoc_eval(steps,error):

    eoc_vec = (np.log(error[:-1]) - np.log(error[1:]) )  / (np.log(steps[:-1]) - np.log(steps[1:]) )
    c_eoc_vec = error[:-1]/steps[:-1]**eoc_vec
    mean = np.mean(eoc_vec)

    return eoc_vec, c_eoc_vec, mean



def run_GL_FEM_conv_test(model_config,training_config, loading_config,FEM_solver_config, experiment_config):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()


    _name = sys.argv[0].split("/")[-1].split(".")[-2]
    if rank == 0:
        Path('log/{_name}.log').mkdir(parents=True, exist_ok=True)
    comm.Barrier()
    log_file = Path(f'log/{_name}.log')
    if rank == 0:
        logging.basicConfig(
        level=logging.DEBUG, datefmt="%m-%d %H:%M", filename=log_file, filemode="w"
        )
    else:
        logging.basicConfig(
        level=logging.DEBUG,
        datefmt="%m-%d %H:%M",
        filename=log_file,
        filemode="a", # append for others
        )
    logger = logging.getLogger(f"{_name} rank {rank}")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # console.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logger.addHandler(console)


    import GL_FEM_initial_values as inits
    import plot_sol    
    cf = conf.ConfigFEM(FEM_solver_config)
    # import dictionaries
    problem_dict = cf.problem_dict
    spaces_config_dict = cf.spaces_config_dict
    minimizer_dict = cf.minimizer_dict


    kappa = problem_dict['kappa']
    only_ord = problem_dict['only_ord']
    geo = problem_dict['geo']
    circ_rad = problem_dict['circ_rad']
    # config = cf.ConfigFEM({})
    # circ_rad = cf.circ_rad
    omega_by_pi_times_ten = problem_dict['omega_by_pi_times_ten']

    h_ref = spaces_config_dict['h_ref']
    M_conv = np.array(spaces_config_dict['M_conv'])
    has = 4/M_conv
    if rank == 0 :
        logger.info(f'has = {has}')

    degree_FEM_ord = spaces_config_dict['degree_FEM_ord']
    Fem_type_mag = spaces_config_dict['Fem_type_mag']
    Nedelec_kind = spaces_config_dict['Nedelec_kind']

    degree_FEM_mag = spaces_config_dict['degree_FEM_mag']
    inc_div = spaces_config_dict['inc_div']
    inc_bc_div_h = spaces_config_dict['inc_bc_div_h']
    corr_div = spaces_config_dict['corr_div']

    if corr_div:
        div_label = 'zero_div' 
    else:
        div_label = 'no_zero_div'   

    tol = minimizer_dict['tol']
    it_num = minimizer_dict['it_num']
    grad_type = minimizer_dict['grad_type']
    tau = minimizer_dict['tau']
    line_search = minimizer_dict['line_search']
    Newton = minimizer_dict['Newton']
    tol_Newton = minimizer_dict['tol_Newton']
    step_Newton_max = minimizer_dict['step_Newton_max']


    mag_scale = problem_dict['mag_scale'] 
    H_type = problem_dict['H_type'] 
    A_type = problem_dict['A_type'] 
    u_type = problem_dict['u_type'] 
    num_holes = problem_dict['num_holes'] 

    grad_type = minimizer_dict['grad_type']
    tau = minimizer_dict['tau']
    line_search = minimizer_dict['line_search']
    grad_type = minimizer_dict['grad_type']


    abs_error = True
    # abs_error = False
    ref_dist = 1

    # h_ref = %has[-1]


    if rank == 0 :
        logger.info(f'h_ref = {h_ref}')

    filename_ref = f'{FEM_solver_config["output_path"]}/ord_{grad_type}_{geo}_{circ_rad}_init_{u_type}_{A_type}_{H_type}_kappa_{kappa}_mag_scale_{mag_scale}_h_{h_ref}_tol_{tol}_{Fem_type_mag}_{div_label}'
    
    if geo == "circle_slice":
        filename_ref = f'{FEM_solver_config["output_path"]}/ord_{grad_type}_{geo}_{circ_rad}_{omega_by_pi_times_ten}_init_{u_type}_{A_type}_{H_type}_kappa_{kappa}_mag_scale_{mag_scale}_h_{h_ref}_tol_{tol}_{Fem_type_mag}_{div_label}'

    if FEM_solver_config['use_NN_initial_guess']:
        filename_ref += '_NN_used'

    filename_ref +='.bp'  
    filename_ref = Path(filename_ref) 

 
    W_cg_ref , W_dg_ref , W_Ned_ref, msh_ref, u_real_ref, u_imag_ref, A_ref = read_function(filename_ref,degree_FEM_ord, degree_FEM_mag,only_ord,comm)

    if abs_error:
        u_abs_ref = fem.Function(W_cg_ref)
        u_abs_ref.x.array[:] =  u_real_ref.x.array[:]**2 + u_imag_ref.x.array[:]**2  
    
    
    rot_A_ref = fem.Function(W_dg_ref)
    rot_A_ref.interpolate(fem.Expression(ufl.curl(A_ref), W_dg_ref.element.interpolation_points))

    if rank == 0 :
        logger.info('prepared reference solution'+'\n')

    #############################################
    errors_ord = np.zeros(len(has)) 
    errors_ord_abs = np.zeros(len(has)) 
    errors_A = np.zeros(len(has)) 


    for i in range(len(has)):
        h = has[i]

        filename = f'{FEM_solver_config["output_path"]}/ord_{grad_type}_{geo}_{circ_rad}_init_{u_type}_{A_type}_{H_type}_kappa_{kappa}_mag_scale_{mag_scale}_h_{h}_tol_{tol}_{Fem_type_mag}_{div_label}'
    
        if geo == "circle_slice":
            filename = f'{FEM_solver_config["output_path"]}/ord_{grad_type}_{geo}_{circ_rad}_{omega_by_pi_times_ten}_init_{u_type}_{A_type}_{H_type}_kappa_{kappa}_mag_scale_{mag_scale}_h_{h}_tol_{tol}_{Fem_type_mag}_{div_label}'

        if FEM_solver_config['use_NN_initial_guess']:
            filename += '_NN_used'

        filename +='.bp'  
        filename = Path(filename) 

        # logger.info(f' filename = {filename}')

        W_cg, W_dg , W_Ned, msh, u_real, u_imag, A = read_function(filename,degree_FEM_ord, degree_FEM_mag,only_ord,comm)

        if abs_error:
            u_abs = fem.Function(W_cg)
            u_abs.x.array[:] =   u_real.x.array[:]**2 + u_imag.x.array[:]**2 

        ########################
        # setup of interpolate_nonmatching provided by Tim Buchholz
        ########################
        num_cells_dg_ref = W_dg_ref.dofmap.list.shape[0]
        cells_dg = np.arange(num_cells_dg_ref,dtype=np.int32)

        interpolation_data_dg = (
            dfx.fem.create_interpolation_data(
                W_dg_ref,
                W_dg,
                cells_dg,
            )
        )

        # rot_A_int =  fem.Function(W_dg_ref)
        # curl_A = ufl.curl(A)

        rot_A_int = fem.Function(W_dg)
        rot_A_int.interpolate(fem.Expression(ufl.curl(A), W_dg.element.interpolation_points))
        rot_A_int_high =  fem.Function(W_dg_ref)
        rot_A_int_high.interpolate_nonmatching(rot_A_int, cells_dg,interpolation_data_dg)

        if rank == 0 :
            logger.info(f'prepared {i+1}-th point of {len(has)} for plot solution'+'\n')

        ########################
        # setup of interpolate_nonmatching provided by Tim Buchholz
        ########################
        num_cells_curl_ref = W_Ned_ref.dofmap.list.shape[0]
        cells_curl = np.arange(num_cells_curl_ref,dtype=np.int32)

        interpolation_data_curl = (
            dfx.fem.create_interpolation_data(
                W_Ned_ref,
                W_Ned,
                cells_curl,
            )
        )

        A_int =  fem.Function(W_Ned_ref)
        A_int.interpolate_nonmatching(A, cells_curl,interpolation_data_curl)  


        ########################
        # compute difference and then L2 norm
        ########################
        # rot_A_diff =  fem.Function(W_dg_ref)
        # rot_A_diff.x.array[:] = rot_A_int_high.x.array[:] - rot_A_ref.x.array[:]
        # errors_A[i] = norm_L2(W_dg_ref,rot_A_diff) 
        # if rank == 0 :
        #     logger.info(f'started to compute errors_A[i]')
        errors_A[i] = error_norm_ref(rot_A_int, rot_A_ref,norm_str='L2',logger = logger, dG = True) 
        # if rank == 0 :
        #     logger.info(f'finished to compute errors_A[i]')

        # if rank == 0 :
        #    logger.info(f'norm_L2(W_dg_ref,rot_A_diff)  = {norm_L2(W_dg_ref,rot_A_diff) }')
        #    logger.info(f'errors_A[i]  = {errors_A[i] }')


        # if rank == 0 :
        #    logger.info(f'norm_L2(W_dg_ref,rot_A_ref)  = {norm_L2(W_dg_ref,rot_A_ref) }'+'\n')

        # errors_A[i] /= norm_L2(W_dg_ref,rot_A_ref) 


       
        errors_ord_abs[i] =   1/kappa * error_norm_ref(u_abs,u_abs_ref,norm_str='H1kappa',kappa = kappa) 
        errors_ord[i] =   1/kappa * error_norm_ref(u_real,u_real_ref,norm_str='H1kappa',kappa = kappa) \
                    + 1/kappa * error_norm_ref(u_imag,u_imag_ref,norm_str='H1kappa',kappa = kappa)

        if rank == 0 :
           logger.info(f'computed errors for {i+1}-th point of {len(has)} for plot solution'+'\n')

           logger.info(f'errors_ord = {errors_ord}')
           logger.info(f'errors_ord_abs = {errors_ord_abs}')
           logger.info(f'errors_A = {errors_A}'+'\n')

        # logger.info(f' errors_ord[i] = {errors_ord[i]} for h = {h} ')
        # logger.info(f' errors_ord_abs[i] = {errors_ord[i]} for h = {h} ')
        # logger.info(f' errors_A[i]   = {errors_A[i]} for h = {h} ')
        # logger.info()

    ########################################
    # present convergence plots
    ########################################
    title_conv_plot_ord = f'conv_plots/ord_{grad_type}_{geo}_{circ_rad}_init_{u_type}_kappa_{kappa}_mag_scale_{mag_scale}_href_{h_ref}_tol_{tol}_{Fem_type_mag}_{div_label}.png'
    title_conv_plot_ord_abs = f'conv_plots/ord_abs_{grad_type}_{geo}_{circ_rad}_init_{u_type}_kappa_{kappa}_mag_scale_{mag_scale}_href_{h_ref}_tol_{tol}_{Fem_type_mag}_{div_label}.png'
    title_conv_plot_rot_A = f'conv_plots/rot_A_{grad_type}_{geo}_{circ_rad}_init_{u_type}_kappa_{kappa}_mag_scale_{mag_scale}_href_{h_ref}_tol_{tol}_{Fem_type_mag}_{div_label}.png'




    if geo == "circle_slice":
        title_conv_plot_ord = f'conv_plots/ord_{grad_type}_{geo}_{circ_rad}_{omega_by_pi_times_ten}_init_{u_type}_kappa_{kappa}_mag_scale_{mag_scale}_href_{h_ref}_tol_{tol}_{Fem_type_mag}_{div_label}.png'
        title_conv_plot_ord_abs = f'conv_plots/ord_abs_{grad_type}_{geo}_{omega_by_pi_times_ten}_{circ_rad}_init_{u_type}_kappa_{kappa}_mag_scale_{mag_scale}_href_{h_ref}_tol_{tol}_{Fem_type_mag}_{div_label}.png'
        title_conv_plot_rot_A = f'conv_plots/rot_A_{grad_type}_{geo}_{circ_rad}_{omega_by_pi_times_ten}_init_{u_type}_kappa_{kappa}_mag_scale_{mag_scale}_href_{h_ref}_tol_{tol}_{Fem_type_mag}_{div_label}.png'

    if rank == 0:

        logger.info(f'errors_ord = {errors_ord}')
        logger.info(f'errors_ord_abs = {errors_ord_abs}')
        logger.info(f'errors_A = {errors_A}'+'\n')
        

        EOC_vec_ord , c_EOC_ord_u, EOC_ord_mean             = eoc_eval(has,errors_ord)
        EOC_vec_ord_abs , c_EOC_ord_abs_u, EOC_ord_abs_mean = eoc_eval(has,errors_ord_abs)
        EOC_vec_A , c_EOC_A, EOC_A_mean                     = eoc_eval(has,errors_A)


        logger.info(f'EOC_ord = {EOC_vec_ord}')
        logger.info(f'EOC_ord_mean = {EOC_ord_mean}')
        logger.info(f'c_EOC_ord_u = {c_EOC_ord_u}'+'\n')

        logger.info(f'EOC_ord_abs = {EOC_vec_ord_abs}')
        logger.info(f'EOC_ord_abs_mean = {EOC_ord_abs_mean}')
        logger.info(f'c_EOC_ord_u = {c_EOC_ord_u}'+'\n')


        logger.info(f'EOC_A = {EOC_vec_A}')
        logger.info(f'EOC_A_mean = {EOC_A_mean}')
        logger.info(f'c_EOC_A = {c_EOC_A}'+'\n')

        # produce plots

        plt.loglog(has,errors_ord,marker = 'o',label='error ord')   
        plt.loglog(has, c_EOC_ord_u[-1]*has**EOC_vec_ord[-1],'--',label=f'order_u = {np.round(EOC_vec_ord[-1],2)}, c = {np.round(c_EOC_ord_u[-1],2)}')
        plt.loglog(has, c_EOC_ord_u[0]*has**EOC_vec_ord[0],'--',label=f'order_u = {np.round(EOC_vec_ord[0],2)}, c = {np.round(c_EOC_ord_u[0],2)}')
        plt.legend()
        plt.title(f'H1kappa error in u,kappa = {kappa}')
        plt.savefig(title_conv_plot_ord)
        plt.show()


        plt.loglog(has,errors_ord_abs,marker = 'o',label='error abs(ord)^2')
        plt.loglog(has, c_EOC_ord_abs_u[-1]*has**EOC_vec_ord_abs[-1],'--',label=f'order_abs_u = {np.round(EOC_vec_ord_abs[-1],2)}, c = {np.round(c_EOC_ord_abs_u[-1],2)}')
        plt.loglog(has, c_EOC_ord_abs_u[0]*has**EOC_vec_ord_abs[0],'--',label=f'order_abs_u = {np.round(EOC_vec_ord_abs[0],2)}, c = {np.round(c_EOC_ord_abs_u[0],2)}')
        plt.legend()
        plt.title(f'H1kappa error in |u|^2,kappa = {kappa}')
        plt.savefig(title_conv_plot_ord_abs)
        plt.show()


        plt.loglog(has,errors_A,marker = 'o',label='error curl A')
        plt.loglog(has, c_EOC_A[-1]*has**EOC_vec_A[-1],'--',label=f'order_rot_a = {np.round(EOC_vec_A[-1],2)}, c = {np.round(c_EOC_A[-1],2)}')
        plt.loglog(has, c_EOC_A[0]*has**EOC_vec_A[0],'--',label=f'order_rot_A = {np.round(EOC_vec_A[0],2)}, c = {np.round(c_EOC_A[0],2)}')
        plt.legend()
        plt.title(f'Error in L^2 for rot A, kappa = {kappa}')
        plt.savefig(title_conv_plot_rot_A)
        plt.show()

        # import pykz
        # pykz.figure()  
        # pykz.loglog(has,errors_A)
        # # pykz.loglog(has,errors_A,label='error curl A')
        # # pykz.loglog(has, c_EOC_A[-1]*has**EOC_vec_A[-1],'--',label=f'order_rot_a = {np.round(EOC_vec_A[-1],2)}, c = {np.round(c_EOC_A[-1],2)}')
        # # pykz.loglog(has, c_EOC_A[0]*has**EOC_vec_A[0],'--',label=f'order_rot_A = {np.round(EOC_vec_A[0],2)}, c = {np.round(c_EOC_A[0],2)}')
        # # pykz.legend()
        # # pykz.title(f'Error in L^2 for rot A, kappa = {kappa}')
        # # pykz.savefig(title_conv_plot_rot_A)
        # # pykz.show()

        # pykz.save("basic_inline.tex", standalone=True)
        # # You could also directly build the pdf
        # pykz.io.export_pdf_from_file("basic_inline.tex")
