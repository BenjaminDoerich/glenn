import c0_GL_FEM_config as conf
import numpy as np
from mpi4py import MPI
import dolfinx as dfx
from dolfinx import mesh, fem, plot, io, default_scalar_type
import GL_FEM_initial_values as inits

import dolfinx.fem.petsc as dfpet

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
import generate_mesh as genmesh


###########################################  
import os 

newpath = "sol_screenshots/"
if not os.path.exists(newpath):
   os.makedirs(newpath)  



def prepare_initial_data(model_config,training_config, loading_config,FEM_solver_config,experiment_config,logger=None):
    cf = conf.ConfigFEM(FEM_solver_config)   
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
    h = has
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
    
    comm = MPI.COMM_WORLD

    if geo == "unit_square":
        mesh_path = Path(f"meshes/unit_square_mesh_{h}.msh")
        if not mesh_path.exists():
            genmesh.build_Cdomain(h,h,write_mesh=True,geometry="unit_square")
        msh = adios4dolfinx.read_mesh(f"meshes/unit_square_mesh_{h}", comm=comm, engine="BP4" )
        

    if geo == "square":
        mesh_path = Path(f"meshes/square_mesh_{h}.msh")
        if not mesh_path.exists():
            genmesh.build_Cdomain(h,h,write_mesh=True,geometry="square")
        msh = adios4dolfinx.read_mesh(f"meshes/square_mesh_{h}", comm=comm, engine="BP4" )
        # msh, cell_tags, facet_tags = gmshio.read_from_msh(f"meshes/square_mesh_{h}.msh", comm, 0, gdim=2)

    if geo == "Lshape":
        mesh_path = Path(f"meshes/Lshape_mesh_{h}.msh")
        if not mesh_path.exists():
            genmesh.build_Cdomain(h,h,write_mesh=True,geometry="Lshape")
        msh = adios4dolfinx.read_mesh(f"meshes/Lshape_mesh_{h}", comm=comm, engine="BP4" ) 

    if geo == "box_hole":
        mesh_path = Path(f"meshes/hole_mesh_{h}.msh")
        if not mesh_path.exists():
            genmesh.build_Cdomain(h,h,write_mesh=True,geometry="box_hole")
        msh = adios4dolfinx.read_mesh(f"meshes/hole_mesh_{h}", comm=comm, engine="BP4" )
        # msh, cell_tags, facet_tags = gmshio.read_from_msh(f"meshes/hole_mesh_{h}.msh", comm, 0, gdim=2)

    if geo == "circle":
        mesh_path = Path(f"meshes/circle_{circ_rad}_mesh_{h}")
        if not mesh_path.exists():
            genmesh.build_Cdomain(h,h,write_mesh=True,geometry="circle")
        msh = adios4dolfinx.read_mesh(f"meshes/circle_{circ_rad}_mesh_{h}.msh", comm=comm, engine="BP4" )
        

    if geo == "circle_slice":
        mesh_path = Path(f"meshes/circle_slice_{circ_rad}_angle_{omega_by_pi_times_ten}_mesh_{h}.msh")
        if not mesh_path.exists():
            if geo == "circle_slice":
                genmesh.build_Cdomain(h,h,write_mesh=True,geometry="circle_slice",circ_rad = circ_rad,omega_by_pi_times_ten=omega_by_pi_times_ten)
        msh = adios4dolfinx.read_mesh(f"meshes/circle_slice_{circ_rad}_angle_{omega_by_pi_times_ten}_mesh_{h}.msh", comm=comm, engine="BP4" )



    if geo == "annulus":
        mesh_path = Path(f"meshes/annulus_mesh_{h}")
        if not mesh_path.exists():
            genmesh.build_Cdomain(h,h,write_mesh=True,geometry="annulus")
        msh = adios4dolfinx.read_mesh(f"meshes/annulus_mesh_{h}.msh", comm=comm, engine="BP4" )
        # msh, cell_tags, facet_tags = gmshio.read_from_msh(f"meshes/annulus_{h}.msh",comm, 0, gdim=2)

    if geo == "L":
        mesh_path = Path(f"meshes/L_mesh_{h}")
        if not mesh_path.exists():
            genmesh.build_Cdomain(h,h,write_mesh=True,geometry="L")
        msh = adios4dolfinx.read_mesh(f"meshes/L_mesh_{h}.msh", comm=comm, engine="BP4" )
  
    FEM_spaces_dict = s_def.def_get_all_V(msh,problem_dict,spaces_config_dict)

    V = FEM_spaces_dict['V']
    V_ord = FEM_spaces_dict['V_ord']
    V_ord_real = FEM_spaces_dict['V_ord_real']
    V_ord_real_col = FEM_spaces_dict['V_ord_real_col']
    V_ord_imag_col = FEM_spaces_dict['V_ord_imag_col']
    V_mag_col = FEM_spaces_dict['V_mag_col']
    V_dG = FEM_spaces_dict['V_dG']
    # h = spaces_config_dict['has'] 
    h_ref = spaces_config_dict['h_ref']   
    comm = V.mesh.comm
    rank = comm.Get_rank()

    ##################################

    msh = V.mesh
    
    u1  = fem.Function(V_ord_real_col)
    u2  = fem.Function(V_ord_imag_col) 
    rot_A_dg = fem.Function(V_dG)
   
    '''
    using reference solution as initial value
    '''
    filename_NN = f'{FEM_solver_config["output_path"]}/{FEM_solver_config["results_filename"]}_init_{u_type}_{A_type}_{H_type}_kappa_{kappa}_h_{h_ref}_tol_{tol}'
    
    if geo == "circle_slice":
        filename_NN = f'{FEM_solver_config["output_path"]}/{FEM_solver_config["results_filename"]}_init_{u_type}_{A_type}_{H_type}_kappa_{kappa}_h_{h_ref}_tol_{tol}'


    NN_string ='_NN_interp'
    filename_NN +=NN_string

    filename_NN +='.bp'  
    filename_NN = Path(filename_NN) 
    
    '''
    using NN solution as initial value
    '''
    
    import fancy_interface  as fi    
    if model_config["multi_kappa"]:
        loading_kappa = loading_config["kappa"][0]

        if not only_ord:
            from train import load_lightning
            A = fem.Function(V_mag_col)
            litPinn, model = load_lightning(experiment_config)
            
            from fancy_interface import CachedPrediction
            predictor = CachedPrediction(litPinn, model_config, loading_kappa)
            f_real,f_imag,A_vals = predictor.f_real, predictor.f_imag, predictor.A_full


            A_direct_np = lambda x: np.vstack((A_vals(x)))
            A.interpolate(A_direct_np)

            
        else:
            from train import load_lightning
            litPinn, model = load_lightning(experiment_config)
            

            from fancy_interface import CachedPrediction
            predictor = CachedPrediction(litPinn, model_config, loading_kappa)
            f_real,f_imag = predictor.f_real, predictor.f_imag

    else:
        if not only_ord:
            from train import load_lightning
            A = fem.Function(V_mag_col)
            litPinn, model = load_lightning(experiment_config)
            
            from fancy_interface import CachedPredictionSingleKappa
            predictor = CachedPredictionSingleKappa(litPinn, loading_config)
            f_real,f_imag,A_vals = predictor.f_real, predictor.f_imag, predictor.A_full


            A_direct_np = lambda x: np.vstack((A_vals(x)))
            A.interpolate(A_direct_np)

        else:
            from train import load_lightning
            litPinn, model = load_lightning(experiment_config)
            

            from fancy_interface import CachedPredictionSingleKappa
            predictor = CachedPredictionSingleKappa(litPinn, loading_config)
            f_real,f_imag = predictor.f_real, predictor.f_imag



    u1.interpolate(f_real)
    u2.interpolate(f_imag)

    if not only_ord:
        u1,u2,A = discrete_divergence.compute_divergence_free_cor(u1, u2, V_ord_real_col.mesh, A,problem_dict,spaces_config_dict, FEM_spaces_dict,cor_all=True)
        _,size = discrete_divergence.compute_discrete_divergence( V_ord_real_col.mesh, A,spaces_config_dict,logger=logger)
    
    
    adios4dolfinx.write_mesh(filename_NN, msh , 'BP4')
    adios4dolfinx.write_function(filename_NN, u1,  name="u1_prep")
    adios4dolfinx.write_function(filename_NN, u2,  name="u2_prep")
    
    u_abs = fem.Function(FEM_spaces_dict['V_ord_real_col'])

    u_abs.x.array[:] = u1.x.array[:]**2 + u2.x.array[:]**2
    if geo != "circle_slice":
        omega_by_pi_times_ten = 0
    
    filename_save = Path(f'{FEM_solver_config["output_path"]}/{FEM_solver_config["results_filename"]}_plot_u1_init_{u_type}_{A_type}_{H_type}_kappa_{kappa}_h_{h}_tol_{tol}_interp_dG.bp')

    with dfx.io.VTXWriter(comm, filename_save, [u1]) as vtx:
        vtx.write(0.0)

    filename_save = Path(f'{FEM_solver_config["output_path"]}/{FEM_solver_config["results_filename"]}_plot_u2_init_{u_type}_{A_type}_{H_type}_kappa_{kappa}_h_{h}_tol_{tol}_interp_dG.bp')
    
    with dfx.io.VTXWriter(comm, filename_save, [u2]) as vtx:
        vtx.write(0.0)

    filename_save = Path(f'{FEM_solver_config["output_path"]}/{FEM_solver_config["results_filename"]}_plot_u_abs_init_{u_type}_{A_type}_{H_type}_kappa_{kappa}_h_{h}_tol_{tol}_interp_dG.bp')

    with dfx.io.VTXWriter(comm, filename_save, [u_abs]) as vtx:
        vtx.write(0.0)


    if not only_ord:
        B_exp = fem.Expression(ufl.rot(A), V_dG.element.interpolation_points)
        rot_A_dg = fem.Function(V_dG)
        rot_A_dg.interpolate(B_exp)
        
        adios4dolfinx.write_function(filename_NN, A,  name="A_prep")
        
        filename_NN_plot = Path(f'{FEM_solver_config["output_path"]}/{FEM_solver_config["results_filename"]}_plot_curl_A_init_{u_type}_{A_type}_{H_type}_kappa_{kappa}_h_{h}_tol_{tol}_interp_dG.bp')
        
        with dfx.io.VTXWriter(comm, filename_NN_plot, [rot_A_dg]) as vtx:
            vtx.write(0.0)
    
        return u1, u2 , A
    return u1,u2