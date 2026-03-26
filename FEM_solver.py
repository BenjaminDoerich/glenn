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
import adios4dolfinx 


import ufl
import time
from pathlib import Path

import scipy.sparse as scp
import numpy as np
# import scipy.sparse.linalg
import scipy.sparse.linalg as ssla


import FEM_solver_one_min as GL_min
import GL_FEM_energies as GL_energy
import spaces_def as s_def
import discrete_divergence as discrete_divergence
import generate_mesh as genmesh

import logging 
import sys
import yaml
from pathlib import Path
import shutil
###########################################  
import os 
def run_FEM_minimzer_post(model_config,training_config, loading_config,FEM_solver_config, experiment_config):
    newpath = FEM_solver_config["output_path"]
    if not os.path.exists(newpath):
        os.makedirs(newpath)  

    aux_path = Path(f"aux/")
    if not aux_path.exists():
        os.makedirs(aux_path)  

    ###########################################


    import GL_FEM_initial_values as inits
    import plot_sol    
    cf = conf.ConfigFEM(FEM_solver_config)
    # import dictionaries
    problem_dict = cf.problem_dict
    spaces_config_dict = cf.spaces_config_dict
    minimizer_dict = cf.minimizer_dict
    use_rand = cf.use_rand


    kappa = problem_dict['kappa']
    only_ord = problem_dict['only_ord']
    geo = problem_dict['geo']
    circ_rad = problem_dict['circ_rad']
    
    omega_by_pi_times_ten = problem_dict['omega_by_pi_times_ten']

    use_ref = spaces_config_dict['use_ref']
    h_ref = spaces_config_dict['h_ref']
    h = spaces_config_dict['has'] 

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


    filename_results_dict = FEM_solver_config['results_filename']+f'h_{h}'
    results_dict = {}


    cache_dir = f"{str(Path.cwd())}/.{filename_results_dict}" #to change to experiment name!
    JIT_OPTIONS = {
                "cffi_extra_compile_args": ["-Ofast", "-march=native"],
                "cache_dir": cache_dir,
                "cffi_libraries": ["m"],
            }

    FEM_solver_config['JIT_OPTIONS'] = JIT_OPTIONS



    ###############################
    # load mesh
    ###############################
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    results_dict['kappa'] = kappa
    results_dict['geo'] = geo
    if geo == 'circle_slice':
        results_dict['omega_by_pi_times_ten'] = omega_by_pi_times_ten


    ###
    # setup logger 

    _name = sys.argv[0].split("/")[-1].split(".")[-2]
    if rank == 0:
        Path("log/{_name}.log").mkdir(parents=True, exist_ok=True)
    comm.Barrier()
    log_file = Path(f"log/{_name}.log")
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



    if geo == "unit_square":
        mesh_path = Path(f"meshes/unit_square_mesh_{h}.msh")
        if not mesh_path.exists():
            genmesh.build_Cdomain(h,h,write_mesh=True,geometry="unit_square")
        msh = adios4dolfinx.read_mesh(f"meshes/unit_square_mesh_{h}", comm=comm, engine="BP4" )
    
    if geo == "Lshape":
        mesh_path = Path(f"meshes/Lshape_mesh_{h}.msh")
        if not mesh_path.exists():
            genmesh.build_Cdomain(h,h,write_mesh=True,geometry="Lshape")
        msh = adios4dolfinx.read_mesh(f"meshes/Lshape_mesh_{h}", comm=comm, engine="BP4" ) 

    if geo == "square":
        mesh_path = Path(f"meshes/square_mesh_{h}.msh")
        if not mesh_path.exists():
            genmesh.build_Cdomain(h,h,write_mesh=True,geometry="square")
        msh = adios4dolfinx.read_mesh(f"meshes/square_mesh_{h}", comm=comm, engine="BP4" )


    if geo == "box_hole":
        mesh_path = Path(f"meshes/hole_mesh_{h}.msh")
        if not mesh_path.exists():
            genmesh.build_Cdomain(h,h,write_mesh=True,geometry="box_hole")
        msh = adios4dolfinx.read_mesh(f"meshes/hole_mesh_{h}", comm=comm, engine="BP4" )

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

    if geo == "L":
        mesh_path = Path(f"meshes/L_mesh_{h}")
        if not mesh_path.exists():
            genmesh.build_Cdomain(h,h,write_mesh=True,geometry="L")
        msh = adios4dolfinx.read_mesh(f"meshes/L_mesh_{h}.msh", comm=comm, engine="BP4" )


  
    FEM_spaces_dict = s_def.def_get_all_V(msh,problem_dict,spaces_config_dict)

    ##############################
    # set test functions
    ##############################

    V_ext = FEM_spaces_dict['V_ext']
    V = FEM_spaces_dict['V']
    V_ord = FEM_spaces_dict['V_ord']
    V_ord_real = FEM_spaces_dict['V_ord_real']
    V_ord_imag = FEM_spaces_dict['V_ord_imag']
    V_mag = FEM_spaces_dict['V_mag']
    V_ord_real_col = FEM_spaces_dict['V_ord_real_col']
    V_ord_imag_col = FEM_spaces_dict['V_ord_imag_col']
    V_mag_col = FEM_spaces_dict['V_mag_col']
    x = FEM_spaces_dict['x']
    bc_V = FEM_spaces_dict['bc_V']
    bc_V_mag = FEM_spaces_dict['bc_V_mag']


    if rank == 0:
        u_dofs = (V_ord.dofmap.index_map.size_global) * V_ord.dofmap.index_map_bs
        all_dofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs

        if only_ord:
            total_dofs = u_dofs
            logger.info(f'start computation with {total_dofs} degrees of freedom for u in C on {size} ranks'+'\n')

        else:
            total_dofs = all_dofs
            logger.info(f'start computation with {total_dofs} degrees of freedom for (u,A) in C x R^2 on {size} ranks')
            logger.info(f'number of dos for u are {u_dofs}'+'\n')

        results_dict['u_dofs'] = u_dofs
        results_dict['total_dofs'] = total_dofs


    




    H = inits.get_H(H_type, mag_scale, x,omega_by_pi_times_ten)

    A1 = inits.get_A1(A_type, 0.5*mag_scale, x,omega_by_pi_times_ten)

    freq = 20
    u_real, u_imag = inits.get_u(u_type,kappa,freq,num_holes,x,msh,V_ord)
    

    comm.Barrier()
    start_time_initials = MPI.Wtime()

    GL_energy.my_scalar_assemble(GL_energy.compute_magn_energy(A1,H),comm)
        
    if only_ord:
        energy_init_bp = GL_energy.compute_energy_ord(u_real,u_imag,A1, kappa,comm)
    else:
        energy_init_bp = GL_energy.compute_energy_full(u_real,u_imag,A1,H, kappa,inc_div,comm)

    if rank == 0:
        logger.info(f'energy_init before initial prep = {energy_init_bp}')
    
    filename_bef = Path(f"aux/inbetween.bp") 
    adios4dolfinx.write_mesh(filename_bef, msh , 'BP4')


    filename = Path(f"aux/prep_inits.bp") 
    if rank == 0:
        local_comm = MPI.COMM_SELF
        msh = adios4dolfinx.read_mesh(filename_bef,  local_comm,'BP4' )
        FEM_spaces_dict = s_def.def_get_all_V(msh,problem_dict,spaces_config_dict)
        x = FEM_spaces_dict['x']

        H = inits.get_H(H_type, mag_scale, x,omega_by_pi_times_ten)
        A1 = inits.get_A1(A_type, 0.5*mag_scale, x,omega_by_pi_times_ten)
        freq = 20
        u_real, u_imag = inits.get_u(u_type,kappa,freq,num_holes,x,msh,FEM_spaces_dict['V_ord'])
        u1_prep, u2_prep, A_prep , energy_init_ap =\
                GL_min.prepare_initial_data(u_real, u_imag, A1 , H,model_config,training_config, loading_config,FEM_solver_config,experiment_config,cf,FEM_spaces_dict,use_rand, 
                                            logger=logger)
        logger.info(f'energy_init after initial prep = {energy_init_ap}')
        
        adios4dolfinx.write_mesh(filename, msh , 'BP4')
        adios4dolfinx.write_function(filename, u1_prep,  name="u1_prep")
        adios4dolfinx.write_function(filename, u2_prep,  name="u2_prep")
        if not only_ord:
            adios4dolfinx.write_function(filename, A_prep,  name="A_prep")
        logger.info(f'rank 0 has prepared initials'+'\n')
    


    end_time_initials = MPI.Wtime()
  

    elapsed_initials = end_time_initials - start_time_initials
    max_time_initials = comm.reduce(elapsed_initials, op=MPI.MAX, root=0)
    results_dict['max_time_initials'] = max_time_initials
    
    comm.Barrier()
    if rank == 0:
        logger.info(f'initials prepared and all ranks here in {max_time_initials:.6f} seconds for {size} ranks'+'\n')

    start_time_gf = MPI.Wtime()


    msh = adios4dolfinx.read_mesh(filename, comm,'BP4' )
    FEM_spaces_dict = s_def.def_get_all_V(msh,problem_dict,spaces_config_dict)
    x = FEM_spaces_dict['x']

    u1 = fem.Function(FEM_spaces_dict['V_ord_real_col'])
    u2 = fem.Function(FEM_spaces_dict['V_ord_real_col'])


    adios4dolfinx.read_function(filename, u1, name="u1_prep")
    adios4dolfinx.read_function(filename, u2, name="u2_prep")
    if not only_ord:
        A = fem.Function(FEM_spaces_dict['V_mag_col'])
        adios4dolfinx.read_function(filename, A, name="A_prep")
    else:
        A = inits.get_A1(A_type, 0.5*mag_scale, x,omega_by_pi_times_ten)
    H = inits.get_H(H_type, mag_scale, x,omega_by_pi_times_ten)
    
    if only_ord:
        energy_init_al = GL_energy.compute_energy_ord(u1,u2,A, kappa,comm)
    else:
        energy_init_al = GL_energy.compute_energy_full(u1,u2,A,H, kappa,inc_div,comm)
    
    if rank == 0: 
        logger.info(f'after loading: energy_init = {energy_init_al} ')

    if rank == 0:
        if minimizer_dict['conjugate']:
            logger.info(f"start with conjugate gradient flow"+'\n')
        else:
            logger.info(f"start with non-conjugate gradient flow"+'\n')



    u1,u2,A ,energy , div_size,  pot_qualy, results_dict =\
                    GL_min.compute_minimum_grad_flow(u1, u2, A , H, cf, FEM_spaces_dict,results_dict ,use_rand, logger=logger)



    comm.Barrier()
    end_time_gf = MPI.Wtime()
    elapsed_gf = end_time_gf - start_time_gf
    max_time_gf = comm.reduce(elapsed_gf, op=MPI.MAX, root=0)

    results_dict['max_time_grad_flow'] = max_time_gf
    results_dict['energy_after_grad_flow'] = energy



    if rank == 0:
        if minimizer_dict['conjugate']:
            logger.info(f"Max time for conjugate gradient flow across processes: {max_time_gf:.6f} seconds for {size} ranks"+'\n')
        else:
            logger.info(f"Max time for gradient flow across processes: {max_time_gf:.6f} seconds for {size} ranks"+'\n')


        



    start_time_Newton = MPI.Wtime()

    if Newton:
        u1,u2,A ,energy , div_size,  pot_qualy, results_dict =\
                        GL_min.compute_minimum_Newton(u1, u2, A , H, FEM_solver_config,cf, FEM_spaces_dict, results_dict, use_rand, energy_GF = energy, logger=logger)
        
   
        results_dict['energy_after_Newton'] = energy
        results_dict['energy_improvement_Newton_minus_grad_flow'] = results_dict['energy_after_Newton'] - results_dict['energy_after_grad_flow']

    
        comm.Barrier()
        end_time_Newton = MPI.Wtime()

        elapsed_Newton = end_time_Newton - start_time_Newton
        max_time_Newton = comm.reduce(elapsed_Newton, op=MPI.MAX, root=0)
        
        results_dict['max_time_Newton'] = max_time_Newton


        if rank == 0:
            logger.info(f"Max time for Newton across processes: {max_time_Newton:.6f} seconds for {size} ranks"+'\n')


    comp_eig = True
    # comp_eig = False


    comm.Barrier()

    if comp_eig:
        start_time_eigs = MPI.Wtime()

        num_eigs  = 7
        if rank == 0:
            logger.info(f'start computing eigenvalues with {total_dofs} dofs')

        eig_vals =\
                            GL_energy.compute_smallest_eigs(u1, u2, A , H, num_eigs,  problem_dict, FEM_spaces_dict,spaces_config_dict,minimizer_dict, logger = logger)
        
        end_time_eigs = MPI.Wtime()
        elapsed_eigs = end_time_eigs - start_time_eigs
        max_time_eigs = comm.reduce(elapsed_eigs, op=MPI.MAX, root=0)

        results_dict['max_time_eigs'] = max_time_eigs


        if eig_vals:
            eig_vals_bool = True
            eig_vals_vec = np.array(eig_vals)

            results_dict['EV_0_smallest_eig_val_duu_E'] = eig_vals_vec[0].tolist()
            results_dict['EV_1_other_eig_vals_duu_E'] = eig_vals_vec[1:].tolist()
            results_dict['EV_2_eig_vals_duu_E_scaled'] = np.round(eig_vals_vec/[abs(eig_vals_vec[0])],2).tolist()
        else:
            eig_vals_bool = False

        comm.Barrier()
        if rank == 0:
            logger.info(f'finished computing eigenvalues in {max_time_eigs:.6f} seconds'+'\n')


    if rank == 0 and comp_eig and eig_vals_bool:
        logger.info(f"smallest eigenvalue: {eig_vals_vec[0]}")
        logger.info(f"other eig_vals: {eig_vals_vec[1:]}"+'\n')
        logger.info(f"eig_vals scaled by smallest: {np.round(eig_vals_vec/[abs(eig_vals_vec[0])],2)}")
        



    NN_string ='_NN_used'
    rand_string ='_rand_used'

    

    filename = f'{FEM_solver_config["output_path"]}/ord_{grad_type}_{geo}_{circ_rad}_init_{u_type}_{A_type}_{H_type}_kappa_{kappa}_mag_scale_{mag_scale}_h_{h}_tol_{tol}_{Fem_type_mag}_{div_label}'
    
    if geo == "circle_slice":
        filename = f'{FEM_solver_config["output_path"]}/ord_{grad_type}_{geo}_{circ_rad}_{omega_by_pi_times_ten}_init_{u_type}_{A_type}_{H_type}_kappa_{kappa}_mag_scale_{mag_scale}_h_{h}_tol_{tol}_{Fem_type_mag}_{div_label}'


    if FEM_solver_config['use_NN_initial_guess']:
        filename +=NN_string

    filename +='.bp'  

    filename = Path(filename) 

    adios4dolfinx.write_mesh(filename, msh , 'BP4')
    adios4dolfinx.write_function(filename, u1,  name="u1")
    adios4dolfinx.write_function(filename, u2,  name="u2")
    if not only_ord:
        adios4dolfinx.write_function(filename, A,  name="MagPot")


    if rank == 0:
        logger.info('saving the solutions')
        local_comm = MPI.COMM_SELF
        msh_self = adios4dolfinx.read_mesh(filename, local_comm,'BP4' )
        FEM_spaces_dict = s_def.def_get_all_V(msh_self,problem_dict,spaces_config_dict)

        u1 = fem.Function(FEM_spaces_dict['V_ord_real_col'])
        u2 = fem.Function(FEM_spaces_dict['V_ord_real_col'])
        u_abs = fem.Function(FEM_spaces_dict['V_ord_real_col'])


        adios4dolfinx.read_function(filename, u1, name="u1")
        adios4dolfinx.read_function(filename, u2, name="u2")


        u_abs.x.array[:] = u1.x.array[:]**2 + u2.x.array[:]**2
        if geo != "circle_slice":
            omega_by_pi_times_ten = 0
        filename_save = Path(f'{FEM_solver_config["output_path"]}/{FEM_solver_config["results_filename"]}_plot_u1_init_{u_type}_{A_type}_{H_type}_kappa_{kappa}_h_{h}_tol_{tol}.bp')

       
        with dfx.io.VTXWriter(local_comm, filename_save, [u1]) as vtx:
            vtx.write(0.0)
   
        filename_save = Path(f'{FEM_solver_config["output_path"]}/{FEM_solver_config["results_filename"]}_plot_u2_init_{u_type}_{A_type}_{H_type}_kappa_{kappa}_h_{h}_tol_{tol}.bp')
        
        with dfx.io.VTXWriter(local_comm, filename_save, [u2]) as vtx:
            vtx.write(0.0)

        filename_save = Path(f'{FEM_solver_config["output_path"]}/{FEM_solver_config["results_filename"]}_plot_u_abs_init_{u_type}_{A_type}_{H_type}_kappa_{kappa}_h_{h}_tol_{tol}.bp')

        with dfx.io.VTXWriter(local_comm, filename_save, [u_abs]) as vtx:
            vtx.write(0.0)

        if not only_ord:

            MagPot = fem.Function(FEM_spaces_dict['V_mag_col'])
            adios4dolfinx.read_function(filename, MagPot, name="MagPot")
            V_dG = FEM_spaces_dict['V_dG']
            if dfx.__version__ == '0.10.0':
               B_exp = fem.Expression(ufl.rot(MagPot), V_dG.element.interpolation_points)
            elif dfx.__version__ == '0.9.0':
                B_exp = fem.Expression(ufl.rot(MagPot), V_dG.element.interpolation_points())
            # B_exp = fem.Expression(ufl.rot(MagPot), V_dG.element.interpolation_points)
            rot_A_dg = fem.Function(V_dG)
            rot_A_dg.interpolate(B_exp)

            filename_save = Path(f'{FEM_solver_config["output_path"]}/{FEM_solver_config["results_filename"]}_plot_curl_A_init_{u_type}_{A_type}_{H_type}_kappa_{kappa}_h_{h}_tol_{tol}.bp')
            
            with dfx.io.VTXWriter(local_comm, filename_save, [rot_A_dg]) as vtx:
                 vtx.write(0.0)
        logger.info('saved the solutions')





        max_time = max_time_initials + max_time_gf
        if Newton:
            max_time += max_time_Newton

        if rank == 0:
            if only_ord:
                logger.info(f'minimizer found after in total= {max_time}s with  {total_dofs} degrees of freedom for u in C on {size} ranks'+'\n')

            else:
                logger.info(f'minimizer found after in total= {max_time}s with {total_dofs} degrees of    freedom for (u,A) in C x R^2 on {size} ranks'+'\n')
            
            with open(FEM_solver_config['output_path']+'/'+filename_results_dict+'.yml', 'w') as outfile:
                yaml.dump(results_dict, outfile, default_flow_style=False)



    if rank == 0 :
        try:
            # Versucht, das Verzeichnis und alle Inhalte zu löschen
            shutil.rmtree(Path(cache_dir))
            logger.info(f"Erfolgreich gelöscht: {cache_dir}")
        except FileNotFoundError:
            logger.info(f"Verzeichnis nicht gefunden: {cache_dir}")
        except OSError as e:
            # Fängt Fehler wie Berechtigungsprobleme oder Datei-in-Verwendung ab
            logger.info(f"Fehler beim Löschen: {e.filename} - {e.strerror}")
