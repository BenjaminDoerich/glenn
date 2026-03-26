import numpy as np
import dolfinx as dfx
from dolfinx import  fem, plot, default_scalar_type
import basix

import ufl

 

###########################################  
# import os 
# newpath = "sol_screenshots/"
# if not os.path.exists(newpath):
#    os.makedirs(newpath)  
###########################################

import pyvista

pyvista.OFF_SCREEN = True


def plot_sol_all_3(msh,u1,u2,degree_FEM_ord,kappa,h,energy,tol,title,init,cf):
 
    
    absol = ufl.inner(u1,u1) + ufl.inner(u2,u2)
   
    
    # element_H1 = ufl.FiniteElement("Lagrange", msh.ufl_cell(), degree = degree_FEM_ord)
    element_H1 = basix.ufl.element("Lagrange", msh.topology.cell_name(), degree = degree_FEM_ord)

    V_plot  = fem.functionspace(msh, element_H1) 
    
    
    absol_expr = fem.Expression(absol, V_plot.element.interpolation_points())
    u_plot = fem.Function(V_plot)
    u_plot.interpolate(absol_expr)
    
    
    u1_expr = fem.Expression(u1, V_plot.element.interpolation_points())

    u1_plot = fem.Function(V_plot)
    u1_plot.interpolate(u1_expr)
    
    u2_expr = fem.Expression(u2, V_plot.element.interpolation_points())

    u2_plot = fem.Function(V_plot)
    u2_plot.interpolate(u2_expr)

    ###############################################
    # plot the absolute value^2 of the minimizer
    ###############################################
    
    transparent = True
    figsize = 800
    
    u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V_plot)
    
    u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
    u_grid.point_data["u"] = u_plot.x.array.real
    u_grid.set_active_scalars("u")
    u_plotter = pyvista.Plotter(off_screen=True)
    
    warped = u_grid.warp_by_scalar()
    # u_plotter.add_mesh(u_grid, show_edges=False, show_scalar_bar=True)
    # u_plotter.add_mesh(u_grid, show_edges=True)
    
    
    
    u_plotter.add_title(f'kappa = {kappa}, h =  {h}, energy = {energy}, tol = {tol}', font_size=10)
    u_plotter.view_xy()
    
    
    # if not pyvista.OFF_SCREEN:
    # u_plotter.show(screenshot=f'sol_screenshots/ord_{geo}_{circ_rad}_init_{u_type}_kappa_{kappa}_mag_scale_{mag_scale}_h_{h}_tol_{tol}_{Fem_type_mag}_{div_label}.png')
     
    
        
    
    subplotter = pyvista.Plotter(shape=(1, 3))
    subplotter.view_xy()
    
    subplotter.subplot(0, 0)
    if init:
        subplotter.add_text("init: abs(u)^2", font_size=14, color="black", position="upper_edge")
    else:
        subplotter.add_text("abs(u)^2", font_size=14, color="black", position="upper_edge")

    subplotter.add_mesh(u_grid, show_edges=False, show_scalar_bar=True)
    subplotter.view_xy()
    
    subplotter.subplot(0, 1)
    subplotter.add_text("Warped function", position="upper_edge", font_size=14, color="black")
    subplotter.set_position([-5, 2.6, -0.1])
    subplotter.set_focus([2, -1, 0.0])
    subplotter.set_viewup([0, 0, 0.1])
    subplotter.add_mesh(warped, show_edges=False, show_scalar_bar=True)
    
    subplotter.add_text(f'kappa = {kappa}, h =  {h}, tol = {tol} , \n energy = {energy},', font_size=10,  position="lower_edge")
    
    
    subplotter.subplot(0, 2)
    phase = fem.Function(V_plot)
    phase.x.array[:] =   np.arctan2(u1_plot.x.array[:] , u2_plot.x.array[:])
    
    phase_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
    phase_grid.point_data["phi"] = phase.x.array.real
    phase_grid.set_active_scalars("phi")
    
    subplotter.add_text("phase of ord", font_size=14, color="black", position="upper_edge")
    subplotter.add_mesh(phase_grid, show_edges=False,  scalar_bar_args={'title': 'phase of ord'})
    subplotter.view_xy()
    
    
    
    if pyvista.OFF_SCREEN:
        subplotter.screenshot(title, transparent_background=transparent , window_size=[2 * figsize, figsize])
    else:
        subplotter.show()
    
    
  ###################### 
    
def plot_sol_real_imag(msh,u_real,u_imag,degree_FEM_ord,kappa,h,energy,tol,title,init,cf):
    
    # element_H1 = ufl.FiniteElement("Lagrange", msh.ufl_cell(), degree = degree_FEM_ord)
    element_H1 = basix.ufl.element("Lagrange", msh.topology.cell_name(), degree = degree_FEM_ord)

    
    V_plot  = fem.functionspace(msh, element_H1) 
    u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V_plot)


    u_real_expr = fem.Expression(u_real, V_plot.element.interpolation_points())
    u_real_plot = fem.Function(V_plot)
    u_real_plot.interpolate(u_real_expr)
    
    
    u_imag_expr = fem.Expression(u_imag, V_plot.element.interpolation_points())
    u_imag_plot = fem.Function(V_plot)
    u_imag_plot.interpolate(u_imag_expr)
    
    
    u_abs_expr = fem.Expression(ufl.inner(u_real,u_real) + ufl.inner(u_imag,u_imag), V_plot.element.interpolation_points())
    u_abs_plot = fem.Function(V_plot)
    u_abs_plot.interpolate(u_abs_expr)
    
   

    subplotter = pyvista.Plotter(shape=(1, 3))
    subplotter.view_xy()
    


    subplotter.subplot(0, 0)

    u_real_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
    u_real_grid.point_data["u"] = u_real_plot.x.array.real
    u_real_grid.set_active_scalars("u")
    u_plotter = pyvista.Plotter(off_screen=True)
    # u_plotter.add_mesh(u_grid, show_edges=True)
    u_plotter.add_mesh(u_real_grid, show_edges=False)
    u_plotter.add_title(f'kappa = {kappa}, h =  {h}, energy = {energy}, tol = {tol}', font_size=10)
    
    if init:
       subplotter.add_text("init: u_real", font_size=14, color="black", position="upper_edge")
    else:
        subplotter.add_text("u_real", font_size=14, color="black", position="upper_edge")
    subplotter.add_mesh(u_real_grid, show_edges=False, show_scalar_bar=True)
    subplotter.view_xy()
    


    u_imag_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
    u_imag_grid.point_data["u"] = u_imag_plot.x.array.real
    u_imag_grid.set_active_scalars("u")
    
   
    subplotter.subplot(0, 1)
    subplotter.add_text("u_imag", font_size=14, color="black", position="upper_edge")
    subplotter.add_mesh(u_imag_grid, show_edges=False, show_scalar_bar=True)
    subplotter.view_xy()
    
    
    subplotter.add_text(f'kappa = {kappa}, h =  {h}, tol = {tol} , \n energy = {energy},', font_size=10,  position="lower_edge")
    

    u_abs_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
    u_abs_grid.point_data["u"] = u_abs_plot.x.array.real
    u_abs_grid.set_active_scalars("u")
   
    subplotter.subplot(0, 2)
    subplotter.add_text("abs(u)^2", font_size=14, color="black", position="upper_edge")
    subplotter.add_mesh(u_abs_grid, show_edges=False, show_scalar_bar=True,scalar_bar_args={'title': 'abs'})
    subplotter.view_xy()    


    subplotter.screenshot(title)

    
    
        


def plot_vorticity(msh,u1,u2,A,degree_FEM_ord,kappa,h,title_vor,cf):

    
   
    cur1 = ufl.rot( (ufl.inner(u1,u1) + ufl.inner(u2,u2) )* A ) + ufl.rot(A)
    cur2 = ufl.rot( u1*ufl.grad(u2) - u2*ufl.grad(u1)  )
    
    
    # element_H1 = ufl.FiniteElement("Lagrange", msh.ufl_cell(), degree = degree_FEM_ord)
    element_H1 = basix.ufl.element("Discontinuous Lagrange", msh.topology.cell_name(), degree = degree_FEM_ord)
    
    V_plot_dg  = fem.functionspace(msh, element_H1) 
    
    
    absol_expr = fem.Expression(cur1 + 1/kappa * cur2, V_plot_dg.element.interpolation_points())
    u_plot = fem.Function(V_plot_dg)
    u_plot.interpolate(absol_expr)
    
    
    ###############################################
    # plot the absolute value^2 of the minimizer
    ###############################################
    import pyvista
    
    u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V_plot_dg)
    
    u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
    u_grid.point_data["u"] = u_plot.x.array.real
    u_grid.set_active_scalars("u")
    u_plotter = pyvista.Plotter(off_screen=True)
    u_plotter.add_mesh(u_grid, show_edges=False)
    
    u_plotter.view_xy()
    
    if pyvista.OFF_SCREEN:
        u_plotter.screenshot(title_vor)
    else:
        u_plotter.show()



def plot_A(A,V,msh,title):


    gdim = msh.geometry.dim
    print(gdim)
    V0 = fem.functionspace(msh, ("Discontinuous Lagrange", 1, (gdim,)))
    A0 = fem.Function(V0, dtype=default_scalar_type)
    A0.interpolate(A)        
    

    with dfx.io.VTXWriter(V.mesh.comm, title , A0, "BP4") as vtx:
        vtx.write(0)
        print('saved solution')  
    
    
    

  