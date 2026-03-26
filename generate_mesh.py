import numpy as np
import dolfinx as dfx
import ufl
from petsc4py import PETSc
from mpi4py import MPI
from typing import Union
import gmsh

if dfx.__version__ == '0.9.0':
    from dolfinx.io import gmshio



import pyvista
from dolfinx.plot import vtk_mesh
from dolfinx import mesh
from dolfinx import fem
import plot_mesh

import adios4dolfinx


pyvista.OFF_SCREEN = True



import os 

newpath = "meshes/"
if not os.path.exists(newpath):
   os.makedirs(newpath) 

def build_Cdomain(h:float, h_corner:float, write_mesh:bool = False, geometry:str = "L",circ_rad:float = 1.0,omega_by_pi_times_ten:int = 10):
    # Initialize Gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal",0)
    model = gmsh.model()
    model.add("my_model")

    lc = h
    lc_edge = h_corner


    if geometry == "unit_square":


    # Create points
        p1 = model.geo.addPoint(0.0, 0.0, 0.0, lc)
        p2 = model.geo.addPoint(1.0, 0.0, 0.0, lc)
        p3 = model.geo.addPoint(1.0, 1.0, 0.0, lc)
        p4 = model.geo.addPoint(0.0, 1.0, 0.0, lc)
        
            
        
        # Create lines
        l1 = model.geo.addLine(p1, p2)
        l2 = model.geo.addLine(p2, p3)
        l3 = model.geo.addLine(p3, p4)
        l4 = model.geo.addLine(p4, p1)
    
    
    
        # Create plane surface
        cl1 = model.geo.addCurveLoop([l1,l2, l3, l4])

        pl = model.geo.addPlaneSurface([cl1])

    if geometry == "box_hole":


    # Create points
        p1 = gmsh.model.geo.addPoint(-1.0, -1.0, 0.0, lc)
        p2 = gmsh.model.geo.addPoint(1.0, -1.0, 0.0, lc)
        p3 = gmsh.model.geo.addPoint(1.0, 1.0, 0.0, lc)
        p4 = gmsh.model.geo.addPoint(-1.0, 1.0, 0.0, lc)
        
        q1 = gmsh.model.geo.addPoint(-0.5, -0.5, 0.0, lc_edge)
        q2 = gmsh.model.geo.addPoint(0.5, -0.5, 0.0, lc_edge)
        q3 = gmsh.model.geo.addPoint(0.5, 0.5, 0.0, lc_edge)
        q4 = gmsh.model.geo.addPoint(-0.5, 0.5, 0.0, lc_edge)
        
            
        
        # Create lines
        l1 = gmsh.model.geo.addLine(p1, p2)
        l2 = gmsh.model.geo.addLine(p2, p3)
        l3 = gmsh.model.geo.addLine(p3, p4)
        l4 = gmsh.model.geo.addLine(p4, p1)
    
        n1 = gmsh.model.geo.addLine(q1, q2)
        n2 = gmsh.model.geo.addLine(q2, q3)
        n3 = gmsh.model.geo.addLine(q3, q4)
        n4 = gmsh.model.geo.addLine(q4, q1)
    
    
        # Create plane surface
        cl1 = gmsh.model.geo.addCurveLoop([l1,l2, l3, l4])
        cl2 = gmsh.model.geo.addCurveLoop([n1,n2, n3, n4])

        pl = gmsh.model.geo.addPlaneSurface([cl1,cl2])

    if geometry == "annulus":
        
        r1 = 1.0
        r2 = 0.5

        # Large circle
        p1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc)
        p2 = gmsh.model.geo.addPoint(r1, 0.0, 0.0, lc)
        p3 = gmsh.model.geo.addPoint(0.0, r1, 0.0, lc)
        p5 = gmsh.model.geo.addPoint(-r1, 0.0, 0.0, lc)    
        p6 = gmsh.model.geo.addPoint(0.0, -r1, 0.0, lc)    


        c1 = gmsh.model.geo.addCircleArc(p2, p1, p3)
        c2 = gmsh.model.geo.addCircleArc(p3, p1, p5)
        c3 = gmsh.model.geo.addCircleArc(p5, p1, p6)
        c4 = gmsh.model.geo.addCircleArc(p6, p1, p2)

        cl1 = gmsh.model.geo.addCurveLoop([c1,c2, c3, c4])

        # small circle

        q1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc)
        q2 = gmsh.model.geo.addPoint(r2, 0.0, 0.0, lc)
        q3 = gmsh.model.geo.addPoint(0.0, r2, 0.0, lc)
        q5 = gmsh.model.geo.addPoint(-r2, 0.0, 0.0, lc)    
        q6 = gmsh.model.geo.addPoint(0.0, -r2, 0.0, lc)    


        d1 = gmsh.model.geo.addCircleArc(q2, q1, q3)
        d2 = gmsh.model.geo.addCircleArc(q3, q1, q5)
        d3 = gmsh.model.geo.addCircleArc(q5, q1, q6)
        d4 = gmsh.model.geo.addCircleArc(q6, q1, q2)


        cl2 = gmsh.model.geo.addCurveLoop([d1,d2, d3, d4])

        #substract from each other
        pl = gmsh.model.geo.addPlaneSurface([cl1,cl2])


    if geometry == "circle":
        
        r1 = circ_rad

        # Large circle
        p1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc)
        p2 = gmsh.model.geo.addPoint(r1, 0.0, 0.0, lc)
        p3 = gmsh.model.geo.addPoint(0.0, r1, 0.0, lc)
        p5 = gmsh.model.geo.addPoint(-r1, 0.0, 0.0, lc)    
        p6 = gmsh.model.geo.addPoint(0.0, -r1, 0.0, lc)    


        c1 = gmsh.model.geo.addCircleArc(p2, p1, p3)
        c2 = gmsh.model.geo.addCircleArc(p3, p1, p5)
        c3 = gmsh.model.geo.addCircleArc(p5, p1, p6)
        c4 = gmsh.model.geo.addCircleArc(p6, p1, p2)

        cl1 = gmsh.model.geo.addCurveLoop([c1,c2, c3, c4])


        #substract from each other
        pl = gmsh.model.geo.addPlaneSurface([cl1])
        
    if geometry == "circle_slice":
        
        r1 = circ_rad
        omega = np.pi / 10 * omega_by_pi_times_ten

        lc_small = 0.2*lc

        # Large circle
        center = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc_small)
        
        num_points = 6 
        angle_step = omega / (num_points-1)

        points_x = np.zeros(num_points)
        points_y = np.zeros(num_points)

        
        for i in range(num_points):
            points_x[i] = r1 * np.cos( i* angle_step)
            points_y[i] = r1 * np.sin( i* angle_step)

        

        p0_add_1 = gmsh.model.geo.addPoint(r1 / 4. , 0.0, 0.0,lc)
        p0_add_2 = gmsh.model.geo.addPoint(r1 / 2. , 0.0, 0.0,lc)

        p5_add_1 = gmsh.model.geo.addPoint(r1 / 4. *np.cos( (num_points-1)* angle_step) , r1 / 4. *np.sin( (num_points-1)* angle_step), 0.0, lc)
        p5_add_2 = gmsh.model.geo.addPoint(r1 / 2. *np.cos( (num_points-1)* angle_step) , r1 / 2. *np.sin( (num_points-1)* angle_step), 0.0, lc)
        
        center = gmsh.model.geo.addPoint(0, 0, 0,lc_small)

        
        p0 = gmsh.model.geo.addPoint(points_x[0], points_y[0], 0.0, lc)
        p1 = gmsh.model.geo.addPoint(points_x[1], points_y[1], 0.0, lc)
        p2 = gmsh.model.geo.addPoint(points_x[2], points_y[2], 0.0, lc)
        p3 = gmsh.model.geo.addPoint(points_x[3], points_y[3], 0.0, lc)
        p4 = gmsh.model.geo.addPoint(points_x[4], points_y[4], 0.0, lc)
        p5 = gmsh.model.geo.addPoint(points_x[5], points_y[5], 0.0, lc)


        c1 = gmsh.model.geo.addCircleArc(p0, center, p1)
        c2 = gmsh.model.geo.addCircleArc(p1, center, p2)
        c3 = gmsh.model.geo.addCircleArc(p2, center, p3)
        c4 = gmsh.model.geo.addCircleArc(p3, center, p4)
        c5 = gmsh.model.geo.addCircleArc(p4, center, p5)

        

        l1 = gmsh.model.geo.addLine(center,p0_add_1)
        l2 = gmsh.model.geo.addLine(p0_add_1,p0_add_2)
        l3 = gmsh.model.geo.addLine(p0_add_2,p0)


        l4 = gmsh.model.geo.addLine(p5,p5_add_2)
        l5 = gmsh.model.geo.addLine(p5_add_2,p5_add_1)
        l6 = gmsh.model.geo.addLine(p5_add_1,center)
        
        cl1 = gmsh.model.geo.addCurveLoop([l1,l2,l3,c1,c2, c3, c4,c5,l4,l5,l6])

        #substract from each other
        pl = gmsh.model.geo.addPlaneSurface([cl1])


    if geometry == "L":
        p1 = gmsh.model.geo.addPoint(0.0, -1.0, 0.0, lc)
        p2 = gmsh.model.geo.addPoint(1.0, -1.0, 0.0, lc)
        p3 = gmsh.model.geo.addPoint(1.0, 1.0, 0.0, lc)
        p4 = gmsh.model.geo.addPoint(-1.0, 1.0, 0.0, lc)
        p5 = gmsh.model.geo.addPoint(-1.0, 0.0, 0.0, lc)
        p6 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc)

        l1 = gmsh.model.geo.addLine(p1, p2)
        l2 = gmsh.model.geo.addLine(p2, p3)
        l3 = gmsh.model.geo.addLine(p3, p4)
        l4 = gmsh.model.geo.addLine(p4, p5)
        l5 = gmsh.model.geo.addLine(p5, p6)
        l6 = gmsh.model.geo.addLine(p6, p1)

        cl1 = gmsh.model.geo.addCurveLoop([l1,l2, l3, l4,l5,l6])
        pl = gmsh.model.geo.addPlaneSurface([cl1])



    if geometry == "Lshape":
        p1 = gmsh.model.geo.addPoint(0.0 , 0.0, 0.0, lc)
        p2 = gmsh.model.geo.addPoint(1.0, 0.0, 0.0, lc)
        p3 = gmsh.model.geo.addPoint(1.0, 0.5, 0.0, lc)
        p4 = gmsh.model.geo.addPoint(0.5,0.5, 0.0, lc)
        p5 = gmsh.model.geo.addPoint(0.5, 1.0, 0.0, lc)
        p6 = gmsh.model.geo.addPoint(0.0, 1.0, 0.0, lc)

        l1 = gmsh.model.geo.addLine(p1, p2)
        l2 = gmsh.model.geo.addLine(p2, p3)
        l3 = gmsh.model.geo.addLine(p3, p4)
        l4 = gmsh.model.geo.addLine(p4, p5)
        l5 = gmsh.model.geo.addLine(p5, p6)
        l6 = gmsh.model.geo.addLine(p6, p1)

        cl1 = gmsh.model.geo.addCurveLoop([l1,l2, l3, l4,l5,l6])
        pl = gmsh.model.geo.addPlaneSurface([cl1])

    model.geo.synchronize()

    # physical group
    if dfx.__version__ == '0.9.0': 
        model.addPhysicalGroup(2,[pl],tag = 1)

    if dfx.__version__ == '0.10.0': 
        model.addPhysicalGroup(2,[pl])


    # Generate the meshs
    model.mesh.generate(2)

    gmsh_model_rank = 0
    mesh_comm = MPI.COMM_WORLD
    if dfx.__version__ == '0.9.0': 
        domain, _, _ = gmshio.model_to_mesh(model, mesh_comm, gmsh_model_rank, gdim=2)
    if dfx.__version__ == '0.10.0':
        domain_data = dfx.io.gmsh.model_to_mesh(model, mesh_comm, gmsh_model_rank, gdim=2)
        domain = domain_data.mesh

    if write_mesh:
        
        if geometry == "unit_square":
            adios4dolfinx.write_mesh(f"meshes/unit_square_mesh_{h}", domain, engine="BP4")
            
        if geometry == "square":
            adios4dolfinx.write_mesh(f"meshes/square_mesh_{h}.msh", domain, engine="BP4")

        
        if geometry == "box_hole":
            adios4dolfinx.write_mesh(f"meshes/hole_mesh_{h}.msh", domain, engine="BP4")

        if geometry == "circle":
            adios4dolfinx.write_mesh(f"meshes/circle_{r1}_mesh_{h}.msh", domain, engine="BP4")

            
        if geometry == "circle_slice":
            adios4dolfinx.write_mesh(f"meshes/circle_slice_{r1}_angle_{omega_by_pi_times_ten}_mesh_{h}.msh", domain, engine="BP4")

            
        if geometry == "annulus":
            adios4dolfinx.write_mesh(f"meshes/annulus_{h}.msh", domain, engine="BP4")

            
        if geometry == "L":
            adios4dolfinx.write_mesh(f"meshes/L_mesh_{h}.msh", domain, engine="BP4")
        
        if geometry == "Lshape":
            adios4dolfinx.write_mesh(f"meshes/Lshape_mesh_{h}", domain, engine="BP4")


    # finalize gmsh
    gmsh.clear()
    gmsh.finalize()

    return domain



if __name__ == "__main__":

    h = 0.5
    h_c = h
    
    Ms = np.array([8,16,32,64,128,256,512,1024])
    Ms = np.array([32]) 
    has  = 4/Ms

    
    geometry = "unit_square"
    geometry = 'circle_slice'
    rad = 1.0
    geometry = "L"
    geometry = "Lshape"

    omega_by_pi_times_ten = 15
    
    
    
    for h in has:
        h_c = h
        
        if geometry == "box_hole":
            build_Cdomain(h,h_c,write_mesh=True,geometry="box_hole")
    
    
        if geometry == "circle":
            build_Cdomain(h,h_c,write_mesh=True,geometry="circle",circ_rad = rad)
            
        if geometry == "circle_slice":
            build_Cdomain(h,h_c,write_mesh=True,geometry="circle_slice",circ_rad = rad,omega_by_pi_times_ten=omega_by_pi_times_ten)
            
        if geometry == "annulus":
            build_Cdomain(h,h_c,write_mesh=True,geometry="annulus")
            
        if geometry == "L":
            build_Cdomain(h,h_c,write_mesh=True,geometry="L")

        if geometry == "Lshape":
            build_Cdomain(h,h_c,write_mesh=True,geometry="Lshape")
    
        if geometry == "square":
            build_Cdomain(h,h_c,write_mesh=True,geometry="square")
            
        if geometry == "unit_square":
            build_Cdomain(h,h_c,write_mesh=True,geometry="unit_square")
    
        
    
        plot_mesh_bool = True
    
        if plot_mesh_bool:
    
            if geometry == "unit_square":
                build_Cdomain(h,h_c,write_mesh=True,geometry="unit_square")
                mesh2 = adios4dolfinx.read_mesh(f"meshes/unit_square_mesh_{h}", comm=MPI.COMM_WORLD, engine="BP4" )
        
                V = fem.functionspace(mesh2, ("CG", 1))
                        
                print('about to plot')
                plot_mesh.plot_sol_pyvista(V,"unit_square_h_"+str(h))
                print('plotted')
            
            
            if geometry == "square":
                build_Cdomain(h,h_c,write_mesh=True,geometry="square")
                mesh2 = adios4dolfinx.read_mesh(f"meshes/square_mesh_{h}.msh", comm=MPI.COMM_WORLD, engine="BP4" )

                V = fem.functionspace(mesh2, ("CG", 1))
                        
                print('about to plot')
                plot_mesh.plot_sol_pyvista(V,"square_h_"+str(h))
                print('plotted')
            
            
            if geometry == "box_hole":
                build_Cdomain(h,h_c,write_mesh=True,geometry="box_hole")
                mesh2 = adios4dolfinx.read_mesh(f"meshes/hole_mesh_{h}.msh", comm=MPI.COMM_WORLD, engine="BP4" )
        
                V = fem.functionspace(mesh2, ("CG", 1))
                        
                print('about to plot')
                plot_mesh.plot_sol_pyvista(V,"box_hole_h_"+str(h))
                print('plotted')

            if geometry == "Lshape":
                build_Cdomain(h,h_c,write_mesh=True,geometry="Lshape")
                mesh2 = adios4dolfinx.read_mesh(f"meshes/Lshape_mesh_{h}", comm=MPI.COMM_WORLD, engine="BP4" )
        
                V = fem.functionspace(mesh2, ("CG", 1))
                        
                print('about to plot')
                plot_mesh.plot_sol_pyvista(V,"Lshape_h_"+str(h))
                print('plotted')
        
        
        
            if geometry == "circle":
                mesh2 = adios4dolfinx.read_mesh(f"meshes/circle_{rad}_mesh_{h}.msh", comm=MPI.COMM_WORLD, engine="BP4" )
        
                V = fem.functionspace(mesh2, ("CG", 1))
                        
                print('about to plot')
                plot_mesh.plot_sol_pyvista(V,"circle_h_"+str(h))
                print('plotted')
                
                
            if geometry == "circle_slice":
                mesh2 = adios4dolfinx.read_mesh(f"meshes/circle_slice_{rad}_angle_{omega_by_pi_times_ten}_mesh_{h}.msh", comm=MPI.COMM_WORLD, engine="BP4" )

                V = fem.functionspace(mesh2, ("CG", 1))
                        
                print('about to plot')
                plot_mesh.plot_sol_pyvista(V,"circle_slice_h_"+str(h)+"_angle_"+str(omega_by_pi_times_ten))
                print('plotted')
                
                
            if geometry == "annulus":
                mesh2 = adios4dolfinx.read_mesh(f"meshes/annulus_{h}.msh", comm=MPI.COMM_WORLD, engine="BP4" )
        
                V = fem.functionspace(mesh2, ("CG", 1))
                        
                print('about to plot')
                plot_mesh.plot_sol_pyvista(V,"annulus_h_"+str(h))
                print('plotted')
                
                
                
            if geometry == "L":
                mesh2 = adios4dolfinx.read_mesh(f"meshes/L_mesh_{h}.msh", comm=MPI.COMM_WORLD, engine="BP4" )
        
                V = fem.functionspace(mesh2, ("CG", 1))
                        
                print('about to plot')
                plot_mesh.plot_sol_pyvista(V,"L_h_"+str(h))
                print('plotted')
    
