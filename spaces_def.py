import numpy as np
from dolfinx import mesh, fem, default_scalar_type
import basix
import ufl




def def_get_all_V(msh,problem_dict,spaces_config_dict):
    use_bc = True
    geo = problem_dict['geo']
    degree_FEM_ord = spaces_config_dict['degree_FEM_ord']
    Fem_type_mag = spaces_config_dict['Fem_type_mag']
    Nedelec_kind = spaces_config_dict['Nedelec_kind']
    degree_FEM_mag = spaces_config_dict['degree_FEM_mag']
    inc_div = spaces_config_dict['inc_div']

    V_div = None



    bc = None
    element_H1 = basix.ufl.element("Lagrange", msh.topology.cell_name(), degree = degree_FEM_ord)
    element_dG = basix.ufl.element(basix.ElementFamily.P, basix.CellType.triangle, 0, lagrange_variant=basix.LagrangeVariant.equispaced, discontinuous=True,shape = (1,))
    V_dG = fem.functionspace(msh,element_dG)

    if Fem_type_mag == 'Nedelec':
        if Nedelec_kind == 1:
            # "Nedelec 1st kind H(curl)", msh.ufl_cell(), 1)
            # https://defelement.com/elements/nedelec1.html
            element_curl = basix.ufl.element(basix.ElementFamily.N1E, basix.CellType.triangle, degree =  degree_FEM_mag)
          
        if Nedelec_kind == 2:
            #https://defelement.com/elements/nedelec2.html
            element_curl = basix.ufl.element(basix.ElementFamily.N2E, basix.CellType.triangle, degree =  degree_FEM_mag)

        V_mag = fem.functionspace(msh, element_curl)
        x= ufl.SpatialCoordinate(V_mag.mesh)

        bc_V = None
      
        
    if Fem_type_mag == 'Nedelec':
        element_H1_div = basix.ufl.element("Lagrange", msh.topology.cell_name(), degree = degree_FEM_mag)
        V_div = fem.functionspace(msh,element_H1_div)

      
      
    if Fem_type_mag == 'Lagrange':
        element_curl = basix.ufl.element(basix.ElementFamily.P, basix.CellType.triangle, degree_FEM_mag, lagrange_variant=basix.LagrangeVariant.equispaced, shape=(2,))
        
     
        
    element_mix =  basix.ufl.mixed_element([element_H1,element_H1, element_curl]) 
    V = fem.functionspace(msh,element_mix)
    x= ufl.SpatialCoordinate(V.mesh)
  
    V_ord = fem.functionspace(msh, basix.ufl.mixed_element([element_H1,element_H1]) )
    V_mag = V.sub(2)
    
    V_ord_real = V.sub(0)
    V_ord_imag = V.sub(1)


    if Fem_type_mag == 'Nedelec':
        element_mix_ext =  basix.ufl.mixed_element([element_H1,element_H1, element_curl,element_H1]) 
        V_ext = fem.functionspace(msh,element_mix_ext)
    else:
        V_ext = V


    if use_bc and Fem_type_mag == 'Lagrange':

        if geo == 'unit_square':
          
            def side2(x):
                return np.logical_and(np.isclose(x[0], 0), x[1] < 1)
            
            def side3(x):
                return np.logical_and(np.isclose(x[0], 1), x[1] < 1)
                       
            def side5(x):
                return np.logical_and(np.isclose(x[1], 0), x[0] < 1)
            
            def side6(x):
                return np.logical_and(np.isclose(x[1], 1), x[0] < 1)
            

          
            boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side2)
            boundary_dofs_x = fem.locate_dofs_topological(V_mag.sub(0), msh.topology.dim - 1, boundary_facets)   
            bc2 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag.sub(0))
            
            boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side3)
            boundary_dofs_x = fem.locate_dofs_topological(V_mag.sub(0), msh.topology.dim - 1, boundary_facets)
            bc3 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag.sub(0))
          
            
          
        
            boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side5)
            boundary_dofs_x = fem.locate_dofs_topological(V_mag.sub(1), msh.topology.dim - 1, boundary_facets)
            bc5 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag.sub(1))
            
            boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side6)
            boundary_dofs_x = fem.locate_dofs_topological(V_mag.sub(1), msh.topology.dim - 1, boundary_facets)
            bc6 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag.sub(1))
        
            bc_V = [bc2, bc3,bc5,bc6]

        if geo == 'square':
          
            def side2(x):
                return np.logical_and(np.isclose(x[0], -1), x[1] < 1)
            
            def side3(x):
                return np.logical_and(np.isclose(x[0], 1), x[1] < 1)
                       
            def side5(x):
                return np.logical_and(np.isclose(x[1], -1), x[0] < 1)
            
            def side6(x):
                return np.logical_and(np.isclose(x[1], 1), x[0] < 1)
            

          
            boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side2)
            boundary_dofs_x = fem.locate_dofs_topological(V_mag.sub(0), msh.topology.dim - 1, boundary_facets)   
            bc2 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag.sub(0))
            
            boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side3)
            boundary_dofs_x = fem.locate_dofs_topological(V_mag.sub(0), msh.topology.dim - 1, boundary_facets)
            bc3 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag.sub(0))
          
            
          
        
            boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side5)
            boundary_dofs_x = fem.locate_dofs_topological(V_mag.sub(1), msh.topology.dim - 1, boundary_facets)
            bc5 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag.sub(1))
            
            boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side6)
            boundary_dofs_x = fem.locate_dofs_topological(V_mag.sub(1), msh.topology.dim - 1, boundary_facets)
            bc6 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag.sub(1))
        
            bc_V = [bc2, bc3,bc5,bc6]
   
        if geo == 'box_hole':

            def side1(x):
                return np.logical_and(np.isclose(x[0], -0.5), x[1] < 1)
            
            def side2(x):
                return np.logical_and(np.isclose(x[0], -1), x[1] < 1)
            
            def side3(x):
                return np.logical_and(np.isclose(x[0], 1), x[1] < 1)
            
            def side4(x):
                return np.logical_and(np.isclose(x[0], 0.5), x[1] < 1)
            
            def side5(x):
                return np.logical_and(np.isclose(x[1], 0.5), x[0] < 1)
            
            def side6(x):
                return np.logical_and(np.isclose(x[1], -0.5), x[0] < 1)
            
            
            def side7(x):
                return np.logical_and(np.isclose(x[1], -1), x[0] < 1)
            
            def side8(x):
                return np.logical_and(np.isclose(x[1], 1), x[0] < 1)
            
            
            
            boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side1)
            boundary_dofs_x = fem.locate_dofs_topological(V_mag.sub(0), msh.topology.dim - 1, boundary_facets)   
            bc1 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag.sub(0))
          
            boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side2)
            boundary_dofs_x = fem.locate_dofs_topological(V_mag.sub(0), msh.topology.dim - 1, boundary_facets)   
            bc2 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag.sub(0))
            
            boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side3)
            boundary_dofs_x = fem.locate_dofs_topological(V_mag.sub(0), msh.topology.dim - 1, boundary_facets)
            bc3 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag.sub(0))
            
            boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side4)
            boundary_dofs_x = fem.locate_dofs_topological(V_mag.sub(0), msh.topology.dim - 1, boundary_facets)
            bc4 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag.sub(0))
          
            
       
            boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side5)
            boundary_dofs_x = fem.locate_dofs_topological(V_mag.sub(1), msh.topology.dim - 1, boundary_facets)
            bc5 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag.sub(1))
            
            boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side6)
            boundary_dofs_x = fem.locate_dofs_topological(V_mag.sub(1), msh.topology.dim - 1, boundary_facets)
            bc6 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag.sub(1))
            
                      
            boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side7)
            boundary_dofs_x = fem.locate_dofs_topological(V_mag.sub(1), msh.topology.dim - 1, boundary_facets)
            bc7 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag.sub(1))
            
                      
            boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side8)
            boundary_dofs_x = fem.locate_dofs_topological(V_mag.sub(1), msh.topology.dim - 1, boundary_facets)
            bc8 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag.sub(1))
        
            # bc = [bc2, bc3,bc5,bc6]
            bc_V = [bc1, bc2,bc3,bc4,bc5,bc6,bc7,bc8]            

        if geo == 'L':

            def side1(x):
                return np.logical_and(np.isclose(x[0], 0), x[1] < 1)
            
            def side2(x):
                return np.logical_and(np.isclose(x[0], -1), x[1] < 1)
            
            def side3(x):
                return np.logical_and(np.isclose(x[0], 1), x[1] < 1)
            
            def side4(x):
                return np.logical_and(np.isclose(x[1], 0), x[0] < 1)
            
            
            def side5(x):
                return np.logical_and(np.isclose(x[1], -1), x[0] < 1)
            
            def side6(x):
                return np.logical_and(np.isclose(x[1], 1), x[0] < 1)
            
            
            
            boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side1)
            boundary_dofs_x = fem.locate_dofs_topological(V_mag.sub(0), msh.topology.dim - 1, boundary_facets)   
            bc1 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag.sub(0))
          
            boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side2)
            boundary_dofs_x = fem.locate_dofs_topological(V_mag.sub(0), msh.topology.dim - 1, boundary_facets)   
            bc2 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag.sub(0))
            
            boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side3)
            boundary_dofs_x = fem.locate_dofs_topological(V_mag.sub(0), msh.topology.dim - 1, boundary_facets)
            bc3 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag.sub(0))
          
            
          
            boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side4)
            boundary_dofs_x = fem.locate_dofs_topological(V_mag.sub(1), msh.topology.dim - 1, boundary_facets)
            bc4 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag.sub(1))
        
            boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side5)
            boundary_dofs_x = fem.locate_dofs_topological(V_mag.sub(1), msh.topology.dim - 1, boundary_facets)
            bc5 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag.sub(1))
            
            boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side6)
            boundary_dofs_x = fem.locate_dofs_topological(V_mag.sub(1), msh.topology.dim - 1, boundary_facets)
            bc6 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag.sub(1))
        
            # bc = [bc1, bc3,bc4,bc6]
            bc_V = [bc1, bc2,bc3,bc4,bc5,bc6]            


        if geo == 'circle':

            def side1(x):
                return  np.logical_and(np.isclose(x[0]*x[0]+x[1]*x[1], 1) ,np.isclose(x[0], - x[1]))  
            
            
            boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side1)
            boundary_dofs_x = fem.locate_dofs_topological(V_mag.sub(0), msh.topology.dim - 1, boundary_facets)   
            bc1 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag.sub(0))
          
                    
          
            boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side1)
            boundary_dofs_x = fem.locate_dofs_topological(V_mag.sub(1), msh.topology.dim - 1, boundary_facets)
            bc4 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag.sub(1))
        
          
            bc_V = [bc1, bc4]
          

        
        if geo == 'annulus':
    
            def side1(x):
                return np.logical_and(np.isclose(x[0], 0), x[1] < 1)
            
            def side2(x):
                return np.logical_and(np.isclose(x[0], -1), x[0] < 1)
            
            def side3(x):
                return np.logical_and(np.isclose(x[0], 1), x[1] < 1)
            
            def side4(x):
                return np.logical_and(np.isclose(x[1], 0), x[0] < 1)
            
            
            def side5(x):
                return np.logical_and(np.isclose(x[1], -1), x[0] < 1)
            
            def side6(x):
                return np.logical_and(np.isclose(x[1], 1), x[0] < 1)
            
            
            
            boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side1)
            boundary_dofs_x = fem.locate_dofs_topological(V_mag.sub(0), msh.topology.dim - 1, boundary_facets)   
            bc1 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag.sub(0))
          
            boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side2)
            boundary_dofs_x = fem.locate_dofs_topological(V_mag.sub(0), msh.topology.dim - 1, boundary_facets)   
            bc2 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag.sub(0))
            
            boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side3)
            boundary_dofs_x = fem.locate_dofs_topological(V_mag.sub(0), msh.topology.dim - 1, boundary_facets)
            bc3 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag.sub(0))
          
            
          
            boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side4)
            boundary_dofs_x = fem.locate_dofs_topological(V_mag.sub(1), msh.topology.dim - 1, boundary_facets)
            bc4 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag.sub(1))
        
            boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side5)
            boundary_dofs_x = fem.locate_dofs_topological(V_mag.sub(1), msh.topology.dim - 1, boundary_facets)
            bc5 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag.sub(1))
            
            boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side6)
            boundary_dofs_x = fem.locate_dofs_topological(V_mag.sub(1), msh.topology.dim - 1, boundary_facets)
            bc6 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag.sub(1))
        
            bc_V = [bc1, bc2,bc3,bc4,bc5,bc6]





    
    V_ord_real_col, _ = V_ord_real.collapse() 
    V_ord_imag_col, _ = V_ord_imag.collapse() 
    
    V_mag_col, _ = V_mag.collapse() 
    
    
    bc_V_mag= None
    
    
    
    
    if use_bc and Fem_type_mag == 'Lagrange':
        if geo == 'unit_square':
          
            def side2(x):
                return np.logical_and(np.isclose(x[0], 0), x[1] < 1)
            
            def side3(x):
                return np.logical_and(np.isclose(x[0], 1), x[1] < 1)
                       
            def side5(x):
                return np.logical_and(np.isclose(x[1], 0), x[0] < 1)
            
            def side6(x):
                return np.logical_and(np.isclose(x[1], 1), x[0] < 1)
            

          
            boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side2)
            boundary_dofs_x = fem.locate_dofs_topological(V_mag_col.sub(0), msh.topology.dim - 1, boundary_facets)   
            bc2 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag_col.sub(0))
            
            boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side3)
            boundary_dofs_x = fem.locate_dofs_topological(V_mag_col.sub(0), msh.topology.dim - 1, boundary_facets)
            bc3 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag_col.sub(0))
          
            
          
        
            boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side5)
            boundary_dofs_x = fem.locate_dofs_topological(V_mag_col.sub(1), msh.topology.dim - 1, boundary_facets)
            bc5 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag_col.sub(1))
            
            boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side6)
            boundary_dofs_x = fem.locate_dofs_topological(V_mag_col.sub(1), msh.topology.dim - 1, boundary_facets)
            bc6 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag_col.sub(1))
        
            bc_V_mag = [bc2, bc3,bc5,bc6]




        if geo == 'square':
          
            def side2(x):
                return np.logical_and(np.isclose(x[0], -1), x[1] < 1)
            
            def side3(x):
                return np.logical_and(np.isclose(x[0], 1), x[1] < 1)
                       
            def side5(x):
                return np.logical_and(np.isclose(x[1], -1), x[0] < 1)
            
            def side6(x):
                return np.logical_and(np.isclose(x[1], 1), x[0] < 1)
            

          
            boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side2)
            boundary_dofs_x = fem.locate_dofs_topological(V_mag_col.sub(0), msh.topology.dim - 1, boundary_facets)   
            bc2 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag_col.sub(0))
            
            boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side3)
            boundary_dofs_x = fem.locate_dofs_topological(V_mag_col.sub(0), msh.topology.dim - 1, boundary_facets)
            bc3 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag_col.sub(0))
          
            
          
        
            boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side5)
            boundary_dofs_x = fem.locate_dofs_topological(V_mag_col.sub(1), msh.topology.dim - 1, boundary_facets)
            bc5 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag_col.sub(1))
            
            boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side6)
            boundary_dofs_x = fem.locate_dofs_topological(V_mag_col.sub(1), msh.topology.dim - 1, boundary_facets)
            bc6 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag_col.sub(1))
        
            bc_V_mag = [bc2, bc3,bc5,bc6]
   
        if geo == 'box_hole':

            def side1(x):
                return np.logical_and(np.isclose(x[0], -0.5), x[1] < 1)
            
            def side2(x):
                return np.logical_and(np.isclose(x[0], -1), x[1] < 1)
            
            def side3(x):
                return np.logical_and(np.isclose(x[0], 1), x[1] < 1)
            
            def side4(x):
                return np.logical_and(np.isclose(x[0], 0.5), x[1] < 1)
            
            def side5(x):
                return np.logical_and(np.isclose(x[1], 0.5), x[0] < 1)
            
            def side6(x):
                return np.logical_and(np.isclose(x[1], -0.5), x[0] < 1)
            
            
            def side7(x):
                return np.logical_and(np.isclose(x[1], -1), x[0] < 1)
            
            def side8(x):
                return np.logical_and(np.isclose(x[1], 1), x[0] < 1)
            
            
            
            boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side1)
            boundary_dofs_x = fem.locate_dofs_topological(V_mag_col.sub(0), msh.topology.dim - 1, boundary_facets)   
            bc1 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag_col.sub(0))
          
            boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side2)
            boundary_dofs_x = fem.locate_dofs_topological(V_mag_col.sub(0), msh.topology.dim - 1, boundary_facets)   
            bc2 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag_col.sub(0))
            
            boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side3)
            boundary_dofs_x = fem.locate_dofs_topological(V_mag_col.sub(0), msh.topology.dim - 1, boundary_facets)
            bc3 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag_col.sub(0))
            
            boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side4)
            boundary_dofs_x = fem.locate_dofs_topological(V_mag_col.sub(0), msh.topology.dim - 1, boundary_facets)
            bc4 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag_col.sub(0))
          
            
       
            boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side5)
            boundary_dofs_x = fem.locate_dofs_topological(V_mag_col.sub(1), msh.topology.dim - 1, boundary_facets)
            bc5 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag_col.sub(1))
            
            boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side6)
            boundary_dofs_x = fem.locate_dofs_topological(V_mag_col.sub(1), msh.topology.dim - 1, boundary_facets)
            bc6 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag_col.sub(1))
            
                      
            boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side7)
            boundary_dofs_x = fem.locate_dofs_topological(V_mag_col.sub(1), msh.topology.dim - 1, boundary_facets)
            bc7 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag_col.sub(1))
            
                      
            boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side8)
            boundary_dofs_x = fem.locate_dofs_topological(V_mag_col.sub(1), msh.topology.dim - 1, boundary_facets)
            bc8 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag_col.sub(1))
        
            bc_V_mag = [bc1, bc2,bc3,bc4,bc5,bc6,bc7,bc8]            

        if geo == 'L':

            def side1(x):
                return np.logical_and(np.isclose(x[0], 0), x[1] < 1)
            
            def side2(x):
                return np.logical_and(np.isclose(x[0], -1), x[1] < 1)
            
            def side3(x):
                return np.logical_and(np.isclose(x[0], 1), x[1] < 1)
            
            def side4(x):
                return np.logical_and(np.isclose(x[1], 0), x[0] < 1)
            
            
            def side5(x):
                return np.logical_and(np.isclose(x[1], -1), x[0] < 1)
            
            def side6(x):
                return np.logical_and(np.isclose(x[1], 1), x[0] < 1)
            
            
            
            boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side1)
            boundary_dofs_x = fem.locate_dofs_topological(V_mag_col.sub(0), msh.topology.dim - 1, boundary_facets)   
            bc1 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag_col.sub(0))
          
            boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side2)
            boundary_dofs_x = fem.locate_dofs_topological(V_mag_col.sub(0), msh.topology.dim - 1, boundary_facets)   
            bc2 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag_col.sub(0))
            
            boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side3)
            boundary_dofs_x = fem.locate_dofs_topological(V_mag_col.sub(0), msh.topology.dim - 1, boundary_facets)
            bc3 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag_col.sub(0))
          
            
          
            boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side4)
            boundary_dofs_x = fem.locate_dofs_topological(V_mag_col.sub(1), msh.topology.dim - 1, boundary_facets)
            bc4 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag_col.sub(1))
        
            boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side5)
            boundary_dofs_x = fem.locate_dofs_topological(V_mag_col.sub(1), msh.topology.dim - 1, boundary_facets)
            bc5 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag_col.sub(1))
            
            boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side6)
            boundary_dofs_x = fem.locate_dofs_topological(V_mag_col.sub(1), msh.topology.dim - 1, boundary_facets)
            bc6 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag_col.sub(1))
        
            # bc = [bc1, bc3,bc4,bc6]
            bc_V_mag = [bc1, bc2,bc3,bc4,bc5,bc6]            


        if geo == 'circle':

            def side1(x):
                return  np.logical_and(np.isclose(x[0]*x[0]+x[1]*x[1], 1) ,np.isclose(x[0], - x[1]))  
            
            
            
            boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side1)
            boundary_dofs_x = fem.locate_dofs_topological(V_mag_col.sub(0), msh.topology.dim - 1, boundary_facets)   
            bc1 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag_col.sub(0))
          
          
            
          
            boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side1)
            boundary_dofs_x = fem.locate_dofs_topological(V_mag_col.sub(1), msh.topology.dim - 1, boundary_facets)
            bc4 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag_col.sub(1))
        
         
            bc_V_mag = [bc1, bc4]
         

        
        if geo == 'annulus':
    
            def side1(x):
                return np.logical_and(np.isclose(x[0], 0), x[1] < 1)
            
            def side2(x):
                return np.logical_and(np.isclose(x[0], -1), x[0] < 1)
            
            def side3(x):
                return np.logical_and(np.isclose(x[0], 1), x[1] < 1)
            
            def side4(x):
                return np.logical_and(np.isclose(x[1], 0), x[0] < 1)
            
            
            def side5(x):
                return np.logical_and(np.isclose(x[1], -1), x[0] < 1)
            
            def side6(x):
                return np.logical_and(np.isclose(x[1], 1), x[0] < 1)
            
            
            
            boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side1)
            boundary_dofs_x = fem.locate_dofs_topological(V_mag_col.sub(0), msh.topology.dim - 1, boundary_facets)   
            bc1 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag_col.sub(0))
          
            boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side2)
            boundary_dofs_x = fem.locate_dofs_topological(V_mag_col.sub(0), msh.topology.dim - 1, boundary_facets)   
            bc2 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag_col.sub(0))
            
            boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side3)
            boundary_dofs_x = fem.locate_dofs_topological(V_mag_col.sub(0), msh.topology.dim - 1, boundary_facets)
            bc3 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag_col.sub(0))
          
            
          
            boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side4)
            boundary_dofs_x = fem.locate_dofs_topological(V_mag_col.sub(1), msh.topology.dim - 1, boundary_facets)
            bc4 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag_col.sub(1))
        
            boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side5)
            boundary_dofs_x = fem.locate_dofs_topological(V_mag_col.sub(1), msh.topology.dim - 1, boundary_facets)
            bc5 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag_col.sub(1))
            
            boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side6)
            boundary_dofs_x = fem.locate_dofs_topological(V_mag_col.sub(1), msh.topology.dim - 1, boundary_facets)
            bc6 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag_col.sub(1))
        
            bc_V_mag = [bc1, bc2,bc3,bc4,bc5,bc6]


    FEM_spaces_dict =  {
            'V_ext': V_ext,
            'V':  V,
            'V_ord': V_ord,
            'V_ord_real': V_ord_real,
            'V_ord_imag': V_ord_imag,
            'V_mag':V_mag,
            'V_ord_real_col':V_ord_real_col,
            'V_ord_imag_col':V_ord_imag_col,
            'V_mag_col':V_mag_col,
            'x':x,
            'bc_V':bc_V,
            'bc_V_mag':bc_V_mag,
            'V_dG' : V_dG,
            'V_div': V_div
       }


    return FEM_spaces_dict





##########################################################################
# space for u
##########################################################################


def get_V_ord(msh,spaces_config_dict):
    degree_FEM_ord = spaces_config_dict['degree_FEM_ord']

    element_H1 = ufl.FiniteElement("Lagrange", msh.ufl_cell(), degree = degree_FEM_ord)

     
    V_ord  = fem.functionspace(msh, element_H1 * element_H1 ) 
    x= ufl.SpatialCoordinate(V_ord.mesh)

    return V_ord , element_H1, x

 



##########################################################################
# space for A
##########################################################################

   
def get_V_mag(msh,problem_dict,spaces_config_dict):

    use_bc = True
    geo = problem_dict['geo']
    Fem_type_mag = spaces_config_dict['Fem_type_mag']
    Nedelec_kind = spaces_config_dict['Nedelec_kind']
    degree_FEM_mag = spaces_config_dict['degree_FEM_mag']
    # inc_div = spaces_config_dict['inc_div']


    bc = None

    if Fem_type_mag == 'Nedelec':
        # inc_div = False
        if Nedelec_kind == 1:
            # "Nedelec 1st kind H(curl)", msh.ufl_cell(), 1)
            # https://defelement.com/elements/nedelec1.html
            element_curl = basix.ufl.element(basix.ElementFamily.N1E, basix.CellType.triangle, degree =  degree_FEM_mag)
          
        if Nedelec_kind == 2:
            #https://defelement.com/elements/nedelec2.html
            element_curl = basix.ufl.element(basix.ElementFamily.N2E, basix.CellType.triangle, degree =  degree_FEM_mag)
            # element_curl = basix.ufl.element(basix.ElementFamily.N2E, basix.CellType.triangle, degree =  degree_FEM_mag, lagrange_variant=basix.LagrangeVariant.equispaced)

        V_mag = fem.functionspace(msh, element_curl)
        x= ufl.SpatialCoordinate(V_mag.mesh)
    
    
    if Fem_type_mag == 'Lagrange':
        # inc_div = True



        element_curl = basix.ufl.element(basix.ElementFamily.P, basix.CellType.triangle, degree_FEM_mag, lagrange_variant=basix.LagrangeVariant.equispaced, shape=(2,))
        
        V_mag = fem.functionspace(msh, element_curl) 

        x= ufl.SpatialCoordinate(V_mag.mesh)

        if use_bc:

            if geo == 'unit_square':
              
                def side2(x):
                    return np.logical_and(np.isclose(x[0], 0), x[1] < 1)
                
                def side3(x):
                    return np.logical_and(np.isclose(x[0], 1), x[1] < 1)
                           
                def side5(x):
                    return np.logical_and(np.isclose(x[1], 0), x[0] < 1)
                
                def side6(x):
                    return np.logical_and(np.isclose(x[1], 1), x[0] < 1)
                
    
              
                boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side2)
                boundary_dofs_x = fem.locate_dofs_topological(V_mag.sub(0), msh.topology.dim - 1, boundary_facets)   
                bc2 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag.sub(0))
                
                boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side3)
                boundary_dofs_x = fem.locate_dofs_topological(V_mag.sub(0), msh.topology.dim - 1, boundary_facets)
                bc3 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag.sub(0))
              
                
              
            
                boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side5)
                boundary_dofs_x = fem.locate_dofs_topological(V_mag.sub(1), msh.topology.dim - 1, boundary_facets)
                bc5 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag.sub(1))
                
                boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side6)
                boundary_dofs_x = fem.locate_dofs_topological(V_mag.sub(1), msh.topology.dim - 1, boundary_facets)
                bc6 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag.sub(1))
            
                bc = [bc2, bc3,bc5,bc6]
    
            if geo == 'square':
              
                def side2(x):
                    return np.logical_and(np.isclose(x[0], -1), x[1] < 1)
                
                def side3(x):
                    return np.logical_and(np.isclose(x[0], 1), x[1] < 1)
                           
                def side5(x):
                    return np.logical_and(np.isclose(x[1], -1), x[0] < 1)
                
                def side6(x):
                    return np.logical_and(np.isclose(x[1], 1), x[0] < 1)
                
    
              
                boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side2)
                boundary_dofs_x = fem.locate_dofs_topological(V_mag.sub(0), msh.topology.dim - 1, boundary_facets)   
                bc2 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag.sub(0))
                
                boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side3)
                boundary_dofs_x = fem.locate_dofs_topological(V_mag.sub(0), msh.topology.dim - 1, boundary_facets)
                bc3 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag.sub(0))
              
                
              
            
                boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side5)
                boundary_dofs_x = fem.locate_dofs_topological(V_mag.sub(1), msh.topology.dim - 1, boundary_facets)
                bc5 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag.sub(1))
                
                boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side6)
                boundary_dofs_x = fem.locate_dofs_topological(V_mag.sub(1), msh.topology.dim - 1, boundary_facets)
                bc6 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag.sub(1))
            
                bc = [bc2, bc3,bc5,bc6]
       
            if geo == 'box_hole':
    
                def side1(x):
                    return np.logical_and(np.isclose(x[0], -0.5), x[1] < 1)
                
                def side2(x):
                    return np.logical_and(np.isclose(x[0], -1), x[1] < 1)
                
                def side3(x):
                    return np.logical_and(np.isclose(x[0], 1), x[1] < 1)
                
                def side4(x):
                    return np.logical_and(np.isclose(x[0], 0.5), x[1] < 1)
                
                def side5(x):
                    return np.logical_and(np.isclose(x[1], 0.5), x[0] < 1)
                
                def side6(x):
                    return np.logical_and(np.isclose(x[1], -0.5), x[0] < 1)
                
                
                def side7(x):
                    return np.logical_and(np.isclose(x[1], -1), x[0] < 1)
                
                def side8(x):
                    return np.logical_and(np.isclose(x[1], 1), x[0] < 1)
                
                
                
                boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side1)
                boundary_dofs_x = fem.locate_dofs_topological(V_mag.sub(0), msh.topology.dim - 1, boundary_facets)   
                bc1 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag.sub(0))
              
                boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side2)
                boundary_dofs_x = fem.locate_dofs_topological(V_mag.sub(0), msh.topology.dim - 1, boundary_facets)   
                bc2 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag.sub(0))
                
                boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side3)
                boundary_dofs_x = fem.locate_dofs_topological(V_mag.sub(0), msh.topology.dim - 1, boundary_facets)
                bc3 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag.sub(0))
                
                boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side4)
                boundary_dofs_x = fem.locate_dofs_topological(V_mag.sub(0), msh.topology.dim - 1, boundary_facets)
                bc4 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag.sub(0))
              
                
           
                boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side5)
                boundary_dofs_x = fem.locate_dofs_topological(V_mag.sub(1), msh.topology.dim - 1, boundary_facets)
                bc5 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag.sub(1))
                
                boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side6)
                boundary_dofs_x = fem.locate_dofs_topological(V_mag.sub(1), msh.topology.dim - 1, boundary_facets)
                bc6 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag.sub(1))
                
                          
                boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side7)
                boundary_dofs_x = fem.locate_dofs_topological(V_mag.sub(1), msh.topology.dim - 1, boundary_facets)
                bc7 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag.sub(1))
                
                          
                boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side8)
                boundary_dofs_x = fem.locate_dofs_topological(V_mag.sub(1), msh.topology.dim - 1, boundary_facets)
                bc8 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag.sub(1))
            
                # bc = [bc2, bc3,bc5,bc6]
                bc = [bc1, bc2,bc3,bc4,bc5,bc6,bc7,bc8]            
    
            if geo == 'L':
    
                def side1(x):
                    return np.logical_and(np.isclose(x[0], 0), x[1] < 1)
                
                def side2(x):
                    return np.logical_and(np.isclose(x[0], -1), x[1] < 1)
                
                def side3(x):
                    return np.logical_and(np.isclose(x[0], 1), x[1] < 1)
                
                def side4(x):
                    return np.logical_and(np.isclose(x[1], 0), x[0] < 1)
                
                
                def side5(x):
                    return np.logical_and(np.isclose(x[1], -1), x[0] < 1)
                
                def side6(x):
                    return np.logical_and(np.isclose(x[1], 1), x[0] < 1)
                
                
                
                boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side1)
                boundary_dofs_x = fem.locate_dofs_topological(V_mag.sub(0), msh.topology.dim - 1, boundary_facets)   
                bc1 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag.sub(0))
              
                boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side2)
                boundary_dofs_x = fem.locate_dofs_topological(V_mag.sub(0), msh.topology.dim - 1, boundary_facets)   
                bc2 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag.sub(0))
                
                boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side3)
                boundary_dofs_x = fem.locate_dofs_topological(V_mag.sub(0), msh.topology.dim - 1, boundary_facets)
                bc3 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag.sub(0))
              
                
              
                boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side4)
                boundary_dofs_x = fem.locate_dofs_topological(V_mag.sub(1), msh.topology.dim - 1, boundary_facets)
                bc4 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag.sub(1))
            
                boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side5)
                boundary_dofs_x = fem.locate_dofs_topological(V_mag.sub(1), msh.topology.dim - 1, boundary_facets)
                bc5 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag.sub(1))
                
                boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side6)
                boundary_dofs_x = fem.locate_dofs_topological(V_mag.sub(1), msh.topology.dim - 1, boundary_facets)
                bc6 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag.sub(1))
            
                # bc = [bc1, bc3,bc4,bc6]
                bc = [bc1, bc2,bc3,bc4,bc5,bc6]            
    
    
            if geo == 'circle':
    
                def side1(x):
                    return  np.logical_and(np.isclose(x[0]*x[0]+x[1]*x[1], 1) ,np.isclose(x[0], - x[1]))  
             
                
                
                boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side1)
                boundary_dofs_x = fem.locate_dofs_topological(V_mag.sub(0), msh.topology.dim - 1, boundary_facets)   
                bc1 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag.sub(0))
              
               
              
                boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side1)
                boundary_dofs_x = fem.locate_dofs_topological(V_mag.sub(1), msh.topology.dim - 1, boundary_facets)
                bc4 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag.sub(1))
            
              
                bc = [bc1, bc4]
          
    
            if geo == 'annulus':

        
                def side1(x):
                    return np.logical_and(np.isclose(x[0], 0), x[1] < 1)
                
                def side2(x):
                    return np.logical_and(np.isclose(x[0], -1), x[0] < 1)
                
                def side3(x):
                    return np.logical_and(np.isclose(x[0], 1), x[1] < 1)
                
                def side4(x):
                    return np.logical_and(np.isclose(x[1], 0), x[0] < 1)
                
                
                def side5(x):
                    return np.logical_and(np.isclose(x[1], -1), x[0] < 1)
                
                def side6(x):
                    return np.logical_and(np.isclose(x[1], 1), x[0] < 1)
                
                
                
                boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side1)
                boundary_dofs_x = fem.locate_dofs_topological(V_mag.sub(0), msh.topology.dim - 1, boundary_facets)   
                bc1 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag.sub(0))
              
                boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side2)
                boundary_dofs_x = fem.locate_dofs_topological(V_mag.sub(0), msh.topology.dim - 1, boundary_facets)   
                bc2 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag.sub(0))
                
                boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side3)
                boundary_dofs_x = fem.locate_dofs_topological(V_mag.sub(0), msh.topology.dim - 1, boundary_facets)
                bc3 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag.sub(0))
              
                
              
                boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side4)
                boundary_dofs_x = fem.locate_dofs_topological(V_mag.sub(1), msh.topology.dim - 1, boundary_facets)
                bc4 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag.sub(1))
            
                boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side5)
                boundary_dofs_x = fem.locate_dofs_topological(V_mag.sub(1), msh.topology.dim - 1, boundary_facets)
                bc5 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag.sub(1))
                
                boundary_facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, side6)
                boundary_dofs_x = fem.locate_dofs_topological(V_mag.sub(1), msh.topology.dim - 1, boundary_facets)
                bc6 = fem.dirichletbc(default_scalar_type(0), boundary_dofs_x, V_mag.sub(1))
            
                bc = [bc1, bc2,bc3,bc4,bc5,bc6]
  

    return V_mag, element_curl, bc, x








