import numpy as np
from mpi4py import MPI
import dolfinx as dfx
from dolfinx import mesh, fem, plot, io, default_scalar_type

import dolfinx.fem.petsc as dfpet
# from dolfinx.io import gmshio
import basix

import ufl

import scipy.sparse as scp
import numpy as np
# import scipy.sparse.linalg
import scipy.sparse.linalg as ssla


import GL_FEM_energies as GL_energy
import spaces_def as s_def
import discrete_divergence as discrete_divergence

atol = 10e-8


def get_H(H_type,mag_scale,x,omega_by_pi_times_ten=None):

    def H():
        if H_type == 1:
            return  np.sqrt(2) * 2 * ufl.pi* ufl.sin(ufl.pi*x[0])*ufl.sin(ufl.pi*x[1]) 

        if H_type == 2:
            return  np.sqrt(2) *  ufl.pi* ufl.cos(0.5*ufl.pi*x[0])*ufl.cos(0.5*ufl.pi*x[1]) 
        
        if H_type == 3:
            return  mag_scale  + x[0] - x[0]  
        
        if H_type == 4:
            return  10 * ufl.sin(ufl.pi*x[0])*ufl.sin(ufl.pi*x[1]) 
 
        
        if H_type == 5:
            omega_inv = 10/omega_by_pi_times_ten    

            r = ufl.sqrt(x[0] * x[0] + x[1] * x[1])
            theta = ufl.conditional(x[1]>0, ufl.atan2(x[1], x[0]) , ufl.atan2(x[1], x[0]) + 2*ufl.pi ) 
   
            H = (2 *  omega_inv +1) *  r**(omega_inv-1) *ufl.sin(omega_inv * theta) 
        
            return ufl.conditional(r>atol, H, 0.0 + x[0]-x[0])
                
        if H_type == 9:
            return  10 * ufl.sin( ufl.pi* x[0] ) * ufl.sin( ufl.pi* x[1] )  
        
        if H_type == 11:
            return  np.sqrt(2) * 2 * ufl.pi  + x[0] - x[0]  
    
    # return ufl.as_vector( (a1 + a2 * x[0] + a3 * x[1] , b1 + b2* x[0] + b3*x[1]) )
        # return  -1.0  + x[0] - x[0]  
        # return ufl.as_vector(( -1.0* x[1] + x[1] ,  1.0 ))
        # return  np.sqrt(2) *  ufl.pi* ufl.cos(0.5*ufl.pi*x[0])*ufl.cos(0.5*ufl.pi*x[1]) 
    
        # 

    return H()    


def get_A1(A1_type,mag_scale,x,omega_by_pi_times_ten=None):

    def A1():
        if A1_type == 1:
            return ufl.as_vector((  np.sqrt(2)* ufl.sin(ufl.pi*x[0])*ufl.cos(ufl.pi*x[1]) , -  np.sqrt(2) * ufl.cos(ufl.pi*x[0])*ufl.sin(ufl.pi*x[1]) ))


        if A1_type == 2:
            return  np.sqrt(2) *  ufl.pi* ufl.cos(0.5*ufl.pi*x[0])*ufl.cos(0.5*ufl.pi*x[1])

        
        if A1_type == 3:
            return ufl.as_vector(( - 1.0 * mag_scale*x[1] , mag_scale*x[0] ))
        
        if A1_type == 4:
            # return ufl.as_vector(( 0.0+ x[0]-x[0] ,  0.0 + x[0] - x[0] ))
            return ufl.as_vector(( 0.01+ x[0]-x[0] ,  0.0 + x[0] - x[0] ))

    
    
    
        if A1_type == 5:
            omega_inv = 10/omega_by_pi_times_ten    

            r = ufl.sqrt(x[0] * x[0] + x[1] * x[1])
            theta = ufl.conditional(x[1]>0, ufl.atan2(x[1], x[0]) , ufl.atan2(x[1], x[0]) + 2*ufl.pi ) 
   
            A_x =   omega_inv * (1-r) * r**(omega_inv-1) * ufl.cos(omega_inv * theta) * ufl.cos(theta) \
                + ( - r**(omega_inv) +  (1-r) * omega_inv * r**(omega_inv-1)  ) \
                            *  ufl.sin(omega_inv * theta) * ufl.sin(theta) 
                            
            A_y = - omega_inv * (1-r) * r**(omega_inv-1) * ufl.cos(omega_inv * theta) * ufl.sin(theta) \
                + ( r**(omega_inv) - (1-r) * omega_inv * r**(omega_inv-1)  ) \
                            *  ufl.sin(omega_inv * theta) * ufl.cos(theta)                    
    
            A_x = ufl.conditional(r>atol, A_x, 0.0 + x[0]-x[0])
            A_y = ufl.conditional(r>atol, A_y, 0.0 + x[0]-x[0])
         
            return ufl.as_vector(( A_x ,A_y))

    return A1()  



##################################################################################


def vec_Pot_x(x):
    #omega = omega_by_pi_times_ten/10 * np.pi
    omega_inv = 10/omega_by_pi_times_ten 
    r, theta = get_polar(x)

    values = omega_inv * (1-r) * r**(omega_inv-1) * np.cos(omega_inv * theta) * np.cos(theta) \
                + ( - r**(omega_inv) +  (1-r) * omega_inv * r**(omega_inv-1)  ) \
                            *  np.sin(omega_inv * theta) * np.sin(theta) 
    values[np.where(np.isclose(r, 0., atol=1e-10))] = 0.

    return values


def vec_Pot_y(x):
    #omega = omega_by_pi_times_ten/10 * np.pi
    omega_inv = 10/omega_by_pi_times_ten 
    r, theta = get_polar(x)
   
    values = - omega_inv * (1-r) * r**(omega_inv-1) * np.cos(omega_inv * theta) * np.sin(theta) \
                + ( r**(omega_inv) - (1-r) * omega_inv * r**(omega_inv-1)  ) \
                            *  np.sin(omega_inv * theta) * np.cos(theta) 
    values[np.where(np.isclose(r, 0., atol=1e-10))] = 0.

    return values





def get_u(u_type, phase_scale , freq, num_holes, x, msh,V_ord):
    # phase_scale = 2.0
    # phase_scale = 0.5

       
    x_vec = np.zeros(num_holes)
    y_vec = np.zeros(num_holes)

    for i in range(num_holes):
        x_vec[i] = 0.5* np.cos(2*i*np.pi / num_holes)
        y_vec[i] = 0.5* np.sin(2*i*np.pi / num_holes)
        

    def pulse(freq,x1,y1):
        return  (1- ufl.exp( -freq* ((x[0]-x1)**2 + (x[1]-y1)**2 )  ))

    def u_real():
        if u_type == 1:
            return  0.8 + x[0] - x[0]  # + 0.0*  x[1] - 0.0*  x[0] 
            # return  0.8 + 0.0*x[0]
            # return  fem.Constant(msh, default_scalar_type(0.8)) #+ x[0]
            # return  fem.Expression(dfx.fem.Constant(V_ord.sub(0).mesh, 0.8), V_ord.sub(0).element.interpolation_points)

        # return ufl.constant.Constant(msh,0.8)
            # return fem.Expression(0.0, V_ord.element.interpolation_points)
    
    

    
    
        if u_type == 2:
            return 1/np.sqrt(2) * x[0]
        
        if u_type == 3:
            return  1.0 + x[0] - x[0] 

        if u_type == 4:
            return  ufl.cos(freq*ufl.pi*x[0]) * ufl.cos(freq*ufl.pi*x[1]) 
        
        if u_type == 5:
             u = pulse(freq,x_vec[0],y_vec[0])
             for m in range(1,num_holes):
                u = u * pulse(freq,x_vec[m],y_vec[m])
             return ufl.cos( phase_scale*ufl.atan2(x[0],x[1]) ) * u
         
        if u_type == 11:
            x_tmp = 2*x[0] - 1
            y_tmp = 2*x[1] - 1
            return 1/np.sqrt(2) * ufl.exp( - x_tmp**2 - y_tmp**2)
        
        if u_type == 12:
            x_tmp = 2*x[0] - 1
            y_tmp = 2*x[1] - 1
            return 1/np.sqrt(ufl.pi) * ((3/2)*x_tmp + 1/2) * ufl.exp( - x_tmp**2 - y_tmp**2)
        
        if u_type == 13:
            x_tmp = 2*x[0] - 1
            y_tmp = 2*x[1] - 1
            return 1/np.sqrt(2) * ufl.cos( 10*(x_tmp**2 + y_tmp**2)) - 1/np.sqrt(2) * ufl.sin( 10*(x_tmp**2 + y_tmp**2))
        
        if u_type == 14:
            x_tmp = 2*x[0] - 1
            y_tmp = 2*x[1] - 1
            return 1/np.sqrt(2) * ( (1 - x_tmp) * ufl.cos( 10*(x_tmp**2 + y_tmp**2)) - (x_tmp + y_tmp) * ufl.sin(10*(x_tmp**2 + y_tmp**2)) )
        
        if u_type == 15:
            return 1/np.sqrt(2) * x[0]
        
    
       
    def u_imag():
        if u_type == 1:
            return   0.6 + x[0] - x[0]  # + 0.0*  x[1] - 0.0*  x[0] 

        # return ufl.as_vector( (a1 + a2 * x[0] + a3 * x[1] , b1 + b2* x[0] + b3*x[1]) )
        if u_type == 2:
            return 1/np.sqrt(2) * x[1]
        
        if u_type == 3:
            return  0.0 + x[0] - x[0] 
        
        if u_type == 4:
            return   ufl.sin(freq*ufl.pi*x[0]) * ufl.sin(freq*ufl.pi*x[1]) 
        
                
        if u_type == 5:
            u = pulse(freq,x_vec[0],y_vec[0])
            for m in range(1,num_holes):
                u = u *  pulse(freq,x_vec[m],y_vec[m])
            return ufl.sin( phase_scale*ufl.atan2(x[0],x[1]) ) * u
        
        if u_type == 11:
            x_tmp = 2*x[0] - 1
            y_tmp = 2*x[1] - 1
            return 1/np.sqrt(2) * ufl.exp( - x_tmp**2 - y_tmp**2)
        
        if u_type == 12:
            x_tmp = 2*x[0] - 1
            y_tmp = 2*x[1] - 1
            return 1/np.sqrt(ufl.pi) * ((3/2)*y_tmp)  * ufl.exp( - x_tmp**2 - y_tmp**2)
        
        if u_type == 13:
            x_tmp = 2*x[0] - 1
            y_tmp = 2*x[1] - 1
            return 1/np.sqrt(2) * ufl.cos( 10*(x_tmp**2 + y_tmp**2)) + 1/np.sqrt(2) * ufl.sin( 10*(x_tmp**2 + y_tmp**2))
        
        if u_type == 14:
            x_tmp = 2*x[0] - 1
            y_tmp = 2*x[1] - 1
            return 1/np.sqrt(2) * ( (1 - x_tmp) * ufl.sin(10*(x_tmp**2 + y_tmp**2)) + (x_tmp + y_tmp) * ufl.cos(10*(x_tmp**2 + y_tmp**2)) )
        
        if u_type == 15:
            return 1/np.sqrt(2) * x[1]

    return u_real(), u_imag()

##################################################################################
##################################################################################
##################################################################################

def get_polar(x):
    r = np.sqrt(x[0] * x[0] + x[1] * x[1])
    theta = np.arctan2(x[1], x[0]) #+ np.pi  / 2. # hier passt etwas nicht?
    theta[theta < 0] = 2* np.pi + theta[theta < 0]   
    return r, theta

##################################################################################


def vec_Pot_x(x):
    #omega = omega_by_pi_times_ten/10 * np.pi
    omega_inv = 10/cf.omega_by_pi_times_ten 
    r, theta = get_polar(x)

    values = omega_inv * (1-r) * r**(omega_inv-1) * np.cos(omega_inv * theta) * np.cos(theta) \
                + ( - r**(omega_inv) +  (1-r) * omega_inv * r**(omega_inv-1)  ) \
                            *  np.sin(omega_inv * theta) * np.sin(theta) 
    values[np.where(np.isclose(r, 0., atol=1e-10))] = 0.

    return values


def vec_Pot_y(x):
    #omega = omega_by_pi_times_ten/10 * np.pi
    omega_inv = 10/cf.omega_by_pi_times_ten 
    r, theta = get_polar(x)
   
    values = - omega_inv * (1-r) * r**(omega_inv-1) * np.cos(omega_inv * theta) * np.sin(theta) \
                + ( r**(omega_inv) - (1-r) * omega_inv * r**(omega_inv-1)  ) \
                            *  np.sin(omega_inv * theta) * np.cos(theta) 
    values[np.where(np.isclose(r, 0., atol=1e-10))] = 0.

    return values