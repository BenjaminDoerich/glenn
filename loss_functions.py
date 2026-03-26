import torch
import lightning as pl
import torch
import math

def vec_pot(x, y):
    A1 =    torch.sqrt(torch.tensor(2.0)) * torch.sin(torch.pi * x) * torch.cos(torch.pi * y) 
    A2 =  - torch.sqrt(torch.tensor(2.0)) * torch.cos(torch.pi * x) * torch.sin(torch.pi * y) 
    return A1, A2


def A_pot(x):
    return torch.stack([torch.sqrt(torch.tensor(2.0)) * torch.sin(torch.pi * x[:,0]) * torch.cos(torch.pi * x[:,1]) , - torch.sqrt(torch.tensor(2.0)) * torch.cos(torch.pi * x[:,0]) * torch.sin(torch.pi * x[:,1]) ]).transpose(1,0)


def H_mag_field_1(x):
    return 2*torch.pi*torch.sqrt(torch.tensor(2.0)) *torch.sin(torch.pi*x[:,0])*torch.sin(torch.pi*x[:,1])
    

def H_mag_field_2(x):
    return torch.pi*torch.sqrt(torch.tensor(2.0)) *torch.cos(0.5*torch.pi*x[:,0])*torch.cos(0.5*torch.pi*x[:,1])


def get_H_mag_field_3(mag_scale):
    def H_mag_field_3(x):
        return mag_scale*torch.ones_like(x[:,0])
    return H_mag_field_3


def H_mag_field_4(x):
    return 10*torch.sin(torch.pi*x[:,0])*torch.sin(torch.pi*x[:,1])

def H_mag_field_5(x):
    return 2.0*torch.sqrt(torch.tensor(2.0))*torch.pi*torch.ones_like(x[:,0])



class MagField:
    def __init__(self,H_type,mag_scale):
        match H_type:
            case 1:
                self.H_mag_field = H_mag_field_1
            case 2:
                self.H_mag_field = H_mag_field_2
            case 3:
                self.H_mag_field = get_H_mag_field_3(mag_scale=mag_scale)
            case 4:
                self.H_mag_field = H_mag_field_4
            case 5:
                self.H_mag_field = H_mag_field_5

    def eval_magfield(self,x):
        return self.H_mag_field(x)
    
    def get_eval_magfield(self):
        return self.H_mag_field

    


def normalized_scaled_train_loss_only_ord(model, source, max_kappa):
    kappa_tens=source[:,[2]]
    tmp = source[:,0:2]
    tmp.requires_grad_(True)
    u_pred = model(torch.cat([tmp,kappa_tens], dim=1))

    u1 = u_pred[:,0]
    u2 = u_pred[:,1]

    A1 , A2 = vec_pot(tmp[:,0], tmp[:,1])
    
    du1 = 1/(max_kappa*kappa_tens) * torch.autograd.grad(u1, tmp, grad_outputs=torch.ones_like(u1), create_graph=True)[0]
    du2 = 1/(max_kappa*kappa_tens) * torch.autograd.grad(u2, tmp, grad_outputs=torch.ones_like(u1), create_graph=True)[0]

    u1_x = du1[:,0]
    u1_y = du1[:,1]
    u2_x = du2[:,0]
    u2_y = du2[:,1]

    res_1 = u1_x**2  + u1_y**2 + u2_x**2 + u2_y**2
    res_2 = (A1**2 + A2**2) * (u1**2 + u2**2)
    res_3 = 2 * u2 *   (A1* u1_x + A2 * u1_y)  - 2*  u1*  (A1* u2_x + A2 * u2_y)
    res_4 = 0.5 * (1.0 - u1**2 - u2**2)**2 
    loss_pde = 0.5*torch.mean((max_kappa*kappa_tens)*(res_1 + res_2 + res_3 +  res_4 ))

    return loss_pde


def normalized_test_loss_only_ord(model, source, max_kappa): # attention: here the kappa is not in [0,1] but the real kappa, so it is scaled.
    kappa_tens=(max_kappa**-1) * source[:,[2]]
    tmp = source[:,0:2].detach().requires_grad_(True)
    u_pred = model(torch.cat([tmp,kappa_tens], dim=1))

    u1 = u_pred[:,0]
    u2 = u_pred[:,1]

    A1 , A2 = vec_pot(tmp[:,0], tmp[:,1])
    
    du1 = 1/source[:,[2]] * torch.autograd.grad(u1, tmp, grad_outputs=torch.ones_like(u1), create_graph=True)[0]
    du2 = 1/source[:,[2]] * torch.autograd.grad(u2, tmp, grad_outputs=torch.ones_like(u1), create_graph=True)[0]

    u1_x = du1[:,0]
    u1_y = du1[:,1]
    u2_x = du2[:,0]
    u2_y = du2[:,1]

    res_1 = u1_x**2  + u1_y**2 + u2_x**2 + u2_y**2
    res_2 = (A1**2 + A2**2) * (u1**2 + u2**2)
    res_3 = 2 * u2 *   (A1* u1_x + A2 * u1_y)  - 2*  u1*  (A1* u2_x + A2 * u2_y)
    res_4 = 0.5 * (1.0 - u1**2 - u2**2)**2 
    loss_pde = 0.5*torch.mean((res_1 + res_2 + res_3 +  res_4 ))
    return loss_pde


def normalized_test_loss_full(model, source, max_kappa, eval_magfield: callable): # attention: here the kappa is not in [0,1] but the real kappa, so it is scaled.
    kappa_tens=(max_kappa**-1) * source[:,[2]]
    tmp = source[:,0:2].detach().requires_grad_(True)

    H_eval_train = eval_magfield(source)
    u_pred = model(torch.cat([tmp,kappa_tens], dim=1))

    u1 = u_pred[:,0]
    u2 = u_pred[:,1]
    A1 =  u_pred[:,2]
    A2 =  u_pred[:,3]
    
    du1 = 1/source[:,[2]] * torch.autograd.grad(u1, tmp, grad_outputs=torch.ones_like(u1), create_graph=True)[0]
    du2 = 1/source[:,[2]] * torch.autograd.grad(u2, tmp, grad_outputs=torch.ones_like(u1), create_graph=True)[0]

    dA_1 = torch.autograd.grad(A1, tmp, grad_outputs=torch.ones_like(A1), create_graph=True)[0]
    dA_2 = torch.autograd.grad(A2, tmp, grad_outputs=torch.ones_like(A2), create_graph=True)[0]

    A1_y = dA_1[:,1]
    A1_x = dA_1[:,0]

    A2_x = dA_2[:,0]
    A2_y = dA_2[:,1]

    u1_x = du1[:,0]
    u1_y = du1[:,1]
    u2_x = du2[:,0]
    u2_y = du2[:,1]

    res_1 = u1_x**2  + u1_y**2 + u2_x**2 + u2_y**2
    res_2 = (A1**2 + A2**2) * (u1**2 + u2**2)
    res_3 = 2 * u2 *   (A1* u1_x + A2 * u1_y)  - 2*  u1*  (A1* u2_x + A2 * u2_y)
    res_4 = 0.5 * (1.0 - u1**2 - u2**2)**2 
    res_curl = (A2_x - A1_y -H_eval_train)**2
    loss_pde = 0.5*torch.mean((res_1 + res_2 + res_3 +  res_4 +res_curl))
    return loss_pde


def normalized_scaled_train_loss_full(model, source, max_kappa, eval_magfield: callable):
    kappa_tens=source[:,[2]]
    tmp = source[:,0:2]
    tmp.requires_grad_(True)
    H_eval_train = eval_magfield(source)
    u_pred = model(torch.cat([tmp,kappa_tens], dim=1))

    u1 = u_pred[:,0]
    u2 = u_pred[:,1]
    A1 =  u_pred[:,2]
    A2 =  u_pred[:,3]
    
    du1 = 1/(max_kappa*kappa_tens) * torch.autograd.grad(u1, tmp, grad_outputs=torch.ones_like(u1), create_graph=True)[0]
    du2 = 1/(max_kappa*kappa_tens) * torch.autograd.grad(u2, tmp, grad_outputs=torch.ones_like(u1), create_graph=True)[0]

    dA_1 = torch.autograd.grad(A1, tmp, grad_outputs=torch.ones_like(A1), create_graph=True)[0]
    dA_2 = torch.autograd.grad(A2, tmp, grad_outputs=torch.ones_like(A2), create_graph=True)[0]


    A1_y = dA_1[:,1]
    A1_x = dA_1[:,0]

    A2_x = dA_2[:,0]
    A2_y = dA_2[:,1]

    u1_x = du1[:,0]
    u1_y = du1[:,1]
    u2_x = du2[:,0]
    u2_y = du2[:,1]

    res_1 = u1_x**2  + u1_y**2 + u2_x**2 + u2_y**2
    res_2 = (A1**2 + A2**2) * (u1**2 + u2**2)
    res_3 = 2 * u2 *   (A1* u1_x + A2 * u1_y)  - 2*  u1*  (A1* u2_x + A2 * u2_y)
    res_4 = 0.5 * (1.0 - u1**2 - u2**2)**2 
    res_curl = (A2_x - A1_y -H_eval_train)**2
    res_divergence = (A1_x + A2_y)**2
    loss_pde = 0.5*torch.mean((max_kappa*kappa_tens)*(res_1 + res_2 + res_3 +  res_4 + res_curl + res_divergence))

    return loss_pde

def get_loss_singleKappaLightningOnlyOrd(kappa):
    def loss_singleKappa(model,x_batched):
        x_batched.requires_grad_(True)
        u_pred = model(x_batched)
        
        u1 = u_pred[:,0]
        u2 = u_pred[:,1]
        
        A1 , A2 = vec_pot(x_batched[:,0], x_batched[:,1])


        du1 = 1/kappa * torch.autograd.grad(u1, x_batched, grad_outputs=torch.ones_like(u1), create_graph=True)[0]
        du2 = 1/kappa * torch.autograd.grad(u2, x_batched, grad_outputs=torch.ones_like(u1), create_graph=True)[0]
        u1_x = du1[:,0]
        u1_y = du1[:,1]
        u2_x = du2[:,0]
        u2_y = du2[:,1]


        res_1 = u1_x**2  + u1_y**2 + u2_x**2 + u2_y**2
        res_2 = (A1**2 + A2**2) * (u1**2 + u2**2)
        res_3 = 2 * u2 *   (A1* u1_x + A2 * u1_y)  - 2*  u1*  (A1* u2_x + A2 * u2_y)
        res_4 = 0.5 * (1.0 - u1**2 - u2**2)**2 

        
        energy = 0.5*torch.mean(res_1 + res_2 + res_3 +  res_4)
        return energy
    return loss_singleKappa


def get_loss_singleKappaLightningFull(kappa):
    def train_loss_singleKappaFull(model, source, eval_magfield: callable):
        source.requires_grad_(True)
        H_eval_train = eval_magfield(source)
        u_pred = model(source)

        u1 = u_pred[:,0]
        u2 = u_pred[:,1]
        A1 =  u_pred[:,2]
        A2 =  u_pred[:,3]
        
        du1 = 1/(kappa) * torch.autograd.grad(u1, source, grad_outputs=torch.ones_like(u1), create_graph=True)[0]
        du2 = 1/(kappa) * torch.autograd.grad(u2, source, grad_outputs=torch.ones_like(u1), create_graph=True)[0]

        dA_1 = torch.autograd.grad(A1, source, grad_outputs=torch.ones_like(A1), create_graph=True)[0]
        dA_2 = torch.autograd.grad(A2, source, grad_outputs=torch.ones_like(A2), create_graph=True)[0]


        A1_y = dA_1[:,1]
        A1_x = dA_1[:,0]

        A2_x = dA_2[:,0]
        A2_y = dA_2[:,1]

        u1_x = du1[:,0]
        u1_y = du1[:,1]
        u2_x = du2[:,0]
        u2_y = du2[:,1]

        res_1 = u1_x**2  + u1_y**2 + u2_x**2 + u2_y**2
        res_2 = (A1**2 + A2**2) * (u1**2 + u2**2)
        res_3 = 2 * u2 *   (A1* u1_x + A2 * u1_y)  - 2*  u1*  (A1* u2_x + A2 * u2_y)
        res_4 = 0.5 * (1.0 - u1**2 - u2**2)**2 
        res_curl = (A2_x - A1_y -H_eval_train)**2
        res_divergence = (A1_x + A2_y)**2
        loss_pde = 0.5*torch.mean(res_1 + res_2 + res_3 +  res_4 + res_curl + res_divergence)

        return loss_pde
    return train_loss_singleKappaFull

