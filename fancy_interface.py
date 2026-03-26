import torch
import numpy as np
from torch.utils.data import DataLoader
import lightning as L


class TensorOnlyDataset(torch.utils.data.Dataset):
    def __init__(self, tensor):
        self.tensor = tensor
    def __len__(self):
        return len(self.tensor)
    def __getitem__(self, idx):
        return self.tensor[idx]


def fast_eval_multiKappa_from_model(model,config,kappa):
    max_kappa = config["max_kappa"]

    model.eval()
    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
        precision="64-true",
        logger=False,
        enable_progress_bar=True,
        use_distributed_sampler=False,
    )

    
    def f_full(x):
        x_proper_shape = torch.from_numpy(x)[0:2,:]
        x_vertical = x_proper_shape.transpose(0,1)
        kappa_tens = (kappa/max_kappa)* torch.ones_like(x_vertical[:,[0]], dtype = torch.float64)
        full_tens = torch.cat((x_vertical, kappa_tens), dim=1)
        dataset=TensorOnlyDataset(full_tens)
        batch_size = 2**18
        preds = []
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        preds = trainer.predict(model, dataloaders=loader)
        eval = torch.cat(preds, dim=0)
        numpy_eval = eval.to('cpu').detach().numpy()
        return numpy_eval 
    return f_full



class CachedPrediction:
    def __init__(self,model,config,kappa):
        self.f_full = fast_eval_multiKappa_from_model(model, config, kappa)
        self.cache = None

    def f_real(self,x):
        self.cache = self.f_full(x)
        return self.cache[:,0].reshape(x.shape[1])
    
    def f_imag(self,x):
        return self.cache[:,1].reshape(x.shape[1])
    
    def A_full(self,x):
        self.cache = self.f_full(x)
        return self.cache[:,2].reshape(x.shape[1]), self.cache[:,3].reshape(x.shape[1])

def fast_eval_singleKappa_from_model(model, config=None):

    model.eval()
    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
        precision="64-true",
        logger=False,
        enable_progress_bar=True,
        use_distributed_sampler=False,
    )


    def f_full(x):
    

        x_proper_shape = torch.from_numpy(np.float32(x))[0:2,:]
        x_vertical = x_proper_shape.transpose(0,1)
        dataset=TensorOnlyDataset(x_vertical)
        batch_size = 2**18

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        preds = trainer.predict(model, dataloaders=loader)
        eval = torch.cat(preds, dim=0)
        numpy_eval = eval.to('cpu').detach().numpy()
        return numpy_eval 
    
    return f_full

class CachedPredictionSingleKappa:
    def __init__(self,model,config):
        self.f_full = fast_eval_singleKappa_from_model(model, config)
        self.cache = None

    def f_real(self,x):
        self.cache = self.f_full(x)
        return self.cache[:,0].reshape(x.shape[1])
    
    def f_imag(self,x):
        return self.cache[:,1].reshape(x.shape[1])
    
    def A_full(self,x):
        self.cache = self.f_full(x)
        return self.cache[:,2].reshape(x.shape[1]), self.cache[:,3].reshape(x.shape[1])


def get_f_real_multiKappaModel(model, config, loading_kappa):
    def f_real(x):
        print("f_real input shape = " + str(x.shape))
        full_function = fast_eval_multiKappa_from_model(model,config,loading_kappa)
        return full_function(x)[:,0].reshape(x.shape[1])
    return f_real
    


def get_f_imag_multikappaModel(model, config, loading_kappa):
    def f_imag(x):
        print("f_imag input shape = " + str(x.shape))
        full_function = fast_eval_multiKappa_from_model(model,config,loading_kappa)
        return full_function(x)[:,1].reshape(x.shape[1])
    return f_imag

def get_A1_multikappaModel(model, config, loading_kappa):
    def A1(x):
        print("A input shape = " + str(x.shape))
        full_function = fast_eval_multiKappa_from_model(model,config,loading_kappa)
        return full_function(x)[:,2].reshape(x.shape[1])
    return A1

def get_A2_multikappaModel(model, config, loading_kappa):
    def A2(x):
        full_function = fast_eval_multiKappa_from_model(model,config,loading_kappa)
        return full_function(x)[:,3].reshape(x.shape[1])
    return A2

def get_Afull_multikappaModel(model, config, loading_kappa):
    def A2(x):
        print("A input shape = " + str(x.shape))
        full_function = fast_eval_multiKappa_from_model(model,config,loading_kappa)
        tmp = full_function(x)
        return tmp[:,2].reshape(x.shape[1]), tmp[:,3].reshape(x.shape[1])
    return A2


def get_uReal_uImag_multikappaModel(model, config,loading_kappa):
    return get_f_real_multiKappaModel(model,config,loading_kappa), get_f_imag_multikappaModel(model,config,loading_kappa)

def get_uA_multikappaModel(model, config,loading_kappa):
    return get_f_real_multiKappaModel(model,config,loading_kappa), get_f_imag_multikappaModel(model,config,loading_kappa), get_A1_multikappaModel(model,config,loading_kappa),get_A2_multikappaModel(model,config,loading_kappa)

def get_uFullA_multikappaModel(model, config,loading_kappa):
    return get_f_real_multiKappaModel(model,config,loading_kappa), get_f_imag_multikappaModel(model,config,loading_kappa), get_Afull_multikappaModel(model,config,loading_kappa)


def get_f_real_singleKappaModel(model,config=None):
    def f_real(x):
        full_function = fast_eval_singleKappa_from_model(model)
        return full_function(x)[:,0].reshape(x.shape[1])
    return f_real

def get_f_imag_singleKappaModel(model,config=None):
    def f_real(x):
        full_function = fast_eval_singleKappa_from_model(model)
        return full_function(x)[:,1].reshape(x.shape[1])
    return f_real

def get_f_full_singleKappaModel(model,config=None):
    return get_f_real_singleKappaModel(model,config), get_f_imag_singleKappaModel(model,config)

