import torch
import lightning as L
from torch import nn
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import random
from models import LitMultiKappaWithWarmup, LitMultiKappaWithWarmupMuon, LitSingleKappaWithWarmup, LitSingleKappaWithWarmupMuon
from models import GatedGLENN, DefaultMLP
from datasets import RefreshableInMemoryDataModule, RefreshLogger, RefreshableInMemoryDataModuleSingleKappa, RefreshableInMemoryRefinementDataModule
from kan import KAN
import yaml

from pytorch_lightning import LightningModule
def inspect_model_dtypes(model: LightningModule):
    float_param_dtypes = set()
    float_buffer_dtypes = set()

    for name, p in model.named_parameters():
        if p.is_floating_point():
            float_param_dtypes.add(p.dtype)
    for name, b in model.named_buffers():
        if b.is_floating_point():
            float_buffer_dtypes.add(b.dtype)
    print("===== Inspecting model torch datatypes =====")
    print("Parameter float dtypes:", float_param_dtypes or "None")
    print("Buffer float dtypes:", float_buffer_dtypes or "None")
    print("============================================")


from pytorch_lightning import LightningDataModule

def inspect_datamodule_dtypes(datamodule: LightningDataModule):
    
    def inspect_loader(name, loader):
        if loader is None:
            print(f"{name}: None")
            return
        batch = next(iter(loader))
        print(f"--- {name} ---")
        if isinstance(batch, (list, tuple)):
            for i, x in enumerate(batch):
                if isinstance(x, torch.Tensor):
                    print(f"batch[{i}] dtype: {x.dtype}")
        elif isinstance(batch, dict):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    print(f"batch['{k}'] dtype: {v.dtype}")
        elif isinstance(batch, torch.Tensor):
            print(f"batch dtype: {batch.dtype}")
        else:
            print(f"{name}: batch type {type(batch)} (not a tensor)")

    print("===== Inspecting datamodule torch datatypes =====")
    inspect_loader("train_dataloader", datamodule.train_dataloader())
    print("============================================")


def create_name(experiment_config):
    general_config = experiment_config["NN"]["general"]
    model_config = experiment_config["NN"]["model"]
    fem_config = experiment_config["FEM_solver_config"]
    problem_dict = fem_config["problem_dict"]
    spacer = "_"
    return model_config["model_name"] + spacer + general_config["tensorboard_logname"] + spacer + "seed"  + model_config["seed"]

def get_trainer(experiment_config):
    """
    Generates the checkpoint callback and trainer from the configuration.
    """
    general_config = experiment_config["NN"]["general"]
    model_config = experiment_config["NN"]["model"]
    fem_config = experiment_config["FEM_solver_config"]
    problem_dict = fem_config["problem_dict"]
    nodes = general_config["nodes"]
    precision = "64-true" if torch.get_default_dtype() == torch.float64 else None
    strategy_string = "ddp_find_unused_parameters_true" if experiment_config["NN"]["model"]["model_name"] == "KAN" else "ddp"
    grad_clip_val = 1.0 if experiment_config["NN"]["training"]["optim"] == "Adam" else None

    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss',
        every_n_epochs=1,
        dirpath=general_config["output_path"],
        filename=general_config["tensorboard_logname"] + general_config["checkpoint_suffix"] +'_{epoch:01d}_-{train_loss:.6f}',
        save_top_k=10,
        mode='min',
        save_last=True,
        save_on_train_epoch_end=True
    )
    logger = TensorBoardLogger(general_config["output_path"] + "/logs", name=general_config["tensorboard_logname"])
    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        num_nodes=nodes,
        strategy=strategy_string,
        max_epochs=experiment_config["NN"]["training"]["total_epochs"],
        logger=logger,
        gradient_clip_val=grad_clip_val,
        callbacks=[checkpoint_callback,RefreshLogger()],
        check_val_every_n_epoch=experiment_config["NN"]["training"]["validate_every_n_epochs"],
        precision=precision,
        benchmark=experiment_config["NN"]["training"]["benchmarking"],
        )
    return trainer


def get_predefined_model(experiment_config):
    """
    Returns one of the predefined torch models depending on the configuration.
    """
    model_config = experiment_config["NN"]["model"]
    block_config = model_config["block_config"]
    problem_dict = experiment_config["FEM_solver_config"]["problem_dict"]

    indim = 3 if model_config["multi_kappa"] else 2
    outdim = 2 if problem_dict["only_ord"] else 4

    match experiment_config["NN"]["model"]["model_name"]:

        case "GatedGLENN_tiny":
            model = GatedGLENN(input_dim=indim, output_dim=outdim,hidden_dim=128,num_blocks=10, block_config=block_config)

        case "GatedGLENN_small":
            model = GatedGLENN(input_dim=indim, output_dim=outdim,hidden_dim=384,num_blocks=20, block_config=block_config)
        
        case "GatedGLENN_medium":
            model = GatedGLENN(input_dim=indim, output_dim=outdim,hidden_dim=512,num_blocks=20 , block_config=block_config)

        case "GatedGLENN_large":
            model = GatedGLENN(input_dim=indim, output_dim=outdim,hidden_dim=512,num_blocks=40 , block_config=block_config)

        case "MLP":
            model = DefaultMLP(input_size=indim, output_size=outdim, num_hidden_layers=7,size_hidden_layers=256,actfun=nn.GELU())

        case "KAN":
            model = KAN(width=[indim,[4,4],[4,4],[3,3],outdim], grid=10, k=5, seed=model_config["seed"],auto_save=False)
            model = model.speed()

    return model


def train_lightning_multiKappa(experiment_config):
    """
    Trains a new model with kappa as input, following the specifications in the experiment config.
    """
    general_config = experiment_config["NN"]["general"]
    model_config = experiment_config["NN"]["model"]
    fem_config = experiment_config["FEM_solver_config"]
    problem_dict = fem_config["problem_dict"]

    if general_config["cluster"]:
        num_workers = 2
    else:
        num_workers = 23

    torch.manual_seed(model_config["seed"])
    L.seed_everything(model_config["seed"], workers=True)
    random.seed(model_config["seed"])

    model = get_predefined_model(experiment_config=experiment_config)

    domain = problem_dict["geo"]
    dm = RefreshableInMemoryDataModule(total_train_samples=eval(experiment_config["NN"]["training"]["in_memory_samples"]), 
                                    global_train_batch_size=eval(experiment_config["NN"]["training"]["batch_size"]), 
                                    global_val_batch_size=eval(experiment_config["NN"]["training"]["validate_batch_size"]), 
                                    val_z_values=experiment_config["NN"]["training"]["kappas_validate"],
                                    val_grid_size=experiment_config["NN"]["training"]["dofs_validate"], 
                                    base_seed=experiment_config["NN"]["training"]["dataset_seed"],
                                    num_workers=num_workers,
                                    refresh_every_epochs=1,
                                    domain = domain
    )
    
    litGLENN = LitMultiKappaWithWarmup(model, experiment_config=experiment_config) if experiment_config["NN"]["training"]["optim"] == "Adam" else LitMultiKappaWithWarmupMuon(model, experiment_config=experiment_config)

    trainer = get_trainer(experiment_config=experiment_config)
    inspect_model_dtypes(litGLENN)
    trainer.fit(litGLENN, datamodule=dm)
    inspect_model_dtypes(litGLENN)
    inspect_datamodule_dtypes(dm)

    last_train_loss = last_train_loss = trainer.callback_metrics["train_loss_epoch"].item()
    for name, param in model.named_parameters():
        print(name, param.dtype)
        break
    with open(general_config["output_path"] + "/" + general_config["tensorboard_logname"] + ".yml", "w") as outfile:
            yaml.dump(experiment_config, outfile,default_flow_style=False)
    return model, litGLENN, last_train_loss




def load_lightning_multiKappa(experiment_config):
    """
    Loads a pre-trained model with kappa as input, following the specificaitons in the experiment config.
    """
    model_config = experiment_config["NN"]["model"]
    general_config = experiment_config["NN"]["general"]

    torch.manual_seed(model_config["seed"])
    L.seed_everything(model_config["seed"], workers=True)
    random.seed(model_config["seed"])

    model = get_predefined_model(experiment_config=experiment_config)

    match experiment_config["NN"]["training"]["optim"]:
        case "Adam":
            litGLENN = LitMultiKappaWithWarmup.load_from_checkpoint(general_config["input_path"] +"/" + general_config["input_file"],model=model, experiment_config=experiment_config)

        case "Muon":
            litGLENN = LitMultiKappaWithWarmupMuon.load_from_checkpoint(general_config["input_path"] +"/" + general_config["input_file"],model=model, experiment_config=experiment_config)

    return litGLENN, model


def plot_all(experiment_config):
    litGLENN, model = load_lightning_multiKappa(experiment_config=experiment_config)
    litGLENN = litGLENN.to('cuda')
    litGLENN.plot_full()
    exit(0)


def train_lightning_singleKappa(experiment_config):
    """
    Trains a new model for a fixed kappa, following the specifications in the experiment config.
    """
    general_config = experiment_config["NN"]["general"]
    model_config = experiment_config["NN"]["model"]

    if general_config["cluster"]:
        num_workers = 2
    else:
        num_workers = 23

    torch.manual_seed(model_config["seed"])
    L.seed_everything(model_config["seed"], workers=True)
    random.seed(model_config["seed"])

    model = get_predefined_model(experiment_config=experiment_config)

    dm = RefreshableInMemoryDataModuleSingleKappa(
                                    total_train_samples=eval(experiment_config["NN"]["training"]["in_memory_samples"]), 
                                    global_train_batch_size=eval(experiment_config["NN"]["training"]["batch_size"]), 
                                    base_seed=experiment_config["NN"]["training"]["dataset_seed"],
                                    num_workers=num_workers,
                                    refresh_every_epochs=1
    )


    litGLENN = LitSingleKappaWithWarmup(model, experiment_config=experiment_config) if experiment_config["NN"]["training"]["optim"] == "Adam" else LitSingleKappaWithWarmupMuon(model, experiment_config=experiment_config)
    trainer = get_trainer(experiment_config=experiment_config)
    inspect_model_dtypes(litGLENN)
    trainer.fit(litGLENN, datamodule=dm)
    last_train_loss = last_train_loss = trainer.callback_metrics["train_loss_epoch"].item()
    inspect_model_dtypes(litGLENN)
    inspect_datamodule_dtypes(dm)

    return model, litGLENN, last_train_loss


def load_lightning_singleKappa(experiment_config):
    """
    Loads a pre-trained model for a fixed kappa, following the specifications in the experiment config.
    """
    general_config = experiment_config["NN"]["general"]
    model_config = experiment_config["NN"]["model"]

    torch.manual_seed(model_config["seed"])
    L.seed_everything(model_config["seed"], workers=True)
    random.seed(model_config["seed"])


    model = get_predefined_model(experiment_config=experiment_config)

    match experiment_config["NN"]["training"]["optim"]:
        case "Adam":
            litGLENN = LitSingleKappaWithWarmup.load_from_checkpoint(general_config["input_path"] +"/" + general_config["input_file"],model=model, experiment_config=experiment_config)
        case "Muon":
            litGLENN = LitSingleKappaWithWarmupMuon.load_from_checkpoint(general_config["input_path"] +"/" + general_config["input_file"],model=model, experiment_config=experiment_config)

    return litGLENN, model

def train_lightning(experiment_config):
    if experiment_config["NN"]["model"]["multi_kappa"]:
        return train_lightning_multiKappa(experiment_config)
    else:
        return train_lightning_singleKappa(experiment_config)
    
def load_lightning(experiment_config):
    if experiment_config["NN"]["model"]["multi_kappa"]:
        return load_lightning_multiKappa(experiment_config)
    else:
        return load_lightning_singleKappa(experiment_config)
    

def refine_lightning_multiKappa(experiment_config):
    """
    Refines a model trained with kappa as input, trained with 32 bits, with 64 bits precision.
    """
    torch.set_default_dtype(torch.float64)

    general_config = experiment_config["NN"]["general"]
    model_config = experiment_config["NN"]["model"]
    fem_config = experiment_config["FEM_solver_config"]
    problem_dict = fem_config["problem_dict"]

    if general_config["cluster"]:
        num_workers = 2
    else:
        num_workers = 23

    torch.manual_seed(model_config["seed"])
    L.seed_everything(model_config["seed"], workers=True)
    random.seed(model_config["seed"])

    litGLENN, model = load_lightning(experiment_config)
    litGLENN = litGLENN.double()

    domain = problem_dict["geo"]
    dm = RefreshableInMemoryDataModule(total_train_samples=eval(experiment_config["NN"]["training"]["in_memory_samples"]), 
                                    global_train_batch_size=eval(experiment_config["NN"]["training"]["batch_size"]), 
                                    global_val_batch_size=eval(experiment_config["NN"]["training"]["validate_batch_size"]), 
                                    val_z_values=experiment_config["NN"]["training"]["kappas_validate"],
                                    val_grid_size=experiment_config["NN"]["training"]["dofs_validate"], 
                                    base_seed=experiment_config["NN"]["training"]["dataset_seed"],
                                    num_workers=num_workers,
                                    refresh_every_epochs=1,
                                    domain = domain
    )
   
    trainer = get_trainer(experiment_config=experiment_config)
    inspect_model_dtypes(litGLENN)
    trainer.fit(litGLENN, datamodule=dm)
    last_train_loss = last_train_loss = trainer.callback_metrics["train_loss_epoch"].item()
    inspect_model_dtypes(litGLENN)
    inspect_datamodule_dtypes(dm)
    return model, litGLENN, last_train_loss



def runtime_refinement_focused_kappa(experiment_config):
    """
    Refines a model that was trained on a range of kappa values for one specific kappa value. Refinement always with 64bit precision.
    Attention: Break of convention here, the kappa used for refinement is the one in the loading config.
    """
    torch.set_default_dtype(torch.float64)

    general_config = experiment_config["NN"]["general"]
    model_config = experiment_config["NN"]["model"]
    loading_config = experiment_config["NN"]["loading_config_for_FEM"]
    refinement_kappa = loading_config["kappa"][0]
    max_kappa = model_config["max_kappa"]
    print(f"refinement kappa = {refinement_kappa}")
    print(f"max kappa = {max_kappa}")

    scaled_kappa = refinement_kappa/max_kappa
    print(f"scaled_kappa = {scaled_kappa}")

    fem_config = experiment_config["FEM_solver_config"]
    problem_dict = fem_config["problem_dict"]

    if general_config["cluster"]:
        num_workers = 2
    else:
        num_workers = 23

    torch.manual_seed(model_config["seed"])
    L.seed_everything(model_config["seed"], workers=True)
    random.seed(model_config["seed"])

    litGLENN, model = load_lightning(experiment_config)
    litGLENN = litGLENN.double()

    domain = problem_dict["geo"]
    dm = RefreshableInMemoryRefinementDataModule(total_train_samples=eval(experiment_config["NN"]["training"]["in_memory_samples"]), 
                                    global_train_batch_size=eval(experiment_config["NN"]["training"]["batch_size"]), 
                                    val_grid_size=experiment_config["NN"]["training"]["dofs_validate"],
                                    val_z_values=[refinement_kappa],
                                    base_seed=experiment_config["NN"]["training"]["dataset_seed"],
                                    num_workers=num_workers,
                                    refresh_every_epochs=experiment_config["NN"]["training"]["total_epochs"],
                                    domain = domain,
                                    kappa = scaled_kappa,
    )
    
    litGLENN = LitMultiKappaWithWarmup(model, experiment_config=experiment_config) if experiment_config["NN"]["training"]["optim"] == "Adam" else LitMultiKappaWithWarmupMuon(model, experiment_config=experiment_config)
    trainer = get_trainer(experiment_config=experiment_config)
    inspect_model_dtypes(litGLENN)
    trainer.fit(litGLENN, datamodule=dm)
    last_train_loss = last_train_loss = trainer.callback_metrics["train_loss_epoch"].item()
    inspect_model_dtypes(litGLENN)
    inspect_datamodule_dtypes(dm)
    return model, litGLENN, last_train_loss