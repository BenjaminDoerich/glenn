import torch
import torch.nn as nn
from matplotlib.backends.backend_pdf import PdfPages
import lightning as L
import matplotlib.pyplot as plt
from loss_functions import MagField
import math
import torch.nn.functional as F
import loss_functions
from torch.optim.lr_scheduler import ExponentialLR
from pathlib import Path
import importlib
import importlib.util



def function_loader(filename,function_name):
    """
    Import a function with a given name from a python file.
    """
    module_filename = filename
    current_path = Path(__file__).resolve().parent
    module_path = current_path / module_filename
    module_name = module_path.stem  # name without .py
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    target_function = getattr(module, function_name)
    return target_function



def split_params_for_muon_and_other(model: nn.Module):
    """
    Splits parameters into AdamW and Muon when using Muon with clean_weight_decay=False.
    """
    muon_params = []
    other_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() == 2:
            muon_params.append(p)
        else:
            other_params.append(p)

    return muon_params, other_params



def split_params_for_muon_and_adamw(model: nn.Module):
    """
    Splits parameters into AdamW and Muon when using Muon with clean_weight_decay=True, i.e., no weight decay on biases and scalings, only on weights.
    """
    muon_params = []
    adam_decay = []
    adam_bias = []
    adam_scales = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        is_2d = (p.dim() == 2)
        is_bias = name.endswith(".bias")
        is_scale_param = ("log_gamma" in name) or ("log_alpha" in name)

        if is_2d and not is_scale_param:
            muon_params.append(p)
        else:
            if is_bias:
                adam_bias.append(p)
            elif is_scale_param:
                adam_scales.append(p)
            else:
                adam_decay.append(p)

    return muon_params, adam_decay, adam_bias, adam_scales



def split_params_for_adamw_only(model: nn.Module):
    """
    Splits parameters when using AdamW only with clean_weight_decay=True, i.e., no weight decay on biases and scalings, only on weights.
    """
    adam_decay = []
    adam_bias = []
    adam_scales = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        is_bias = name.endswith(".bias")
        is_scale_param = ("log_gamma" in name) or ("log_alpha" in name)

        if is_bias:
            adam_bias.append(p)
        elif is_scale_param:
            adam_scales.append(p)
        else:
            adam_decay.append(p)

    return adam_decay, adam_bias, adam_scales



def get_correct_loss_functions(experiment_config):
    """
    Helper function. Since we use a variety of different loss functions depending on the 
    situation, this helper prevents the accidental usage of an incorrect loss.
    experiment_config: Full config dict of the experiment, including the FEM_solver_config.
    """
    model_config = experiment_config["NN"]["model"]
    problem_dict = experiment_config["FEM_solver_config"]["problem_dict"]
    if model_config["multi_kappa"]:
        if not problem_dict["only_ord"]:
            train_loss = function_loader("loss_functions.py","normalized_scaled_train_loss_full")
            test_loss = function_loader("loss_functions.py","normalized_test_loss_full")
        else:
            train_loss = function_loader("loss_functions.py","normalized_scaled_train_loss_only_ord")
            test_loss = function_loader("loss_functions.py","normalized_test_loss_only_ord")
    else:
        if not problem_dict["only_ord"]:
            train_loss = loss_functions.get_loss_singleKappaLightningFull(experiment_config["NN"]["model"]["max_kappa"])
            test_loss = loss_functions.get_loss_singleKappaLightningFull(experiment_config["NN"]["model"]["max_kappa"]) # TODO
        else:
            train_loss = loss_functions.get_loss_singleKappaLightningOnlyOrd(experiment_config["NN"]["model"]["max_kappa"])
            test_loss = loss_functions.get_loss_singleKappaLightningOnlyOrd(experiment_config["NN"]["model"]["max_kappa"]) # TODO   
    return train_loss, test_loss



def init_layer(layer, activation):
    """
    Initialize a Linear layer according to the chosen activation function.
    Bias is always zero-initialized.
    """
    if isinstance(activation, nn.ReLU):
        nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
    
    elif isinstance(activation, nn.GELU):
        nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
    
    elif isinstance(activation, nn.SiLU):
        nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
    
    elif isinstance(activation, nn.Tanh):
        nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('tanh'))
    
    else:
        nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('linear'))
    
    nn.init.zeros_(layer.bias)



def init_half_zero_half_sigmoid(layer: nn.Linear):
    """Initialize first half of a linear layer's outputs to 0,
    and the second half with Xavier init tuned for sigmoid."""
    if not isinstance(layer, nn.Linear):
        return

    with torch.no_grad():
        out_features = layer.weight.size(0)
        half = out_features // 2
        layer.weight[:half].zero_()
        gain = nn.init.calculate_gain('sigmoid')
        nn.init.xavier_uniform_(layer.weight[half:], gain=gain)

        if layer.bias is not None:
            layer.bias.zero_()



def init_half_small_half_sigmoid(layer: nn.Linear, value_gain: float = 0.1):
    """
    Initialize first half of a linear layer with small Xavier,
    second half tuned for sigmoid.
    """
    if not isinstance(layer, nn.Linear):
        return
    with torch.no_grad():
        out_features = layer.weight.size(0)
        half = out_features // 2

        nn.init.xavier_uniform_(layer.weight[:half], gain=value_gain)

        gate_gain = nn.init.calculate_gain('sigmoid')
        nn.init.xavier_uniform_(layer.weight[half:], gain=gate_gain)

        if layer.bias is not None:
            layer.bias.zero_()




class PerChannelScale(nn.Module):
    def __init__(
        self,
        d_model: int,
        init_scale: float = 1e-4,
        max_scale: float | None = 1.0,
    ):
        super().__init__()
        log_init = math.log(init_scale)
        self.log_gamma = nn.Parameter(torch.full((d_model,), log_init))
        self.max_scale = max_scale


    def forward(self) -> torch.Tensor:
        gamma = torch.exp(self.log_gamma)
        if self.max_scale is not None:
            gamma = torch.clamp(gamma, max=self.max_scale)
        return gamma



class ScalarScale(nn.Module):
    def __init__(
        self,
        init_scale: float = 1e-4,
        max_scale: float | None = 1.0,
    ):
        super().__init__()
        log_init = math.log(init_scale)
        self.log_alpha = nn.Parameter(torch.tensor(log_init))
        self.max_scale = max_scale


    def forward(self) -> torch.Tensor:
        alpha = torch.exp(self.log_alpha)
        if self.max_scale is not None:
            alpha = torch.clamp(alpha, max=self.max_scale)
        return alpha



class GatedMLPBlockFancy(nn.Module):

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        activation: str = "gelu",
        variant: str = "gated_output",
        scale_mode: str = "fixed_scalar",
        scale_init: float = 1.0,
        max_scale: float | None = None,
    ):
        super().__init__()
        if d_ff is None:
            d_ff = d_model
        self.d_model = d_model
        self.d_ff = d_ff
        self.variant = variant
        self.scale_mode = scale_mode

        if activation == "silu":
            self.activation = F.silu
            self.activation_module = nn.SiLU()
        elif activation == "gelu":
            self.activation = F.gelu
            self.activation_module = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        if self.variant == "classical_swiglu":
            self.w_in = nn.Linear(d_model, 2 * d_ff, bias=False)
            self.w_out = nn.Linear(d_ff, d_model, bias=False)
            self.fc1 = None
            self.fc2 = None
            self._init_scale(scale_init, max_scale)
            self.reset_parameters()

        elif self.variant == "gated_output":
            self.fc1 = nn.Linear(d_model, d_ff, bias=False)
            self.fc2 = nn.Linear(d_ff, 2 * d_model, bias=False)
            self.w_in = None
            self.w_out = None
            self._init_scale(scale_init, max_scale)
            self.reset_parameters()

        elif self.variant == "gated_output_stable":
            assert d_ff == d_model, "For stable variant, d_ff must equal d_model."
            self.fc1 = nn.Linear(d_model, d_model, bias=True)
            self.fc2 = nn.Linear(d_model, 2 * d_model, bias=True)
            self.w_in = None
            self.w_out = None
            assert self.scale_mode == "fixed_scalar", "Stable variant expects fixed_scalar scale."
            self.scale = scale_init
            # init_half_zero_half_sigmoid(self.fc2) # original zero init was stable and worked well
            init_half_small_half_sigmoid(self.fc2, value_gain=0.1)  # alternative init that is not constant zero
            init_layer(self.fc1, self.activation_module)

        else:
            raise ValueError(f"Unknown variant: {self.variant}")


    def _init_scale(self, scale_init: float, max_scale: float | None):
        if self.scale_mode == "learned_scalar":
            self.scale = ScalarScale(init_scale=scale_init, max_scale=max_scale)
        elif self.scale_mode == "learned_channel":
            self.scale = PerChannelScale(self.d_model, init_scale=scale_init, max_scale=max_scale)
        elif self.scale_mode == "fixed_scalar":
            self.register_buffer("scale", torch.full((), scale_init))
        else:
            raise ValueError(f"Unknown scale_mode: {self.scale_mode}")


    def reset_parameters(self):
        if self.variant == "classical_swiglu":
            nn.init.xavier_normal_(self.w_in.weight, gain=1.0)
            nn.init.xavier_normal_(self.w_out.weight, gain=1.0)
        elif self.variant == "gated_output":
            nn.init.xavier_normal_(self.fc1.weight, gain=1.0)
            nn.init.xavier_normal_(self.fc2.weight, gain=1.0)
        elif self.variant == "gated_output_stable":
            return
        else:
            raise RuntimeError("Invalid variant state")


    def _get_scale(self) -> torch.Tensor:
        if isinstance(self.scale, (PerChannelScale, ScalarScale)):
            return self.scale()
        else:
            return self.scale


    def _forward_classical_swiglu(self, x: torch.Tensor) -> torch.Tensor:
        g, v = self.w_in(x).chunk(2, dim=-1)
        h = self.activation(g) * v
        f = self.w_out(h)
        scale = self._get_scale()
        return x + scale * f


    def _forward_gated_output(self, x: torch.Tensor) -> torch.Tensor:
        h = self.activation(self.fc1(x))
        value, gate_pre = self.fc2(h).chunk(2, dim=-1)
        gate = self.activation(gate_pre)
        scale = self._get_scale()
        return x + scale * gate * value


    def _forward_gated_output_stable(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.fc1(x)
        residual = self.activation(residual)
        value, gate_pre = self.fc2(residual).chunk(2, dim=-1)
        gate = torch.sigmoid(gate_pre)
        scale = self._get_scale()
        return x + scale * gate * value


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.variant == "classical_swiglu":
            return self._forward_classical_swiglu(x)
        elif self.variant == "gated_output":
            return self._forward_gated_output(x)
        elif self.variant == "gated_output_stable":
            return self._forward_gated_output_stable(x)
        else:
            raise RuntimeError("Invalid variant state")
        
 
     
class GatedGLENN(nn.Module):
    def __init__(self, 
                  input_dim,
                  hidden_dim, 
                  num_blocks, 
                  output_dim=1, 
                  block_config=None,):

        super().__init__()
        scale = 1.0 / (2*num_blocks)

        self.input_layer = nn.Linear(input_dim, hidden_dim)

        d_ff = hidden_dim if block_config["block_dim"] == "same" else None


        self.blocks = nn.ModuleList([
                                GatedMLPBlockFancy(d_model=hidden_dim, 
                                              d_ff=d_ff, 
                                              variant=block_config["swiglu_variant"],
                                              activation=block_config["activation"], 
                                              scale_mode=block_config["scale_mode"], 
                                              scale_init=scale ) 
                                for _ in range(num_blocks)
                            ])
  
        self.output_layer = nn.Linear(hidden_dim, output_dim)


    def forward(self, x):
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_layer(x)
        return x
    


class LitMultiKappaWithWarmup(L.LightningModule):
    def __init__(self, model, experiment_config={}, train_loss_function = None, test_loss_function = None):
        super().__init__()
        self.model = model
        self.experiment_config = experiment_config
        self.lr = float(self.experiment_config["NN"]["training"]["initial_optim_lr"])
        self.lr_scales = float(self.experiment_config["NN"]["training"]["initial_optim_lr_scales"])

        self.loss_fun, self.test_loss = get_correct_loss_functions(experiment_config=experiment_config)

        self.training_losses = []
        self.plotting_configured = ("plotting" in self.experiment_config["NN"])
        self.plotter = function_loader("fancy_plotters.py", self.experiment_config["NN"]["plotting"]["plotting_function_name"])
        self.max_kappa = self.experiment_config["NN"]["model"]["max_kappa"]
        self.warmup_steps = eval(str(self.experiment_config["NN"]["training"]["warmup_steps"]))
        self.cosine_decay_steps = eval(str(self.experiment_config["NN"]["training"]["cosine_decay_steps"]))
        self.eta_min = float(self.experiment_config["NN"]["training"]["eta_min"])
        self.weight_decay = float(self.experiment_config["NN"]["training"]["weight_decay"])
        self.cooldown_steps = eval(str(self.experiment_config["NN"]["training"]["cooldown_steps"]))

        self.only_ord = self.experiment_config["FEM_solver_config"]["problem_dict"]["only_ord"]
        self.magpot = MagField(self.experiment_config["FEM_solver_config"]["problem_dict"]["H_type"], self.experiment_config["FEM_solver_config"]["problem_dict"]["mag_scale"])
        print(type(self.magpot.get_eval_magfield()))


    def forward(self,x):
        return self.model(x)
    

    def configure_plotters(self, experiment_config):
        self.plotting_configured = True
        self.experiment_config = experiment_config


    def training_step(self, batch, batch_idx):
        x = batch
        if not self.only_ord:
            loss = self.loss_fun(self.model, x, self.max_kappa, eval_magfield=self.magpot.get_eval_magfield())
        else:
            loss = self.loss_fun(self.model, x, self.max_kappa)

        self.training_losses.append(loss.detach())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        optimizer = self.trainer.optimizers[0]
        use_clean_decay = self.experiment_config["NN"]["training"]["clean_weight_decay"]


        lr_main = optimizer.param_groups[0]["lr"]
        self.log(
            "lr_AdamW_main",
            lr_main,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )


        # should be more robust, without biases we log the scales lr as bias lr:
        if use_clean_decay:

            if len(optimizer.param_groups) >= 2:
                lr_bias = optimizer.param_groups[1]["lr"]
                self.log(
                    "lr_AdamW_bias",
                    lr_bias,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True,
                )

            if len(optimizer.param_groups) >= 3:
                lr_scales = optimizer.param_groups[2]["lr"]
                self.log(
                    "lr_AdamW_scales",
                    lr_scales,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True,
                )

        return loss
        

    def plot_full(self):
        general_config = self.experiment_config["NN"]["general"]
        plotting_config = self.experiment_config["NN"]["plotting"]
        import numpy as np
        from fancy_plotters import allplot
        plotting_config["plotting_kappas"] = np.arange(0.05*self.max_kappa, self.max_kappa + 0.5, 0.5).tolist() 
        for frame_idx, kappa in enumerate(plotting_config["plotting_kappas"], start=1):
            fig, ax = allplot(
                model=self,
                kappa=kappa,
                ax=None,
                max_kappa=self.max_kappa,
                n=200,
                only_rank0=True,
                add_colorbar=True,
                add_title=True,
            )

            fig.savefig(
                f"{general_config['output_path']}/final_kap_{frame_idx:04d}.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)

            
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        step = self.global_step
        if not self.only_ord:
            with torch.enable_grad():
                test_loss = self.test_loss(self.model, batch, self.max_kappa,eval_magfield=self.magpot.get_eval_magfield())
        else:
            with torch.enable_grad():
                test_loss = self.test_loss(self.model, batch, self.max_kappa)


        z_val = self.trainer.datamodule.val_z_values[dataloader_idx]

        self.log(
            f"val_loss@kappa={z_val}",
            test_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        if self.plotting_configured:
            general_config = self.experiment_config["NN"]["general"]
            plotting_config = self.experiment_config["NN"]["plotting"]


            if batch_idx == 0:
                fig, axes = plt.subplots(2, 4, figsize=(20, 10))
                axes = axes.flatten()

                for ax, kappa in zip(axes, plotting_config["plotting_kappas"]):
                    self.plotter(self, kappa, ax=ax, max_kappa=self.max_kappa)

                plt.tight_layout()

                step = self.global_step
                self.logger.experiment.add_figure(
                    f"val/abs_value_plots",
                    fig,
                    global_step=step
                )

                pdf_path = str(general_config["output_path"]) + "/val_abs_plots_epoch" + str(self.current_epoch) + "_step" + str(step) + ".pdf"
                with PdfPages(pdf_path) as pdf:
                    pdf.savefig(fig)
                fig.savefig(str(general_config["output_path"]) + "/val_abs_plots_epoch" + str(self.current_epoch) + "_step" + str(step) + ".svg", format='svg')
                fig.savefig(str(general_config["output_path"]) + "/val_abs_plots_epoch" + str(self.current_epoch) + "_step" + str(step) + ".png", format='png')

                plt.close(fig)
        return test_loss
    

    def configure_optimizers(self):
        use_clean_decay = self.experiment_config["NN"]["training"]["clean_weight_decay"]

        if use_clean_decay:
            adam_decay, adam_bias, adam_scales = split_params_for_adamw_only(self)

            param_groups = []
            if len(adam_decay) > 0:
                param_groups.append(
                    {
                        "params": adam_decay,
                        "lr": self.lr,
                        "weight_decay": self.weight_decay,
                    }
                )
            if len(adam_bias) > 0:
                param_groups.append(
                    {
                        "params": adam_bias,
                        "lr": self.lr,
                        "weight_decay": 0.0,
                    }
                )
            if len(adam_scales) > 0:
                param_groups.append(
                    {
                        "params": adam_scales,
                        "lr": self.lr_scales,
                        "weight_decay": 0.0,
                    }
                )

            optimizer = torch.optim.AdamW(
                param_groups,
            )
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )

        scheduler1 = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=self.experiment_config["NN"]["training"]["warmup_factor"],
            end_factor=1.0,
            total_iters=self.warmup_steps,
        )

        if self.experiment_config["NN"]["training"]["scheduler"] == "exponential":
            factor = 10.0
            steps_per_factor = eval(str(self.experiment_config["NN"]["training"]["exp_steps"]))
            gamma = factor ** (-1.0 / steps_per_factor)
            scheduler2 = ExponentialLR(optimizer, gamma=gamma)
        else:
            scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.cosine_decay_steps,
                eta_min=self.eta_min,
            )

        start_factor = self.eta_min / self.lr
        end_factor = self.experiment_config["NN"]["training"]["cooldown_factor"] * start_factor
        scheduler3 = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=start_factor,
            end_factor=end_factor,
            total_iters=self.cooldown_steps,
        )

        lr = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[scheduler1, scheduler2, scheduler3],
            milestones=[self.warmup_steps, self.warmup_steps + self.cosine_decay_steps],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr,
                "interval": "step",
                "frequency": 1,
            },
        }



class LitMultiKappaWithWarmupMuon(L.LightningModule):
    def __init__(self, model, experiment_config={}):
        super().__init__()
        self.model = model
        self.experiment_config = experiment_config
        self.lr = float(self.experiment_config["NN"]["training"]["initial_optim_lr"])
        self.lr_scales = float(self.experiment_config["NN"]["training"]["initial_optim_lr_scales"])

        self.loss_fun, self.test_loss = get_correct_loss_functions(experiment_config=experiment_config)
        
        self.training_losses = []
        self.plotting_configured = ("plotting" in self.experiment_config["NN"])
        self.plotter = function_loader("fancy_plotters.py", self.experiment_config["NN"]["plotting"]["plotting_function_name"])
        self.max_kappa = self.experiment_config["NN"]["model"]["max_kappa"]
        self.warmup_steps = eval(str(self.experiment_config["NN"]["training"]["warmup_steps"]))
        self.cosine_decay_steps = eval(self.experiment_config["NN"]["training"]["cosine_decay_steps"])
        self.eta_min = float(self.experiment_config["NN"]["training"]["eta_min"])
        self.weight_decay = float(self.experiment_config["NN"]["training"]["weight_decay"])
        self.automatic_optimization = False
        self.cooldown_steps = eval(str(self.experiment_config["NN"]["training"]["cooldown_steps"]))
        self.only_ord = self.experiment_config["FEM_solver_config"]["problem_dict"]["only_ord"]
        self.magpot = MagField(self.experiment_config["FEM_solver_config"]["problem_dict"]["H_type"], self.experiment_config["FEM_solver_config"]["problem_dict"]["mag_scale"])


    def forward(self,x):
        return self.model(x)
    

    def configure_plotters(self, experiment_config):
        self.plotting_configured = True
        self.experiment_config = experiment_config


    def training_step(self, batch, batch_idx):
        x = batch
        if not self.only_ord:
            loss = self.loss_fun(self.model, x, self.max_kappa, eval_magfield=self.magpot.get_eval_magfield())
        else:
            loss = self.loss_fun(self.model, x, self.max_kappa)

        opt_w, opt_b = self.optimizers(use_pl_optimizer=True)
        sched_weights, sched_biases = self.lr_schedulers()
        opt_w.zero_grad(set_to_none=True)
        opt_b.zero_grad(set_to_none=True)

        self.manual_backward(loss)

        opt_w.step()
        sched_weights.step()
        opt_b.step()
        sched_biases.step()

        self.training_losses.append(loss.detach())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        lr_muon_main = opt_w.param_groups[0]["lr"]
        lr_adam_main = opt_b.param_groups[0]["lr"]

        self.log("lr_Muon_main", lr_muon_main,
                on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("lr_AdamW_main", lr_adam_main,
                on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        use_clean_decay = self.experiment_config["NN"]["training"]["clean_weight_decay"]

        if use_clean_decay:

            if len(opt_b.param_groups) >= 2:
                # If we add biases and want to log their LR
                # lr_bias = opt_b.param_groups[1]["lr"]
                # self.log("lr_AdamW_bias", lr_bias, ...)
                pass

            if len(opt_b.param_groups) >= 2:
                lr_scales = opt_b.param_groups[-1]["lr"]
                self.log("lr_AdamW_scales", lr_scales,
                        on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss
        

    def plot_full(self):
        general_config = self.experiment_config["NN"]["general"]
        plotting_config = self.experiment_config["NN"]["plotting"]
        import numpy as np
        from fancy_plotters import allplot
        plotting_config["plotting_kappas"] = np.arange(0.05*self.max_kappa, self.max_kappa + 0.5, 0.5).tolist() 
        for frame_idx, kappa in enumerate(plotting_config["plotting_kappas"], start=1):
            fig, ax = allplot(
                model=self,
                kappa=kappa,
                ax=None,
                max_kappa=self.max_kappa,
                n=200,
                only_rank0=True,
                add_colorbar=True,
                add_title=True,
            )

            fig.savefig(
                f"{general_config['output_path']}/final_kap_{frame_idx:04d}.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)


    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        step = self.global_step
        if not self.only_ord:
            with torch.enable_grad():
                test_loss = self.test_loss(self.model, batch, self.max_kappa,eval_magfield=self.magpot.get_eval_magfield())
        else:
            with torch.enable_grad():
                test_loss = self.test_loss(self.model, batch, self.max_kappa)

        z_val = self.trainer.datamodule.val_z_values[dataloader_idx]

        self.log(
            f"val_loss@kappa={z_val}",
            test_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        if self.plotting_configured:
            general_config = self.experiment_config["NN"]["general"]
            plotting_config = self.experiment_config["NN"]["plotting"]


            if batch_idx == 0 and self.trainer.global_rank == 0:
                fig, axes = plt.subplots(2, 4, figsize=(20, 10))
                axes = axes.flatten()
                for ax, kappa in zip(axes, plotting_config["plotting_kappas"]):
                    self.plotter(self, kappa, ax=ax, max_kappa=self.max_kappa)


                plt.tight_layout()

                step = self.global_step
                self.logger.experiment.add_figure(
                    f"val/abs_value_plots",
                    fig,
                    global_step=step
                )

                pdf_path = str(general_config["output_path"]) + "/val_abs_plots_epoch" + str(self.current_epoch) + "_step" + str(step) + ".pdf"
                with PdfPages(pdf_path) as pdf:
                    pdf.savefig(fig)
                fig.savefig(str(general_config["output_path"]) + "/val_abs_plots_epoch" + str(self.current_epoch) + "_step" + str(step) + ".svg", format='svg')
                fig.savefig(str(general_config["output_path"]) + "/val_abs_plots_epoch" + str(self.current_epoch) + "_step" + str(step) + ".png", format='png')

                plt.close(fig)
        return test_loss
    

    def configure_optimizers(self):
        use_clean_decay = self.experiment_config["NN"]["training"]["clean_weight_decay"]

        optimizers = []
        schedulers = []

        if use_clean_decay:
            muon_params, adam_decay, adam_bias, adam_scales = split_params_for_muon_and_adamw(self.model)
            self.print(
                f"[cfg] muon params: {sum(p.numel() for p in muon_params)} | "
                f"adam_decay: {sum(p.numel() for p in adam_decay)} | "
                f"adam_bias: {sum(p.numel() for p in adam_bias)} | "
                f"adam_scales: {sum(p.numel() for p in adam_scales)}"
            )
            assert len(muon_params) > 0, "No weight params found for Muon (are there any 2D weights?)"
            assert (len(adam_decay) + len(adam_bias) + len(adam_scales)) > 0, "No non-muon params found"
        else:
            muon_params, other_params = split_params_for_muon_and_other(self.model)
            self.print(
                f"[cfg] muon params: {sum(p.numel() for p in muon_params)} | "
                f"other params: {sum(p.numel() for p in other_params)}"
            )
            assert len(muon_params) > 0, "No weight params found"
            assert len(other_params) > 0, "No bias params found"

            adam_decay = other_params
            adam_bias = []
            adam_scales = []

        opt_w = torch.optim.Muon(
            muon_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            adjust_lr_fn="match_rms_adamw"
        )
        optimizers.append(opt_w)

        scheduler1W = torch.optim.lr_scheduler.LinearLR(
            opt_w,
            start_factor=self.experiment_config["NN"]["training"]["warmup_factor"],
            end_factor=1.0,
            total_iters=self.warmup_steps,
        )

        if self.experiment_config["NN"]["training"]["scheduler"] == "exponential":
            factor = 10.0
            steps_per_factor = eval(str(self.experiment_config["NN"]["training"]["exp_steps"]))
            gamma = factor ** (-1.0 / steps_per_factor)
            scheduler2W = ExponentialLR(opt_w, gamma=gamma)
        else:
            scheduler2W = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt_w,
                T_max=self.cosine_decay_steps,
                eta_min=self.eta_min,
            )

        start_factorW = self.eta_min / self.lr
        end_factorW = self.experiment_config["NN"]["training"]["cooldown_factor"] * start_factorW
        scheduler3W = torch.optim.lr_scheduler.LinearLR(
            opt_w,
            start_factor=start_factorW,
            end_factor=end_factorW,
            total_iters=self.cooldown_steps,
        )

        lrW = torch.optim.lr_scheduler.SequentialLR(
            opt_w,
            schedulers=[scheduler1W, scheduler2W, scheduler3W],
            milestones=[self.warmup_steps, self.warmup_steps + self.cosine_decay_steps],
        )
        schedulers.append(lrW)

        if use_clean_decay:
            adamw_param_groups = []
            if len(adam_decay) > 0:
                adamw_param_groups.append(
                    {
                        "params": adam_decay,
                        "lr": self.lr,
                        "weight_decay": self.weight_decay,
                    }
                )
            if len(adam_bias) > 0:
                adamw_param_groups.append(
                    {
                        "params": adam_bias,
                        "lr": self.lr,
                        "weight_decay": 0.0,
                    }
                )
            if len(adam_scales) > 0:
                adamw_param_groups.append(
                    {
                        "params": adam_scales,
                        "lr": self.lr_scales,
                        "weight_decay": 0.0,
                    }
                )
            opt_b = torch.optim.AdamW(adamw_param_groups)
        else:
            opt_b = torch.optim.AdamW(
                adam_decay,
                lr=self.lr,
                weight_decay=self.weight_decay,
            )

        optimizers.append(opt_b)

        scheduler1B = torch.optim.lr_scheduler.LinearLR(
            opt_b,
            start_factor=self.experiment_config["NN"]["training"]["warmup_factor"],
            end_factor=1.0,
            total_iters=self.warmup_steps,
        )

        if self.experiment_config["NN"]["training"]["scheduler"] == "exponential":
            factor = 10.0
            steps_per_factor = eval(str(self.experiment_config["NN"]["training"]["exp_steps"]))
            gamma = factor ** (-1.0 / steps_per_factor)
            scheduler2B = ExponentialLR(opt_b, gamma=gamma)
        else:
            scheduler2B = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt_b,
                T_max=self.cosine_decay_steps,
                eta_min=self.eta_min,
            )

        start_factorB = self.eta_min / self.lr
        end_factorB = self.experiment_config["NN"]["training"]["cooldown_factor"] * start_factorB
        scheduler3B = torch.optim.lr_scheduler.LinearLR(
            opt_b,
            start_factor=start_factorB,
            end_factor=end_factorB,
            total_iters=self.cooldown_steps,
        )

        lrB = torch.optim.lr_scheduler.SequentialLR(
            opt_b,
            schedulers=[scheduler1B, scheduler2B, scheduler3B],
            milestones=[self.warmup_steps, self.warmup_steps + self.cosine_decay_steps],
        )
        schedulers.append(lrB)

        return [opt_w, opt_b], [
            {"scheduler": lrW, "interval": "step", "name": "sch_w"},
            {"scheduler": lrB, "interval": "step", "name": "sch_b"},
        ]



class DefaultMLP(nn.Module):
    def __init__(self, input_size, output_size, num_hidden_layers, size_hidden_layers, actfun):
        super().__init__()
        self.input_size = input_size

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.input_size, size_hidden_layers))
        self.layers.append(actfun)
        for _ in range(num_hidden_layers):
            self.layers.append(nn.Linear(size_hidden_layers, size_hidden_layers))
            self.layers.append(actfun)
        self.layers.append(nn.Linear(size_hidden_layers, output_size))

        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.layers:
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.uniform_(m.bias)


    def forward(self, input_data):
        for layer in self.layers:
            input_data = layer(input_data)
        return input_data
    


class LitSingleKappaWithWarmup(L.LightningModule):
    def __init__(self, model, experiment_config={}):
        super().__init__()
        self.model = model
        self.experiment_config = experiment_config
        self.lr = float(self.experiment_config["NN"]["training"]["initial_optim_lr"])
        self.lr_scales = float(self.experiment_config["NN"]["training"]["initial_optim_lr_scales"])

        self.training_losses = []
        self.plotting_configured = ("plotting" in self.experiment_config["NN"])
        self.plotter = function_loader("fancy_plotters.py", self.experiment_config["NN"]["plotting"]["plotting_function_name"])
        self.max_kappa = self.experiment_config["NN"]["model"]["max_kappa"]
        from loss_functions import get_loss_singleKappaLightningFull, get_loss_singleKappaLightningOnlyOrd
        if self.experiment_config["FEM_solver_config"]["problem_dict"]["only_ord"]:
            self.loss_fun = get_loss_singleKappaLightningOnlyOrd(self.max_kappa)
        else:
            self.loss_fun = get_loss_singleKappaLightningFull(self.max_kappa)

        self.magpot = MagField(self.experiment_config["FEM_solver_config"]["problem_dict"]["H_type"], self.experiment_config["FEM_solver_config"]["problem_dict"]["mag_scale"])
            
        self.warmup_steps = eval(str(self.experiment_config["NN"]["training"]["warmup_steps"]))
        self.cosine_decay_steps = eval(self.experiment_config["NN"]["training"]["cosine_decay_steps"])
        self.eta_min = float(self.experiment_config["NN"]["training"]["eta_min"])
        self.weight_decay = float(self.experiment_config["NN"]["training"]["weight_decay"])
        self.cooldown_steps = eval(str(self.experiment_config["NN"]["training"]["cooldown_steps"]))


    def forward(self,x):
        return self.model(x)


    def configure_plotters(self, experiment_config):
        self.plotting_configured = True
        self.experiment_config = experiment_config


    def training_step(self, batch, batch_idx):
        x = batch
        if self.experiment_config["FEM_solver_config"]["problem_dict"]["only_ord"]:
            loss = self.loss_fun(self.model, x)
        else:
            loss = self.loss_fun(self.model, x, eval_magfield=self.magpot.get_eval_magfield())

        self.training_losses.append(loss.detach())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        optimizer = self.trainer.optimizers[0]
        use_clean_decay = self.experiment_config["NN"]["training"]["clean_weight_decay"]

        lr_main = optimizer.param_groups[0]["lr"]
        self.log(
            "lr_AdamW_main",
            lr_main,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        if use_clean_decay:

            if len(optimizer.param_groups) >= 2:
                lr_bias = optimizer.param_groups[1]["lr"]
                self.log(
                    "lr_AdamW_bias",
                    lr_bias,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True,
                )

            if len(optimizer.param_groups) >= 3:
                lr_scales = optimizer.param_groups[2]["lr"]
                self.log(
                    "lr_AdamW_scales",
                    lr_scales,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True,
                )

        return loss
        
    
    def configure_optimizers(self):
        use_clean_decay = self.experiment_config["NN"]["training"]["clean_weight_decay"]

        if use_clean_decay:
            adam_decay, adam_bias, adam_scales = split_params_for_adamw_only(self)

            param_groups = []
            if len(adam_decay) > 0:
                param_groups.append(
                    {
                        "params": adam_decay,
                        "lr": self.lr,
                        "weight_decay": self.weight_decay,
                    }
                )
            if len(adam_bias) > 0:
                param_groups.append(
                    {
                        "params": adam_bias,
                        "lr": self.lr,
                        "weight_decay": 0.0,
                    }
                )
            if len(adam_scales) > 0:
                param_groups.append(
                    {
                        "params": adam_scales,
                        "lr": self.lr_scales,
                        "weight_decay": 0.0,
                    }
                )

            optimizer = torch.optim.AdamW(
                param_groups,
            )
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )

        scheduler1 = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=self.experiment_config["NN"]["training"]["warmup_factor"],
            end_factor=1.0,
            total_iters=self.warmup_steps,
        )

        if self.experiment_config["NN"]["training"]["scheduler"] == "exponential":
            factor = 10.0
            steps_per_factor = eval(str(self.experiment_config["NN"]["training"]["exp_steps"]))
            gamma = factor ** (-1.0 / steps_per_factor)
            scheduler2 = ExponentialLR(optimizer, gamma=gamma)
        else:
            scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.cosine_decay_steps,
                eta_min=self.eta_min,
            )

        start_factor = self.eta_min / self.lr
        end_factor = self.experiment_config["NN"]["training"]["cooldown_factor"] * start_factor
        scheduler3 = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=start_factor,
            end_factor=end_factor,
            total_iters=self.cooldown_steps,
        )

        lr = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[scheduler1, scheduler2, scheduler3],
            milestones=[self.warmup_steps, self.warmup_steps + self.cosine_decay_steps],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr,
                "interval": "step",
                "frequency": 1,
            },
        }



class LitSingleKappaWithWarmupMuon(L.LightningModule):
    def __init__(self, model, experiment_config={}):
        super().__init__()
        self.model = model
        self.experiment_config = experiment_config
        self.lr = float(self.experiment_config["NN"]["training"]["initial_optim_lr"])
        self.lr_scales = float(self.experiment_config["NN"]["training"]["initial_optim_lr_scales"])

        self.training_losses = []
        self.plotting_configured = ("plotting" in self.experiment_config["NN"])
        self.plotter = function_loader("fancy_plotters.py", self.experiment_config["NN"]["plotting"]["plotting_function_name"])
        self.max_kappa = self.experiment_config["NN"]["model"]["max_kappa"]
        self.warmup_steps = eval(str(self.experiment_config["NN"]["training"]["warmup_steps"]))
        self.cosine_decay_steps = eval(self.experiment_config["NN"]["training"]["cosine_decay_steps"])
        self.eta_min = float(self.experiment_config["NN"]["training"]["eta_min"])
        self.weight_decay = float(self.experiment_config["NN"]["training"]["weight_decay"])
        self.automatic_optimization = False
        self.cooldown_steps = eval(str(self.experiment_config["NN"]["training"]["cooldown_steps"]))

        self.magpot = MagField(self.experiment_config["FEM_solver_config"]["problem_dict"]["H_type"], self.experiment_config["FEM_solver_config"]["problem_dict"]["mag_scale"])
        from loss_functions import get_loss_singleKappaLightningFull, get_loss_singleKappaLightningOnlyOrd
        if self.experiment_config["FEM_solver_config"]["problem_dict"]["only_ord"]:
            self.loss_fun = get_loss_singleKappaLightningOnlyOrd(self.max_kappa)
        else:
            self.loss_fun = get_loss_singleKappaLightningFull(self.max_kappa)

    def forward(self,x):
        return self.model(x)

    def configure_plotters(self, experiment_config):
        self.plotting_configured = True
        self.experiment_config = experiment_config

    def training_step(self, batch, batch_idx):
        x = batch
        if self.experiment_config["FEM_solver_config"]["problem_dict"]["only_ord"]:
            loss = self.loss_fun(self.model, x)
        else:
            loss = self.loss_fun(self.model, x, eval_magfield=self.magpot.get_eval_magfield())
        opt_w, opt_b = self.optimizers(use_pl_optimizer=True)
        sched_weights, sched_biases = self.lr_schedulers()
        opt_w.zero_grad(set_to_none=True)
        opt_b.zero_grad(set_to_none=True)

        self.manual_backward(loss)

        opt_w.step()
        sched_weights.step()
        opt_b.step()
        sched_biases.step()

        self.training_losses.append(loss.detach())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        lr_muon_main = opt_w.param_groups[0]["lr"]
        lr_adam_main = opt_b.param_groups[0]["lr"]

        self.log("lr_Muon_main", lr_muon_main,
                on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("lr_AdamW_main", lr_adam_main,
                on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        use_clean_decay = self.experiment_config["NN"]["training"]["clean_weight_decay"]

        if use_clean_decay:

            if len(opt_b.param_groups) >= 2:
                pass

            if len(opt_b.param_groups) >= 2:
                lr_scales = opt_b.param_groups[-1]["lr"]
                self.log("lr_AdamW_scales", lr_scales,
                        on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss
        
    
    def configure_optimizers(self):
        use_clean_decay = self.experiment_config["NN"]["training"]["clean_weight_decay"]


        optimizers = []
        schedulers = []

        if use_clean_decay:
            muon_params, adam_decay, adam_bias, adam_scales = split_params_for_muon_and_adamw(self.model)
            self.print(
                f"[cfg] muon params: {sum(p.numel() for p in muon_params)} | "
                f"adam_decay: {sum(p.numel() for p in adam_decay)} | "
                f"adam_bias: {sum(p.numel() for p in adam_bias)} | "
                f"adam_scales: {sum(p.numel() for p in adam_scales)}"
            )
            assert len(muon_params) > 0, "No weight params found for Muon (are there any 2D weights?)"
            assert (len(adam_decay) + len(adam_bias) + len(adam_scales)) > 0, "No non-muon params found"
        else:
            muon_params, other_params = split_params_for_muon_and_other(self.model)
            self.print(
                f"[cfg] muon params: {sum(p.numel() for p in muon_params)} | "
                f"other params: {sum(p.numel() for p in other_params)}"
            )
            assert len(muon_params) > 0, "No weight params found"
            assert len(other_params) > 0, "No bias params found"

            adam_decay = other_params
            adam_bias = []
            adam_scales = []

        opt_w = torch.optim.Muon(
            muon_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            adjust_lr_fn="match_rms_adamw"
        )
        optimizers.append(opt_w)

        scheduler1W = torch.optim.lr_scheduler.LinearLR(
            opt_w,
            start_factor=self.experiment_config["NN"]["training"]["warmup_factor"],
            end_factor=1.0,
            total_iters=self.warmup_steps,
        )

        if self.experiment_config["NN"]["training"]["scheduler"] == "exponential":
            factor = 10.0
            steps_per_factor = eval(str(self.experiment_config["NN"]["training"]["exp_steps"]))
            gamma = factor ** (-1.0 / steps_per_factor)
            scheduler2W = ExponentialLR(opt_w, gamma=gamma)
        else:
            scheduler2W = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt_w,
                T_max=self.cosine_decay_steps,
                eta_min=self.eta_min,
            )

        start_factorW = self.eta_min / self.lr
        end_factorW = self.experiment_config["NN"]["training"]["cooldown_factor"] * start_factorW
        scheduler3W = torch.optim.lr_scheduler.LinearLR(
            opt_w,
            start_factor=start_factorW,
            end_factor=end_factorW,
            total_iters=self.cooldown_steps,
        )

        lrW = torch.optim.lr_scheduler.SequentialLR(
            opt_w,
            schedulers=[scheduler1W, scheduler2W, scheduler3W],
            milestones=[self.warmup_steps, self.warmup_steps + self.cosine_decay_steps],
        )
        schedulers.append(lrW)

        if use_clean_decay:
            adamw_param_groups = []
            if len(adam_decay) > 0:
                adamw_param_groups.append(
                    {
                        "params": adam_decay,
                        "lr": self.lr,
                        "weight_decay": self.weight_decay,
                    }
                )
            if len(adam_bias) > 0:
                adamw_param_groups.append(
                    {
                        "params": adam_bias,
                        "lr": self.lr,
                        "weight_decay": 0.0,
                    }
                )
            if len(adam_scales) > 0:
                adamw_param_groups.append(
                    {
                        "params": adam_scales,
                        "lr": self.lr_scales,
                        "weight_decay": 0.0,
                    }
                )
            opt_b = torch.optim.AdamW(adamw_param_groups)
        else:
            opt_b = torch.optim.AdamW(
                adam_decay,
                lr=self.lr,
                weight_decay=self.weight_decay,
            )

        optimizers.append(opt_b)

        scheduler1B = torch.optim.lr_scheduler.LinearLR(
            opt_b,
            start_factor=self.experiment_config["NN"]["training"]["warmup_factor"],
            end_factor=1.0,
            total_iters=self.warmup_steps,
        )

        if self.experiment_config["NN"]["training"]["scheduler"] == "exponential":
            factor = 10.0
            steps_per_factor = eval(str(self.experiment_config["NN"]["training"]["exp_steps"]))
            gamma = factor ** (-1.0 / steps_per_factor)
            scheduler2B = ExponentialLR(opt_b, gamma=gamma)
        else:
            scheduler2B = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt_b,
                T_max=self.cosine_decay_steps,
                eta_min=self.eta_min,
            )

        start_factorB = self.eta_min / self.lr
        end_factorB = self.experiment_config["NN"]["training"]["cooldown_factor"] * start_factorB
        scheduler3B = torch.optim.lr_scheduler.LinearLR(
            opt_b,
            start_factor=start_factorB,
            end_factor=end_factorB,
            total_iters=self.cooldown_steps,
        )

        lrB = torch.optim.lr_scheduler.SequentialLR(
            opt_b,
            schedulers=[scheduler1B, scheduler2B, scheduler3B],
            milestones=[self.warmup_steps, self.warmup_steps + self.cosine_decay_steps],
        )
        schedulers.append(lrB)

        return [opt_w, opt_b], [
            {"scheduler": lrW, "interval": "step", "name": "sch_w"},
            {"scheduler": lrB, "interval": "step", "name": "sch_b"},
        ]
    