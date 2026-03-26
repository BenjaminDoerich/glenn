import torch
import numpy as np
import matplotlib.pyplot as plt
import copy
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch._dynamo as dynamo

@dynamo.disable
def _model_device(model):
    try:
        dev = next(model.parameters()).device
    except StopIteration:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dev.type == "cpu" and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return dev

@dynamo.disable
def abs_value_plotter_multiKappa(
    model,
    kappa,
    ax=None,
    max_kappa=110,
    n=200,
    only_rank0=True,
):
    if only_rank0 and torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() != 0:
            return None

    actual_kappa = float(kappa)
    kappa_norm = actual_kappa / float(max_kappa)

    device = _model_device(model)
    was_training = model.training
    model.eval()

    x = torch.linspace(0.0, 1.0, n, device=device)
    y = torch.linspace(0.0, 1.0, n, device=device)
    Xg, Yg = torch.meshgrid(x, y, indexing="ij")
    Kg = torch.full_like(Xg, fill_value=kappa_norm)

    XYK = torch.stack((Xg, Yg, Kg), dim=-1).reshape(-1, 3)

    with torch.no_grad():
        u = model(XYK)[:, :2]

    u_abs = (u[:, 0]**2 + u[:, 1]**2).sqrt().reshape(n, n)

    X_plot = Xg.cpu().numpy()
    Y_plot = Yg.cpu().numpy()
    U_plot = u_abs.cpu().numpy()

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    cf = ax.contourf(X_plot, Y_plot, U_plot, levels=100, cmap="jet")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = ax.figure.colorbar(cf, cax=cax)
    cbar.set_label("|u(x,y)|")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"|u(x,y)|, kappa={actual_kappa:g}")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    if was_training:
        model.train()

    return ax



from mpl_toolkits.axes_grid1 import make_axes_locatable
@dynamo.disable
def allplot(
    model,
    kappa,
    ax=None,
    max_kappa=110,
    n=200,
    only_rank0=True,
    add_colorbar=True,
    add_title=True,
):
    if only_rank0 and torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() != 0:
            return None

    actual_kappa = float(kappa)
    kappa_norm = actual_kappa / float(max_kappa)

    device = model.device

    was_training = model.training
    model.eval()

    x = torch.linspace(0.0, 1.0, n, device=device)
    y = torch.linspace(0.0, 1.0, n, device=device)
    Xg, Yg = torch.meshgrid(x, y, indexing="ij")
    Kg = torch.full_like(Xg, fill_value=kappa_norm)
    XYK = torch.stack((Xg, Yg, Kg), dim=-1).reshape(-1, 3)

    with torch.no_grad():
        u = model(XYK)[:, :2]

    u_abs = (u[:, 0]**2 + u[:, 1]**2).sqrt().reshape(n, n)

    X_plot = Xg.cpu().numpy()
    Y_plot = Yg.cpu().numpy()
    U_plot = u_abs.cpu().numpy()

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.get_figure()

    cf = ax.contourf(X_plot, Y_plot, U_plot, levels=100, cmap="jet")

    if add_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(cf, cax=cax)
        cbar.set_label(r"$|u(x,y)|$")

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")

    if add_title:
        ax.set_title(rf"$|u(x,y)|$, $\kappa = {actual_kappa:g}$")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    if was_training:
        model.train()

    return fig, ax


def plot_function_on_samples(samples, fvals, cmap="viridis"):

    xy = samples.detach().cpu()
    z = fvals.detach().cpu()

    x = xy[:, 0].numpy()
    y = xy[:, 1].numpy()
    c = z.numpy()

    plt.figure(figsize=(5, 4))
    sc = plt.scatter(x, y, c=c, s=10, cmap=cmap)
    plt.gca().set_aspect("equal", adjustable="box")

    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.title("Function values on L-shaped domain")

    cb = plt.colorbar(sc)
    cb.set_label(r"$f(x, y)$")

    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig("testplotlshape.png")


def _is_main_process():
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True

@dynamo.disable
def abs_value_plotter_multiKappaLshape(
    model,
    kappa,
    ax=None,
    max_kappa=110,
    n=200,
    only_rank0=True,
):
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() != 0:
            return None

    if only_rank0 and torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() != 0:
            return None

    actual_kappa = float(kappa)
    kappa_norm = actual_kappa / float(max_kappa)

    device = _model_device(model)
    was_training = model.training
    model.eval()

    x = torch.linspace(0.0, 1.0, n, device=device)
    y = torch.linspace(0.0, 1.0, n, device=device)
    Xg, Yg = torch.meshgrid(x, y, indexing="ij")
    xyv = torch.stack([Xg.reshape(-1), Yg.reshape(-1)], dim=-1)

    scale = torch.tensor((0.5, 0.5), device=device)
    shift2 = torch.tensor((0.5, 0.0), device=device)
    shift3 = torch.tensor((0.0, 0.5), device=device)

    data_1 = scale * xyv
    data_2 = scale * xyv + shift2
    data_3 = scale * xyv + shift3
    data = torch.cat([data_1, data_2, data_3], dim=0)

    Kg = torch.full_like(data[:, [0]], fill_value=kappa_norm)
    XYK = torch.cat((data, Kg), dim=-1)
    with torch.no_grad():
        u = model(XYK)[:, :2]

    u_abs = (u[:, 0]**2 + u[:, 1]**2).sqrt()

    del Xg, Yg, xyv, data_1, data_2, data_3, Kg, XYK, u
    torch.cuda.empty_cache()

    samples_cpu = data[:, 0:2].detach().cpu()
    u_abs_cpu = u_abs.detach().cpu()
    del data, u_abs

    x_plot = samples_cpu[:, 0].numpy()
    y_plot = samples_cpu[:, 1].numpy()
    c_plot = u_abs_cpu.numpy()

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    sc = ax.scatter(x_plot, y_plot, c=c_plot, s=5, cmap="jet")
    cbar = ax.figure.colorbar(sc, cax=cax)
    cbar.set_label(r"$|u(x,y)|$")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(fr"$|u(x,y)|$, $\kappa={actual_kappa:g}$")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    if was_training:
        model.train()

    return ax



def abs_value_plotter_multiKappa_old(model, kappa, ax=None,max_kappa=110):

    actual_kappa = kappa
    kappa = kappa/max_kappa
    model_cpu = copy.deepcopy(model).to("cpu").eval()

    x_vals_plot = torch.linspace(0, 1, 200)
    y_vals_plot = torch.linspace(0, 1, 200)
    kappa_mesh_plot = torch.linspace(kappa, kappa, 1)

    KAPPA_plot, X_plot, Y_plot = np.meshgrid(kappa_mesh_plot, x_vals_plot, y_vals_plot)

    X_tensor_plot = torch.tensor(X_plot.flatten(), device="cpu").unsqueeze(1)
    Y_tensor_plot = torch.tensor(Y_plot.flatten(), device="cpu").unsqueeze(1)
    KAPPA_tensor_plot = torch.tensor(KAPPA_plot.flatten(), device="cpu").unsqueeze(1)

    XY_test_plot = torch.cat([X_tensor_plot, Y_tensor_plot, KAPPA_tensor_plot], dim=1)

    with torch.no_grad():
        u_pred_plot = model_cpu(XY_test_plot.to('cpu'))[:,0:2].cpu().detach()

    u_abs_plot = torch.sqrt(u_pred_plot[:, 0] ** 2 + u_pred_plot[:, 1] ** 2)
    u_abs_mesh = u_abs_plot.reshape(200, 200).numpy()

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    cf = ax.contourf(
        X_plot.squeeze(),
        Y_plot.squeeze(),
        u_abs_mesh,
        levels=100,
        cmap="jet"
    )
    plt.colorbar(cf, ax=ax, label="|u(x,y)|")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Abs GLE, kappa={actual_kappa}")

    return ax
    