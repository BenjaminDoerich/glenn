import dolfinx as dfx
import ufl
from petsc4py import PETSc
from mpi4py import MPI
from typing import Union
import gmsh
# from dolfinx.io import gmshio
import pyvista
from dolfinx.plot import vtk_mesh
from dolfinx import mesh
from dolfinx import fem #, io
from typing import Optional, Tuple
pyvista.OFF_SCREEN = True

import os
if not os.path.exists('plots'):
   os.makedirs('plots')


    
def plot_sol_pyvista(
    V: dfx.fem.function.functionspace,
    name: str = "u",
    sol: dfx.fem.function.Function = None,
    cmap: Optional[str] = None,
    clim: Optional[Tuple[float, float]] = None,
    show_cbar: bool = True,
    warp_by_scalar: bool = False,
    warping_factor: float = 0.5,
    show_edges: bool = True,
    # jupyter_backend: Optional[str] = None,  # "static"
):
    """
    Simple plotting with pyvista and dolfinx
    Args:
        V (dfx.fem.function.functionspace): dfx.fem.functionspace
        sol (dfx.fem.function.Function): solution function
        cmap (Optional[str], optional): colormap. Defaults to None.
        clim (Optional[Tuple[float,float]], optional): (vmin,vmax). Defaults to None.
        name (str, optional): name of solution. Defaults to "u".
        show_cbar (bool, optional):  Defaults to True.
        warp_by_scalar (bool, optional):  Defaults to False.
        warping_factor (float, optional):  Defaults to 0.5.
        show_edges (bool, optional):  Defaults to False.
        jupyter_backend (Optional[str], optional): Valid: 'trame','static','client','server','none'; Default: 'static'.
    """
    cells, types, x = dfx.plot.vtk_mesh(V)
    if dfx.__version__ == '0.9.0':
        pyvista.start_xvfb()
    grid = pyvista.UnstructuredGrid(cells, types, x)
    if sol == None:
        grid.point_data[name] = 1.0
        clim = (0.0, 1.0)
    else:
        grid.point_data[name] = sol.x.array.real
        # clim = (0.0, 0.5)
    grid.set_active_scalars(name)

    plotter = pyvista.Plotter()
    plotter.add_mesh(
        grid,
        scalars=name,
        show_edges=show_edges,
        cmap=cmap,
        clim=clim,
        show_scalar_bar=show_cbar and not warp_by_scalar,
    )
    if warp_by_scalar == True:
        warped = grid.warp_by_scalar(factor=warping_factor)
        plotter.add_mesh(
            warped, cmap='viridis', clim=clim, show_edges=show_edges, show_scalar_bar=False
        )
        if show_cbar:
            plotter.add_scalar_bar(name)
    else:
        plotter.view_xy()

    if pyvista.OFF_SCREEN:
        if dfx.__version__ == '0.9.0':
            pyvista.start_xvfb()
            pyvista.start_xvfb(wait=0.1)
        plotter.screenshot(f"plots/{name}.png", window_size=[900, 900])
    else:
        # if jupyter_backend is not None:
        #     plotter.show(jupyter_backend=jupyter_backend)
        # else:
        plotter.show()


