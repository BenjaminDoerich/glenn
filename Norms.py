import numpy as np
from mpi4py import MPI
import dolfinx as dfx
import ufl
from petsc4py import PETSc
from typing import Callable
import basix


def norm_L2(V: dfx.fem.function.Function, u: dfx.fem.function.Function):
    # Integrate the error
    error = dfx.fem.form(ufl.inner(u, u) * ufl.dx)
    error_local = dfx.fem.assemble_scalar(error)
    error_global = V.mesh.comm.allreduce(error_local, op=MPI.SUM)
    return np.sqrt(error_global)


def norm_H10(V: dfx.fem.function.Function, u: dfx.fem.function.Function):
    # Integrate the error
    error = dfx.fem.form(ufl.dot(ufl.grad(u), ufl.grad(u)) * ufl.dx(domain=V.mesh))
    error_local = dfx.fem.assemble_scalar(error)
    error_global = V.mesh.comm.allreduce(error_local, op=MPI.SUM)
    return np.sqrt(error_global)


def norm_H1(V: dfx.fem.function.Function, u: dfx.fem.function.Function):
    # Integrate the error
    return np.sqrt(norm_H10(V, u) ** 2 + norm_L2(V, u) ** 2)


def norm_H1kappa(V: dfx.fem.function.Function, u: dfx.fem.function.Function, kappa: int ):
    # Integrate the error
    return np.sqrt( 1.0/kappa**2 * norm_H10(V, u) ** 2 + norm_L2(V, u) ** 2)


def norm_Hminus1(
    V: dfx.fem.function.FunctionSpace,
    func_vector: dfx.fem.function.Function,
    boundary_dofs=None,
):
    invlapl_p1 = dfx.fem.Function(V)
    psi = ufl.TrialFunction(V)
    phi = ufl.TestFunction(V)
    lapl = ufl.inner(ufl.grad(psi), ufl.grad(phi)) * ufl.dx
    rhs = func_vector * phi * ufl.dx
    zero = dfx.fem.Constant(V.mesh, 0.0)

    if boundary_dofs is None:
        tdim = V.mesh.topology.dim
        fdim = tdim - 1
        V.mesh.topology.create_connectivity(fdim, tdim)
        boundary_facets = dfx.mesh.exterior_facet_indices(V.mesh.topology)
        boundary_dofs = dfx.fem.locate_dofs_topological(V, fdim, boundary_facets)

    bc = dfx.fem.dirichletbc(zero, boundary_dofs, V)
    problem = dfx.fem.petsc.LinearProblem(
        lapl, rhs, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )
    invlapl_p1 = problem.solve()

    return norm_H10(V, invlapl_p1)


def error_norm_ref(
    uh: dfx.fem.function.Function,
    u_ref: dfx.fem.function.Function,
    norm_str: str,
    degree_raise=2,
    kappa =1, 
    logger = None,
    dG = False
):
    # Create higher order function space
    # if dfx.__version__ < "0.8.0":
    #     degree_ref = u_ref.function_space.ufl_element().degree()
    #     family_ref = u_ref.function_space.ufl_element().family()
    # else:
    degree_ref = u_ref.function_space.ufl_element().degree
    family_ref = u_ref.function_space.ufl_element().element_family
    mesh_ref = u_ref.function_space.mesh

    # if logger:
    #     logger.info(f'family_ref = {family_ref}')

    if not dG:
        V = dfx.fem.functionspace(mesh_ref, (family_ref, degree_ref))
        W = dfx.fem.functionspace(mesh_ref, (family_ref, degree_ref + degree_raise))

    if dG:
        element_dG_ref = basix.ufl.element(basix.ElementFamily.P, basix.CellType.triangle, degree_ref, lagrange_variant=basix.LagrangeVariant.equispaced, discontinuous=True,shape = (1,))
        element_dG_ref_raise = basix.ufl.element(basix.ElementFamily.P, basix.CellType.triangle, degree_ref + degree_raise , lagrange_variant=basix.LagrangeVariant.equispaced, discontinuous=True,shape = (1,))
    
        V = dfx.fem.functionspace(mesh_ref, element_dG_ref)
        W = dfx.fem.functionspace(mesh_ref, element_dG_ref_raise)
        # V_dG = fem.functionspace(msh,element_dG)

    # Interpolate approximate solution
    u_W = dfx.fem.Function(W)
    degree = uh.function_space.ufl_element().degree
    mesh = uh.function_space.mesh

    if not dG:
        family = uh.function_space.ufl_element().element_family
        V_h = dfx.fem.functionspace(mesh, (family, degree))

    if dG:
        element_dG = basix.ufl.element(basix.ElementFamily.P, basix.CellType.triangle, degree, lagrange_variant=basix.LagrangeVariant.equispaced, discontinuous=True,shape = (1,))
        V_h = dfx.fem.functionspace(mesh, element_dG)

    
    num_cells_local = W.dofmap.list.shape[0]
    cells = np.arange(num_cells_local, dtype=np.int32)
    interpolation_data = dfx.fem.create_interpolation_data(
        W,
        V_h,
        cells,
    )
    u_W.interpolate_nonmatching(
        uh,
        cells,
        interpolation_data,
    )
    # Interpolate reference solution
    u_W_ref = dfx.fem.Function(W)
    u_W_ref.interpolate(u_ref)
    # Compute the error in the higher order function space
    e_W = dfx.fem.Function(W)
    e_W.x.array[:] = u_W.x.array - u_W_ref.x.array

    if norm_str == "L2":
        return norm_L2(V, e_W)
    elif norm_str == "H10":
        return norm_H10(V, e_W)
    elif norm_str == "H1":
        return norm_H1(V, e_W)
    elif norm_str == "Hminus1":
        return norm_Hminus1(V, e_W)
    elif norm_str == "H1kappa":
        return norm_H1kappa(V, e_W,kappa=kappa)



def error_norm(
    uh: dfx.fem.function.Function,
    u_ex: Callable[[np.ndarray | float], float] | ufl.core.expr.Expr,
    norm_str: str,
    degree_raise=2,
):
    # Create higher order function space
    if dfx.__version__ < "0.8.0":
        degree = uh.function_space.ufl_element().degree()
        family = uh.function_space.ufl_element().family()
    else:
        degree = uh.function_space.ufl_element().degree
        family = uh.function_space.ufl_element().element_family

    mesh = uh.function_space.mesh
    V = dfx.fem.functionspace(mesh, (family, degree))
    W = dfx.fem.functionspace(mesh, (family, degree + degree_raise))
    # Interpolate approximate solution
    u_W = dfx.fem.Function(W)
    u_W.interpolate(uh)

    # Interpolate exact solution, special handling if exact solution
    # is a ufl expression or a python lambda function
    u_ex_W = dfx.fem.Function(W)
    if isinstance(u_ex, ufl.core.expr.Expr):
        u_expr = dfx.fem.Expression(u_ex, W.element.interpolation_points())
        u_ex_W.interpolate(u_expr)
    else:
        u_ex_W.interpolate(u_ex)

    # Compute the error in the higher order function space
    e_W = dfx.fem.Function(W)
    e_W.x.array[:] = u_W.x.array - u_ex_W.x.array

    if norm_str == "L2":
        return norm_L2(V, e_W)
    elif norm_str == "H10":
        return norm_H10(V, e_W)
    elif norm_str == "H1":
        return norm_H1(V, e_W)
    elif norm_str == "Hminus1":
        return norm_Hminus1(V, e_W)


def error_infinity(
    u_h: dfx.fem.function.Function,
    u_ex: Callable[[np.ndarray | float], float] | ufl.core.expr.Expr,
):
    # Interpolate exact solution, special handling if exact solution
    # is a ufl expression or a python lambda function
    comm = u_h.function_space.mesh.comm
    u_ex_V = dfx.fem.Function(u_h.function_space)
    if isinstance(u_ex, ufl.core.expr.Expr):
        u_expr = dfx.fem.Expression(
            u_ex, u_h.function_space.element.interpolation_points()
        )
        u_ex_V.interpolate(u_expr)
    else:
        u_ex_V.interpolate(u_ex)
    # Compute infinity norm, furst local to process, then gather the max
    # value over all processes
    error_max_local = np.max(np.abs(u_h.x.array - u_ex_V.x.array))
    error_max = comm.allreduce(error_max_local, op=MPI.MAX)
    return error_max


def EOC(t: np.ndarray, e: np.ndarray) -> float:
    # p = np.log10(e[1:-1] / e[2:]) / np.log10(t[1:-1] / t[2:])
    p = np.log10(e[1:] / e[:-1]) / np.log10(t[1:] / t[:-1])
    return np.median(p)


def combine_norms(norm_1: np.ndarray, norm_2: np.ndarray) -> np.ndarray:
    return np.sqrt(norm_1 * norm_1 + norm_2 * norm_2)
