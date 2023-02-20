import numpy as np
from mpi4py import MPI
from dolfinx import fem
import ufl

def error_L2(uh, u_ex, degree_raise=3):
    """ Adapted from: 
    https://jsdokken.com/dolfinx-tutorial/chapter4/convergence.html """

    degree = uh.function_space.ufl_element().degree()
    family = uh.function_space.ufl_element().family()
    mesh = uh.function_space.mesh
    W = fem.FunctionSpace(mesh, (family, degree+degree_raise))

    uh_W = fem.Function(W)
    uh_W.interpolate(uh)

    u_ex_W = fem.Function(W)
    u_ex_W.interpolate(u_ex)

    e_W = fem.Function(W)
    e_W.vector.array[:] = uh_W.vector.array - u_ex_W.vector.array

    # Integrate the error
    error = fem.form(ufl.inner(e_W, e_W) * ufl.dx)
    error_local = fem.assemble_scalar(error)
    error_global = mesh.comm.allreduce(error_local, op=MPI.SUM)
    return np.sqrt(error_global)

def error_H1(uh, u_ex, degree_raise=3):
    """ Adapted from: 
    https://jsdokken.com/dolfinx-tutorial/chapter4/convergence.html """

    degree = uh.function_space.ufl_element().degree()
    family = uh.function_space.ufl_element().family()
    mesh = uh.function_space.mesh
    W = fem.FunctionSpace(mesh, (family, degree+degree_raise))

    uh_W = fem.Function(W)
    uh_W.interpolate(uh)

    u_ex_W = fem.Function(W)
    u_ex_W.interpolate(u_ex)

    e_W = fem.Function(W)
    e_W.vector.array[:] = uh_W.vector.array - u_ex_W.vector.array

    # Integrate the error
    error = fem.form(ufl.inner(e_W, e_W) * ufl.dx + \
                     ufl.inner(ufl.grad(e_W), ufl.grad(e_W)) * ufl.dx)
    error_local = fem.assemble_scalar(error)
    error_global = mesh.comm.allreduce(error_local, op=MPI.SUM)
    return np.sqrt(error_global)
