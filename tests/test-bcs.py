import numpy as np
import matplotlib.pyplot as plt

from mpi4py import MPI
from dolfinx import mesh

from dolfinx import fem
from petsc4py.PETSc import ScalarType
import ufl


def create_mesh(N):

    domain = mesh.create_unit_square(MPI.COMM_SELF, N, N)

    return domain


def create_problem(domain, V, u_ex):

    # Create boundary conditions:
    # Dirichlet u = u_D for x=0 and for x=1.
    # Neumann du/dn = g for y=0 and y=1 is imposed in the rhs functional.

    u_bc = fem.Function(V)
    u_bc.interpolate(u_ex)

    def boundary_left_D(x):
        return np.isclose(x[0], 0.0)
    def boundary_right_D(x):
        return np.isclose(x[0], 1.0)
    
    dofs_left_D = fem.locate_dofs_geometrical(V, boundary_left_D)
    dofs_right_D = fem.locate_dofs_geometrical(V, boundary_right_D)

    bc_left_D = fem.dirichletbc(u_bc, dofs_left_D)
    bc_right_D = fem.dirichletbc(u_bc, dofs_right_D)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    n = ufl.FacetNormal(domain)

    f = -ufl.div(ufl.grad(u_ex))
    g = ufl.inner(n, ufl.grad(u_ex))

    a = ufl.inner( ufl.grad(u), ufl.grad(v) ) * ufl.dx
    L = f * v * ufl.dx + g * v * ufl.ds

    problem = fem.petsc.LinearProblem(a, L, bcs=[bc_left_D, bc_right_D], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

    return problem



N = 33
domain = create_mesh(N)

V = fem.FunctionSpace(domain, ("CG", 1))
V2 = fem.FunctionSpace(domain, ("CG", 2))

u_ex_func = lambda x: 1 + x[0]**2 + x[1]**2
u_ex = fem.Function(V2)
u_ex.interpolate(u_ex_func)


problem = create_problem(domain, V, u_ex)

uh = problem.solve()

uu = uh.vector.array
xx = uh.function_space.mesh.geometry.x

if xx.shape[0] > 1000:
    step = xx.shape[0] // 1000
else:
    step = 1

ax = plt.figure().add_subplot(projection='3d')
ax.plot_trisurf(xx[::step,0], xx[::step,1], uu[::step], cmap=plt.cm.viridis)
ax.set_title("$u_h$")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")

ax = plt.figure().add_subplot(projection='3d')
ax.plot_trisurf(xx[::step,0], xx[::step,1], uu[::step] - u_ex_func([xx[::step,0], xx[::step,1]]), cmap=plt.cm.viridis)
ax.set_title(r"$u_h - u_\mathrm{ex}$")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")



plt.show()
