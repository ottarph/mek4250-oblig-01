import numpy as np
import matplotlib.pyplot as plt

from mpi4py import MPI
from dolfinx import mesh

from dolfinx import fem
import ufl

from helpers.errors import error_L2, error_H1

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

    problem = fem.petsc.LinearProblem(a, L, \
                            bcs=[bc_left_D, bc_right_D], \
                            petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

    return problem



N = 11
domain = create_mesh(N)

order = 2
V = fem.FunctionSpace(domain, ("CG", order))
V2 = fem.FunctionSpace(domain, ("CG", order+1))

def u_ex_func(x):
    return 1 + x[0]**2 + x[1]**2

u_ex = fem.Function(V2)
u_ex.interpolate(u_ex_func)


problem = create_problem(domain, V, u_ex)

uh = problem.solve()

V_lin = fem.FunctionSpace(domain, ("CG", 1))
uh_l = fem.Function(V_lin)
uh_l.interpolate(uh)

uu = uh_l.vector.array
xx = uh_l.function_space.mesh.geometry.x

if xx.shape[0] > 1000:
    step = xx.shape[0] // 1000
else:
    step = 1

ax = plt.figure().add_subplot(projection='3d')
ax.plot_trisurf(xx[::step,0], xx[::step,1], uu[::step], cmap=plt.cm.viridis)
ax.set_title("$u_h,$ pointwise")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")

ax = plt.figure().add_subplot(projection='3d')
ax.plot_trisurf(xx[::step,0], xx[::step,1], \
                uu[::step] - u_ex_func([xx[::step,0], xx[::step,1]]), \
                    cmap=plt.cm.viridis)
ax.set_title(r"$u_h - u_\mathrm{ex},$ pointwise")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")


fine_mesh = create_mesh(4*N)
V_f = fem.FunctionSpace(fine_mesh, ("CG", 1))
uh_f = fem.Function(V_f)
uh_f.interpolate(uh)
uh_f.interpolate(uh)

xxx = uh_f.function_space.mesh.geometry.x
uuu = uh_f.vector.array

ax = plt.figure().add_subplot(projection='3d')
ax.plot_trisurf(xxx[::step,0], xxx[::step,1], \
                uuu[::step] - u_ex_func([xxx[::step,0], xxx[::step,1]]), \
                      cmap=plt.cm.viridis)
ax.set_title(r"$u_h - u_\mathrm{ex}$, finer mesh")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")


if domain.comm.rank == 0:
    print(f"{order=}, {N=}")
    print(f"L2-error: {error_L2(uh, u_ex):.2e}")
    print(f"H1-error: {error_H1(uh, u_ex):.2e}")

plt.show()
