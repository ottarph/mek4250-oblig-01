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


def create_problem(domain, V, mu_val, SUPG=False):

    # Create boundary conditions:
    # Dirichlet u=0 for x=0, Dirichlet u=1 for x=1.
    # Neumann du/dn = 0 for y=0 and y=1 is imposed in the rhs functional.

    def boundary_left_D(x):
        return np.isclose(x[0], 0.0)
    def boundary_right_D(x):
        return np.isclose(x[0], 1.0)
    
    dofs_left_D = fem.locate_dofs_geometrical(V, boundary_left_D)
    dofs_right_D = fem.locate_dofs_geometrical(V, boundary_right_D)

    bc_left_D = fem.dirichletbc(fem.Constant(domain, \
                                             ScalarType(0.0)), dofs_left_D, V)
    bc_right_D = fem.dirichletbc(fem.Constant(domain, \
                                             ScalarType(1.0)), dofs_right_D, V)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    if isinstance(mu_val, fem.Constant):
        mu_val = mu_val.value
    
    mu = fem.Constant(domain, mu_val)
    w = fem.Constant(domain, ScalarType((1.0, 0.0)))

    if SUPG:
        h = 1 / ( np.sqrt(domain.geometry.x.shape[0]) - 1.0 )
        beta = 0.5
        v = v + beta*h * ufl.inner(w, ufl.grad(v))

    f = fem.Constant(domain, ScalarType(0.0))
    g = fem.Constant(domain, ScalarType(0.0))

    a = mu * ufl.inner( ufl.grad(u), ufl.grad(v) ) * ufl.dx \
        + ufl.inner(w, ufl.grad(u)) * v * ufl.dx
    L = f * v * ufl.dx + g * v * ufl.ds

    problem = fem.petsc.LinearProblem(a, L, \
                            bcs=[bc_left_D, bc_right_D], \
                            petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

    return problem


def main():

    from helpers.plots import trisurf
    from helpers.errors import error_L2, error_H1
    from helpers.solutions import ex01_sol as u_ex_func


    mu_val = 0.01
    N = 100
    order = 1
    SUPG = False
    domain = create_mesh(N)


    V = fem.FunctionSpace(domain, ("CG", order))
    V2 = fem.FunctionSpace(domain, ("CG", order+1))

    problem = create_problem(domain, V, mu_val, SUPG=SUPG)

    uh = problem.solve()

    plotmesh = create_mesh(40)


    trisurf(uh, plotmesh, title="$u_h$")

    trisurf(u_ex_func(mu_val), plotmesh, title=r"$u_\mathrm{ex}$")


    u_ex = fem.Function(V)
    u_ex.interpolate(u_ex_func(mu_val))

    e = fem.Function(V)
    e.vector.array[:] = uh.vector.array[:] - u_ex.vector.array[:]


    trisurf(e, plotmesh, title=r"$u_h - u_\mathrm{ex},$ pointwise")


    plt.show()

if __name__ == "__main__":
    main()
