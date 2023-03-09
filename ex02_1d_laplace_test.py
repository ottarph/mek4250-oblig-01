import numpy as np

from mpi4py import MPI

from dolfinx import fem
from petsc4py import PETSc
import ufl


def create_problem(mesh, ft):

    V = fem.FunctionSpace(mesh, ("CG", 1))

    tdim = mesh.topology.dim
    fdim = tdim - 1

    bcs = []
    for marker, val in zip([2, 3, 4, 5], [0, 0, 0, 2]):
        dofs = fem.locate_dofs_topological(V, fdim, ft.find(marker))
        bc = fem.dirichletbc(fem.Constant(mesh, PETSc.ScalarType(val)), dofs, V)
        bcs.append(bc)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    f = fem.Constant(mesh, PETSc.ScalarType(1.0))

    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    l = ufl.inner(f, v) * ufl.dx

    problem = fem.petsc.LinearProblem(a, l, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

    return problem



def main():

    from ex02_create_mesh import create_mesh_static, create_mesh_variable

    triangles = True

    import gmsh
    gmsh.initialize()
    mesh, ct, ft = create_mesh_variable(triangles=triangles)
    gmsh.finalize()

    problem = create_problem(mesh, ft)

    uh = problem.solve()
    uh.name = "u_h"

    from dolfinx import io
    with io.XDMFFile(mesh.comm, "output/ex02_uh_test.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(uh)

    return

if __name__ == "__main__":
    main()
