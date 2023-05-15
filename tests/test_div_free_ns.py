import numpy as np
import matplotlib.pyplot as plt

import ufl
import dolfinx
from dolfinx import fem
from petsc4py import PETSc
from mpi4py import MPI

from solvers.implicit_ipcs import implicit_IPCS

import gmsh
from helpers.ex02_create_mesh import create_mesh_variable
from helpers.solutions import ex02_inlet_flow_BC as inlet_flow_BC


class MySolver(implicit_IPCS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.div_L2_sqr_form = fem.form( ufl.div(self.u_) * ufl.div(self.u_) * ufl.dx )
        self.div_u_L2 = np.zeros(np.ceil((self.T - self.t0) / self.dt)\
                                    .astype(int)+1, dtype=PETSc.ScalarType)
        self.ts = np.zeros_like(self.div_u_L2)

        self.div_space = fem.FunctionSpace(self.mesh, ("DG", 0))
        self.u_div = fem.Function(self.div_space)
        self.u_div.name = "div u"

        self.div_trial, self.div_test = ufl.TrialFunction(self.div_space), ufl.TestFunction(self.div_space)

        self.div_lhs_form = fem.form( self.div_trial * self.div_test * ufl.dx )
        self.div_rhs_form = fem.form( ufl.div(self.u_) * self.div_test * ufl.dx )

        self.A_div = fem.petsc.assemble_matrix(self.div_lhs_form)
        self.A_div.assemble()
        self.b_div = fem.petsc.create_vector(self.div_rhs_form)

        self.solver_div = PETSc.KSP().create(self.mesh.comm)
        self.solver_div.setOperators(self.A_div)
        self.solver_div.getPC().setType(PETSc.PC.Type.JACOBI)
        self.solver_div.setType(PETSc.KSP.Type.CG)
        """ Basically a mass matrix, CG-Jacobi should be okay. """

        return
    
    def compute_div_norm(self):
        div_L2_sqr = fem.assemble_scalar(self.div_L2_sqr_form)
        return np.sqrt(self.mesh.comm.reduce(div_L2_sqr, op=MPI.SUM, root=0))
    
    def compute_div(self):

        # self.u_ is `scatter_forward()`-ed from `self.step()` before `self.callback()`

        with self.b_div.localForm() as loc_b:
            loc_b.set(0)
        fem.petsc.assemble_vector(self.b_div, self.div_rhs_form)
        self.b_div.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, 
                              mode=PETSc.ScatterMode.REVERSE)
        self.solver_div.solve(self.b_div, self.u_div.vector)

        self.u_div.x.scatter_forward()

        return
    
    def callback(self):

        self.div_u_L2[self.it] = self.compute_div_norm()
        self.ts[self.it] = self.t

        self.compute_div()

        self.xdmf.write_function(self.u_div, self.t)

        return


def main():


    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    mesh, ct, ft = create_mesh_variable(triangles=True, lf=1.0)
    gmsh.finalize()

    V_el = ufl.VectorElement("CG", mesh.ufl_cell(), 2)
    Q_el = ufl.FiniteElement("CG", mesh.ufl_cell(), 1)

    dt = 0.05
    U_m = 1.5
    U_inlet = inlet_flow_BC(U_m)
    """ 2D-2, unsteady flow """


    solver = MySolver(
        mesh, ft, V_el, Q_el, U_inlet, dt=dt, T=5.0,
        log_interval=10,
        fname="output/divergence.xdmf", data_fname=None,
        do_warm_up=True, warm_up_iterations=20
    )

    solver.run()

    print(solver.div_u_L2)

    plt.figure()
    plt.plot(solver.ts, solver.div_u_L2, 'k')
    plt.xlabel("$t$")
    plt.ylabel(r"$||\nabla \cdot u||_{L^2}$")
    plt.show()

    return

if __name__ == "__main__":
    main()
