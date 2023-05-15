import numpy as np
import matplotlib.pyplot as plt

import ufl
import dolfinx
from dolfinx import fem
from petsc4py import PETSc
from mpi4py import MPI

# from solvers.navier_stokes_solver import NS_Solver
from solvers.implicit_ipcs import implicit_IPCS

import gmsh
from helpers.ex02_create_mesh import create_mesh_variable
from helpers.solutions import ex02_inlet_flow_BC as inlet_flow_BC


class MySolver(implicit_IPCS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # self.div_u = np.zeros_like(self.drag_forces)
        self.div_L2_sqr_form = fem.form( ufl.div(self.u_) * ufl.div(self.u_) * ufl.dx )
        self.div_u = np.zeros(np.ceil((self.T - self.t0) / self.dt)\
                                    .astype(int)+1, dtype=PETSc.ScalarType)
        self.ts = np.zeros_like(self.div_u)
        

        return
    
    def compute_div(self):
        div_L2_sqr = fem.assemble_scalar(self.div_L2_sqr_form)
        return np.sqrt(self.mesh.comm.reduce(div_L2_sqr, op=MPI.SUM, root=0))
    
    def callback(self):

        self.div_u[self.it] = self.compute_div()
        self.ts[self.it] = self.t

        return


def main():


    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    mesh, ct, ft = create_mesh_variable(triangles=True, lf=1.0)
    gmsh.finalize()

    V_el = ufl.VectorElement("CG", mesh.ufl_cell(), 2)
    Q_el = ufl.FiniteElement("CG", mesh.ufl_cell(), 1)

    dt = 0.05
    U_m = 2.5
    U_inlet = inlet_flow_BC(U_m)
    """ 2D-2, unsteady flow """


    solver = MySolver(
        mesh, ft, V_el, Q_el, U_inlet, dt=dt, T=5.0,
        log_interval=10,
        fname=None, data_fname=None,
        do_warm_up=True, warm_up_iterations=20
    )

    solver.run()

    print(solver.div_u)

    plt.figure()
    plt.plot(solver.ts, solver.div_u, 'k')
    plt.xlabel("$t$")
    plt.ylabel(r"$||\nabla \cdot u||_{L^2}$")
    plt.show()

    return

if __name__ == "__main__":
    main()
