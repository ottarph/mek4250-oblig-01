import numpy as np

import ufl
import dolfinx
from dolfinx import fem
from petsc4py import PETSc

from solvers.implicit_ipcs import implicit_IPCS


def main():

    import gmsh
    from helpers.ex02_create_mesh import create_mesh_variable, create_mesh_static, create_mesh_basic
    from helpers.solutions import ex02_inlet_flow_BC as inlet_flow_BC

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    mesh, ct, ft = create_mesh_variable(triangles=True, lf=1.0)
    h = 0.01
    dt = 1 / 160
    # print(f"{dt=}")
    # mesh, ct, ft = create_mesh_static(h=h, triangles=True)
    gmsh.finalize()
    
    """ Change the below variables to test different polynomial orders for 
        velocity and pressure.
        Default values of u_order=2 and p_order=1 work well. 
        With u_order=1, p_order=1, derived quantities are inaccurate. 
            The drag coefficients and pressure drops are severely overestimated
            and the lift coefficient is severely underestimated.
    """
    u_order = 1
    p_order = 1

    V_el = ufl.VectorElement("CG", mesh.ufl_cell(), u_order)
    Q_el = ufl.FiniteElement("CG", mesh.ufl_cell(), p_order)
    """ Taylor-Hook P2-P1 elements. """

    U_m = 1.5
    U_inlet = inlet_flow_BC(U_m)
    """ 2D-2, unsteady flow """

    solver = implicit_IPCS(
        mesh, ft, V_el, Q_el, U_inlet, dt=dt, T=5.0,
        log_interval=100,
        fname="output/SI_elem_comp_IPCS.xdmf", data_fname="data/SI_elem_comp_IPCS.npy",
        do_warm_up=False, warm_up_iterations=20
    )

    solver.run()

    print(solver.compute_energy())

    if solver.data_fname is not None:
        import matplotlib.pyplot as plt
        burn_in = solver.ts.shape[0] // 5
        C_D = solver.drag_forces[burn_in:] * 2 / (solver.rho.value * solver.U_bar**2 * solver.D)
        _, axs = plt.subplots(1,3)
        axs[0].plot(range(C_D.shape[0]), C_D, 'k-')
        axs[0].set_title("Drag coefficients")
        C_L = solver.lift_forces[burn_in:] * 2 / (solver.rho.value * solver.U_bar**2 * solver.D)
        axs[1].plot(range(C_L.shape[0]), C_L, 'k-')
        axs[1].set_title("Lift coefficients")
        p_diffs = solver.pressure_diffs[burn_in:]
        axs[2].plot(range(p_diffs.shape[0]), p_diffs, 'k-')
        axs[2].set_title("Pressure differences")
        plt.show()

    return

if __name__ == "__main__":
    main()
