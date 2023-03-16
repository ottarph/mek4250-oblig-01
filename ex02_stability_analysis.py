import numpy as np
import matplotlib.pyplot as plt
import gmsh

import ufl

from solvers.navier_stokes_solver import NS_Solver
from solvers.explicit_ipcs import explicit_IPCS
from solvers.implicit_ipcs import implicit_IPCS

from ex02_create_mesh import create_mesh_static
from helpers.solutions import ex02_inlet_flow_BC as inlet_flow_BC


def stability_analysis(solver_class, hs, ks):

    if not issubclass(solver_class, NS_Solver):
        raise ValueError()

    U_m = 1.5
    U_inlet = inlet_flow_BC(U_m)
    """ 2D-2, unsteady flow """

    blown_up_arr = np.zeros((len(hs), len(ks)))
    u_inf_arr = np.zeros((len(hs), len(ks)))

    for i, h in enumerate(hs):

        gmsh.initialize()
        mesh, ct, ft = create_mesh_static(h=h, triangles=True)
        gmsh.finalize()


        V_el = ufl.VectorElement("CG", mesh.ufl_cell(), 2)
        Q_el = ufl.FiniteElement("CG", mesh.ufl_cell(), 1)
        """ Taylor-Hook P2-P1 elements. """

        for j, k in enumerate(ks):

            print(f"{h=}, {k=}")
            solver = solver_class(mesh, ft, V_el, Q_el, U_inlet, dt=k, T=8,
                log_interval=np.inf,
                fname=None, data_fname=None,
                do_warm_up=False
            )
            solver.run()

            if solver.blown_up:
                blown_up_arr[i,j] = 1.0
            u_inf = np.amax(np.abs(solver.u_.vector.array))
            u_inf_arr[i,j] = u_inf


    return blown_up_arr, u_inf_arr



def main():

    hs = [2**(-1-i) for i in range(6)]
    ks = [2**(-5-j) for j in range(6)]

    hs = np.logspace(np.log10(0.5), np.log10(0.15), 20)
    ks = np.logspace(np.log10(0.005), np.log10(0.001), 20)
    # hs = np.logspace(np.log10(0.5), np.log10(0.05*2), 6)
    # ks = np.logspace(np.log10(0.005), np.log10(0.001*4), 6)
    print(hs)
    print(ks)

    hh = np.zeros((hs.shape[0], ks.shape[0]))
    for j in range(ks.shape[0]):
        hh[:,j] = hs
    kk = np.zeros((hs.shape[0], ks.shape[0]))
    for i in range(hs.shape[0]):
        kk[i,:] = ks

    print(hh)
    print(kk)

    # quit()

    blown_up_arr, u_inf_arr = stability_analysis(explicit_IPCS, hs, ks)

    with np.printoptions(precision=3):
        print(blown_up_arr)
        print(u_inf_arr)

    plt.figure()

    plt.scatter(np.log(hh), np.log(kk), c=blown_up_arr)
    plt.xlabel("$\log(h)$")
    plt.ylabel("$\log(\Delta t)$")

    plt.show()

    return


if __name__ == "__main__":
    main()
