import numpy as np
import matplotlib.pyplot as plt
import gmsh

import ufl

from solvers.navier_stokes_solver import NS_Solver
from solvers.explicit_ipcs import explicit_IPCS
from solvers.implicit_ipcs import implicit_IPCS

from helpers.ex02_create_mesh import create_mesh_static, create_mesh_basic
from helpers.solutions import ex02_inlet_flow_BC as inlet_flow_BC


# BASELINE_ENERGY = 0.5674871879670002# At t=5.0
BASELINE_ENERGY = 0.5557884793150839# At t=1.0, SI-IPCS, variable mesh, lf=1.0, dt=1/160.


def stability_analysis(solver_class, hs, ks, T=1.0):

    if not issubclass(solver_class, NS_Solver):
        raise ValueError()

    U_m = 1.5
    U_inlet = inlet_flow_BC(U_m)
    """ 2D-2, unsteady flow """

    energy_arr = np.zeros((len(hs), len(ks)))

    for i, h in enumerate(hs):

        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 0)
        mesh, ct, ft = create_mesh_basic(h=h, triangles=True)
        gmsh.finalize()


        V_el = ufl.VectorElement("CG", mesh.ufl_cell(), 2)
        Q_el = ufl.FiniteElement("CG", mesh.ufl_cell(), 1)
        """ Taylor-Hook P2-P1 elements. """

        def U_0(x):
            H = U_inlet.H
            U_m = U_inlet.U_m
            values = np.zeros((2, x.shape[1]))
            for i in range(x.shape[1]):
                x_tmp = [x[0,i], x[1,i], x[2,i]]
                values[0,i] = 4.0 * U_m * x_tmp[1] * (H - x_tmp[1]) / H**2
            return values


        for j, k in enumerate(ks):

            print(f"{h=}, {k=}")
            solver = solver_class(mesh, ft, V_el, Q_el, U_inlet, U_0=U_0, dt=k,
                                  T=max(T, 100*k*1),
                log_interval=np.inf,
                # fname=f"output/imp_stab.xdmf", data_fname=None,
                do_warm_up=False
            )
            solver.run()

            # if solver.blown_up:
            energy_arr[i,j] = solver.compute_energy()


    return energy_arr



def main():

    hs = np.array([0.41/2 * 2**(-i) for i in range(6)])
    ks = np.array([0.02*2**(-j) for j in range(6)])

    hs = np.logspace(np.log10(0.41/2), np.log10(0.05), 10)
    ks = np.logspace(np.log10(0.02), np.log10(0.002), 10)

    """ For explicit solver. """
    hs = np.logspace(np.log10(0.41/2), np.log10(0.05), 20)
    ks = np.logspace(np.log10(0.02), np.log10(0.002), 20)

    # hs = np.array([0.41/2, 0.41/4, 0.41/8])
    """ For implicit solver. """
    hs = np.logspace(np.log10(0.41/16), np.log10(0.01), 10)
    # ks = np.logspace(np.log10(0.4), np.log10(0.05), 10)
    ks = np.logspace(np.log10(1.5), np.log10(0.01), 10)
    # hs = np.logspace(np.log10(0.41/2), np.log10(0.1), 10)
    # ks = np.logspace(np.log10(0.02), np.log10(0.01), 10)

    print(hs)
    print(ks)

    # quit()

    hh = np.zeros((hs.shape[0], ks.shape[0]))
    for j in range(ks.shape[0]):
        hh[:,j] = hs
    kk = np.zeros((hs.shape[0], ks.shape[0]))
    for i in range(hs.shape[0]):
        kk[i,:] = ks

    print(hh)
    print(kk)

    # quit()

    energy_arr = stability_analysis(explicit_IPCS, hs, ks, T=1.0)
    energy_arr = stability_analysis(implicit_IPCS, hs, ks, T=2.0)

    with np.printoptions(precision=3):
        print(energy_arr / BASELINE_ENERGY)


    import matplotlib as mpl
    # plt.figure()

    fig, ax = plt.subplots(figsize=(12,10))
    im = ax.matshow(energy_arr / BASELINE_ENERGY, norm=mpl.colors.Normalize(vmin=1.0, vmax=4))
    plt.colorbar(im)
    # x_labels = hs # labels you want to see
    plt.yticks(np.arange(len(hs)), np.round(np.log10(hs), 2))
    plt.xticks(np.arange(len(ks)), np.round(np.log10(ks), 2))
    plt.ylabel(r"$\log_{10} h$")
    plt.xlabel(r"$\log_{10} \Delta t$")
    plt.gca().xaxis.set_label_position('top')
    # plt.title("Kinetic energy at $t = 1.0$, relative to baseline")

    plt.savefig("images/implicit_stab_anal_tmp.png", dpi=400)

    plt.show()

    return


if __name__ == "__main__":
    main()
