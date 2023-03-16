import numpy as np
import matplotlib.pyplot as plt
import gmsh

import ufl

from solvers.navier_stokes_solver import NS_Solver
from solvers.explicit_ipcs import explicit_IPCS
from solvers.implicit_ipcs import implicit_IPCS

from ex02_create_mesh import create_mesh_static
from helpers.solutions import ex02_inlet_flow_BC as inlet_flow_BC


BASELINE_ENERGY = 0.5674871879670002# At t=5.0
BASELINE_ENERGY = 0.5557884793150839# At t=1.0

def energy(solver):
    return np.linalg.norm(solver.u_.vector.array, ord=2) / solver.u_.vector.array.shape[0]

def stability_check1(solver:NS_Solver, warm_up_steps=40):

    for _ in range(warm_up_steps):
        if np.amax(np.abs(solver.u_.x.array)) > 10: break
        solver.step()

    energy_last = energy(solver)
    solver.step()
    energy_curr = energy(solver)
    
    blown_up = False
    for _ in range(100):
        
        energy_last = energy_curr
        energy_curr = energy(solver)
        if energy_curr > 1.5 * energy_last:
            blown_up = True

    return blown_up

def stability_check2(solver):

    return energy(solver) > 5 * BASELINE_ENERGY


def stability_analysis(solver_class, hs, ks):

    if not issubclass(solver_class, NS_Solver):
        raise ValueError()

    U_m = 1.5
    U_inlet = inlet_flow_BC(U_m)
    """ 2D-2, unsteady flow """

    blown_up_arr = np.zeros((len(hs), len(ks)))
    u_inf_arr = np.zeros((len(hs), len(ks)))
    energy_arr = np.zeros_like(blown_up_arr)

    for i, h in enumerate(hs):

        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 0)
        mesh, ct, ft = create_mesh_static(h=h, triangles=True)
        gmsh.finalize()


        V_el = ufl.VectorElement("CG", mesh.ufl_cell(), 2)
        Q_el = ufl.FiniteElement("CG", mesh.ufl_cell(), 1)
        """ Taylor-Hook P2-P1 elements. """

        for j, k in enumerate(ks):

            print(f"{h=}, {k=}")
            solver = solver_class(mesh, ft, V_el, Q_el, U_inlet, dt=k,
                                  T=max(1.0, 20*k*0),
                log_interval=np.inf,
                # fname=f"stabanal/exp_{h}_{k}.xdmf", data_fname=None,
                do_warm_up=False
            )
            solver.run()

            # if solver.blown_up:
            energy_arr[i,j] = solver.compute_energy()
            if stability_check2(solver):
                blown_up_arr[i,j] = 1.0
            u_inf = np.amax(np.abs(solver.u_.vector.array))
            u_inf_arr[i,j] = u_inf


    return blown_up_arr, u_inf_arr, energy_arr



def main():

    hs = np.array([0.41/2 * 2**(-i) for i in range(4)])
    ks = np.array([0.002*2**(-j) for j in range(4)])

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

    blown_up_arr, u_inf_arr, energy_arr = stability_analysis(explicit_IPCS, hs, ks)

    with np.printoptions(precision=3):
        print(blown_up_arr)
        print(u_inf_arr)
        print(energy_arr / BASELINE_ENERGY)

    # plt.figure()

    # plt.scatter(np.log(hh), np.log(kk), c=blown_up_arr)
    # plt.xlabel("$\log(h)$")
    # plt.ylabel("$\log(\Delta t)$")

    # for i in range(len(hs)):
    #     for j in range(len(ks)):
    #         energy_arr[i,j] = ( i*len(ks) + j ) * BASELINE_ENERGY

    import matplotlib as mpl
    # plt.figure()
    im = plt.matshow(energy_arr / BASELINE_ENERGY, norm=mpl.colors.Normalize(vmin=0.8*BASELINE_ENERGY, vmax=10*BASELINE_ENERGY))
    plt.colorbar(im)
    # x_labels = hs # labels you want to see
    plt.yticks(np.arange(len(hs)), hs)
    plt.xticks(np.arange(len(ks)), ks)
    plt.ylabel("$h$")
    plt.xlabel("$\Delta t$")
    plt.gca().xaxis.set_label_position('top')
    plt.title("Kinetic energy at $t = 1.0$, relative to baseline")

    plt.show()

    return


if __name__ == "__main__":
    main()
