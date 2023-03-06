import numpy as np
import matplotlib.pyplot as plt

from dolfinx import fem

from helpers.solutions import ex01_sol as u_ex_func
from helpers.errors import error_H1, error_L2
from ex01_solver import create_mesh, create_problem


def conv_test(mus, Ns, SUPG=False, title=None):

    data = np.zeros((len(mus), len(Ns), 2))

    for i, mu_val in enumerate(mus):
        for j, N in enumerate(Ns):
            
            order = 1
            domain = create_mesh(N)
            V = fem.FunctionSpace(domain, ("CG", order))
            problem = create_problem(domain, V, mu_val, SUPG=SUPG)
            
            uh = problem.solve()
            
            L2 = error_L2(uh, u_ex_func(mu_val))
            H1 = error_H1(uh, u_ex_func(mu_val))

            data[i, j, 0] = np.copy(L2)
            data[i, j, 1] = np.copy(H1)

    hs = 1 / np.array(Ns) # N facets per side

    fig, ax = plt.subplots()

    for i, mu in enumerate(mus):
        ax.loglog(hs, data[i, :, 0], '-o', label=f"$\mu={mu}, V=L^2$")
    for i, mu in enumerate(mus):
        ax.loglog(hs, data[i, :, 1], '--o', label=f"$\mu={mu}, V=H^1$")

    ax.legend()
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel("$h$")
    ax.set_ylabel(r"$\Vert u - u_h \Vert_V$")

    const_fits = np.zeros((len(mus), 2))
    order_fits = np.zeros((len(mus), 2))
    for i in range(len(mus)):
        poly = np.polynomial.polynomial.Polynomial.fit(
            np.log(hs), np.log(data[i,:,0]), deg=1
        )
        const_fits[i,0] = np.exp(poly.coef[0])
        order_fits[i,0] = poly.coef[1]

        poly = np.polynomial.polynomial.Polynomial.fit(
            np.log(hs), np.log(data[i,:,1]), deg=1
        )
        const_fits[i,1] = np.exp(poly.coef[0])
        order_fits[i,1] = poly.coef[1]

    return const_fits, order_fits, fig, ax

mus = [1.0, 0.3, 0.1]
Ns = [8, 16, 32, 64, 128, 256]

SUPG = False
print(f"{SUPG=}")
const_fits, order_fits, fig_cg, ax_cg = conv_test(mus, Ns, SUPG=SUPG, title="w/o SUPG")
for i, mu in enumerate(mus):
    print(f"{mu=}, ||e||_L^2 ≈ {const_fits[i,0]:.2e}*h^{order_fits[i,0]:.2f}")
for i, mu in enumerate(mus):
    print(f"{mu=}, ||e||_H^1 ≈ {const_fits[i,1]:.2e}*h^{order_fits[i,1]:.2f}")

print()

SUPG = True
print(f"{SUPG=}")
const_fits, order_fits, fig_supg, ax_supg = conv_test(mus, Ns, SUPG=SUPG, title="w/ SUPG")
for i, mu in enumerate(mus):
    print(f"{mu=}, ||e||_L^2 ≈ {const_fits[i,0]:.2e}*h^{order_fits[i,0]:.2f}")
for i, mu in enumerate(mus):
    print(f"{mu=}, ||e||_H^1 ≈ {const_fits[i,1]:.2e}*h^{order_fits[i,1]:.2f}")


plt.show()
