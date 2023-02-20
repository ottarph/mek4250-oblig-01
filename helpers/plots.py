import matplotlib.pyplot as plt
from dolfinx import fem


def trisurf(u, mesh=None, title=None):

    if mesh is None:
        mesh = u.function_space.mesh

    V = fem.FunctionSpace(mesh, ("CG", 1))
    uh = fem.Function(V)
    uh.interpolate(u)

    uu = uh.vector.array
    xx = mesh.geometry.x

    ax = plt.figure().add_subplot(projection='3d')
    ax.plot_trisurf(xx[:,0], xx[:,1], uu[:], cmap=plt.cm.viridis)
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

    return ax