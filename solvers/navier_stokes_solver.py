import numpy as np

import ufl
import dolfinx
from dolfinx import fem, io, geometry
from petsc4py import PETSc
from mpi4py import MPI


class NS_Solver:
    """
    Super-class defining Navier-Stokes solver functionality.
    """
    def __init__(self, mesh:dolfinx.mesh.Mesh, facet_tags,
                 V_el:ufl.VectorElement, Q_el:ufl.FiniteElement, 
                 U_inlet:callable, U_0:callable=None, t0:float=0.0, 
                 T:float=8.0, dt:float=1/1600, fname=None, data_fname=None, 
                 do_warm_up=False, warm_up_iterations=None):
        """
        Super-class defining Navier-Stokes solver functionality.
        ```
            U_inlet: x -> u
        ```
        """

        self.mesh = mesh
        """ Computational mesh of problem. """
        self.facet_tags = facet_tags
        """ Tags of the different parts of the boundary """
        self.fluid_marker = 1
        self.inlet_marker = 2
        self.outlet_marker = 3
        self.wall_marker = 4
        self.obstacle_marker = 5

        self.H = 0.41
        self.D = 0.1
        self.L = 2.2
        self.S_c_x = 0.2
        self.S_c_y = 0.2
        """ Problem geometry """

        self.U_inlet = U_inlet
        """ Boundary condition ``u(0,y,t) = U_inlet(0,y)`` """

        if U_0 is None:
            U_0 = self.U_inlet.as_IC
        self.U_0 = U_0
        """ Function to set initial condition of fluid flow """

        self.rho = fem.Constant(mesh, PETSc.ScalarType(1.0))
        self.mu = fem.Constant(mesh, PETSc.ScalarType(1e-3))
        self.nu = self.mu / self.rho
        self.U_m = self.U_inlet([0.0, self.H/2])
        self.U_bar = 2 / 3 * self.U_m
        self.Re = self.U_bar * self.D * self.rho.value / self.mu.value
        """ Problem physical parameters """

        self.p_probe_1 = np.array([0.15, 0.2, 0.0])
        self.p_probe_2 = np.array([0.25, 0.2, 0.0])

        tree = geometry.BoundingBoxTree(mesh, mesh.geometry.dim)
        points = np.array([self.p_probe_1, self.p_probe_2])
        cell_candidates = geometry.compute_collisions(tree, points)
        colliding_cells = geometry.compute_colliding_cells(mesh, 
                                            cell_candidates, points)
        self.front_cells = colliding_cells.links(0)
        self.back_cells = colliding_cells.links(1)

        self.V_el = V_el
        """ Fluid element """
        self.V = fem.FunctionSpace(self.mesh, self.V_el)
        """ Fluid finite element function space """
        self.Q_el = Q_el
        """ Pressure element """
        self.Q = fem.FunctionSpace(self.mesh, self.Q_el)
        """ Pressure finite element function space """

        self.u_ = fem.Function(self.V)
        self.u_.name = "uh"
        self.u_.vector.array[:] = 0.0
        self.p_ = fem.Function(self.Q)
        self.p_.name = "ph"
        self.p_.vector.array[:] = 0.0

        self.bcs_u, self.bcs_p = self.make_boundary_conditions()

        self.dObs = ufl.Measure("exterior_facet", domain=self.mesh, 
                                subdomain_id=self.obstacle_marker, 
                                subdomain_data=self.facet_tags)
        """ Integration measure over cylinder obstacle. """
        self.n = -ufl.FacetNormal(self.mesh)
        """ Inward pointing boundary normal vector. """
        self.t = ufl.as_vector((self.n[1], -self.n[0]))
        """ Boundary tangent vector. """
        self.u_t = ufl.inner(self.t, self.u_)
        """ Flow tangential velocity. """
        self.drag_form = fem.form( (
            self.mu * ufl.inner(ufl.grad(self.u_t), self.n) * self.n[1] \
            - self.p_ * self.n[0]
        ) * self.dObs )
        """ Form for computing drag. """
        self.lift_form = fem.form((
            -self.mu * ufl.inner(ufl.grad(self.u_t), self.n) * self.n[0] \
            - self.p_ * self.n[1]
        ) * self.dObs )
        """ Form for computing lift. """

        self.t0 = t0
        """ Starting time """
        self.T = T
        """ End time. """
        self.dt = dt
        """ Time step size """
        self.t = t0
        """ Current time step """
        self.it = 0
        """ Current iterate number """

        self.do_warm_up = do_warm_up
        if warm_up_iterations is None:
            warm_up_iterations = np.ceil(self.L / (self.U_m*self.dt)).astype(int)
        self.warm_up_iterations = warm_up_iterations

        self.fname = fname
        self.data_fname = data_fname

        if self.data_fname is not None:
            self.drag_forces = np.zeros(np.ceil((self.T - self.t0) / self.dt)\
                                        .astype(int)+1, dtype=PETSc.ScalarType)
            self.lift_forces = np.zeros_like(self.drag_forces)
            self.pressure_diffs = np.zeros_like(self.drag_forces)
            self.ts = np.zeros_like(self.drag_forces)

        if self.fname is not None:
            self.xdmf = io.XDMFFile(mesh.comm, self.fname, "w")
            self.xdmf.write_mesh(self.mesh)

        return
    
    def make_boundary_conditions(self):
        """ Makes the Dirichlet boundary conditions. """
        tdim = self.mesh.topology.dim
        fdim = tdim - 1

        u_inlet = fem.Function(self.V)
        u_inlet.interpolate(self.U_inlet)
        bc_u_inlet = fem.dirichletbc(u_inlet, fem.locate_dofs_topological(
            self.V, fdim, self.facet_tags.find(self.inlet_marker))
        )

        bc_u_nonslip_walls = fem.dirichletbc(
            fem.Constant(self.mesh, PETSc.ScalarType((0.0, 0.0))),
            fem.locate_dofs_topological(
                self.V, fdim, self.facet_tags.find(self.wall_marker)
            ), self.V
        )

        bc_u_nonslip_obstacle = fem.dirichletbc(
            fem.Constant(self.mesh, PETSc.ScalarType((0.0, 0.0))),
            fem.locate_dofs_topological(
                self.V, fdim, self.facet_tags.find(self.obstacle_marker)
            ), self.V
        )

        bc_p_outlet = fem.dirichletbc(
            fem.Constant(self.mesh, PETSc.ScalarType(0.0)),
            fem.locate_dofs_topological(
                self.Q, fdim, self.facet_tags.find(self.outlet_marker)
            ), self.Q
        )

        bcs_u = {
            "inlet": bc_u_inlet,
            "walls": bc_u_nonslip_walls,
            "obstacle": bc_u_nonslip_obstacle
        }
        bcs_p = {"outlet": bc_p_outlet}

        return bcs_u, bcs_p
    
    def compute_drag(self):
        F_d = fem.assemble_scalar(self.drag_form)
        return self.mesh.comm.reduce(F_d, op=MPI.SUM, root=0)
    
    def compute_lift(self):
        F_l = fem.assemble_scalar(self.lift_form)
        return self.mesh.comm.reduce(F_l, op=MPI.SUM, root=0)
    
    def compute_pressure_difference():
        raise NotImplementedError()
    
    def initialize(self):

        self.u_.interpolate(self.U_0)

        if self.do_warm_up:
            for _ in range(self.warm_up_iterations):
                self.step()

        if self.fname is not None:
            self.xdmf.write_function(self.u_, self.t0)
            self.xdmf.write_function(self.p_, self.t0)

        if self.data_fname is not None:
            self.ts[0] = self.t0
            self.drag_forces[0] = self.compute_drag()
            self.lift_forces[0] = self.compute_lift()
            try:
                """ Temporary, until implemented. """
                self.pressure_diffs[0] = self.compute_pressure_difference()
            except:
                pass

        return
    
    def finalize(self):

        if self.fname is not None:
            self.xdmf.close()

        if self.data_fname is not None:
            arr = np.zeros((self.drag_forces.shape[0], 4), dtype=float)
            arr[:,0] = self.ts
            arr[:,1] = self.drag_forces
            arr[:,2] = self.lift_forces
            arr[:,3] = self.pressure_diffs
            np.savetxt(self.data_fname, arr)

        return

    def run(self):

        self.initialize()

        eps = 1e-3 * self.dt
        while self.t < self.T - eps:
            self.step()
            self.t += self.dt
            self.it += 1

            self.U_inlet.t = self.t

            if self.data_fname is not None:
                self.ts[self.it] = self.t
                self.drag_forces[self.it] = self.compute_drag()
                self.lift_forces[self.it] = self.compute_lift()
                try:
                    """ Temporary, until implemented. """
                    self.pressure_diffs[self.it] = self.compute_pressure_difference()
                except:
                    pass

            if self.fname is not None:
                self.xdmf.write_function(self.u_, self.t)
                self.xdmf.write_function(self.p_, self.t)

        self.finalize()

        return
    
    def step(self):
        raise NotImplementedError()
    

    
def main():

    import gmsh
    from ex02_create_mesh import create_mesh_variable
    from helpers.solutions import ex02_inlet_flow_BC as inlet_flow_BC

    gmsh.initialize()
    mesh, ct, ft = create_mesh_variable(triangles=True)    
    gmsh.finalize()

    V_el = ufl.VectorElement("CG", mesh.ufl_cell(), 2)
    Q_el = ufl.FiniteElement("CG", mesh.ufl_cell(), 1)
    """ Taylor-Hook P2-P1 elements. """

    U_m = 0.3
    U_inlet = inlet_flow_BC(U_m)

    solver = NS_Solver(
        mesh, ft, V_el, Q_el, U_inlet, dt=0.1,
        fname="output/generic_NS.xdmf", data_fname="data/generic_NS.npy"
    )

    return

    
if __name__ == "__main__":
    main()
