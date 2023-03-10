import numpy as np

import ufl
import dolfinx
from dolfinx import fem
from petsc4py import PETSc

from solvers.navier_stokes_solver import NS_Solver

class explicit_IPCS(NS_Solver):

    def __init__(self, extra_arg, *args, extra_kwarg=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra_arg = extra_arg
        self.extra_kwarg = extra_kwarg

        def U_0(x):
            values = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)
            for i in range(x.shape[1]):
                if np.isclose(x[0,i], 0.0):
                    x_tmp = [x[0,i], x[1,i], x[2,i]]
                    values[0,i] = self.U_inlet(x_tmp)
            return values
        
        self.u_.interpolate(U_0)

        self.xdmf.write_function(self.u_, self.t)


        # self.u_s = fem.Function(self.V)
        """ Initial guess velocity """

        self.u = ufl.TrialFunction(self.V)
        self.v = ufl.TestFunction(self.V)
        self.phi = ufl.TrialFunction(self.Q)
        self.q = ufl.TestFunction(self.Q)

        self.phi_ = fem.Function(self.Q)

        self.u_s = self.u_ + self.dt * \
        (-ufl.dot(self.u_, ufl.nabla_grad(self.u_)) - 
         1/self.rho * ufl.nabla_grad(self.p_) +
         self.nu * ufl.div(ufl.grad(self.u_)))
        
        self.u_c = fem.Function(self.V)
        
        self.f1 = -self.rho / self.dt * ufl.div(self.u_s)

        self.a1 = fem.form(ufl.inner(ufl.grad(self.phi), ufl.grad(self.q)) * ufl.dx)
        self.l1 = fem.form(self.f1 * self.q * ufl.dx)

        self.A1 = fem.petsc.assemble_matrix(self.a1, bcs=[self.bcs_p["outlet"]])
        self.A1.assemble()
        self.b1 = fem.petsc.create_vector(self.l1)

        self.solver1 = PETSc.KSP().create(self.mesh.comm)
        self.solver1.setOperators(self.A1)
        self.solver1.setType(PETSc.KSP.Type.PREONLY)
        self.solver1.getPC().setType(PETSc.PC.Type.LU)

        self.a_uc = fem.form(ufl.inner(self.u, self.v) * ufl.dx)
        self.l_uc = fem.form(ufl.inner( self.dt / self.rho \
                                       * ufl.grad(self.phi_), self.v) * ufl.dx)
        """ Variational problem to compute u_c from phi. """

        self.A_uc = fem.petsc.assemble_matrix(self.a_uc, bcs=[
            self.bcs_u["inlet"], self.bcs_u["walls"], self.bcs_u["obstacle"]])
        self.A_uc.assemble()
        self.b_uc = fem.petsc.create_vector(self.l_uc)

        self.solver_uc = PETSc.KSP().create(self.mesh.comm)
        self.solver_uc.setOperators(self.A_uc)
        self.solver_uc.setType(PETSc.KSP.Type.PREONLY)
        self.solver_uc.getPC().setType(PETSc.PC.Type.LU)

        self.u_maxs = []

        return
    
    def step(self):

        # Update the right hand side reusing the initial vector
        with self.b1.localForm() as loc_b:
            loc_b.set(0)
        fem.petsc.assemble_vector(self.b1, self.l1)

        # Apply Dirichlet boundary condition to the vector
        fem.petsc.apply_lifting(self.b1, [self.a1], [[self.bcs_p["outlet"]]])
        self.b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, 
                            mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(self.b1, [self.bcs_p["outlet"]])    

        # Solve linear problem
        self.solver1.solve(self.b1, self.phi_.vector)
        self.phi_.x.scatter_forward()

        # Update pressure vector
        self.p_.x.array[:] += self.phi_.x.array[:]
        self.p_.x.scatter_forward()
        # alternative:
        # p_.vector.axpy(1, phi.vector)
        # p_.x.scatter_forward()

        with self.b_uc.localForm() as loc_b:
            loc_b.set(0)
        fem.petsc.assemble_vector(self.b_uc, self.l_uc)

        # Apply Dirichlet boundary condition to the vector
        fem.petsc.apply_lifting(self.b_uc, [self.a_uc], 
            [[self.bcs_u["inlet"], self.bcs_u["walls"], 
                                self.bcs_u["obstacle"]]])
        self.b_uc.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, 
                              mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(self.b_uc, 
            [self.bcs_u["inlet"], self.bcs_u["walls"], self.bcs_u["obstacle"]])    

        # Solve linear problem to obtain velocity correction.
        self.solver_uc.solve(self.b_uc, self.u_c.vector)
        self.u_c.x.scatter_forward()

        self.u_.x.array[:] += self.u_c.x.array[:]
        self.u_.x.scatter_forward()

        self.u_maxs.append(np.amax(self.u_.x.array))
        

        return



def main():

    import gmsh
    from ex02_create_mesh import create_mesh_variable, create_mesh_static
    from helpers.solutions import ex02_inlet_flow_BC as inlet_flow_BC

    gmsh.initialize()
    # mesh, ct, ft = create_mesh_variable(triangles=True)
    mesh, ct, ft = create_mesh_static(h=0.01, triangles=True)
    gmsh.finalize()
    
    V_el = ufl.VectorElement("CG", mesh.ufl_cell(), 2)
    Q_el = ufl.FiniteElement("CG", mesh.ufl_cell(), 1)
    """ Taylor-Hook P2-P1 elements. """

    U_m = 0.3
    U_inlet = inlet_flow_BC(U_m)

    solver = explicit_IPCS(0.0,
        mesh, ft, V_el, Q_el, U_inlet, dt=0.000_000_01, T=0.000_000_1,
        extra_kwarg=3.0,
        fname="output/IPCS.xdmf", data_fname="data/IPCS.npy",
        do_initialize=False
        
    )

    print(f"{solver.extra_arg=}")
    print(f"{solver.extra_kwarg=}")

    # for _ in range(5):
    #     print("bar")
    #     solver.step()
    solver.run()
    print(solver.u_maxs)

    return

if __name__ == "__main__":
    main()
