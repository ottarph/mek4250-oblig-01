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


        self.u = ufl.TrialFunction(self.V)
        self.v = ufl.TestFunction(self.V)
        self.p = ufl.TrialFunction(self.Q)
        self.q = ufl.TestFunction(self.Q)

        self.u_s = fem.Function(self.V)
        self.phi = fem.Function(self.Q)

        # F_us_lhs = ufl.inner(self.u, self.v) * ufl.dx
        # F_us_rhs = ufl.inner(self.u_, self.v) * ufl.dx
        # F_us_rhs -= self.dt * ufl.inner(ufl.dot(self.u_, ufl.nabla_grad(self.u_)), self.v) * ufl.dx
        # # F_us_rhs -= self.dt / self.rho * ufl.inner(ufl.nabla_grad(self.p_), self.v) * ufl.dx
        # F_us_rhs += self.dt / self.rho * self.p_ * ufl.div(self.v) * ufl.dx
        # # F_us_rhs += self.dt * self.nu * ufl.inner(ufl.div(ufl.nabla_grad(self.u_)), self.v) * ufl.dx@
        # F_us_rhs += self.dt * self.nu * ufl.inner(ufl.div(ufl.grad(self.u_)), self.v) * ufl.dx

        F_us_lhs = ufl.inner(self.u, self.v) * ufl.dx
        F_us_rhs = ufl.inner(self.u_, self.v) * ufl.dx
        F_us_rhs -= self.dt * ufl.inner( ufl.dot(self.u_, ufl.nabla_grad(self.u_)), self.v ) * ufl.dx
        F_us_rhs += self.dt / self.rho * self.p_ * ufl.div(self.v) * ufl.dx
        F_us_rhs -= self.nu * self.dt * ufl.inner(self.u_, self.v) * ufl.dx

        F_us = F_us_lhs - F_us_rhs
        self.a_us = fem.form(ufl.lhs(F_us))
        self.l_us = fem.form(ufl.rhs(F_us))
        self.bcs_us = [self.bcs_u["inlet"], self.bcs_u["walls"], self.bcs_u["obstacle"]]
        """ Variational problem to compute u_s from u_, p_ """

        self.A_us = fem.petsc.assemble_matrix(self.a_us, bcs=self.bcs_us)
        self.A_us.assemble()
        self.b_us = fem.petsc.create_vector(self.l_us)

        self.solver_us = PETSc.KSP().create(self.mesh.comm)
        self.solver_us.setOperators(self.A_us)
        self.solver_us.setType(PETSc.KSP.Type.BCGS)
        self.solver_us.getPC().setType(PETSc.PC.Type.JACOBI)

        
        self.f1 = -self.rho / self.dt * ufl.div(self.u_s)

        self.a_phi = fem.form(ufl.inner(ufl.grad(self.p), ufl.grad(self.q)) * ufl.dx)
        self.l_phi = fem.form(self.f1 * self.q * ufl.dx)
        self.bcs_phi = [self.bcs_p["outlet"]]
        """ Variational problem to compute phi from u_s """

        self.A_phi = fem.petsc.assemble_matrix(self.a_phi, bcs=self.bcs_phi)
        self.A_phi.assemble()
        self.b_phi = fem.petsc.create_vector(self.l_phi)

        self.solver_phi = PETSc.KSP().create(self.mesh.comm)
        self.solver_phi.setOperators(self.A_phi)
        self.solver_phi.setType(PETSc.KSP.Type.MINRES)
        self.solver_phi.getPC().setType(PETSc.PC.Type.HYPRE)
        self.solver_phi.getPC().setHYPREType("boomeramg")


        F_uc_lhs = ufl.inner(self.u, self.v) * ufl.dx
        F_uc_rhs = ufl.inner(self.u_s, self.v) * ufl.dx
        F_uc_rhs -= self.dt/self.rho * ufl.inner(ufl.grad(self.phi), self.v) * ufl.dx
        # F_uc_rhs += self.dt/self.rho * self.phi * ufl.div(self.v) * ufl.dx
        self.a_uc = fem.form(ufl.lhs(F_uc_lhs - F_uc_rhs))
        self.l_uc = fem.form(ufl.rhs(F_uc_lhs - F_uc_rhs))
        # self.bcs_uc = [self.bcs_u["inlet"], self.bcs_u["walls"], self.bcs_u["obstacle"]]
        """ Variational problem to compute u_ from phi, u_s. """

        self.A_uc = fem.petsc.assemble_matrix(self.a_uc)
        self.A_uc.assemble()
        self.b_uc = fem.petsc.create_vector(self.l_uc)

        self.solver_uc = PETSc.KSP().create(self.mesh.comm)
        self.solver_uc.setOperators(self.A_uc)
        self.solver_uc.setType(PETSc.KSP.Type.CG)
        self.solver_uc.getPC().setType(PETSc.PC.Type.SOR)

        self.u_maxs = []

        return
    
    def step(self):

        # self.A_us.zeroEntries()
        # fem.petsc.assemble_matrix(self.A_us, self.a_us, 
        #                           bcs=self.bcs_us)
        # self.A_us.assemble()

        with self.b_us.localForm() as loc_b:
            loc_b.set(0)
        fem.petsc.assemble_vector(self.b_us, self.l_us)
        fem.petsc.apply_lifting(self.b_us, [self.a_us], [self.bcs_us])
        self.b_us.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, 
                            mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(self.b_us, self.bcs_us) 


        self.solver_us.solve(self.b_us, self.u_s.vector)
        self.u_s.x.scatter_forward()


        # Update the right hand side reusing the initial vector
        with self.b_phi.localForm() as loc_b:
            loc_b.set(0)
        fem.petsc.assemble_vector(self.b_phi, self.l_phi)

        # Apply Dirichlet boundary condition to the vector
        fem.petsc.apply_lifting(self.b_phi, [self.a_phi], [self.bcs_phi])
        self.b_phi.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, 
                            mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(self.b_phi, self.bcs_phi)    


        # Solve linear problem
        self.solver_phi.solve(self.b_phi, self.phi.vector)
        self.phi.x.scatter_forward()


        # Update pressure vector
        self.p_.vector.axpy(1, self.phi.vector)
        self.p_.x.scatter_forward()


        with self.b_uc.localForm() as loc_b:
            loc_b.set(0)
        fem.petsc.assemble_vector(self.b_uc, self.l_uc)
        self.b_uc.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, 
                              mode=PETSc.ScatterMode.REVERSE)

        self.solver_uc.solve(self.b_uc, self.u_.vector)
        self.u_.x.scatter_forward()


        self.u_maxs.append(np.amax(self.u_.x.array))
        if self.u_maxs[-1] > 1e3:
            print("Blow-up")
            self.finalize()
            quit()
        # print(self.u_maxs[-1])

        if self.it % 1000 == 0:
            print(self.t)
        return



def main():

    import gmsh
    from ex02_create_mesh import create_mesh_variable, create_mesh_static
    from helpers.solutions import ex02_inlet_flow_BC as inlet_flow_BC

    gmsh.initialize()
    # mesh, ct, ft = create_mesh_variable(triangles=True)
    H = 0.03175
    U_inf = 0.3
    dt = 0.5 * 0.1 / U_inf * H
    dt = 1/3200
    h = 0.01054
    dt = 0.0003215
    mesh, ct, ft = create_mesh_static(h=h, triangles=True)
    gmsh.finalize()
    
    V_el = ufl.VectorElement("CG", mesh.ufl_cell(), 2)
    Q_el = ufl.FiniteElement("CG", mesh.ufl_cell(), 1)
    """ Taylor-Hook P2-P1 elements. """

    U_m = 0.3
    U_inlet = inlet_flow_BC(U_m)

    solver = explicit_IPCS(0.0,
        mesh, ft, V_el, Q_el, U_inlet, dt=dt, T=5.0,
        extra_kwarg=3.0,
        fname="output/IPCS.xdmf", data_fname="data/IPCS.npy",
        do_initialize=False
        
    )

    solver.run()
    # print(solver.u_maxs)

    return

if __name__ == "__main__":
    main()