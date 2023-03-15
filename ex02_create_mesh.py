"""
Mesh generation code is adapted from the FEniCSx tutorial by Jørgen S. Dokken
https://jsdokken.com/dolfinx-tutorial/chapter2/ns_code2.html
published under CCA 4.0 license
http://creativecommons.org/licenses/by/4.0/.
"""

import numpy as np

from mpi4py import MPI

from dolfinx import io

import gmsh

def create_mesh_variable(triangles=True, lf=1.0):

    # Rectangle dimensions
    L = 2.2
    H = 0.41

    # Cylinder geometry
    c_x = 0.2
    c_y = 0.2
    r = 0.05

    gdim = 2
    fluid_marker = 1
    inlet_marker = 2
    outlet_marker = 3
    wall_marker = 4
    obstacle_marker = 5
    inflow, outflow, walls, obstacle = [], [], [], []

    res_min = r / 3

    mesh_comm = MPI.COMM_WORLD
    model_rank = 0
    if mesh_comm.rank == model_rank:
        rectangle = gmsh.model.occ.addRectangle(0, 0, 0, L, H)
        cylinder_obstacle = gmsh.model.occ.addDisk(c_x, c_y, 0, r, r)

        fluid = gmsh.model.occ.cut([(gdim, rectangle)], [(gdim, cylinder_obstacle)])
        gmsh.model.occ.synchronize()

        volumes = gmsh.model.getEntities(dim=gdim)
        assert(len(volumes) == 1)
        gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], fluid_marker)

        boundaries = gmsh.model.getBoundary(volumes, oriented=False)
        for boundary in boundaries:
            center_of_mass = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])
            if np.allclose(center_of_mass, [0, H/2, 0]):
                inflow.append(boundary[1])
            elif np.allclose(center_of_mass, [L, H/2, 0]):
                outflow.append(boundary[1])
            elif np.allclose(center_of_mass, [L/2, H, 0]) or np.allclose(center_of_mass, [L/2, 0, 0]):
                walls.append(boundary[1])
            else:
                obstacle.append(boundary[1])
        gmsh.model.addPhysicalGroup(1, walls, wall_marker)
        gmsh.model.setPhysicalName(1, wall_marker, "Walls")
        gmsh.model.addPhysicalGroup(1, inflow, inlet_marker)
        gmsh.model.setPhysicalName(1, inlet_marker, "Inlet")
        gmsh.model.addPhysicalGroup(1, outflow, outlet_marker)
        gmsh.model.setPhysicalName(1, outlet_marker, "Outlet")
        gmsh.model.addPhysicalGroup(1, obstacle, obstacle_marker)
        gmsh.model.setPhysicalName(1, obstacle_marker, "Obstacle")


        distance_field = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distance_field, "EdgesList", obstacle)
        threshold_field = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(threshold_field, "IField", distance_field)
        gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", res_min)
        gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", 0.25 * H)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", r)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 2 * H)
        min_field = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [threshold_field])
        gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

        if not triangles:
            # Don't know what these do, but they make the mesh cells quadrilateral
            # Seems to make a nice triangular mesh when not in use.
            gmsh.option.setNumber("Mesh.Algorithm", 8)
            gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
            gmsh.option.setNumber("Mesh.RecombineAll", 1)
            gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)

        gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", lf)
        gmsh.model.mesh.generate(gdim)
        gmsh.model.mesh.setOrder(2)
        
        gmsh.model.mesh.optimize("Netgen")
        
        
    mesh, ct, ft = io.gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)
    ft.name = "Facet markers"

    return mesh, ct, ft


def create_mesh_static(h=0.05, triangles=True):

    # Rectangle dimensions
    L = 2.2
    H = 0.41

    # Cylinder geometry
    c_x = 0.2
    c_y = 0.2
    r = 0.05

    gdim = 2
    fluid_marker = 1
    inlet_marker = 2
    outlet_marker = 3
    wall_marker = 4
    obstacle_marker = 5
    inflow, outflow, walls, obstacle = [], [], [], []

    mesh_comm = MPI.COMM_WORLD
    model_rank = 0
    if mesh_comm.rank == model_rank:
        rectangle = gmsh.model.occ.addRectangle(0, 0, 0, L, H)
        cylinder_obstacle = gmsh.model.occ.addDisk(c_x, c_y, 0, r, r)

        fluid = gmsh.model.occ.cut([(gdim, rectangle)], [(gdim, cylinder_obstacle)])
        gmsh.model.occ.synchronize()

        volumes = gmsh.model.getEntities(dim=gdim)
        assert(len(volumes) == 1)
        gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], fluid_marker)

        boundaries = gmsh.model.getBoundary(volumes, oriented=False)
        for boundary in boundaries:
            center_of_mass = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])
            if np.allclose(center_of_mass, [0, H/2, 0]):
                inflow.append(boundary[1])
            elif np.allclose(center_of_mass, [L, H/2, 0]):
                outflow.append(boundary[1])
            elif np.allclose(center_of_mass, [L/2, H, 0]) or np.allclose(center_of_mass, [L/2, 0, 0]):
                walls.append(boundary[1])
            else:
                obstacle.append(boundary[1])
        gmsh.model.addPhysicalGroup(1, walls, wall_marker)
        gmsh.model.setPhysicalName(1, wall_marker, "Walls")
        gmsh.model.addPhysicalGroup(1, inflow, inlet_marker)
        gmsh.model.setPhysicalName(1, inlet_marker, "Inlet")
        gmsh.model.addPhysicalGroup(1, outflow, outlet_marker)
        gmsh.model.setPhysicalName(1, outlet_marker, "Outlet")
        gmsh.model.addPhysicalGroup(1, obstacle, obstacle_marker)
        gmsh.model.setPhysicalName(1, obstacle_marker, "Obstacle")

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)

        gmsh.model.mesh.generate(gdim)
        gmsh.model.mesh.setOrder(2)
        
        gmsh.model.mesh.optimize("Netgen")
        

    mesh, ct, ft = io.gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)
    ft.name = "Facet markers"

    return mesh, ct, ft



def main():

    gmsh.initialize()

    triangles = True
    mesh, ct, ft = create_mesh_variable(triangles=triangles, lf=0.5)
    # mesh, ct, ft = create_mesh_static(h=0.05, triangles=triangles)

    gmsh.finalize()

    with io.XDMFFile(MPI.COMM_WORLD, "output/ex02_mesh_test.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(ft)
        xdmf.write_meshtags(ct)

    return


if __name__ == "__main__":
    main()
