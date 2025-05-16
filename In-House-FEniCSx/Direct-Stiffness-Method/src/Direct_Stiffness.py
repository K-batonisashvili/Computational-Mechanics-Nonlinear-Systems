"""
Project:        Direct-Stifness Matrix Method
Author:         Kote Batonisashvili
Description:    Assignment 2.1. Direct-Stifness Matrix Method which creates Frames with their respective Nodes and Elements.
"""

# Standard Imports
import numpy as np
import matplotlib.pyplot as plt


def check_unit_vector(vec: np.ndarray):
    """
    """
    if np.isclose(np.linalg.norm(vec), 1.0):
        return
    else:
        raise ValueError("Expected a unit vector for reference vector.")


def check_parallel(vec_1: np.ndarray, vec_2: np.ndarray):
    """
    """
    if np.isclose(np.linalg.norm(np.cross(vec_1, vec_2)), 0.0):
        raise ValueError("Reference vector is parallel to beam axis.")
    else:
        return

class Nodes():
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

          # Initialize default nodal loads to zero
        self.Fx = 0.0
        self.Fy = 0.0
        self.Fz = 0.0
        self.Mx = 0.0
        self.My = 0.0
        self.Mz = 0.0

        # Initialize default displacements to zero
        self.u = 0.0
        self.v = 0.0
        self.w = 0.0
        self.theta_x = 0.0
        self.theta_y = 0.0
        self.theta_z = 0.0

        # Initialize boundary conditions (all DOFs free by default)
        self.boundary_conditions = np.array([False, False, False, False, False, False])

    def set_nodal_load(self, Fx, Fy, Fz, Mx, My, Mz):
        self.Mz = Mz
        self.My = My
        self.Mx = Mx
        self.Fx = Fx
        self.Fy = Fy
        self.Fz = Fz

    def get_nodal_load(self):
        return np.array([self.Fx, self.Fy, self.Fz, self.Mx, self.My, self.Mz])
    
    def get_coordinates(self):
        return np.array([self.x, self.y, self.z])
    
    def get_displacements(self):
        return np.array([self.u, self.v, self.w, self.theta_x, self.theta_y, self.theta_z])

    def set_boundary_constraints(self, constraints: np.ndarray = None):
        if constraints is None:
            self.boundary_conditions = np.array([False, False, False, False, False, False])
        else:
            self.boundary_conditions = constraints

class Elements():
    def __init__(self, node1, node2, E, v, A, I_z, I_y, I_p, J, local_z_axis):
        self.node1 = node1
        self.node2 = node2
        self.E = E          # Young's Modulus
        self.v = v          # Poisson's Ratio
        self.A = A          # Area
        self.I_z = I_z      # Inertia about z-axis
        self.I_y = I_y      # Inertia about y-axis
        self.I_p = I_p      # Polar moment of inertia
        self.J = J          # Torsional Constant
        self.local_z_axis = local_z_axis

   
    def local_geometric_stiffness_matrix_3D_beam(self) -> np.ndarray:
        """
        local element geometric stiffness matrix
        source: p. 258 of McGuire's Matrix Structural Analysis 2nd Edition
        Given:
            material and geometric parameters:
                L, A, I_rho (polar moment of inertia)
            element forces and moments:
                Fx2, Mx2, My1, Mz1, My2, Mz2
        Context:
            load vector:
                [Fx1, Fy1, Fz1, Mx1, My1, Mz1, Fx2, Fy2, Fz2, Mx2, My2, Mz2]
            DOF vector:
                [u1, v1, w1, th_x1, th_y1, th_z1, u2, v2, w2, th_x2, th_y2, th_z2]
            Equation:
                [load vector] = [stiffness matrix] @ [DOF vector]
        Returns:
            12 x 12 geometric stiffness matrix k_g
        """
        Fx1, Fy1, Fz1, Mx1, My1, Mz1 = self.node1.get_nodal_load()
        Fx2, Fy2, Fz2, Mx2, My2, Mz2 = self.node2.get_nodal_load()

        L = np.linalg.norm(self.node1.get_coordinates() - self.node2.get_coordinates())
        k_g = np.zeros((12, 12))
        # upper triangle off diagonal terms
        k_g[0, 6] = -Fx2 / L
        k_g[1, 3] = My1 / L
        k_g[1, 4] = Mx2 / L
        k_g[1, 5] = Fx2 / 10.0
        k_g[1, 7] = -6.0 * Fx2 / (5.0 * L)
        k_g[1, 9] = My2 / L
        k_g[1, 10] = -Mx2 / L
        k_g[1, 11] = Fx2 / 10.0
        k_g[2, 3] = Mz1 / L
        k_g[2, 4] = -Fx2 / 10.0
        k_g[2, 5] = Mx2 / L
        k_g[2, 8] = -6.0 * Fx2 / (5.0 * L)
        k_g[2, 9] = Mz2 / L
        k_g[2, 10] = -Fx2 / 10.0
        k_g[2, 11] = -Mx2 / L
        k_g[3, 4] = -1.0 * (2.0 * Mz1 - Mz2) / 6.0
        k_g[3, 5] = (2.0 * My1 - My2) / 6.0
        k_g[3, 7] = -My1 / L
        k_g[3, 8] = -Mz1 / L
        k_g[3, 9] = -Fx2 * self.I_p  / (self.A * L)
        k_g[3, 10] = -1.0 * (Mz1 + Mz2) / 6.0
        k_g[3, 11] = (My1 + My2) / 6.0
        k_g[4, 7] = -Mx2 / L
        k_g[4, 8] = Fx2 / 10.0
        k_g[4, 9] = -1.0 * (Mz1 + Mz2) / 6.0
        k_g[4, 10] = -Fx2 * L / 30.0
        k_g[4, 11] = Mx2 / 2.0
        k_g[5, 7] = -Fx2 / 10.0
        k_g[5, 8] = -Mx2 / L
        k_g[5, 9] = (My1 + My2) / 6.0
        k_g[5, 10] = -Mx2 / 2.0
        k_g[5, 11] = -Fx2 * L / 30.0
        k_g[7, 9] = -My2 / L
        k_g[7, 10] = Mx2 / L
        k_g[7, 11] = -Fx2 / 10.0
        k_g[8, 9] = -Mz2 / L
        k_g[8, 10] = Fx2 / 10.0
        k_g[8, 11] = Mx2 / L
        k_g[9, 10] = (Mz1 - 2.0 * Mz2) / 6.0
        k_g[9, 11] = -1.0 * (My1 - 2.0 * My2) / 6.0
        # add in the symmetric lower triangle
        k_g = k_g + k_g.transpose()
        # add diagonal terms
        k_g[0, 0] = Fx2 / L
        k_g[1, 1] = 6.0 * Fx2 / (5.0 * L)
        k_g[2, 2] = 6.0 * Fx2 / (5.0 * L)
        k_g[3, 3] = Fx2 * self.I_p  / (self.A * L)
        k_g[4, 4] = 2.0 * Fx2 * L / 15.0
        k_g[5, 5] = 2.0 * Fx2 * L / 15.0
        k_g[6, 6] = Fx2 / L
        k_g[7, 7] = 6.0 * Fx2 / (5.0 * L)
        k_g[8, 8] = 6.0 * Fx2 / (5.0 * L)
        k_g[9, 9] = Fx2 * self.I_p  / (self.A * L)
        k_g[10, 10] = 2.0 * Fx2 * L / 15.0
        k_g[11, 11] = 2.0 * Fx2 * L / 15.0
        
        return k_g

    def local_elastic_stiffness_matrix_3D_beam(self) -> np.ndarray:
        """
        local element elastic stiffness matrix
        source: p. 73 of McGuire's Matrix Structural Analysis 2nd Edition
        Given:
            material and geometric parameters:
                A, L, Iy, Iz, J, nu, E
        Context:
            load vector:
                [Fx1, Fy1, Fz1, Mx1, My1, Mz1, Fx2, Fy2, Fz2, Mx2, My2, Mz2]
            DOF vector:
                [u1, v1, w1, th_x1, th_y1, th_z1, u2, v2, w2, th_x2, th_y2, th_z2]
            Equation:
                [load vector] = [stiffness matrix] @ [DOF vector]
        Returns:
            12 x 12 elastic stiffness matrix k_e
        """
        L = np.linalg.norm(self.node1.get_coordinates() - self.node2.get_coordinates())
        k_e = np.zeros((12, 12))
        # Axial terms - extension of local x axis
        axial_stiffness = self.E * self.A / L
        k_e[0, 0] = axial_stiffness
        k_e[0, 6] = -axial_stiffness
        k_e[6, 0] = -axial_stiffness
        k_e[6, 6] = axial_stiffness
        # Torsion terms - rotation about local x axis
        torsional_stiffness = self.E * self.J / (2.0 * (1 + self.v) * L)
        k_e[3, 3] = torsional_stiffness
        k_e[3, 9] = -torsional_stiffness
        k_e[9, 3] = -torsional_stiffness
        k_e[9, 9] = torsional_stiffness
        # Bending terms - bending about local z axis
        k_e[1, 1] = self.E * 12.0 * self.I_z / L ** 3.0
        k_e[1, 7] = self.E * -12.0 * self.I_z / L ** 3.0
        k_e[7, 1] = self.E * -12.0 * self.I_z / L ** 3.0
        k_e[7, 7] = self.E * 12.0 * self.I_z / L ** 3.0
        k_e[1, 5] = self.E * 6.0 * self.I_z / L ** 2.0
        k_e[5, 1] = self.E * 6.0 * self.I_z / L ** 2.0
        k_e[1, 11] = self.E * 6.0 * self.I_z / L ** 2.0
        k_e[11, 1] = self.E * 6.0 * self.I_z / L ** 2.0
        k_e[5, 7] = self.E * -6.0 * self.I_z / L ** 2.0
        k_e[7, 5] = self.E * -6.0 * self.I_z / L ** 2.0
        k_e[7, 11] = self.E * -6.0 * self.I_z / L ** 2.0
        k_e[11, 7] = self.E * -6.0 * self.I_z / L ** 2.0
        k_e[5, 5] = self.E * 4.0 * self.I_z / L
        k_e[11, 11] = self.E * 4.0 * self.I_z / L
        k_e[5, 11] = self.E * 2.0 * self.I_z / L
        k_e[11, 5] = self.E * 2.0 * self.I_z / L
        # Bending terms - bending about local y axis
        k_e[2, 2] = self.E * 12.0 * self.I_y / L ** 3.0
        k_e[2, 8] = self.E * -12.0 * self.I_y / L ** 3.0
        k_e[8, 2] = self.E * -12.0 * self.I_y / L ** 3.0
        k_e[8, 8] = self.E * 12.0 * self.I_y / L ** 3.0
        k_e[2, 4] = self.E * -6.0 * self.I_y / L ** 2.0
        k_e[4, 2] = self.E * -6.0 * self.I_y / L ** 2.0
        k_e[2, 10] = self.E * -6.0 * self.I_y / L ** 2.0
        k_e[10, 2] = self.E * -6.0 * self.I_y / L ** 2.0
        k_e[4, 8] = self.E * 6.0 * self.I_y / L ** 2.0
        k_e[8, 4] = self.E * 6.0 * self.I_y / L ** 2.0
        k_e[8, 10] = self.E * 6.0 * self.I_y / L ** 2.0
        k_e[10, 8] = self.E * 6.0 * self.I_y / L ** 2.0
        k_e[4, 4] = self.E * 4.0 * self.I_y / L
        k_e[10, 10] = self.E * 4.0 * self.I_y / L
        k_e[4, 10] = self.E * 2.0 * self.I_y / L
        k_e[10, 4] = self.E * 2.0 * self.I_y / L
        return k_e
    

    def rotation_matrix_3D(self) -> np.ndarray:
        """
        3D rotation matrix
        source: Chapter 5.1 of McGuire's Matrix Structural Analysis 2nd Edition
        Given:
            optional: reference z vector direction v_temp to orthonormalize the local y and z axis
                if v_temp is not given, VVVV
        Compute:
            where l, m, n are defined as direction cosines:
            gamma = [[lx'=cos alpha_x', mx'=cos beta_x', nx'=cos gamma_x'],
                    [ly'=cos alpha_y', my'=cos beta_y', ny'=cos gamma_y'],
                    [lz'=cos alpha_z', mz'=cos beta_z', nz'=cos gamma_z']]
        """
        x1, y1, z1 = self.node1.get_coordinates()
        x2, y2, z2 = self.node2.get_coordinates()
        L = np.sqrt((x2 - x1) ** 2.0 + (y2 - y1) ** 2.0 + (z2 - z1) ** 2.0)
        lxp = (x2 - x1) / L
        mxp = (y2 - y1) / L
        nxp = (z2 - z1) / L
        local_x = np.asarray([lxp, mxp, nxp])

        # Use the provided local_z_axis
        v_temp = self.local_z_axis
        check_unit_vector(v_temp)
        check_parallel(local_x, v_temp)
        
        # Compute the local y-axis
        local_y = np.cross(v_temp, local_x)
        local_y = local_y / np.linalg.norm(local_y)

        # Compute the local z-axis
        local_z = np.cross(local_x, local_y)
        local_z = local_z / np.linalg.norm(local_z)

        # Assemble rotation matrix
        gamma = np.vstack((local_x, local_y, local_z))
        
        return gamma
    
    def transformation_matrix_3D(self, gamma: np.ndarray) -> np.ndarray:
        """
        3D transformation matrix
        source: Chapter 5.1 of McGuire's Matrix Structural Analysis 2nd Edition
        Given:
            gamma -- the 3x3 rotation matrix
        Compute:
            Gamma -- the 12x12 transformation matrix
        """
        Gamma = np.zeros((12, 12))
        Gamma[0:3, 0:3] = gamma
        Gamma[3:6, 3:6] = gamma
        Gamma[6:9, 6:9] = gamma
        Gamma[9:12, 9:12] = gamma
        return Gamma


class Frame():
    def __init__(self, nodes, elements):
        self.nodes = nodes
        self.elements = elements

    def local_stiffness_matrix(self, element):
        return element.local_elastic_stiffness_matrix_3D_beam()
    
    def global_stiffness_matrix(self):
        """
        Variables:
                    k_e -- local element stiffness matrix
                    k_g -- global element stiffness matrix
                    K -- global stiffness matrix
        Returns:    
                    K -- global stiffness
        """
        num_dofs = len(self.nodes) * 6  # Total number of DOFs
        K = np.zeros((num_dofs, num_dofs))  # Initialize global stiffness matrix

        for element in self.elements:
            gamma = element.rotation_matrix_3D()
            Transformation_Matrix_Gamma = element.transformation_matrix_3D(gamma)
            k_e = self.local_stiffness_matrix(element)
            k_g = Transformation_Matrix_Gamma.T @ k_e @ Transformation_Matrix_Gamma

            # Get node indices dynamically
            idx1 = self.nodes.index(element.node1)
            idx2 = self.nodes.index(element.node2)

            # Convert to DOF indices
            dof1, dof2 = idx1 * 6, idx2 * 6

            # Assemble global stiffness matrix
            K[dof1:dof1+6, dof1:dof1+6] += k_g[0:6, 0:6]
            K[dof1:dof1+6, dof2:dof2+6] += k_g[0:6, 6:12]
            K[dof2:dof2+6, dof1:dof1+6] += k_g[6:12, 0:6]
            K[dof2:dof2+6, dof2:dof2+6] += k_g[6:12, 6:12]

        return K
    
    def global_force_matrix(self):
        """
        Assembles the global force matrix F by extracting nodal loads.
        Returns:
            F (np.ndarray): The global force matrix.
        """
        F = np.zeros(len(self.nodes) * 6)  # Initialize force matrix

        # Assemble global force vector
        for i, node in enumerate(self.nodes):
            nodal_loads = node.get_nodal_load()  # Get [Fx, Fy, Fz, Mx, My, Mz]
            F[i * 6: i * 6 + 6] = nodal_loads  # Assign to correct indices

        return F  # Return the assembled global force matrix


    def get_displacements(self, F, K):
        """
        Variables:
                    K -- global stiffness matrix
                    F -- global force matrix
                    U -- global displacement matrix
        Returns:
                    U -- global displacement matrix
        """
        U = np.linalg.solve(K, F)
        
        for i, node in enumerate(self.nodes):
            node.u = U[i * 6]
            node.v = U[i * 6 + 1]
            node.w = U[i * 6 + 2]
            node.theta_x = U[i * 6 + 3]
            node.theta_y = U[i * 6 + 4]
            node.theta_z = U[i * 6 + 5]

        return U  # Return displacement vector


    def get_reactions(self, U, original_K, F):
        """
        Variables:
                    K -- global stiffness matrix
                    U -- global displacement matrix
                    F -- global force matrix
        Returns:
                    R -- global reaction matrix
        """
      
        R = original_K  @ U - F  # Compute full reaction force vector
        
        # Zero out reactions at free DOFs
        for i, node in enumerate(self.nodes):
            for dof in range(6):  # Loop through all 6 DOFs per node
                if not node.boundary_conditions[dof]:  # If DOF is not constrained
                    R[i * 6 + dof] = 0  # Ensure reaction force is zero

        return R  # Return reaction forces/moments at supports
    

    def calculations(self):
        original_K = self.global_stiffness_matrix()  # Save unmodified stiffness matrix
        K = original_K.copy()  # Copy to apply boundary conditions
        F = self.global_force_matrix()

        # Apply boundary conditions
        for i, node in enumerate(self.nodes):
            for dof, constrained in enumerate(node.boundary_conditions):
                if constrained:
                    index = i * 6 + dof
                    K[index, :] = 0
                    K[:, index] = 0
                    K[index, index] = 1  # Keep diagonal entry for numerical stability
                    F[index] = 0  # Ensure force vector correctly handles constraints


        U = self.get_displacements(F, K)        
        R = self.get_reactions(U, original_K, F)  # Use original K
    
        return U, R
    
    def compute_internal_forces(self):
        """
        Compute internal forces and moments in local coordinate system, including elastic and geometric stiffness.
        """

        internal_forces = []  # Initialize list to store internal forces for each element

        for element in self.elements:
            gamma = element.rotation_matrix_3D()
            Transformation_Matrix_Gamma = element.transformation_matrix_3D(gamma)
            k_e_local = element.local_elastic_stiffness_matrix_3D_beam()
            k_g_local = element.local_geometric_stiffness_matrix_3D_beam()

            # Transform local stiffness matrices to global coordinate system
            k_e_global = Transformation_Matrix_Gamma.T @ k_e_local @ Transformation_Matrix_Gamma
            k_g_global = Transformation_Matrix_Gamma.T @ k_g_local @ Transformation_Matrix_Gamma

            # Get displacements for the element
            displacements = np.concatenate((element.node1.get_displacements(), element.node2.get_displacements()))

            # Compute internal forces in global coordinate system
            internal_forces_global = k_e_global @ displacements + k_g_global @ displacements

            # Transform internal forces to local coordinate system
            internal_forces_local = Transformation_Matrix_Gamma @ internal_forces_global

            # Append internal forces for the element to the list
            internal_forces.append(internal_forces_local)

        return internal_forces
    

    def plot(self):
        """
        Plot original shape.
        """
        from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot each element in 3D
        for element in self.elements:
            x_coords = [element.node1.x, element.node2.x]
            y_coords = [element.node1.y, element.node2.y]
            z_coords = [element.node1.z, element.node2.z]
            ax.plot(x_coords, y_coords, z_coords, marker='o')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Frame Structure (Original)')
        plt.show()

    def plot_deformed(self):
        """
        Plot deformed shape.
        """
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot each element in 3D
        for element in self.elements:
            x_coords = [element.node1.x + element.node1.u, element.node2.x + element.node2.u]
            y_coords = [element.node1.y + element.node1.v, element.node2.y + element.node2.v]
            z_coords = [element.node1.z + element.node1.w, element.node2.z + element.node2.w]
            ax.plot(x_coords, y_coords, z_coords, marker='o')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Frame Structure (Deformed)')
        plt.show()

    def plot_internal_forces(self):
        """
        Plot internal forces and moments for each element.
        """
        internal_forces = self.compute_internal_forces()
        fig, ax = plt.subplots()

        for i, forces in enumerate(internal_forces):
            ax.plot(np.arange(len(forces)), forces, label=f'Element {i}', marker='o')

        ax.set_xlabel('DOF')
        ax.set_ylabel('Internal Force/Moment')
        ax.legend()
        plt.show()