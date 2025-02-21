"""
Project:        Direct-Stifness Matrix Method
Author:         Kote Batonisashvili
Description:    Assignment 2.1. Direct-Stifness Matrix Method which creates Frames with their respective Nodes and Elements.
"""

# Standard Imports
import numpy as np
import matplotlib.pyplot as plt
from math_utils import * 


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

    def __nodal_load__(self, Fx, Fy, Fz, Mx, My, Mz):
        self.Mz = Mz
        self.My = My
        self.Mx = Mx
        self.Fx = Fx
        self.Fy = Fy
        self.Fz = Fz

    def __nodal_displacement__(self, u, v, w, theta_x, theta_y, theta_z):
        self.u = u
        self.v = v
        self.w = w
        self.theta_x = theta_x
        self.theta_y = theta_y
        self.theta_z = theta_z

    def get_nodal_load(self):
        return np.array([self.Fx, self.Fy, self.Fz, self.Mx, self.My, self.Mz])
    
    def get_coordinates(self):
        return np.array([self.x, self.y, self.z])
    
    def get_displacements(self):
        return np.array([self.u, self.v, self.w, self.theta_x, self.theta_y, self.theta_z])

    def set_boundary_constraints(self, constraints: np.ndarray = None):
        if constraints is None:
            return np.array([False, False, False, False, False, False])
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
    

    def rotation_matrix_3D(self, v_temp: np.ndarray = None) -> np.ndarray:
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

        # choose a vector to orthonormalize the y axis if one is not given
        if v_temp is None:
            # if the beam is oriented vertically, switch to the global y axis
            if np.isclose(lxp, 0.0) and np.isclose(mxp, 0.0):
                v_temp = np.array([0, 1.0, 0.0])
            else:
                # otherwise use the global z axis
                v_temp = np.array([0, 0, 1.0])
        else:
            # check to make sure that given v_temp is a unit vector
            check_unit_vector(v_temp)
            # check to make sure that given v_temp is not parallel to the local x axis
            check_parallel(local_x, v_temp)
        
        # compute the local y axis
        local_y = np.cross(v_temp, local_x)
        local_y = local_y / np.linalg.norm(local_y)

        # compute the local z axis
        local_z = np.cross(local_x, local_y)
        local_z = local_z / np.linalg.norm(local_z)

        # assemble R
        gamma = np.vstack((local_x, local_y, local_z))
        
        return gamma
    
    def transformation_matrix_3D(gamma: np.ndarray) -> np.ndarray:
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
        K = np.zeros((len(self.nodes) * 6, len(self.nodes) * 6))
        for element in self.elements:
            gamma = element.rotation_matrix_3D()
            Transofrmation_Matrix_Gamma = transformation_matrix_3D(gamma)
            k_e = self.local_stiffness_matrix(element)
            k_g = Transofrmation_Matrix_Gamma.T @ k_e @ Transofrmation_Matrix_Gamma
            K[element.node1 * 6:element.node1 * 6 + 6, element.node1 * 6:element.node1 * 6 + 6] += k_g[0:6, 0:6]
            K[element.node1 * 6:element.node1 * 6 + 6, element.node2 * 6:element.node2 * 6 + 6] += k_g[0:6, 6:12]
            K[element.node2 * 6:element.node2 * 6 + 6, element.node1 * 6:element.node1 * 6 + 6] += k_g[6:12, 0:6]
            K[element.node2 * 6:element.node2 * 6 + 6, element.node2 * 6:element.node2 * 6 + 6] += k_g[6:12, 6:12]
        
        # Apply boundary conditions
        for i, node in enumerate(self.nodes):
            for dof, constrained in enumerate(node.boundary_conditions):
                if constrained:
                    index = i * 6 + dof
                    K[index, :] = 0
                    K[:, index] = 0
                    K[index, index] = 1
        
        return K

    def plot(self):
        _, ax = plt.subplots()
        for element in self.elements:
            ax.plot([element.node1.x, element.node2.x], [element.node1.y, element.node2.y], marker='o')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Frame Structure')
        plt.grid(True)
        plt.show()

def main(frame):

    frame.nodes[0].set_boundary_conditions([True, True, True, True, True, True])  # Node 0, all DOFs constrained
    frame.nodes[1].set_boundary_conditions([False, True, True, False, False, False])  # Node 1, some DOFs constrained
    frame.nodes[2].set_boundary_conditions([False, True, True, False, False, False])  # Node 2, some DOFs constrained

    K = frame.global_stiffness_matrix()
    K.plot()

