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

        # Apply boundary conditions
        for i, node in enumerate(self.nodes):
            for dof, constrained in enumerate(node.boundary_conditions):
                if constrained:
                    index = i * 6 + dof
                    K[index, :] = 0
                    K[:, index] = 0
                    K[index, index] = 1  # Set to 1 for stability

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


    def get_reactions(self, U, K, F):
        """
        Variables:
                    K -- global stiffness matrix
                    U -- global displacement matrix
                    F -- global force matrix
                    R -- global reaction matrix
        Returns:
                    R -- global reaction matrix
        """
      
        R = np.zeros_like(F)  # Initialize reaction force vector

        # Extract reaction forces only at constrained DOFs
        for i, node in enumerate(self.nodes):
            for dof in range(6):  # Loop through 6 DOFs per node
                if node.boundary_conditions[dof]:  # If DOF is constrained
                    index = i * 6 + dof
                    R[index] = F[index]  # Store reaction force/moment

        return R  # Return reaction forces/moments at supports
    

    def calculations(self):
        K = self.global_stiffness_matrix()
        print(f"global stifness matrix: {K}")
        F = self.global_force_matrix()
        print(f"global force matrix: {F}")
        U = self.get_displacements(F,K)
        print(f"displacements: {U}")
        R = self.get_reactions(U, K, F)
        return U, R
    

    def plot(self):
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
        ax.set_title('3D Frame Structure')
        plt.show()

def manual_input():
    """
    This method allows the user to have pop-up dialog options to manually input values for the nodes and elements.    
    """

     # Define Nodes
    while True:
        try:
            num_of_nodes = int(input("Enter number of Nodes: "))
            break
        except ValueError:
            print("Invalid input. Please enter an integer value for the number of nodes.")

    nodes = []

    for i in range(num_of_nodes):
        while True:
            try:
                x = float(input(f"Enter x-coordinate for Node {i}: "))
                y = float(input(f"Enter y-coordinate for Node {i}: "))
                z = float(input(f"Enter z-coordinate for Node {i}: "))
                nodes.append(Nodes(x, y, z))
                break
            except ValueError:
                print("Invalid input. Please enter numeric values for the coordinates.")

    # Define Elements
    elements = []
    for i in range(2):
        print(f"Nodes start from 0 to {num_of_nodes - 1}, please enter accordingly.")
        while True:
            try:
                node1 = nodes[int(input(f"Please enter Node 1 for Element {i}: "))]
                node2 = nodes[int(input(f"Please enter Node 2 for Element {i}: "))]
                break
            except (ValueError, IndexError):
                print("Invalid input. Please enter valid node indices.")

        while True:
            try:
                E = float(input(f"Enter Young's Modulus for Element {i}: "))
                v = float(input(f"Enter Poisson's Ratio for Element {i}: "))
                A = float(input(f"Enter Area for Element {i}: "))
                I_z = float(input(f"Enter Inertia about z-axis for Element {i}: "))
                I_y = float(input(f"Enter Inertia about y-axis for Element {i}: "))
                I_p = float(input(f"Enter Polar Moment of Inertia for Element {i}: "))
                J = float(input(f"Enter Torsional Constant for Element {i}: "))
                local_z_axis = [float(x) for x in input(f"Enter Local z-axis for Element {i}: ").split()]
                elements.append(Elements(node1, node2, E, v, A, I_z, I_y, I_p, J, local_z_axis))
                break
            except ValueError:
                print("Invalid input. Please enter numeric values for the element properties.")

    # Set Nodal Loads
    for i in range(num_of_nodes):
        while True:
            try:
                print("For boundary conditions, enter 1 for constrained, or 0 for not. Your input will look like this: 1 0 0 0 0 0")
                constraints = [bool(int(x)) for x in input(f"Enter Boundary Constraints for Node {i} (0 or 1 for each DOF): ").split()]
                if len(constraints) != 6:
                    raise ValueError("You must enter exactly 6 values.")
                nodes[i].set_boundary_constraints(constraints)
                break
            except ValueError:
                print("Invalid input. Please enter 6 binary values (0 or 1) for the boundary constraints.")

    return nodes, elements



def main():

    # # User Input for Nodes, Elements, and Boundary Conditions
    # nodes, elements = manual_input()

    # Define Nodes
    nodes = []
    nodes.append(Nodes(0.0, 0.0, 0.0))
    nodes.append(Nodes(-5.0, 1.0, 10.0))
    nodes.append(Nodes(-1.0, 5.0, 13.0))
    nodes.append(Nodes(-3.0, 7.0, 11.0))
    nodes.append(Nodes(6.0, 9.0, 5.0))

    # Apply boundary conditions
    nodes[0].set_boundary_constraints([False, False, True, True, True, True])  # Node 1 fully constrained
    nodes[1].set_boundary_constraints([False, False, False, False, False, False])  # Node 2 constrained in Y, Z
    nodes[2].set_boundary_constraints([False, False, False, False, False, False])  # Node 2 constrained in Y, Z
    nodes[3].set_boundary_constraints([True, True, True, True, True, True])  # Node 2 constrained in Y, Z
    nodes[4].set_boundary_constraints([True, True, True, False, False, False])  # Node 2 constrained in Y, Z


    nodes[2].set_nodal_load(0.1, -0.05, -0.075, 0.5, -0.1, 0.3)

    # Define Elements
    E = 500  # Young's Modulus in Pascals
    v = 0.3  # Poisson's Ratio
    A = (np.pi)  # Cross-sectional Area in square meters
    I_z = (np.pi/4)  # Moment of Inertia about z-axis in meters^4
    I_y = (np.pi/4)  # Moment of Inertia about y-axis in meters^4
    I_p = (np.pi/2)  # Polar Moment of Inertia in meters^4
    J = (np.pi/2)  # Torsional Constant in meters^4
    local_z_axis = [0, 0, 1]  # Local z-axis direction

    elements = []
    elements.append(Elements(nodes[0], nodes[1], E, v, A, I_z, I_y, I_p, J, local_z_axis))
    elements.append(Elements(nodes[1], nodes[2], E, v, A, I_z, I_y, I_p, J, local_z_axis))
    elements.append(Elements(nodes[3], nodes[2], E, v, A, I_z, I_y, I_p, J, local_z_axis))
    elements.append(Elements(nodes[2], nodes[4], E, v, A, I_z, I_y, I_p, J, local_z_axis))


    # Create Frame and Compute Results
    frame = Frame(nodes, elements)
    U, R = frame.calculations()

    # Output Results
    print("\nNodal Displacements & Rotations:")
    for i, node in enumerate(nodes):
        print(f"Node {i}: u={U[i * 6]:.6f}, v={U[i * 6 + 1]:.6f}, w={U[i * 6 + 2]:.6f}, "
                f"θx={U[i * 6 + 3]:.6f}, θy={U[i * 6 + 4]:.6f}, θz={U[i * 6 + 5]:.6f}")

    print("\nReaction Forces & Moments at Supports:")
    for i, node in enumerate(nodes):
        if any(node.boundary_conditions):  # Only print for constrained nodes
            print(f"Node {i}: Fx={R[i * 6]:.2f}, Fy={R[i * 6 + 1]:.2f}, Fz={R[i * 6 + 2]:.2f}, "
                    f"Mx={R[i * 6 + 3]:.2f}, My={R[i * 6 + 4]:.2f}, Mz={R[i * 6 + 5]:.2f}")
            
    frame.plot()

if __name__ == "__main__":
    main()