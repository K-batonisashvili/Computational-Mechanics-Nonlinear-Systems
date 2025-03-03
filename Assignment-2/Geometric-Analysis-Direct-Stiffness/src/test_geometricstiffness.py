"""
Project:        Geometric Analysis Direct Stiffness
Author:         Kote Batonisashvili
Description:    Assignment 2.2 test file which has test cases to ensure the main code has no errors.
"""

import pytest
import numpy as np
import scipy
from scipy.linalg import eigh
from Direct_Stiffness import Nodes, Elements, Frame

def test_node_initialization():
    """Test that nodes are initialized correctly."""
    node = Nodes(1.0, 2.0, 3.0)
    assert np.array_equal(node.get_coordinates(), np.array([1.0, 2.0, 3.0])), "Node coordinates are incorrect"
    assert np.array_equal(node.get_nodal_load(), np.zeros(6)), "Node loads are not initialized to zero"
    assert np.array_equal(node.get_displacements(), np.zeros(6)), "Node displacements are not initialized to zero"

def test_element_initialization():
    """Test that elements are initialized correctly along with their nodes and youngs modulus specifically."""
    node1 = Nodes(0.0, 0.0, 0.0)
    node2 = Nodes(1.0, 0.0, 0.0)
    element = Elements(node1, node2, 200e9, 0.3, 0.01, 0.01, 0.01, 0.01, 0.01, [0, 0, 1])
    assert element.node1 == node1, "Element node1 is incorrect"
    assert element.node2 == node2, "Element node2 is incorrect"
    assert element.E == 200e9, "Element Young's Modulus is incorrect"

def test_global_stiffness_matrix():
    """Test the global stiffness matrix calculations and ensure that the shape is 12x12."""
    nodes = [Nodes(0, 0, 0), Nodes(1, 0, 0)]
    elements = [Elements(nodes[0], nodes[1], 200e9, 0.3, 0.01, 0.01, 0.01, 0.01, 0.01, [0, 0, 1])]
    frame = Frame(nodes, elements)
    K = frame.global_stiffness_matrix()
    assert K.shape == (12, 12), "Global stiffness matrix size is incorrect"

def test_boundary_conditions():
    """Test that boundary conditions are applied correctly."""
    node = Nodes(0.0, 0.0, 0.0)
    node.set_boundary_constraints([True, True, True, True, True, True])
    assert np.array_equal(node.boundary_conditions, [True, True, True, True, True, True]), "Boundary conditions are not set correctly"

def test_displacements():
    """Test the displacement calculation."""
    nodes = [Nodes(0, 0, 0), Nodes(1, 0, 0)]
    elements = [Elements(nodes[0], nodes[1], 200e9, 0.3, 0.01, 0.01, 0.01, 0.01, 0.01, [0, 0, 1])]
    frame = Frame(nodes, elements)
    nodes[0].set_boundary_constraints([True, True, True, True, True, True])
    nodes[1].set_boundary_constraints([False, True, True, False, False, False])
    nodes[1].set_nodal_load(1000, 0, 0, 0, 0, 0)
    K = frame.global_stiffness_matrix()
    F = frame.global_force_matrix()

    # Apply boundary conditions
    for i, node in enumerate(nodes):
        for dof, constrained in enumerate(node.boundary_conditions):
            if constrained:
                index = i * 6 + dof
                K[index, :] = 0
                K[:, index] = 0
                K[index, index] = 1  # Keep diagonal entry for numerical stability
                F[index] = 0  # Ensure force vector correctly handles constraints

    U = frame.get_displacements(F, K)
    assert U.shape == (12,), "Displacement vector size is incorrect"

def test_reactions():
    """Test the reaction force calculation."""
    nodes = [Nodes(0, 0, 0), Nodes(1, 0, 0)]
    elements = [Elements(nodes[0], nodes[1], 200e9, 0.3, 0.01, 0.01, 0.01, 0.01, 0.01, [0, 0, 1])]
    frame = Frame(nodes, elements)
    nodes[0].set_boundary_constraints([True, True, True, True, True, True])
    nodes[1].set_boundary_constraints([False, True, True, False, False, False])
    nodes[1].set_nodal_load(1000, 0, 0, 0, 0, 0)
    K = frame.global_stiffness_matrix()
    F = frame.global_force_matrix()

    # Apply boundary conditions
    for i, node in enumerate(nodes):
        for dof, constrained in enumerate(node.boundary_conditions):
            if constrained:
                index = i * 6 + dof
                K[index, :] = 0
                K[:, index] = 0
                K[index, index] = 1  # Keep diagonal entry for numerical stability
                F[index] = 0  # Ensure force vector correctly handles constraints

    U = frame.get_displacements(F, K)
    R = frame.get_reactions(U, K, F)
    assert R.shape == (12,), "Reaction force vector size is incorrect"

# Define nodes with coordinates (x, y, z)
node1 = Nodes(0.0, 0.0, 0.0)
node2 = Nodes(1.0, 0.0, 0.0)
node3 = Nodes(1.0, 1.0, 0.0)

# Set boundary conditions (e.g., fixed at node1)
node1.set_boundary_constraints(np.array([True, True, True, True, True, True]))
node2.set_boundary_constraints(np.array([False, False, False, False, False, False]))
node3.set_boundary_constraints(np.array([False, False, False, False, False, False]))


# List of nodes
nodes = [node1, node2, node3]

# Define material and geometric properties
E = 500  # Young's Modulus in Pascals
v = 0.3    # Poisson's Ratio
A = 0.01   # Cross-sectional area in square meters
I_z = 1e-6 # Moment of inertia about z-axis in meters^4
I_y = 1e-6 # Moment of inertia about y-axis in meters^4
I_p = 2e-6 # Polar moment of inertia in meters^4
J = 1e-6   # Torsional constant in meters^4
local_z_axis = np.array([0.0, 0.0, 1.0])  # Local z-axis

# Create elements
element1 = Elements(node1, node2, E, v, A, I_z, I_y, I_p, J, local_z_axis)
element2 = Elements(node2, node3, E, v, A, I_z, I_y, I_p, J, local_z_axis)

# List of elements
elements = [element1, element2]

def test_symmetry():
    frame = Frame(nodes, elements)
    K = frame.global_stiffness_matrix()
    Kg = frame.global_geometric_stiffness_matrix()
    assert np.allclose(K, K.T), "Global stiffness matrix K is not symmetric"
    assert np.allclose(Kg, Kg.T), "Global geometric stiffness matrix Kg is not symmetric"

def test_positive_definiteness():
    frame = Frame(nodes, elements)
    K = frame.global_stiffness_matrix()
    try:
        np.linalg.cholesky(K)
        print("Global stiffness matrix K is positive definite")
    except np.linalg.LinAlgError:
        assert False, "Global stiffness matrix K is not positive definite"

def test_positive_definiteness_global_matrix():
    frame = Frame(nodes, elements)
    K = frame.global_stiffness_matrix()
    Kg = frame.global_geometric_stiffness_matrix()
    K_global = np.block([[K, np.zeros_like(K)], [np.zeros_like(K), Kg]])
    try:
        np.linalg.cholesky(Kg)
        print("Global matrix [K, Kg] is positive definite")
    except np.linalg.LinAlgError:
        assert False, "Global matrix [K, Kg] is not positive definite"

def test_boundary_conditions():
    frame = Frame(nodes, elements)
    K = frame.global_stiffness_matrix()
    Kg = frame.global_geometric_stiffness_matrix()
    for i, node in enumerate(frame.nodes):
        for dof, constrained in enumerate(node.boundary_conditions):
            if constrained:
                index = i * 6 + dof
                assert K[index, index] == 1, "Boundary condition not correctly applied to K"
                assert Kg[index, index] == 1, "Boundary condition not correctly applied to Kg"

def test_eigenvalue_problem():
    frame = Frame(nodes, elements)
    K = frame.global_stiffness_matrix()
    Kg = frame.global_geometric_stiffness_matrix()
    
    # Ensure matrices are not empty and have the correct dimensions
    assert K.size > 0, "Global stiffness matrix K is empty"
    assert Kg.size > 0, "Global geometric stiffness matrix Kg is empty"
    assert K.shape == Kg.shape, "Matrices K and Kg must have the same dimensions"
    
    try:
        eigenvalues, eigenvectors = scipy.linalg.eig(K, -Kg)
        assert np.all(eigenvalues > 0), "Eigenvalues should be positive"
    except np.linalg.LinAlgError as e:
        assert False, f"Eigenvalue computation failed: {e}"

def test_critical_load_factor():
    frame = Frame(nodes, elements)
    critical_load_factor = frame.elastic_critical_load()
    assert critical_load_factor > 0, "Critical load factor should be positive"

def test_transformation_matrix():
    element = elements[0]  # Test with the first element
    gamma = element.rotation_matrix_3D()
    Transformation_Matrix_Gamma = element.transformation_matrix_3D(gamma)
    assert Transformation_Matrix_Gamma.shape == (12, 12), "Transformation matrix should be 12x12"

def test_internal_forces():
    frame = Frame(nodes, elements)
    internal_forces = frame.compute_internal_forces()
    for forces in internal_forces:
        assert forces.shape == (12,), "Internal forces vector should have 12 components"

if __name__ == "__main__":
    pytest.main()
