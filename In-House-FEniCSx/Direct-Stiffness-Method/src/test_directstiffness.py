"""
Project:        Direct-Stifness Matrix Method
Author:         Kote Batonisashvili
Description:    Assignment 2.1 test file which has test cases to ensure the main code has no errors.
"""

import pytest
import numpy as np
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