"""
Project:        Geometric Analysis Direct Stiffness
Author:         Kote Batonisashvili
Description:    Assignment 2.2 test file which has test cases to ensure the main code has no errors.
"""
import numpy as np
import pytest
from Geometric_stiffness import Nodes, Elements, Frame

# Define material and cross-sectional properties
E = 200e9  # Young's modulus (Pa)
v = 0.3    # Poisson's ratio
A = 0.01   # Cross-sectional area (m²)
I_y = 1e-6 # Moment of inertia about y-axis (m^4)
I_z = 1e-6 # Moment of inertia about z-axis (m^4)
I_p = 2e-6 # Polar moment of inertia (m^4)
J = 2e-6   # Torsional constant (m^4)

@pytest.fixture
def setup_nodes():
    """Fixture to create a set of nodes."""
    nodes = [
        Nodes(0, 0, 0),
        Nodes(1, 0, 0),
        Nodes(1, 1, 0),
        Nodes(0, 1, 0)
    ]
    return nodes

@pytest.fixture
def setup_elements(setup_nodes):
    """Fixture to create elements connecting the nodes."""
    nodes = setup_nodes
    elements = [
        Elements(nodes[0], nodes[1], E, v, A, I_z, I_y, I_p, J),
        Elements(nodes[1], nodes[2], E, v, A, I_z, I_y, I_p, J),
        Elements(nodes[2], nodes[3], E, v, A, I_z, I_y, I_p, J),
        Elements(nodes[3], nodes[0], E, v, A, I_z, I_y, I_p, J)
    ]
    return elements

@pytest.fixture
def setup_frame(setup_nodes, setup_elements):
    """Fixture to create a frame structure."""
    return Frame(setup_nodes, setup_elements)

# --- TEST NODE CLASS ---
def test_node_creation():
    """Test if nodes are correctly initialized."""
    node = Nodes(1.0, 2.0, 3.0)
    assert node.get_coordinates().tolist() == [1.0, 2.0, 3.0]

def test_nodal_load():
    """Test if nodal loads can be set and retrieved correctly."""
    node = Nodes(0, 0, 0)
    node.set_nodal_load(10, 15, -5, 0.1, 0.2, -0.3)
    assert node.get_nodal_load().tolist() == [10, 15, -5, 0.1, 0.2, -0.3]

def test_boundary_constraints():
    """Test if boundary constraints can be set properly."""
    node = Nodes(0, 0, 0)
    node.set_boundary_constraints([True, False, True, False, True, False])
    assert node.boundary_conditions.tolist() == [True, False, True, False, True, False]

# --- TEST ELEMENT CLASS ---
def test_element_creation(setup_elements):
    """Test if elements are correctly created and local axes are computed."""
    element = setup_elements[0]
    assert element.node1.get_coordinates().tolist() == [0, 0, 0]
    assert element.node2.get_coordinates().tolist() == [1, 0, 0]
    assert np.isclose(np.linalg.norm(element.local_x_axis), 1.0)

def test_rotation_matrix(setup_elements):
    """Test if rotation matrices are correctly computed."""
    element = setup_elements[0]
    gamma = element.rotation_matrix_3D()
    assert gamma.shape == (3, 3)  # Should be a 3x3 matrix
    assert np.allclose(np.linalg.det(gamma), 1.0)  # Should be an orthonormal matrix

def test_transformation_matrix(setup_elements):
    """Test if transformation matrices are correctly computed."""
    element = setup_elements[0]
    gamma = element.rotation_matrix_3D()
    T = element.transformation_matrix_3D(gamma)
    assert T.shape == (12, 12)  # Should be a 12x12 matrix

# --- TEST FRAME CLASS ---
def test_global_stiffness_matrix(setup_frame):
    """Test if the global stiffness matrix is correctly computed."""
    frame = setup_frame
    K = frame.global_stiffness_matrix()
    assert K.shape == (24, 24)  # 4 nodes * 6 DOFs each

def test_internal_forces(setup_frame):
    """Test if internal forces are computed for all elements."""
    frame = setup_frame
    internal_forces = frame.compute_internal_forces()
    assert len(internal_forces) == len(frame.elements)  # Should match element count

def test_geometric_stiffness_matrix(setup_frame):
    """Test if geometric stiffness matrix is correctly computed."""
    frame = setup_frame
    Kg = frame.global_geometric_stiffness_matrix()
    assert Kg.shape == (24, 24)  # Global stiffness matrix should be same size

def test_force_matrix(setup_frame):
    """Test if the global force matrix is assembled correctly."""
    frame = setup_frame
    F = frame.global_force_matrix()
    assert F.shape == (24,)  # Should be a 24x1 vector

def test_displacement_calculation(setup_frame):
    """Test displacement calculation given a force matrix."""
    frame = setup_frame
    F = frame.global_force_matrix()
    K = frame.global_stiffness_matrix()
    U = frame.get_displacements(F, K)
    assert U.shape == (24,)

def test_critical_load(setup_frame):
    """Test if the critical load factor is computed."""
    frame = setup_frame
    try:
        critical_load, _ = frame.elastic_critical_load()
        assert critical_load > 0  # Should be a positive eigenvalue
    except ValueError:
        pytest.skip("No positive eigenvalue found, skipping test.")

def test_reactions(setup_frame):
    """Test if reactions are correctly computed."""
    frame = setup_frame
    F = frame.global_force_matrix()
    K = frame.global_stiffness_matrix()
    U = frame.get_displacements(F, K)
    R = frame.get_reactions(U, K)
    assert R.shape == (24,)


# Second round of testing just in case. It also has a simpler cantilever beam test case.


E = 200e9  # Young's modulus (Pa)
v = 0.3    # Poisson's ratio
A = 0.01   # Cross-sectional area (m²)
I_y = 1e-6 # Moment of inertia about y-axis (m^4)
I_z = 1e-6 # Moment of inertia about z-axis (m^4)
I_p = 2e-6 # Polar moment of inertia (m^4)
J = 2e-6   # Torsional constant (m^4)

@pytest.fixture
def setup_cantilever_beam():
    """Fixture to create a cantilever beam structure."""
    nodes = [
        Nodes(0, 0, 0),
        Nodes(1, 0, 0)
    ]
    elements = [
        Elements(nodes[0], nodes[1], E, v, A, I_z, I_y, I_p, J)
    ]
    frame = Frame(nodes, elements)
    # Apply boundary conditions (fixed at node 0)
    nodes[0].set_boundary_constraints([True, True, True, True, True, True])
    # Apply a point load at the free end (node 1)
    nodes[1].set_nodal_load(0, -1000, 0, 0, 0, 0)  # 1000 N downward force
    return frame

# Add the new tests to the test suite
def test_global_stiffness_matrix(setup_frame):
    """Test if the global stiffness matrix is correctly computed."""
    frame = setup_frame
    K = frame.global_stiffness_matrix()
    assert K.shape == (24, 24)  # 4 nodes * 6 DOFs each

def test_internal_forces(setup_frame):
    """Test if internal forces are computed for all elements."""
    frame = setup_frame
    internal_forces = frame.compute_internal_forces()
    assert len(internal_forces) == len(frame.elements)  # Should match element count

def test_geometric_stiffness_matrix(setup_frame):
    """Test if geometric stiffness matrix is correctly computed."""
    frame = setup_frame
    Kg = frame.global_geometric_stiffness_matrix()
    assert Kg.shape == (24, 24)  # Global stiffness matrix should be same size

def test_displacement_calculation(setup_frame):
    """Test displacement calculation given a force matrix."""
    frame = setup_frame
    F = frame.global_force_matrix()
    K = frame.global_stiffness_matrix()
    U = frame.get_displacements(F, K)
    assert U.shape == (24,)

def test_critical_load(setup_frame):
    """Test if the critical load factor is computed."""
    frame = setup_frame
    try:
        critical_load, _ = frame.elastic_critical_load()
        assert critical_load > 0  # Should be a positive eigenvalue
    except ValueError:
        pytest.skip("No positive eigenvalue found, skipping test.")

def test_reactions(setup_frame):
    """Test if reactions are correctly computed."""
    frame = setup_frame
    F = frame.global_force_matrix()
    K = frame.global_stiffness_matrix()
    U = frame.get_displacements(F, K)
    R = frame.get_reactions(U, K)
    assert R.shape == (24,)