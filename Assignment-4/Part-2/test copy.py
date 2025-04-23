from dolfinx import mesh, fem, log, plot, default_scalar_type
from dolfinx.fem import Function
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI
import numpy as np
import ufl
import pyvista
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells

# Problem dimensions
L, H = 387.0, 10.0  # Length and height of the beam
nx, ny = 100, 8  # Mesh resolution
E = default_scalar_type(167800.0)  # Young's modulus
nu = default_scalar_type(0.6666666666666667)  # Poisson's ratio
q = -0.2  # Uniform load (N/m)

# Create a 2D rectangular mesh
domain = mesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [L, H]], [nx, ny], mesh.CellType.quadrilateral)
V = fem.functionspace(domain, ("Lagrange", 2, (domain.geometry.dim, )))

# Boundary conditions
def left_boundary(x):
    return np.isclose(x[0], 0.0)

left_dofs = fem.locate_dofs_geometrical(V, left_boundary)
zero_displacement = np.array([0.0, 0.0], dtype=default_scalar_type)
bcs = [fem.dirichletbc(zero_displacement, left_dofs, V)]

# Material properties
mu = fem.Constant(domain, E / (2 * (1 + nu)))
lmbda = fem.Constant(domain, E * nu / ((1 + nu) * (1 - 2 * nu)))

# Define the weak form
u = fem.Function(V)  # Displacement
v = ufl.TestFunction(V)  # Test function
I = ufl.variable(ufl.Identity(2))
F = ufl.variable(I + ufl.grad(u))
C = ufl.variable(F.T * F)
Ic = ufl.variable(ufl.tr(C))
J = ufl.variable(ufl.det(F))
psi = (mu / 2) * (Ic - 2) - mu * ufl.ln(J) + (lmbda / 2) * (ufl.ln(J))**2
P = ufl.diff(psi, F)

# External force: Uniform load applied on the top edge
ds = ufl.Measure("ds", domain=domain)
dx = ufl.Measure("dx", domain=domain)
top_edge = lambda x: np.isclose(x[1], H)
top_facets = mesh.locate_entities_boundary(domain, domain.topology.dim - 1, top_edge)
load = fem.Constant(domain, np.array([0.0, q], dtype=default_scalar_type))
F_form = ufl.inner(ufl.grad(v), P) * dx - ufl.dot(v, load) * ds(2)

# Solver setup
problem = NonlinearProblem(F_form, u, bcs)
solver = NewtonSolver(domain.comm, problem)
solver.atol = 1e-8
solver.rtol = 1e-8
solver.convergence_criterion = "incremental"

# Solve the problem
log.set_log_level(log.LogLevel.INFO)
num_its, converged = solver.solve(u)
assert converged, "Solver did not converge"
u.x.scatter_forward()

# Analytical solution for tip deflection
E_eff = E / (1 - nu**2)
I = H**3 / 12.0
w_analytical = q * L**4 / (8.0 * E_eff * I)

if domain.comm.rank == 0:
    # Create query point with shape (1, 3) even in 2D
    tip_point = np.array([[L, H / 2, 0.0]], dtype=domain.geometry.x.dtype)

    # Create bounding box tree using the correct API
    tree = bb_tree(mesh=domain, dim=domain.topology.dim)

    # Compute candidate cells for collision
    cell_candidates = compute_collisions_points(tree, tip_point)

    # Narrow down to actual colliding cells
    colliding_cells = compute_colliding_cells(domain, cell_candidates, tip_point)

    if len(colliding_cells.links(0)) > 0:
        cell = colliding_cells.links(0)[0]
        # Allocate array for result
        tip_disp = np.zeros(domain.geometry.dim, dtype=domain.geometry.x.dtype)

        # Evaluate displacement at the tip point
        u.eval(tip_disp, tip_point[0][:2], cell)
        print(f"Computed tip deflection (y): {tip_disp[1]:.6f}")
        print(f"Analytical Euler-Bernoulli deflection: {w_analytical:.6f}")
        print(f"Error between computed vs analytical: {abs(tip_disp[1] - w_analytical):.6f}")
    else:
        print("Tip point is not inside the domain.")

# Visualization setup
pyvista.start_xvfb()
plotter = pyvista.Plotter()
topology, cells, geometry = plot.vtk_mesh(u.function_space)
function_grid = pyvista.UnstructuredGrid(topology, cells, geometry)
function_grid["u"] = u.x.array.reshape(geometry.shape[0], 2)
function_grid.set_active_vectors("u")
actor = plotter.add_mesh(function_grid, show_edges=True, lighting=False, clim=[0, 0.1])
plotter.show()