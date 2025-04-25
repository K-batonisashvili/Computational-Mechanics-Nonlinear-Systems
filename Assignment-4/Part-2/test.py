from dolfinx import mesh, fem, log, plot, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx import geometry
from mpi4py import MPI
import matplotlib.pyplot as plt
import numpy as np
import ufl

# Bridge dimensions
L, W, H = 10.0, 1.0, 0.5  # Length, width, height
num_elements = [50, 20, 20]  # Mesh resolution 

# Create a 3D bridge mesh
domain = mesh.create_box(MPI.COMM_WORLD, [[0.0, 0.0, 0.0], [L, W, H]], num_elements, mesh.CellType.hexahedron)
V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim, )))
dim = domain.topology.dim

# Material properties for bridge
rho = 800  # Density 
E = fem.Constant(domain, 2.1e11)  
nu = fem.Constant(domain, 0.3)
mu = (E / (2 * (1 + nu)))
lmbda = (E * nu / ((1 + nu) * (1 - 2 * nu)))

# Define the weak form
u = ufl.TrialFunction(V)  # Displacement as a vector field
v = ufl.TestFunction(V)  # Test function
solution = fem.Function(V)  # Solution vector

def epsilon(v):
    return ufl.sym(ufl.grad(v))


def sigma(v):
    return lmbda * ufl.tr(epsilon(v)) * ufl.Identity(dim) + 2 * mu * epsilon(v)

# Boundary conditions: Fix both ends of the bridge
def left_end(x):
    return np.isclose(x[0], 0.0)

def right_end(x):
    return np.isclose(x[0], L)


left_dofs = fem.locate_dofs_geometrical(V, left_end)
right_dofs = fem.locate_dofs_geometrical(V, right_end)
zero_displacement = np.array([0.0, 0.0, 0.0], dtype=default_scalar_type)
bcs = [fem.dirichletbc(zero_displacement, left_dofs, V),
       fem.dirichletbc(zero_displacement, right_dofs, V)]

# External force: Vertical force applied uniformly
q = fem.Constant(domain, default_scalar_type((0.0, 0.0, -1e4)))  # Force in the negative z-direction
dx = ufl.Measure("dx", domain=domain)
a = ufl.inner(sigma(u), epsilon(v)) * dx
L_form = ufl.inner(q, v) * dx

# Solver setup using a linear solver
problem = LinearProblem(a, L_form, bcs=bcs, u=solution)
problem.solve()

# Define the moment of inertia for the beam's cross-section
Inertia = W * H**3 / 12  # Assuming a rectangular cross-section

# Bounding box tree for collision detection
bb_tree = geometry.bb_tree(domain, domain.topology.dim)

# Define points along the beam's length for displacement evaluation
x = np.linspace(0, L, 100)
points = np.zeros((3, 100))
points[0] = x
points[1] = W / 2  # Mid-width of the beam
points[2] = H / 2  # Mid-height of the beam

cells = []
points_on_proc = []
u_values = []

# Find cells containing the points
cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
for i, point in enumerate(points.T):
    if len(colliding_cells.links(i)) > 0:
        points_on_proc.append(point)
        cells.append(colliding_cells.links(i)[0])

points_on_proc = np.array(points_on_proc, dtype=np.float64)
u_values = solution.eval(points_on_proc, cells)

# Plot FEA results
fig = plt.figure()
plt.plot(points_on_proc[:, 0], u_values[:, 2], "k", linewidth=2, label="FEA")  # Z-displacement
plt.grid(True)

# Define the analytic solution for comparison
u_analytic = lambda x: (q.value[2] / (24 * E.value * Inertia)) * (x**4 - 2 * L * x**3 + L**3 * x)

# Ensure x-coordinates match
x_analytic = np.linspace(0, L, len(points_on_proc[:, 0]))
u_analytic_values = u_analytic(x_analytic)

# Compute and save the error
error = np.max(np.abs(u_values[:, 2] - u_analytic_values))
print(f"Error: {error}")

# Plot analytic solution
plt.plot(x_analytic, u_analytic_values, "r", linewidth=2, label="Analytic")
plt.grid(True)
plt.xlabel("x")
plt.ylabel("Displacement (Z)")
plt.legend()
output_png = "Computed_vs_Analytic_Error.png"
plt.savefig(output_png)