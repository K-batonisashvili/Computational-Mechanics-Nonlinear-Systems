from dolfinx import mesh, fem, log, plot, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI
import numpy as np
import ufl
import pyvista

# Bridge dimensions
L, W, H = 10.0, 1.0, 0.5  # Length, width, height
num_elements = [24, 2, 2]  # Mesh resolution

# Create a 3D bridge mesh
domain = mesh.create_box(MPI.COMM_WORLD, [[0.0, 0.0, 0.0], [L, W, H]], num_elements, mesh.CellType.hexahedron)
V = fem.functionspace(domain, ("Lagrange", 2, (domain.geometry.dim, )))

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

# Material properties
rho = 7850  # Density (kg/m^3)
E = default_scalar_type(2.1e8)  # Young's modulus (Pa)
nu = default_scalar_type(0.3)
mu = fem.Constant(domain, E / (2 * (1 + nu)))
lmbda = fem.Constant(domain, E * nu / ((1 + nu) * (1 - 2 * nu)))

# Define the weak form
u = fem.Function(V)  # Displacement
v = ufl.TestFunction(V) # Test function
u_t = fem.Function(V)  # Velocity
u_tt = fem.Function(V)  # Acceleration

I = ufl.variable(ufl.Identity(domain.geometry.dim)) 
F = ufl.variable(I + ufl.grad(u)) 
C = ufl.variable(F.T * F) 
Ic = ufl.variable(ufl.tr(C)) 
J = ufl.variable(ufl.det(F)) 
psi = (mu / 2) * (Ic - 3) - mu * ufl.ln(J) + (lmbda / 2) * (ufl.ln(J))**2
P = ufl.diff(psi, F) 

# External force: Vertical force applied at the center of the bridge
force_center = fem.Constant(domain, default_scalar_type((0.0, 0.0, -1e4)))  # Force in the negative z-direction
ds = ufl.Measure("ds", domain=domain)
dx = ufl.Measure("dx", domain=domain)
F_form = rho * ufl.dot(u_tt, v) * dx + ufl.inner(ufl.grad(v), P) * dx - ufl.dot(v, force_center) * ds

# Time-stepping parameters
dt = 1  # Time step size
T_end = 20.0  # Total simulation time
num_steps = int(T_end / dt)

# Analytical solution for displacement at the middle of the beam
def analytical_displacement(L, E, I, P):
    return (P * L**3) / (48 * E * I)

# Moment of inertia for a rectangular cross-section
I = (W * H**3) / 12

# Calculate analytical displacement at the middle of the beam
analytical_disp = analytical_displacement(L, E, I, 1e4)
# if domain.comm.rank == 0:
#     print(f"Analytical displacement at the middle of the beam: {analytical_disp:.6f} m")

# Solver setup
problem = NonlinearProblem(F_form, u, bcs)
solver = NewtonSolver(domain.comm, problem)
solver.atol = 1e-8
solver.rtol = 1e-8
solver.convergence_criterion = "incremental"

# Visualization setup
pyvista.start_xvfb()
plotter = pyvista.Plotter()
plotter.open_gif("bridge_dynamics_p_mesh.gif", fps=10)

topology, cells, geometry = plot.vtk_mesh(u.function_space)
function_grid = pyvista.UnstructuredGrid(topology, cells, geometry)
function_grid["u"] = np.zeros((geometry.shape[0], 3))
function_grid.set_active_vectors("u")

actor = plotter.add_mesh(function_grid, show_edges=True, lighting=False, clim=[0, 0.1])

# Time-stepping loop
log.set_log_level(log.LogLevel.INFO)
for step in range(num_steps):
    t = step * dt
    print(f"Time step {step + 1}/{num_steps}, Time: {t:.2f}s")
    
    # Solve for the displacement
    num_its, converged = solver.solve(u)
    assert converged, f"Solver did not converge at step {step + 1}"
    u.x.scatter_forward()

    # Extract computed displacement at the middle of the beam
    if domain.comm.rank == 0:
        mid_point = np.array([L / 2, W / 2, H / 2])
        mid_disp = u.eval(mid_point)
        print(f"Computed displacement at the middle of the beam: {mid_disp[2]:.6f} m")
        print(f"Difference from analytical solution: {abs(mid_disp[2] - analytical_disp):.6f} m")

    # Update velocity and acceleration
    u_tt.x.array[:] = (u.x.array - u_t.x.array) / dt
    u_t.x.array[:] = u.x.array

    # Update visualization
    function_grid["u"][:, :3] = u.x.array.reshape(geometry.shape[0], 3)
    warped = function_grid.warp_by_vector("u", factor=1)
    plotter.update_coordinates(warped.points, render=False)
    plotter.write_frame()

plotter.close()