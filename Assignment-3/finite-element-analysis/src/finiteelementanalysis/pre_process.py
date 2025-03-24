import gmsh
import numpy as np


def generate_rect_mesh_2d(
    ele_type: str,
    x_lower: float,
    y_lower: float,
    x_upper: float,
    y_upper: float,
    nx: int,
    ny: int
):
    """
    Generate a 2D rectangular mesh for one of the following element types:
      - D2_nn3_tri   : 3-node linear triangles (tri3)
      - D2_nn6_tri   : 6-node quadratic triangles (tri6)
      - D2_nn4_quad  : 4-node bilinear quadrilaterals (quad4)
      - D2_nn8_quad  : 8-node quadratic quadrilaterals (quad8)

    The domain is [x_lower, x_upper] x [y_lower, y_upper]. The integer nx, ny
    specify how many element slices along x and y. For example:
      - If ele_type='D2_nn4_quad' and nx=3, ny=2, you get a 3 x 2 grid of quad4
        elements => total 3*2=6 elements.
      - If ele_type='D2_nn3_tri', each rectangular cell is split into 2 triangles,
        so total elements = 2 * nx * ny, and so on.

    Parameters
    ----------
    ele_type : str
        One of {'D2_nn3_tri', 'D2_nn6_tri', 'D2_nn4_quad', 'D2_nn8_quad'}.
    x_lower, y_lower : float
        Coordinates of the lower-left corner of the domain.
    x_upper, y_upper : float
        Coordinates of the upper-right corner of the domain.
    nx, ny : int
        Number of subdivisions (elements in each direction) along x and y.

    Returns
    -------
    coords : numpy.ndarray
        Node coordinates, shape (n_nodes, 2).
    connect : numpy.ndarray
        Element connectivity, shape depends on element type:
          - tri3  -> (n_elem, 3)
          - tri6  -> (n_elem, 6)
          - quad4 -> (n_elem, 4)
          - quad8 -> (n_elem, 8)

    Notes
    -----
    - Indices in `connect` are 0-based.
    - For the quadratic elements (tri6, quad8), this code automatically
      generates mid-edge nodes. The approach is uniform and assumes a
      structured rectangular grid. Each element cell places the extra
      mid-edge nodes by subdividing edges in half.
    """
    # Dispatch to the appropriate helper
    if ele_type == "D2_nn3_tri":
        return generate_tri3_mesh(x_lower, y_lower, x_upper, y_upper, nx, ny)
    elif ele_type == "D2_nn6_tri":
        return generate_tri6_mesh(x_lower, y_lower, x_upper, y_upper, nx, ny)
    elif ele_type == "D2_nn4_quad":
        return generate_quad4_mesh(x_lower, y_lower, x_upper, y_upper, nx, ny)
    elif ele_type == "D2_nn8_quad":
        return generate_quad8_mesh(x_lower, y_lower, x_upper, y_upper, nx, ny)
    else:
        raise ValueError(f"Unknown element type: {ele_type}")


# --------------------------------------------------------------------------
#   FUNCTIONS FOR EACH ELEMENT TYPE
# --------------------------------------------------------------------------

def generate_tri3_mesh(xl, yl, xh, yh, nx, ny):
    """
    Generate a simple tri3 (3-node) mesh by subdividing each rectangular cell
    into two triangles.
    """
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    n_nodes_x = nx + 1
    n_nodes_y = ny + 1

    # Create the coordinates array
    coords_list = []
    for iy in range(n_nodes_y):
        for ix in range(n_nodes_x):
            xcoord = xl + ix * dx
            ycoord = yl + iy * dy
            coords_list.append((xcoord, ycoord))
    coords = np.array(coords_list, dtype=float)  # (n_nodes, 2)

    # Create the connectivity
    connectivity_list = []
    for iy in range(ny):
        for ix in range(nx):
            node0 = iy * n_nodes_x + ix
            node1 = iy * n_nodes_x + (ix + 1)
            node2 = (iy + 1) * n_nodes_x + ix
            node3 = (iy + 1) * n_nodes_x + (ix + 1)

            # two triangles
            connectivity_list.append([node0, node1, node2])
            connectivity_list.append([node2, node1, node3])

    connect = np.array(connectivity_list, dtype=int)
    return coords, connect


def generate_tri6_mesh(xl, yl, xh, yh, nx, ny):
    """
    Generate a tri6 (6-node) mesh by subdividing each rectangular cell into
    two triangles, adding mid-edge nodes in the correct shape function order.
    """
    dx = (xh - xl) / float(nx)
    dy = (yh - yl) / float(ny)

    # Refined grid has (2*nx+1) points in x, (2*ny+1) points in y
    npx = 2 * nx + 1
    npy = 2 * ny + 1

    # Build refined coordinates
    coords_list = [(xl + 0.5 * ix * dx, yl + 0.5 * iy * dy)
                   for iy in range(npy) for ix in range(npx)]
    coords = np.array(coords_list, dtype=float)

    def node_id(ix, iy):
        return iy * npx + ix

    connectivity_list = []
    
    for celly in range(ny):
        for cellx in range(nx):
            ix0, iy0 = 2 * cellx, 2 * celly

            # -- First triangle ----------------------------------------
            # Corner nodes (from shape function order)
            N3 = node_id(ix0,   iy0)     # Bottom-left
            N1 = node_id(ix0+2, iy0)     # Bottom-right
            N2 = node_id(ix0,   iy0+2)   # Top-left
            
            # Mid-edge nodes
            N4 = node_id(ix0+1, iy0+1)   # midpoint (N1->N2) diagonal
            N5 = node_id(ix0,   iy0+1)   # midpoint (N2->N3) left vertical
            N6 = node_id(ix0+1, iy0)     # midpoint (N3->N1) bottom horizontal

            connectivity_list.append([N1, N2, N3, N4, N5, N6])

            # -- Second triangle ---------------------------------------
            # Corner nodes
            N3_2 = node_id(ix0+2, iy0+2)   # Top-right
            N1_2 = node_id(ix0,   iy0+2)   # Top-left  (same as N2 above)
            N2_2 = node_id(ix0+2, iy0)     # Bottom-right (same as N1 above)

            # Mid-edge nodes for second triangle
            # (N1_2 -> N2_2) = (top-left -> bottom-right) => diagonal
            # (N2_2 -> N3_2) = (bottom-right -> top-right) => right vertical
            # (N3_2 -> N1_2) = (top-right -> top-left) => top horizontal
            N4_2 = node_id(ix0+1, iy0+1)   # mid-edge (N1_2->N2_2) diagonal
            N5_2 = node_id(ix0+2, iy0+1)   # mid-edge (N2_2->N3_2) right vertical
            N6_2 = node_id(ix0+1, iy0+2)   # mid-edge (N3_2->N1_2) top horizontal

            connectivity_list.append([N1_2, N2_2, N3_2, N4_2, N5_2, N6_2])

    connect = np.array(connectivity_list, dtype=int)  # shape (n_elems, 6)
    return coords, connect


def generate_quad4_mesh(xl, yl, xh, yh, nx, ny):
    """
    Generate a 2D mesh of 4-node quadrilaterals (bilinear quad).
    """
    dx = (xh - xl) / float(nx)
    dy = (yh - yl) / float(ny)
    n_nodes_x = nx + 1
    n_nodes_y = ny + 1

    # Create node coordinates
    coords_list = []
    for iy in range(n_nodes_y):
        for ix in range(n_nodes_x):
            xcoord = xl + ix * dx
            ycoord = yl + iy * dy
            coords_list.append((xcoord, ycoord))
    coords = np.array(coords_list, dtype=float)  # (n_nodes, 2)

    # Connectivity
    connectivity_list = []
    for iy in range(ny):
        for ix in range(nx):
            node0 = iy * n_nodes_x + ix
            node1 = iy * n_nodes_x + (ix + 1)
            node2 = (iy + 1) * n_nodes_x + (ix + 1)
            node3 = (iy + 1) * n_nodes_x + ix
            # Quad element (node0, node1, node2, node3)
            connectivity_list.append([node0, node1, node2, node3])

    connect = np.array(connectivity_list, dtype=int)  # shape (n_elems, 4)
    return coords, connect


def generate_quad8_mesh(xl, yl, xh, yh, nx, ny):
    """
    Generate a 2D mesh of 8-node quadrilaterals (quadratic quad).
    Each cell has corner + mid-edge nodes, excluding the central node.
    """
    dx = (xh - xl) / float(nx)
    dy = (yh - yl) / float(ny)
    npx = 2 * nx + 1  # number of points in x-direction
    npy = 2 * ny + 1  # number of points in y-direction

    # Dictionary to map old node indices to new node indices
    node_map = {}
    new_coords = []
    new_index = 0

    # Build refined coordinates, skipping central nodes
    for iy in range(npy):
        for ix in range(npx):
            # Skip center nodes at (ix0+1, iy0+1) in 2x2 blocks
            if ix % 2 == 1 and iy % 2 == 1:
                continue
            node_map[(ix, iy)] = new_index  # Store new index mapping
            new_coords.append((xl + 0.5 * ix * dx, yl + 0.5 * iy * dy))
            new_index += 1

    coords = np.array(new_coords, dtype=float)

    def node_id(ix, iy):
        return node_map[(ix, iy)]

    connectivity_list = []
    for celly in range(ny):
        for cellx in range(nx):
            ix0 = 2 * cellx
            iy0 = 2 * celly

            # Define the 8-node connectivity for the quadratic quadrilateral
            connectivity_list.append([
                node_id(ix0,   iy0),   # bottom-left
                node_id(ix0+2, iy0),   # bottom-right
                node_id(ix0+2, iy0+2), # top-right
                node_id(ix0,   iy0+2), # top-left
                node_id(ix0+1, iy0),   # mid-edge bottom
                node_id(ix0+2, iy0+1), # mid-edge right
                node_id(ix0+1, iy0+2), # mid-edge top
                node_id(ix0,   iy0+1)  # mid-edge left
            ])

    connect = np.array(connectivity_list, dtype=int)  # (n_elems, 8)
    return coords, connect


def mesh_outline(
    outline_points: list[tuple[float, float]],
    element_type: str,
    mesh_name: str,
    mesh_size: float = 0.05,
):
    """
    Generate a 2D mesh of the specified element type (D2_nn3_tri or D2_nn6_tri)
    for a user-defined shape outline using the gmsh Python API.

    Parameters
    ----------
    outline_points : list of (float, float)
        The polygon or spline points defining the shape's outline in XY.
        If not closed (first point != last point), the function appends
        the first point to the end.
    element_type : str
        Either 'D2_nn3_tri' (linear triangles) or 'D2_nn6_tri' (quadratic triangles).
    mesh_name : str
        A name for the gmsh model.
    mesh_size : float
        Characteristic length scale for the outline points.

    Returns
    -------
    coords : numpy.ndarray of shape (n_nodes, 2)
        The (x, y) coordinates of each node in the 2D mesh.
    connectivity : numpy.ndarray of shape (n_elems, n_nodes_per_elem)
        The triangular element connectivity (either 3 or 6 nodes/element),
        with 0-based node indices.

    Raises
    ------
    ValueError
        If an unsupported element_type is provided.
    RuntimeError
        If no elements of the requested type are found in the final mesh.
    """
    gmsh.initialize()
    gmsh.model.add(mesh_name)
    
    # Ensure the shape is properly closed
    if outline_points[0] != outline_points[-1]:
        outline_points.append(outline_points[0])
    
    # Create gmsh points
    point_tags = []
    for kk in range(0, len(outline_points) - 1):
        x = outline_points[kk][0]
        y = outline_points[kk][1]
        pt_tag = gmsh.model.geo.addPoint(x, y, 0.0, mesh_size)
        point_tags.append(pt_tag)
    
    # Create lines
    curve_tags = []
    for i in range(len(point_tags) - 1):
        start_pt = point_tags[i]
        end_pt = point_tags[i + 1]
        line_tag = gmsh.model.geo.addLine(start_pt, end_pt)
        curve_tags.append(line_tag)
    
    start_pt = point_tags[-1]
    end_pt = point_tags[0]
    line_tag = gmsh.model.geo.addLine(start_pt, end_pt)
    curve_tags.append(line_tag)

    # Make a closed loop from these lines
    loop_tag = gmsh.model.geo.addCurveLoop(curve_tags)
    
    # Create a plane surface from the loop
    surface_tag = gmsh.model.geo.addPlaneSurface([loop_tag])
    
    # Optionally, define a physical group for the surface
    # so Gmsh understands it as a meshable 2D region.
    surf_group = gmsh.model.addPhysicalGroup(2, [surface_tag])
    gmsh.model.setPhysicalName(2, surf_group, "MySurface")
    
    # Set element polynomial order
    if element_type == 'D2_nn3_tri':
        gmsh.model.mesh.setOrder(1)
        tri_wanted_type = 2   # Gmsh code for 3-node triangles
    elif element_type == 'D2_nn6_tri':
        gmsh.model.mesh.setOrder(2)
        tri_wanted_type = 9   # Gmsh code for 6-node (quadratic) triangles
    else:
        gmsh.finalize()
        raise ValueError(f"Unknown element type: {element_type}")
    
    # Synchronize geometry
    gmsh.model.geo.synchronize()

    # Generate 2D mesh
    gmsh.model.mesh.generate(dim=2)

    # Seems like a bug in gmsh, need to call this again if quadratic
    if element_type == 'D2_nn6_tri':
        gmsh.model.mesh.setOrder(2)

    # Ensure quadratic elements get generated
    gmsh.model.mesh.optimize()
    gmsh.model.mesh.renumberNodes()
    
    # Extract node coordinates and connectivity
    types, elem_tags, node_tags = gmsh.model.mesh.getElements(dim=2, tag=surface_tag)
    
    # Find the index for the desired triangle type
    index_in_list = None
    for i, t in enumerate(types):
        if t == tri_wanted_type:
            index_in_list = i
            break
    if index_in_list is None:
        gmsh.finalize()
        raise RuntimeError(f"No elements of type {tri_wanted_type} found in mesh.")
    
    these_elem_tags = elem_tags[index_in_list]  # element IDs (not needed for connectivity)
    these_node_tags = node_tags[index_in_list]  # node IDs, flattened
    
    # Gmsh global nodes and their coordinates
    all_node_indices, all_node_coords, _ = gmsh.model.mesh.getNodes()
    # Build a map from gmsh node ID -> local index
    id2local = {node_id: i for i, node_id in enumerate(all_node_indices)}
    
    # Convert from (x,y,z) to (x,y)
    all_node_coords_3d = all_node_coords.reshape(-1, 3)
    coords = all_node_coords_3d[:, :2]
    
    # Build connectivity array
    n_nodes_per_elem = 3 if element_type == 'D2_nn3_tri' else 6
    n_elems = len(these_elem_tags)
    connectivity = np.zeros((n_elems, n_nodes_per_elem), dtype=int)
    
    # each element has n_nodes_per_elem node IDs in these_node_tags
    for e in range(n_elems):
        for k in range(n_nodes_per_elem):
            gmsh_node_id = these_node_tags[e * n_nodes_per_elem + k]
            connectivity[e, k] = id2local[gmsh_node_id]

    # reverse ordering for D2_nn3_tri elements -- gmsh automatically return clockwise nodes
    if element_type == "D2_nn3_tri":
        connectivity[:, [0, 1]] = connectivity[:, [1, 0]]  # reverse 0 and 1 for CCW
    elif element_type == "D2_nn6_tri":
        # Reverse corner nodes: [0, 1, 2] → [0, 2, 1]
        connectivity[:, [1, 2]] = connectivity[:, [2, 1]]
        # Reverse mid-edge nodes: [3, 4, 5] → [5, 4, 3]
        connectivity[:, [3, 5]] = connectivity[:, [5, 3]]

    gmsh.finalize()
    return coords, connectivity


def get_cloud_outline():
    """
    Return a list of (x, y) coordinate pairs for a bulldog head outline,
    as extracted from Inkscape.

    The coordinates below were copied directly from an Inkscape path export.
    You can further clean or scale them as needed.

    Returns
    -------
    outline_points : list of (float, float)
        The bulldog outline, stored as a list of XY pairs.
    """
    # Raw coordinate string (from Inkscape)
    raw_coords = """
    118.32,139.855811023622
    115.79342,139.741821023622
    113.28723,139.401841023622
    110.82168999999999,138.83821102362202
    108.41668999999999,138.055561023622
    106.09154999999998,137.06045102362202
    103.86483999999999,135.861121023622
    101.75436999999998,134.467281023622
    99.77464999999998,132.887841023622
    97.94469999999998,131.13706102362198
    96.27948999999998,129.22892102362198
    94.79252999999999,127.17881102362199
    93.49549999999998,125.003531023622
    91.33667999999997,126.426441023622
    89.01266999999997,127.55965102362201
    86.56215999999998,128.384381023622
    84.02589999999998,128.88691102362202
    81.44599999999998,129.058811023622
    78.88955999999999,128.897371023622
    76.37450999999999,128.411861023622
    73.94258999999998,127.607561023622
    71.63394999999998,126.497871023622
    69.48590999999999,125.10244102362199
    67.53249,123.445301023622
    65.79992999999999,121.547741023622
    64.32309,119.445031023622
    63.12732999999999,117.170711023622
    62.37387699999999,115.210351023622
    61.83123799999999,113.18149102362199
    61.50494199999999,111.10681102362199
    61.39655299999999,109.00939102362199
    61.41939299999999,108.37857102362199
    59.91249299999999,108.45387102362199
    57.34429299999999,108.26288102362199
    54.83376299999999,107.68928102362199
    52.43850299999999,106.74350102362197
    50.21269299999999,105.44824102362196
    48.205502999999986,103.83477102362195
    46.461032999999986,101.94032102362195
    45.01868299999999,99.80686102362193
    44.043995999999986,97.81125102362194
    43.33717399999998,95.70583102362193
    42.91038899999998,93.52630102362193
    42.76850399999998,91.30986102362192
    42.92391799999998,88.99217102362192
    43.39089999999998,86.71675102362192
    44.16327199999998,84.52609102362192
    45.226281999999976,82.46075102362192
    46.55885199999997,80.55807102362192
    48.30123199999997,78.69725102362193
    50.29897199999997,77.11371102362193
    52.50906199999997,75.84333102362194
    54.88366199999997,74.91623102362195
    57.370101999999974,74.35414102362193
    59.912481999999976,74.16700102362194
    62.533271999999975,74.20560102362194
    65.15409199999998,74.24100102362195
    67.77497199999998,74.27290102362196
    70.39589199999998,74.30080102362194
    73.01686199999997,74.32420102362195
    75.63786199999997,74.34260102362197
    78.01725199999997,74.35450102362196
    80.39666199999996,74.36150102362197
    82.77608199999996,74.36250102362197
    85.15550199999996,74.35750102362198
    87.53490199999996,74.34680102362196
    89.91423199999996,74.33030102362196
    92.37657199999995,74.30790102362195
    94.83887199999995,74.28170102362196
    97.30115199999995,74.25380102362197
    99.76345199999994,74.22690102362196
    102.22578999999995,74.20330102362198
    104.68816999999994,74.18480102362199
    107.15058999999994,74.17240102362197
    109.61304999999993,74.16640102362197
    111.64466999999993,74.16640102362197
    113.67628999999994,74.16640102362197
    115.70790999999994,74.16640102362197
    117.73952999999995,74.16640102362197
    120.27196999999994,74.16640102362197
    122.80440999999993,74.16640102362197
    125.33684999999993,74.16640102362197
    127.86928999999992,74.16640102362197
    130.40172999999993,74.16640102362197
    132.58729999999994,74.16640102362197
    134.77286999999995,74.16640102362197
    136.95843999999997,74.16640102362197
    139.14400999999998,74.16640102362197
    141.32958,74.16640102362197
    143.82394,74.34647102362197
    146.2655,74.88746102362197
    148.60146,75.78035102362196
    150.78186,77.00505102362197
    152.76122,78.53355102362198
    154.49859,80.33241102362197
    155.89402,82.26159102362197
    157.00878,84.36542102362196
    157.81965,86.60400102362198
    158.31026,88.93383102362196
    158.47358,91.30926102362196
    158.30537,93.70242102362195
    157.80441000000002,96.04858102362195
    156.97986000000003,98.30148102362196
    155.84870000000004,100.41713102362195
    154.43378000000004,102.35462102362195
    152.69358000000005,104.13777102362195
    150.71325000000004,105.64975102362195
    148.53416000000004,106.85778102362195
    146.20277000000004,107.73687102362194
    146.42546000000004,109.69525102362195
    146.51076000000003,111.66439102362196
    146.40003000000004,114.18020102362195
    146.06658000000004,116.67624102362194
    145.51096000000004,119.13239102362195
    144.73679000000004,121.52865102362196
    143.75039000000004,123.84562102362196
    142.56028000000003,126.06489102362195
    141.15847000000002,128.19432102362197
    139.57058000000004,130.18885102362196
    137.80990000000003,132.03265102362195
    135.89077000000003,133.71089102362197
    133.82846000000004,135.20976102362195
    131.81896000000003,136.41916102362194
    129.71570000000003,137.45696102362194
    127.53306000000002,138.31534102362195
    125.28621000000003,138.98789102362196
    122.99095000000003,139.47001102362196
    120.66348000000002,139.75917102362195
    118.31999000000002,139.85517102362195
    118.32,139.855811023622
    """
    # split the raw string by whitespace
    tokens = raw_coords.strip().split()
    # parse each token as "x,y"
    outline_points = []
    for t in tokens:
        x_str, y_str = t.split(",")
        x_val = float(x_str)
        y_val = float(y_str) 
        outline_points.append([x_val, y_val])
    return outline_points


def identify_rect_boundaries(
    coords: np.ndarray,
    connect: np.ndarray,
    ele_type: str,
    x_lower: float,
    x_upper: float,
    y_lower: float,
    y_upper: float,
    tol: float = 1e-12
):
    """
    Identify boundary nodes, elements, and faces for a rectangular 2D domain
    mesh. Boundaries are labeled as 'left', 'right', 'bottom', or 'top' based
    on coordinate checks against x_lower, x_upper, y_lower, y_upper.

    Parameters
    ----------
    coords : numpy.ndarray of shape (n_nodes, 2)
        The node coordinates array, typically from generate_rect_mesh_2d(...).
    connect : numpy.ndarray
        The element connectivity array, shape depends on ele_type:
          - tri3  -> (n_elems, 3)
          - tri6  -> (n_elems, 6)
          - quad4 -> (n_elems, 4)
          - quad8 -> (n_elems, 8)
    ele_type : str
        One of {'D2_nn3_tri', 'D2_nn6_tri', 'D2_nn4_quad', 'D2_nn8_quad'}.
    x_lower, x_upper : float
        The domain boundaries in x.
    y_lower, y_upper : float
        The domain boundaries in y.
    tol : float, optional
        Tolerance for comparing floating-point coordinates. If a node is
        within `tol` of a boundary, it's considered on that boundary.

    Returns
    -------
    boundary_nodes : dict of {str -> set of int}
        Keys are 'left','right','bottom','top'. Values are sets of node indices
        that lie on that boundary.
    boundary_edges : dict of {str -> list of (elem_id, face_id)}
        Keys are 'left','right','bottom','top'. Each entry is a list of
        tuples (element_id, local_face_id) indicating which element-face
        belongs to that boundary.

    Notes
    -----
    - For triangular elements, each face/edge is defined by consecutive nodes
      in the connectivity. For tri3, edges are (0,1), (1,2), (2,0); for tri6,
      edges are (0,1,3), (1,2,4), (2,0,5).
    - For quadrilateral elements, each face is defined by consecutive nodes
      in the connectivity array. For quad4, faces are (0,1), (1,2), (2,3), (3,0);
      for quad8, faces are (0,1,4), (1,2,5), (2,3,6), (3,0,7).
    - This function focuses on a strictly rectangular domain. We identify
      boundary nodes by checking x or y vs. x_lower, x_upper, y_lower, y_upper
      within a tolerance. Then, we find which element edges/faces connect
      these boundary nodes to label them accordingly.
    """

    n_nodes = coords.shape[0]
    n_elems = connect.shape[0]

    # 1. Identify boundary nodes by coordinate
    #    We'll store them in sets to avoid duplicates
    left_nodes = set()
    right_nodes = set()
    bottom_nodes = set()
    top_nodes = set()

    for nid in range(n_nodes):
        xval, yval = coords[nid]
        # Compare with tolerance
        if abs(xval - x_lower) < tol:
            left_nodes.add(nid)
        if abs(xval - x_upper) < tol:
            right_nodes.add(nid)
        if abs(yval - y_lower) < tol:
            bottom_nodes.add(nid)
        if abs(yval - y_upper) < tol:
            top_nodes.add(nid)

    # 2. Determine how faces are enumerated for each element type
    #    We'll define a helper that, given 'ele_type', returns a list of "faces"
    #    as tuples of local node indices in the connectivity array.
    face_definitions = local_faces_for_element_type(ele_type)

    # 3. Identify boundary edges/faces by checking if *all* the nodes
    #    in that face belong to the same boundary set. Because if an entire
    #    face is on x_lower => all face nodes must have x ~ x_lower, etc.
    #    We'll store the result as a dict: { boundary : list of (elem_id, face_id) }
    boundary_edges = {
        'left': [],
        'right': [],
        'bottom': [],
        'top': []
    }

    for e in range(n_elems):
        # Each face is a list of local node indices in the connectivity
        for face_id, face_lnodes in enumerate(face_definitions):
            # The actual global node ids for this face
            face_nodes = connect[e, face_lnodes]
            # We'll see if they are on left, right, bottom, top.
            # In a rectangular domain, if all these face nodes are in left_nodes,
            # that face is a left boundary, etc. But watch out for corner elements
            # that might belong to left *and* bottom, for instance.
            # Typically, we consider it "left" if all face nodes are in left_nodes
            # We'll do it that way for simplicity. 
            # A corner face might appear in two boundary sets if it's degenerate, 
            # but usually an element face won't be "on" two boundaries at once 
            # unless the domain is extremely coarse.
            if all(fn in left_nodes for fn in face_nodes):
                boundary_edges['left'].append((e, face_id))
            if all(fn in right_nodes for fn in face_nodes):
                boundary_edges['right'].append((e, face_id))
            if all(fn in bottom_nodes for fn in face_nodes):
                boundary_edges['bottom'].append((e, face_id))
            if all(fn in top_nodes for fn in face_nodes):
                boundary_edges['top'].append((e, face_id))

    # 4. Return the results
    boundary_nodes = {
        'left': left_nodes,
        'right': right_nodes,
        'bottom': bottom_nodes,
        'top': top_nodes
    }

    return boundary_nodes, boundary_edges


def local_faces_for_element_type(ele_type: str):
    """
    Return a list of "faces" for the given 2D element type, where each
    face is defined by a tuple of local connectivity indices.
    
    For example, tri3 has 3 edges:
       face0 = (0,1)
       face1 = (1,2)
       face2 = (2,0)

    tri6 (quadratic triangle) has 3 edges each with 3 nodes:
       face0 = (0,1,3)
       face1 = (1,2,4)
       face2 = (2,0,5)

    quad4 has 4 edges:
       face0 = (0,1)
       face1 = (1,2)
       face2 = (2,3)
       face3 = (3,0)

    quad8 (quadratic quad) has 4 edges each with 3 nodes:
       face0 = (0,1,4)
       face1 = (1,2,5)
       face2 = (2,3,6)
       face3 = (3,0,7)
    """
    if ele_type == "D2_nn3_tri":
        # 3-node triangle
        return [
            (0, 1),
            (1, 2),
            (2, 0)
        ]
    elif ele_type == "D2_nn6_tri":
        # 6-node triangle
        return [
            (0, 1, 3),
            (1, 2, 4),
            (2, 0, 5)
        ]
    elif ele_type == "D2_nn4_quad":
        # 4-node quad
        return [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0)
        ]
    elif ele_type == "D2_nn8_quad":
        # 8-node quad
        return [
            (0, 1, 4),
            (1, 2, 5),
            (2, 3, 6),
            (3, 0, 7)
        ]
    else:
        raise ValueError(f"Unknown element type: {ele_type}")


def assign_fixed_nodes_rect(
    boundary_nodes: dict[str, set[int]],
    boundary: str,
    dof_0_disp: float = None,
    dof_1_disp: float = None,
    dof_2_disp: float = None
) -> np.ndarray:
    """
    Build a (3, n_fixed) array of prescribed boundary conditions for all nodes
    on a specified boundary of a rectangular 2D mesh.

    Parameters
    ----------
    boundary_nodes : dict of {str -> set of int}
        A dictionary mapping each boundary ('left','right','bottom','top') to 
        a set of node indices on that boundary.
    boundary : str
        Which boundary name in boundary_nodes to apply these DOF constraints to 
        (e.g. 'left', 'top', etc.).
    dof_0_disp : float or None, optional
        If not None, fix DOF #0 of each node at the given displacement.
    dof_1_disp : float or None, optional
        If not None, fix DOF #1 of each node at the given displacement.
    dof_2_disp : float or None, optional
        If not None, fix DOF #2 of each node at the given displacement.
        In a 2D problem, typically dof_2_disp is None by default.

    Returns
    -------
    fixed_nodes : numpy.ndarray, shape (3, n_fixed)
        The prescribed boundary conditions. Each column has:
          [ node_id, dof_index, displacement_value ].

    Notes
    -----
    - Only DOFs for which a non-None displacement is provided will be fixed.
    - For 2D (ncoord=2, ndof=2), typically dof_2_disp is unused.
    - If boundary_nodes[boundary] is empty, this function returns an empty array.
    """
    # Get all node indices on the specified boundary
    node_ids = boundary_nodes.get(boundary, set())
    if not node_ids:
        # No nodes on this boundary => return empty array
        return np.empty((3, 0), dtype=float)

    # Build a list of constraints
    constraints = []
    for node_id in node_ids:
        if dof_0_disp is not None:
            constraints.append((node_id, 0, dof_0_disp))
        if dof_1_disp is not None:
            constraints.append((node_id, 1, dof_1_disp))
        if dof_2_disp is not None:
            constraints.append((node_id, 2, dof_2_disp))

    # If no constraints were added (all disp = None), return empty
    if not constraints:
        return np.empty((3, 0), dtype=float)

    # Convert list to numpy array and transpose to shape (3, n_fixed)
    fixed_array = np.array(constraints, dtype=float).T  # shape => (3, n_fixed)
    return fixed_array


def assign_uniform_load_rect(
    boundary_edges: dict[str, list[tuple[int, int]]],
    boundary: str,
    dof_0_load: float = 0.0,
    dof_1_load: float = 0.0,
    dof_2_load: float = 0.0
) -> np.ndarray:
    """
    Create a distributed-load specification for a boundary in a 2D or 3D mesh,
    returning an array dload_info of shape (ndof+2, n_face_loads).

    Each column of dload_info describes a uniform traction load on a single
    element-face along the specified boundary. The format:
      - dload_info[0, j] => element index (elem_id)
      - dload_info[1, j] => local face ID (face_id) on that element
      - dload_info[2, j], dload_info[3, j], [dload_info[4, j]] => the traction
        components for dof=0,1,[2].

    Parameters
    ----------
    boundary_edges : dict of {str -> list of (elem_id, face_id)}
        Keys are 'left','right','bottom','top'. Each entry is a list of
        (element_id, local_face_id) pairs indicating which element-face
        belongs to that boundary.
    boundary : str
        The boundary name in boundary_edges to which the uniform traction
        is applied (e.g. 'left', 'top', etc.).
    dof_0_load : float, optional
        The traction in the dof=0 direction (e.g., x-direction in 2D).
    dof_1_load : float, optional
        The traction in the dof=1 direction (e.g., y-direction in 2D).
    dof_2_load : float, optional
        The traction in the dof=2 direction (if 3D). If you are strictly 2D,
        this should be 0 (the default).

    Returns
    -------
    dload_info : numpy.ndarray, shape (ndof+2, n_face_loads)
        The distributed face load info. Each column corresponds to a single face
        along `boundary`. The top rows contain the (element_id, face_id),
        followed by the traction components. If no boundary faces exist or the
        traction is zero in all directions and you prefer to omit them, you can
        filter accordingly.

    Notes
    -----
    - If dof_2_load is nonzero, we assume ndof=3. Otherwise, ndof=2.
    - If the boundary has no faces in boundary_edges[boundary], returns an
      empty array with shape (ndof+2, 0).
    - In a typical 2D code with tri or quad elements, face_id might range
      from 0..2 or 0..3, etc.
    - The traction is uniform. If you want a variable traction, you might
      compute different values per face.
    """
    # check boundary faces
    faces = boundary_edges.get(boundary, [])
    n_face_loads = len(faces)
    ndof = 3  # assumes ndof = 3, if code is 2D final entry will be ignored
    if n_face_loads == 0:
        # no faces => return empty
        return np.empty((ndof+2, 0), dtype=float)

    # build the traction vector for each face
    # shape => (ndof+2, n_face_loads)
    dload_info = np.zeros((ndof+2, n_face_loads), dtype=float)

    # dof loads in a list
    load_list = [dof_0_load, dof_1_load, dof_2_load]

    # iterate through faces and add to dload_info vector
    for j, (elem_id, face_id) in enumerate(faces):
        # element_id, face_id
        dload_info[0, j] = elem_id
        dload_info[1, j] = face_id
        # traction components
        for i in range(ndof):
            dload_info[i + 2, j] = load_list[i]

    return dload_info
