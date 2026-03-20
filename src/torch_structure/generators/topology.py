import torch

from torch_structure.data import StructData
from torch_structure.generators.helpers import calculate_line, line_intersection
import re


def grid_side(data: StructData, name: str, side: str) -> list:
    """
    Returns the ordered list of node names on a given side of a named grid in data.

    Args:
        data: StructData containing the grid nodes.
        name: Grid name prefix used when the grid was created.
        side: One of "x0", "x1", "y0", "y1".
    """
    pattern = re.compile(rf"^{re.escape(name)}_(\d+)_node_(\d+)$")
    nodes = {}
    for node_name in data.metadata["node_name_to_index"]:
        m = pattern.match(node_name)
        if m:
            nodes[(int(m.group(1)), int(m.group(2)))] = node_name

    if not nodes:
        raise ValueError(f"No grid nodes found with name '{name}'")

    max_i = max(i for i, _ in nodes)
    max_j = max(j for _, j in nodes)

    if side == "x0":
        return [nodes[(0, j)] for j in range(max_j + 1)]
    elif side == "x1":
        return [nodes[(max_i, j)] for j in range(max_j + 1)]
    elif side == "y0":
        return [nodes[(i, 0)] for i in range(max_i + 1)]
    elif side == "y1":
        return [nodes[(i, max_j)] for i in range(max_i + 1)]
    else:
        raise ValueError(f"Invalid side '{side}'. Must be one of: x0, x1, y0, y1.")


def weld_grids(data: StructData, nodes_a: list, nodes_b: list):
    """
    Adds edges between two ordered lists of nodes, welding two grid sides together.

    Args:
        data: StructData containing both grids.
        nodes_a: Ordered node names from the first grid side.
        nodes_b: Ordered node names from the second grid side.
    """
    if len(nodes_a) != len(nodes_b):
        raise ValueError(
            f"Cannot weld sides of different lengths: {len(nodes_a)} vs {len(nodes_b)}"
        )
    for a, b in zip(nodes_a, nodes_b):
        data.add_edge(a, b)


def merge_nodes_by_coords(data: StructData, tolerance: float = 1e-6):
    """
    Merges all nodes that share the same coordinates into a single node.

    Args:
        data: StructData to merge nodes in.
        tolerance: Distance threshold for considering two nodes coincident.
    """
    groups = {}
    for name, idx in data.metadata["node_name_to_index"].items():
        key = tuple((data.coords[idx] / tolerance).round().long().tolist())
        groups.setdefault(key, []).append(name)

    for names in groups.values():
        if len(names) > 1:
            data.merge_nodes(names[0], names[1:])


def grid(
    data: StructData,
    nx: int,
    ny: int,
    name: str = "grid",
    origin: torch.Tensor = torch.tensor([0.0, 0.0, 0.0]),
    dx: torch.Tensor = torch.tensor([1.0, 0.0, 0.0]),
    dy: torch.Tensor = torch.tensor([0.0, 1.0, 0.0]),
):
    """
    Adds a rectangular grid of nodes and edges to an existing StructData object.

    Creates (nx+1) chains of length ny along dy, offset by dx, then connects
    corresponding nodes across adjacent chains.

    Args:
        data: StructData to add the grid to.
        nx: Number of edges in the x direction.
        ny: Number of edges in the y direction.
        name: Prefix for node names.
        origin: Coordinates of the (0,0) corner node.
        dx: Step vector between adjacent chains.
        dy: Step vector between nodes within a chain.
    """
    for i in range(nx + 1):
        chain(data, ny, name=f"{name}_{i}", origin=origin + i * dx, direction=dy)

    for i in range(nx):
        for j in range(ny + 1):
            data.add_edge(f"{name}_{i}_node_{j}", f"{name}_{i+1}_node_{j}")


def chain(
    data: StructData,
    length: int,
    name: str = "chain",
    origin: torch.Tensor = torch.tensor([0.0, 0.0, 0.0]),
    direction: torch.Tensor = torch.tensor([1.0, 0.0, 0.0]),
):
    """
    Adds a linear chain of nodes connected by edges to an existing StructData object.

    Node i is placed at origin + i * direction. Nodes are named
    "{name}_node_0" through "{name}_node_{length}".

    Args:
        data: StructData to add the chain to.
        length: Number of edges (resulting in length + 1 nodes).
        name: Prefix for node names.
        origin: Coordinates of the first node. Default: (0, 0, 0).
        direction: Step vector from one node to the next. Default: (1, 0, 0).
    """
    for i in range(length + 1):
        data.add_node(f"{name}_node_{i}", coords=origin + i * direction)

    for i in range(length):
        data.add_edge(f"{name}_node_{i}", f"{name}_node_{i + 1}")






def _resolve_node_kwargs(node_kwargs, j):
    return node_kwargs(j) if callable(node_kwargs) else node_kwargs


def _resolve_node_load(node_load, load, is_last):
    if node_load is not None:
        return node_load
    return torch.zeros(3, dtype=torch.float) if is_last else load


def _resolve_force_sign(force_sign, j):
    try:
        return float(force_sign[j - 1])
    except (TypeError, KeyError):
        return force_sign


def build_trail(
    data: StructData,
    name: str,
    n_nodes: int,
    origin_coords: torch.Tensor,
    load: torch.Tensor = torch.zeros(3, dtype=torch.float),
    node_load: torch.Tensor = None,
    force_sign=-1.0,
    length: float = None,
    origin_node_kwargs: dict = {},
    node_kwargs={},
    edge_kwargs: dict = {},
    last_edge_kwargs: dict = None,
):
    """
    Adds a CEM trail to data: one origin node followed by n_nodes subsequent nodes,
    connected by trail edges. The last node automatically gets support_condition=[True,True,True].
    """

    base_edge_kwargs = {}
    if length is not None:
        base_edge_kwargs["length"] = length
    base_edge_kwargs.update(edge_kwargs)

    # Origin node
    data.add_node(
        f"{name}_node_0",
        coords=origin_coords,
        is_origin_node=torch.tensor(True),
        sequence=torch.tensor(0, dtype=torch.long),
        load=load,
        **origin_node_kwargs,
    )

    # Subsequent nodes & edges
    for j in range(1, n_nodes + 1):
        is_last = j == n_nodes

        data.add_node(
            f"{name}_node_{j}",
            is_origin_node=torch.tensor(False),
            sequence=torch.tensor(j, dtype=torch.long),
            load=_resolve_node_load(node_load, load, is_last),
            support_condition=(
                torch.tensor([True, True, True])
                if is_last
                else torch.tensor([False, False, False])
            ),
            **_resolve_node_kwargs(node_kwargs, j),
        )

        this_edge_kwargs = base_edge_kwargs if not is_last or last_edge_kwargs is None \
            else {**base_edge_kwargs, **last_edge_kwargs}

        edge_call_kwargs = {"is_trail_edge": torch.tensor(True), **this_edge_kwargs}
        fs = _resolve_force_sign(force_sign, j)
        if fs is not None:
            edge_call_kwargs["force_sign"] = torch.tensor(fs)

        data.add_edge(
            f"{name}_node_{j - 1}",
            f"{name}_node_{j}",
            **edge_call_kwargs,
        )

def add_chain(graph, start_node, end_node, points, prefix, node_kwargs=None, edge_kwargs=None):
    """
    Adds a chain of nodes from start_node through points to end_node.

    Each intermediate node is named f"{prefix}_{j}" and receives pattern_coords=point
    plus any extra node_kwargs. Edges connect consecutive nodes and the final node to
    end_node, all receiving edge_kwargs.
    """
    if node_kwargs is None:
        node_kwargs = {}
    if edge_kwargs is None:
        edge_kwargs = {}

    prev_node = start_node
    for j, point in enumerate(points):
        next_node = f"{prefix}_{j}"
        graph.add_node(next_node, pattern_coords=point, **node_kwargs)
        graph.add_edge(prev_node, next_node, **edge_kwargs)
        prev_node = next_node
    graph.add_edge(prev_node, end_node, **edge_kwargs)


def fix_graph(data, n_trails):
    """Merges all trail origin nodes into a single centroid node at [0,0,0]."""
    data.add_node(
        "centroid",
        coords=torch.tensor([0.0, 0.0, 0.0]),
        load=torch.tensor([0.0, 0.0, -1.0]),
    )
    merge_nodes = [f"trail_{i}_node_0" for i in range(n_trails)]
    data.merge_nodes("centroid", merge_nodes)


def add_quad_mesh(graph, n, pattern, boundary_density, centroid, diagonals):
    """
    Adds the sub-quad mesh nodes and edges for all n panels.
    Operates on nodes already present in graph (corners, boundary, skeleton, opening).
    """
    _no_edge = {"is_boundary_edge": torch.tensor(False)}

    for i in range(n):
        boundary_1 = [f"skeleton_{i}_{j}" for j in range(boundary_density)]
        boundary_2 = [f"boundary_{(i + 1) % n}_{j}" for j in range(boundary_density)]
        boundary_3 = [f"skeleton_{(i + 1) % n}_{j}" for j in range(boundary_density)]
        boundary_4 = list(reversed([f"boundary_{i}_{j}" for j in range(boundary_density + 1, 2 * boundary_density + 1)]))
        boundary_5 = [f"boundary_{i}_{j}" for j in range(2 * boundary_density + 1)]

        # add nodes
        if pattern == "standard":
            for j in range(boundary_density):
                for k in range(boundary_density):
                    line_1 = calculate_line(graph.nodes[boundary_1[j]]["pattern_coords"], graph.nodes[boundary_2[j]]["pattern_coords"])
                    line_2 = calculate_line(graph.nodes[boundary_3[k]]["pattern_coords"], graph.nodes[boundary_4[k]]["pattern_coords"])
                    graph.add_node(f"quad_{i}_{j}_{k}", pattern_coords=line_intersection(line_1, line_2), is_support=torch.tensor(False), is_boundary=torch.tensor(False))
        elif pattern == "singularity":
            for j in range(boundary_density):
                for k in range(2 * boundary_density + 1):
                    line_1 = calculate_line(graph.nodes[boundary_1[j]]["pattern_coords"], graph.nodes[boundary_3[j]]["pattern_coords"])
                    line_2 = calculate_line(graph.nodes[boundary_5[k]]["pattern_coords"], centroid)
                    graph.add_node(f"quad_{i}_{j}_{k}", pattern_coords=line_intersection(line_1, line_2), is_support=torch.tensor(False), is_boundary=torch.tensor(False))
        elif pattern == "opening":
            boundary_opening = [f"opening_skeleton_{i}_{j}" for j in range(2 * boundary_density + 1)]
            for j in range(boundary_density):
                for k in range(2 * boundary_density + 1):
                    line_1 = calculate_line(graph.nodes[boundary_1[j]]["pattern_coords"], graph.nodes[boundary_3[j]]["pattern_coords"])
                    line_2 = calculate_line(graph.nodes[boundary_5[k]]["pattern_coords"], graph.nodes[boundary_opening[k]]["pattern_coords"])
                    graph.add_node(f"quad_{i}_{j}_{k}", pattern_coords=line_intersection(line_1, line_2), is_support=torch.tensor(False), is_boundary=torch.tensor(False))
        else:
            raise ValueError(f"Invalid pattern: {pattern}")

        # add edges in first direction
        if pattern == "standard":
            for j in range(boundary_density):
                prev_node = boundary_2[j]
                for k in range(boundary_density):
                    next_node = f"quad_{i}_{j}_{k}"
                    graph.add_edge(prev_node, next_node, **_no_edge)
                    prev_node = next_node
                graph.add_edge(prev_node, boundary_1[j], **_no_edge)
        elif pattern == "singularity":
            for j in range(2 * boundary_density + 1):
                prev_node = boundary_5[j]
                for k in range(boundary_density):
                    next_node = f"quad_{i}_{k}_{j}"
                    graph.add_edge(prev_node, next_node, **_no_edge)
                    prev_node = next_node
                graph.add_edge(prev_node, "centroid", **_no_edge)
        elif pattern == "opening":
            for j in range(2 * boundary_density + 1):
                prev_node = boundary_5[j]
                for k in range(boundary_density):
                    next_node = f"quad_{i}_{k}_{j}"
                    graph.add_edge(prev_node, next_node, **_no_edge)
                    prev_node = next_node
                graph.add_edge(prev_node, boundary_opening[j], **_no_edge)
        else:
            raise ValueError(f"Invalid pattern: {pattern}")

        # add edges in second direction
        if pattern == "standard":
            for j in range(boundary_density):
                prev_node = boundary_4[j]
                for k in range(boundary_density):
                    next_node = f"quad_{i}_{k}_{j}"
                    graph.add_edge(prev_node, next_node, **_no_edge)
                    prev_node = next_node
                graph.add_edge(prev_node, boundary_3[j], **_no_edge)
        elif pattern in ["singularity", "opening"]:
            for j in range(boundary_density):
                prev_node = boundary_1[j]
                for k in range(2 * boundary_density + 1):
                    next_node = f"quad_{i}_{j}_{k}"
                    graph.add_edge(prev_node, next_node, is_boundary_edge=torch.tensor(False), ring=torch.tensor(j))
                    prev_node = next_node
                graph.add_edge(prev_node, boundary_3[j], is_boundary_edge=torch.tensor(False), ring=torch.tensor(j))
        else:
            raise ValueError(f"Invalid pattern: {pattern}")

        # add diagonal edges
        if diagonals:
            if pattern != "standard":
                raise ValueError("Diagonals are only supported for the standard pattern.")
            prev_node = f"corner_{(i + 1) % n}"
            for j in range(boundary_density):
                next_node = f"quad_{i}_{j}_{j}"
                graph.add_edge(prev_node, next_node, is_boundary_edge=torch.tensor(False), is_diagonal_edge=torch.tensor(True))
                prev_node = next_node
            graph.add_edge(prev_node, "centroid", is_boundary_edge=torch.tensor(False), is_diagonal_edge=torch.tensor(True))
