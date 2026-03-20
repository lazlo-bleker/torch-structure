import torch
import math
import numpy as np

from torch_structure.data import StructData
from torch_structure.generators.base_generator import BaseGenerator, InvalidSampleError
from torch_structure.generators.helpers import nd_linspace, sample_unit_circle, sample_rectangle, polygon_centroid, compute_optimal_rotation, circular_arc, quadratic_bezier, sample_corner_angles
from torch_structure.generators.topology import add_chain, add_quad_mesh


class GridShellGenerator(BaseGenerator):
    def __init__(self, **overrides):
        super().__init__(**overrides)
        self.max_attempts = 100
        self.node_attrs = {
            "pattern_coords": torch.empty((0, 2), dtype=torch.float),
            "is_support": torch.empty((0, 1), dtype=torch.bool),
            "is_boundary": torch.empty((0, 1), dtype=torch.bool),
        }
        self.edge_attrs = {
            "is_boundary_edge": torch.empty((0, 1), dtype=torch.bool),
            "is_diagonal_edge": torch.empty((0, 1), dtype=torch.bool),
            "is_opening_edge": torch.empty((0, 1), dtype=torch.bool),
            "ring": torch.empty((0, 1), dtype=torch.int),
        }
        self.default_attrs = {
            "is_diagonal_edge": torch.tensor(False),
            "is_opening_edge": torch.tensor(False),
            "ring": torch.tensor(torch.nan),
        }

    def validate_input(
        self,
        n,
        boundary_density,
        pattern,
        diagonals,
        centroid_support,
        unsupported_boundaries,
        rectangle,
        square,
        curve_boundaries,
        circle,
        q_target_field,
        q_target_boundary,
        opening_radius,
        boundary_support_list,
        boundary_curvature_list,
        rectangle_width,
        rectangle_height,
        square_size,
        corner_angle_list,
    ):
        if pattern not in ["standard", "singularity", "opening"]:
            raise ValueError(f"Invalid pattern: {pattern}")
        if rectangle and n != 4:
            raise ValueError("Rectangle pattern requires n=4.")
        if square and not rectangle:
            raise ValueError("Square pattern requires rectangle=True.")
        if circle and pattern not in ["singularity", "opening"]:
            raise ValueError("Circle requires pattern='singularity' or 'opening'.")
        if circle and (square or rectangle):
            raise ValueError("Circle is incompatible with square or rectangle")
        if pattern == "opening" and not centroid_support:
            raise ValueError("Opening pattern requires centroid support.")

    def sample_input(
        self,
        n=None,
        boundary_density=None,
        pattern=None,
        diagonals=None,
        centroid_support=None,
        unsupported_boundaries=True,
        rectangle=False,
        square=False,
        curve_boundaries=True,
        circle=False,
        q_target_field=None,
        q_target_boundary=None,
        opening_radius=None,
        boundary_support_list=None,
        boundary_curvature_list=None,
        rectangle_width=None,
        rectangle_height=None,
        square_size=None,
        corner_angle_list=None,
    ):
        if n is None:
            if rectangle or square:
                n = 4
            else:
                n = np.random.randint(3, 7)
        if boundary_density is None:
            boundary_density = np.random.randint(3, 8)
        if centroid_support is None:
            centroid_support = np.random.choice([True, False], p=[0.3, 0.7])
        if pattern is None:
            pattern = (
                np.random.choice(["standard", "singularity", "opening"])
                if centroid_support
                else "standard"
            )
        if diagonals is None:
            diagonals = (
                np.random.choice([True, False]) if pattern == "standard" else False
            )
        if q_target_field is None:
            q_target_field = np.random.uniform(-15.0, -25.0)
        if q_target_boundary is None:
            q_target_boundary = np.random.uniform(-80.0, -120.0)
        if opening_radius is None:
            if pattern == "opening":
                opening_radius = np.random.rand() * 0.05 + 0.1
            else:
                opening_radius = np.nan
        if boundary_support_list is None:
            if unsupported_boundaries:
                boundary_support_list = np.random.choice(
                    [True, False], size=n, p=[0.5, 0.5]
                )
            else:
                boundary_support_list = [True] * n
        if boundary_curvature_list is None:
            if curve_boundaries:
                boundary_curvature_list = np.empty(n)
                boundary_curvature_list[boundary_support_list] = (
                    np.random.rand(boundary_support_list.sum()) - 0.5
                )
                boundary_curvature_list[~boundary_support_list] = (
                    np.random.rand((~boundary_support_list).sum()) * 0.2 + 0.3
                )
            else:
                boundary_curvature_list = np.zeros(n)
        if rectangle_width is None:
            if rectangle:
                rectangle_width = torch.rand(1) * 0.9 + 0.1
            else:
                rectangle_width = torch.nan
        if rectangle_height is None:
            if rectangle:
                rectangle_height = torch.rand(1) * 0.9 + 0.1
            else:
                rectangle_height = torch.nan
        if square_size is None:
            if square:
                square_size = torch.rand(1) * 0.9 + 0.1
            else:
                square_size = torch.nan
        if corner_angle_list is None:
            corner_angle_list = sample_corner_angles(n)

        input = {
            "n": n,
            "boundary_density": boundary_density,
            "pattern": pattern,
            "diagonals": torch.tensor(diagonals),
            "centroid_support": torch.tensor(centroid_support),
            "unsupported_boundaries": torch.tensor(unsupported_boundaries),
            "rectangle": torch.tensor(rectangle),
            "square": torch.tensor(square),
            "curve_boundaries": torch.tensor(curve_boundaries),
            "circle": torch.tensor(circle),
            "q_target_field": torch.tensor(q_target_field, dtype=torch.float),
            "q_target_boundary": torch.tensor(q_target_boundary, dtype=torch.float),
            "opening_radius": torch.tensor(opening_radius, dtype=torch.float),
            "boundary_support_list": torch.tensor(
                boundary_support_list, dtype=torch.bool
            ),
            "boundary_curvature_list": torch.tensor(
                boundary_curvature_list, dtype=torch.float
            ),
            "rectangle_width": torch.tensor(rectangle_width, dtype=torch.float),
            "rectangle_height": torch.tensor(rectangle_height, dtype=torch.float),
            "square_size": torch.tensor(square_size, dtype=torch.float),
            "corner_angle_list": torch.tensor(corner_angle_list, dtype=torch.float),
        }

        return input

    def generate(
        self,
        n,
        boundary_density,
        pattern,
        diagonals,
        centroid_support,
        unsupported_boundaries,
        rectangle,
        square,
        curve_boundaries,
        circle,
        q_target_field,
        q_target_boundary,
        opening_radius,
        boundary_support_list,
        boundary_curvature_list,
        rectangle_width,
        rectangle_height,
        square_size,
        corner_angle_list,
    ):
        graph = StructData(
            node_attrs=self.node_attrs,
            edge_attrs=self.edge_attrs,
            default_attrs=self.default_attrs,
        )

        if rectangle:
            if square:
                corner_points = sample_rectangle(width=square_size, square=True)
            else:
                corner_points = sample_rectangle(
                    width=rectangle_width, height=rectangle_height, square=False
                )
        elif circle:
            corner_points = sample_unit_circle(
                n, angles=torch.linspace(0, 2 * math.pi, n + 1)[:-1]
            )
        else:
            corner_points = sample_unit_circle(n, angles=corner_angle_list)
        looped_corner_points = corner_points + [corner_points[0]]
        centroid = polygon_centroid(looped_corner_points)
        if pattern in ["standard", "singularity"]:
            graph.add_node(
                "centroid",
                pattern_coords=centroid,
                is_support=centroid_support,
                is_boundary=torch.tensor(False),
            )
        elif pattern == "opening":
            angles = torch.zeros(n)
            for i in range(n):
                angles[i] = torch.arctan2(
                    corner_points[i][1] - centroid[1], corner_points[i][0] - centroid[0]
                )
            circle_angles = compute_optimal_rotation(angles)
            for i in range(n):
                pattern_coord = (
                    torch.tensor(
                        [torch.cos(circle_angles[i]), torch.sin(circle_angles[i])]
                    )
                    * opening_radius
                    + centroid
                )
                graph.add_node(
                    f"opening_{i}",
                    pattern_coords=pattern_coord,
                    is_support=centroid_support,
                    is_boundary=torch.tensor(True),
                )

        # add corner points
        for i, point in enumerate(corner_points):
            graph.add_node(
                f"corner_{i}",
                pattern_coords=point,
                is_support=torch.tensor(True),
                is_boundary=torch.tensor(True),
            )

        # add mesh skeleton points
        for i in range(n):
            is_support = boundary_support_list[i]
            start_boundary, end_boundary = (
                looped_corner_points[i],
                looped_corner_points[i + 1],
            )
            center_boundary = 0.5 * (start_boundary + end_boundary)
            curvature_param = boundary_curvature_list[i]
            control_point = center_boundary + curvature_param * (
                centroid - center_boundary
            )

            # add external boundary points
            if circle:
                boundary_curve = circular_arc(
                    start_boundary, end_boundary, centroid, boundary_density * 2 + 3
                )[1:-1]
            else:
                boundary_curve = quadratic_bezier(
                    start_boundary,
                    control_point,
                    end_boundary,
                    boundary_density * 2 + 3,
                )[1:-1]
            prev_node = f"corner_{i}"

            for j, point in enumerate(boundary_curve):
                next_node = f"boundary_{i}_{j}"
                graph.add_node(
                    next_node,
                    pattern_coords=point,
                    is_support=is_support,
                    is_boundary=torch.tensor(True),
                )
                if not is_support:
                    graph.add_edge(
                        prev_node, next_node, is_boundary_edge=torch.tensor(True)
                    )
                prev_node = next_node
            if not is_support:
                graph.add_edge(
                    next_node,
                    f"corner_{(i + 1) % n}",
                    is_boundary_edge=torch.tensor(True),
                )

            # add internal boundary points
            _skele_kwargs = {"is_support": torch.tensor(False), "is_boundary": torch.tensor(False)}
            _inner_edge_kwargs = {"is_boundary_edge": torch.tensor(False)}
            if pattern == "standard":
                add_chain(
                    graph,
                    start_node=f"boundary_{i}_{len(boundary_curve) // 2}",
                    end_node="centroid",
                    points=nd_linspace(boundary_curve[len(boundary_curve) // 2], centroid, boundary_density + 2)[1:-1],
                    prefix=f"skeleton_{i}",
                    node_kwargs=_skele_kwargs,
                    edge_kwargs=_inner_edge_kwargs,
                )
            elif pattern == "singularity":
                add_chain(
                    graph,
                    start_node=f"corner_{i}",
                    end_node="centroid",
                    points=nd_linspace(corner_points[i], centroid, boundary_density + 2)[1:-1],
                    prefix=f"skeleton_{i}",
                    node_kwargs=_skele_kwargs,
                    edge_kwargs=_inner_edge_kwargs,
                )
            elif pattern == "opening":
                add_chain(
                    graph,
                    start_node=f"corner_{i}",
                    end_node=f"opening_{i}",
                    points=nd_linspace(corner_points[i], graph.nodes[f"opening_{i}"]["pattern_coords"], boundary_density + 2)[1:-1],
                    prefix=f"skeleton_{i}",
                    node_kwargs=_skele_kwargs,
                    edge_kwargs=_inner_edge_kwargs,
                )
                add_chain(
                    graph,
                    start_node=f"opening_{i}",
                    end_node=f"opening_{(i + 1) % n}",
                    points=circular_arc(graph.nodes[f"opening_{i}"]["pattern_coords"], graph.nodes[f"opening_{(i + 1) % n}"]["pattern_coords"], centroid, 2 * boundary_density + 3)[1:-1],
                    prefix=f"opening_skeleton_{i}",
                    node_kwargs={"is_support": centroid_support, "is_boundary": torch.tensor(True)},
                    edge_kwargs={"is_boundary_edge": torch.tensor(False), "is_opening_edge": torch.tensor(True)},
                )
            else:
                raise ValueError(f"Invalid pattern: {pattern}")

        add_quad_mesh(graph, n, pattern, boundary_density, centroid, diagonals)

        load = torch.zeros((graph.num_nodes, 3), dtype=torch.float)
        load[~graph.is_support.view(-1)] = torch.tensor(
            [0.0, 0.0, -1.0], dtype=torch.float
        )
        graph.load = load

        graph.coords = torch.cat(
            [graph.pattern_coords, torch.zeros((graph.pattern_coords.shape[0], 1))],
            dim=1,
        )

        # Compute distance to centroid
        graph.centroid_distance = torch.linalg.norm(
            graph.pattern_coords - centroid, dim=1, keepdim=True
        )

        graph.xy_laplacian_smoothing(is_fixed=graph.is_boundary, verbose=False)
        q_target = q_target_field * torch.ones(graph.num_edges)
        q_target[graph.is_boundary_edge.view(-1)] = q_target_boundary
        graph.q_target = q_target.unsqueeze(1)

        graph = graph.tna(verbose=False)

        if graph.bbox[0, 2] < 0 or graph.bbox[1, 2] > 3.0:
            raise InvalidSampleError(
                f"Z-coordinates out of bounds: ({graph.bbox[0, 2]}, {graph.bbox[1, 2]})"
            )

        if graph.force_density.max() > 0.0:
            raise InvalidSampleError(f"Tension element(s) present: {graph.force_density.max()}")

        return graph

    def is_internal_edge(self, u, v):
        """Check if both nodes of the edge are not on the outer boundary."""
        return self.is_internal_node(u) or self.is_internal_node(v)

    def is_internal_node(self, node):
        """Check if a node is internal (not on the outer boundary)."""
        x, y = node
        return 0 < x < self.n_cell_x and 0 < y < self.n_cell_y

