import torch
import math
import numpy as np
import random

from torch_structure.data import StructData
from torch_structure.generators.base_generator import BaseGenerator, InvalidSampleError


class CableNetGenerator(BaseGenerator):
    def __init__(self, **overrides):
        super().__init__(**overrides)
        self.max_attempts = 100
        self.node_attrs = {
            "pattern_coords": torch.empty((0, 2), dtype=torch.float),
            "is_support": torch.empty((0, 1), dtype=torch.bool),
            "is_boundary": torch.empty((0, 1), dtype=torch.bool),
            "z_coord": torch.empty((0, 1), dtype=torch.float),
        }
        self.edge_attrs = {
            "is_boundary_edge": torch.empty((0, 1), dtype=torch.bool),
            "is_diagonal_edge": torch.empty((0, 1), dtype=torch.bool),
            "is_opening_edge": torch.empty((0, 1), dtype=torch.bool),
            "ring": torch.empty((0, 1), dtype=torch.int),
        }
        self.default_attrs = {
            "z_coord": torch.tensor(0.0),
            "is_diagonal_edge": torch.tensor(False),
            "is_opening_edge": torch.tensor(False),
            "ring": torch.tensor(torch.nan),
        }

    def validate_inputs(
        self,
        n,
        boundary_density,
        support_height,
        pattern,
        diagonals,
        centroid_support,
        unsupported_boundaries,
        rectangle,
        square,
        curve_boundaries,
        circle,
        q_field,
        q_boundary,
        q_diagonal,
        rectangle_width,
        rectangle_height,
        square_size,
        corner_angle_list,
        corner_support_sequence,
    ):
        if n < 4:
            raise ValueError("n must be at least 4")
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
        if len(corner_support_sequence) != n:
            raise ValueError("corner_support_sequence must have length n")

    def sample_input(
        self,
        n=None,
        boundary_density=None,
        support_height=None,
        pattern="standard",
        diagonals=None,
        centroid_support=False,
        unsupported_boundaries=True,
        rectangle=False,
        square=False,
        curve_boundaries=True,
        circle=False,
        q_field=None,
        q_boundary=None,
        q_diagonal=None,
        rectangle_width=None,
        rectangle_height=None,
        square_size=None,
        corner_angle_list=None,
        corner_support_sequence=None,
    ):
        if n is None:
            if rectangle or square:
                n = 4
            else:
                n = np.random.randint(4, 7)
        if boundary_density is None:
            boundary_density = np.random.randint(3, 8)
        if support_height is None:
            support_height = np.random.uniform(0.35, 0.8)
        if diagonals is None:
            diagonals = (
                np.random.choice([True, False]) if pattern == "standard" else False
            )
        if q_field is None:
            q_field = np.random.uniform(30.0, 50.0)
        if q_boundary is None:
            q_boundary = np.random.uniform(200.0, 300.0)
        if q_diagonal is None:
            q_diagonal = np.random.uniform(30.0, 100.0)
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
            min_angle = torch.tensor(math.pi) / 8
            corner_angle_list = []
            while len(corner_angle_list) < n:
                angle = (torch.rand(1) * 2 * math.pi).item()
                # Check if the new angle is far enough from all others
                if all(
                    abs((angle - a + math.pi) % (2 * math.pi) - math.pi) >= min_angle
                    for a in corner_angle_list
                ):
                    corner_angle_list.append(angle)
            corner_angle_list.sort()
        if corner_support_sequence is None:
            corner_support_sequence = self.generate_random_sequence(n)

        input = {
            "n": n,
            "boundary_density": boundary_density,
            "support_height": support_height,
            "pattern": pattern,
            "diagonals": diagonals,
            "centroid_support": centroid_support,
            "unsupported_boundaries": unsupported_boundaries,
            "rectangle": rectangle,
            "square": square,
            "curve_boundaries": curve_boundaries,
            "circle": circle,
            "q_field": q_field,
            "q_boundary": q_boundary,
            "q_diagonal": q_diagonal,
            "rectangle_width": torch.tensor(rectangle_width, dtype=torch.float),
            "rectangle_height": torch.tensor(rectangle_height, dtype=torch.float),
            "square_size": torch.tensor(square_size, dtype=torch.float),
            "corner_angle_list": torch.tensor(corner_angle_list, dtype=torch.float),
            "corner_support_sequence": torch.tensor(
                corner_support_sequence, dtype=torch.long
            ),
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
        support_height,
        q_field,
        q_boundary,
        q_diagonal,
        rectangle_width,
        rectangle_height,
        square_size,
        corner_angle_list,
        corner_support_sequence,
    ):
        graph = StructData(
            node_attrs=self.node_attrs,
            edge_attrs=self.edge_attrs,
            default_attrs=self.default_attrs,
        )

        if rectangle:
            if square:
                corner_points = self.sample_rectangle(width=square_size, square=True)
            else:
                corner_points = self.sample_rectangle(
                    width=rectangle_width, height=rectangle_height, square=False
                )
        elif circle:
            corner_points = self.sample_unit_circle(
                n, angles=torch.linspace(0, 2 * math.pi, n + 1)[:-1]
            )
        else:
            corner_points = self.sample_unit_circle(n, angles=corner_angle_list)
        looped_corner_points = corner_points + [corner_points[0]]
        centroid = self.polygon_centroid(looped_corner_points)
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
            circle_angles = self.compute_optimal_rotation(angles)
            opening_radius = torch.rand(1) * 0.05 + 0.1
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
        support_sequence = corner_support_sequence * support_height
        for i, point in enumerate(corner_points):
            graph.add_node(
                f"corner_{i}",
                pattern_coords=point,
                is_support=torch.tensor(True),
                is_boundary=torch.tensor(True),
                z_coord=support_sequence[i],
            )

        # add mesh skeleton points
        for i in range(n):
            if unsupported_boundaries:
                is_support = False
            else:
                is_support = torch.tensor(True)
            start_boundary, end_boundary = (
                looped_corner_points[i],
                looped_corner_points[i + 1],
            )
            center_boundary = 0.5 * (start_boundary + end_boundary)
            if curve_boundaries:
                if is_support:
                    curvature_param = torch.rand(1) - 0.5
                else:
                    curvature_param = torch.rand(1) * 0.2 + 0.3
            else:
                curvature_param = 0.0
            control_point = center_boundary + curvature_param * (
                centroid - center_boundary
            )

            # add external boundary points
            if circle:
                boundary_curve = self.circular_arc(
                    start_boundary, end_boundary, centroid, boundary_density * 2 + 3
                )[1:-1]
            else:
                boundary_curve = self.quadratic_bezier(
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
            if pattern == "standard":
                internal_boundary = self.nd_linspace(
                    boundary_curve[len(boundary_curve) // 2],
                    centroid,
                    boundary_density + 2,
                )[1:-1]
                prev_node = f"boundary_{i}_{len(boundary_curve) // 2}"
                for j, point in enumerate(internal_boundary):
                    next_node = f"skeleton_{i}_{j}"
                    graph.add_node(
                        next_node,
                        pattern_coords=point,
                        is_support=torch.tensor(False),
                        is_boundary=torch.tensor(False),
                    )
                    graph.add_edge(
                        prev_node, next_node, is_boundary_edge=torch.tensor(False)
                    )
                    prev_node = next_node
                graph.add_edge(
                    next_node, "centroid", is_boundary_edge=torch.tensor(False)
                )
            elif pattern == "singularity":
                internal_boundary = self.nd_linspace(
                    corner_points[i], centroid, boundary_density + 2
                )[1:-1]
                prev_node = f"corner_{i}"
                for j, point in enumerate(internal_boundary):
                    next_node = f"skeleton_{i}_{j}"
                    graph.add_node(
                        next_node,
                        pattern_coords=point,
                        is_support=torch.tensor(False),
                        is_boundary=torch.tensor(False),
                    )
                    graph.add_edge(
                        prev_node, next_node, is_boundary_edge=torch.tensor(False)
                    )
                    prev_node = next_node
                graph.add_edge(
                    next_node, "centroid", is_boundary_edge=torch.tensor(False)
                )
            elif pattern == "opening":
                internal_boundary = self.nd_linspace(
                    corner_points[i],
                    graph.nodes[f"opening_{i}"]["pattern_coords"],
                    boundary_density + 2,
                )[1:-1]
                prev_node = f"corner_{i}"
                for j, point in enumerate(internal_boundary):
                    next_node = f"skeleton_{i}_{j}"
                    graph.add_node(
                        next_node,
                        pattern_coords=point,
                        is_support=torch.tensor(False),
                        is_boundary=torch.tensor(False),
                    )
                    graph.add_edge(
                        prev_node, next_node, is_boundary_edge=torch.tensor(False)
                    )
                    prev_node = next_node
                graph.add_edge(
                    next_node, f"opening_{i}", is_boundary_edge=torch.tensor(False)
                )
                opening_boundary = self.circular_arc(
                    graph.nodes[f"opening_{i}"]["pattern_coords"],
                    graph.nodes[f"opening_{(i + 1) % n}"]["pattern_coords"],
                    centroid,
                    2 * boundary_density + 3,
                )[1:-1]
                prev_node = f"opening_{i}"
                for j, point in enumerate(opening_boundary):
                    next_node = f"opening_skeleton_{i}_{j}"
                    graph.add_node(
                        next_node,
                        pattern_coords=point,
                        is_support=centroid_support,
                        is_boundary=torch.tensor(True),
                    )
                    graph.add_edge(
                        prev_node,
                        next_node,
                        is_boundary_edge=torch.tensor(False),
                        is_opening_edge=torch.tensor(True),
                    )
                    prev_node = next_node
                graph.add_edge(
                    next_node,
                    f"opening_{(i + 1) % n}",
                    is_boundary_edge=torch.tensor(False),
                    is_opening_edge=torch.tensor(True),
                )
            else:
                raise ValueError(f"Invalid pattern: {pattern}")

        # add sub quad meshes
        for i in range(n):
            boundary_1 = [f"skeleton_{i}_{j}" for j in range(boundary_density)]
            boundary_2 = [
                f"boundary_{(i + 1) % n}_{j}" for j in range(0, boundary_density)
            ]
            boundary_3 = [
                f"skeleton_{(i + 1) % n}_{j}" for j in range(boundary_density)
            ]
            boundary_4 = list(
                reversed(
                    [
                        f"boundary_{i}_{j}"
                        for j in range(boundary_density + 1, 2 * boundary_density + 1)
                    ]
                )
            )
            boundary_5 = [
                f"boundary_{i}_{j}" for j in range(0, 2 * boundary_density + 1)
            ]

            # add nodes
            if pattern == "standard":
                for j in range(boundary_density):
                    for k in range(boundary_density):
                        line_1 = self.calculate_line(
                            graph.nodes[boundary_1[j]]["pattern_coords"],
                            graph.nodes[boundary_2[j]]["pattern_coords"],
                        )
                        line_2 = self.calculate_line(
                            graph.nodes[boundary_3[k]]["pattern_coords"],
                            graph.nodes[boundary_4[k]]["pattern_coords"],
                        )
                        intersection = self.line_intersection(line_1, line_2)
                        graph.add_node(
                            f"quad_{i}_{j}_{k}",
                            pattern_coords=intersection,
                            is_support=torch.tensor(False),
                            is_boundary=torch.tensor(False),
                        )
            elif pattern == "singularity":
                for j in range(boundary_density):
                    for k in range(2 * boundary_density + 1):
                        line_1 = self.calculate_line(
                            graph.nodes[boundary_1[j]]["pattern_coords"],
                            graph.nodes[boundary_3[j]]["pattern_coords"],
                        )
                        line_2 = self.calculate_line(
                            graph.nodes[boundary_5[k]]["pattern_coords"], centroid
                        )
                        intersection = self.line_intersection(line_1, line_2)
                        graph.add_node(
                            f"quad_{i}_{j}_{k}",
                            pattern_coords=intersection,
                            is_support=torch.tensor(False),
                            is_boundary=torch.tensor(False),
                        )
            elif pattern == "opening":
                boundary_opening = [
                    f"opening_skeleton_{i}_{j}" for j in range(2 * boundary_density + 1)
                ]
                for j in range(boundary_density):
                    for k in range(2 * boundary_density + 1):
                        line_1 = self.calculate_line(
                            graph.nodes[boundary_1[j]]["pattern_coords"],
                            graph.nodes[boundary_3[j]]["pattern_coords"],
                        )
                        line_2 = self.calculate_line(
                            graph.nodes[boundary_5[k]]["pattern_coords"],
                            graph.nodes[boundary_opening[k]]["pattern_coords"],
                        )
                        intersection = self.line_intersection(line_1, line_2)
                        graph.add_node(
                            f"quad_{i}_{j}_{k}",
                            pattern_coords=intersection,
                            is_support=torch.tensor(False),
                            is_boundary=torch.tensor(False),
                        )
            else:
                raise ValueError(f"Invalid pattern: {pattern}")

            # add edges in first direction
            if pattern == "standard":
                for j in range(boundary_density):
                    prev_node = boundary_2[j]
                    for k in range(boundary_density):
                        next_node = f"quad_{i}_{j}_{k}"
                        graph.add_edge(
                            prev_node, next_node, is_boundary_edge=torch.tensor(False)
                        )
                        prev_node = next_node
                    graph.add_edge(
                        prev_node, boundary_1[j], is_boundary_edge=torch.tensor(False)
                    )
            elif pattern == "singularity":
                for j in range(2 * boundary_density + 1):
                    prev_node = boundary_5[j]
                    for k in range(boundary_density):
                        next_node = f"quad_{i}_{k}_{j}"
                        graph.add_edge(
                            prev_node, next_node, is_boundary_edge=torch.tensor(False)
                        )
                        prev_node = next_node
                    graph.add_edge(
                        prev_node,
                        "centroid",
                        is_boundary_edge=torch.tensor(False),
                    )
            elif pattern == "opening":
                for j in range(2 * boundary_density + 1):
                    prev_node = boundary_5[j]
                    for k in range(boundary_density):
                        next_node = f"quad_{i}_{k}_{j}"
                        graph.add_edge(
                            prev_node, next_node, is_boundary_edge=torch.tensor(False)
                        )
                        prev_node = next_node
                    graph.add_edge(
                        prev_node,
                        boundary_opening[j],
                        is_boundary_edge=torch.tensor(False),
                    )
            else:
                raise ValueError(f"Invalid pattern: {pattern}")

            # add edges in second direction
            if pattern == "standard":
                for j in range(boundary_density):
                    prev_node = boundary_4[j]
                    for k in range(boundary_density):
                        next_node = f"quad_{i}_{k}_{j}"
                        graph.add_edge(
                            prev_node, next_node, is_boundary_edge=torch.tensor(False)
                        )
                        prev_node = next_node
                    graph.add_edge(
                        prev_node, boundary_3[j], is_boundary_edge=torch.tensor(False)
                    )
            elif pattern in ["singularity", "opening"]:
                for j in range(boundary_density):
                    prev_node = boundary_1[j]
                    for k in range(2 * boundary_density + 1):
                        next_node = f"quad_{i}_{j}_{k}"
                        graph.add_edge(
                            prev_node,
                            next_node,
                            is_boundary_edge=torch.tensor(False),
                            ring=torch.tensor(j),
                        )
                        prev_node = next_node
                    graph.add_edge(
                        prev_node,
                        boundary_3[j],
                        is_boundary_edge=torch.tensor(False),
                        ring=torch.tensor(j),
                    )
            else:
                raise ValueError(f"Invalid pattern: {pattern}")

            # add diagonal edges
            if diagonals:
                if pattern != "standard":
                    raise ValueError(
                        "Diagonals are only supported for the standard pattern."
                    )
                prev_node = f"corner_{(i + 1) % n}"
                for j in range(boundary_density):
                    next_node = f"quad_{i}_{j}_{j}"
                    graph.add_edge(
                        prev_node,
                        next_node,
                        is_boundary_edge=torch.tensor(False),
                        is_diagonal_edge=torch.tensor(True),
                    )
                    prev_node = next_node
                graph.add_edge(
                    prev_node,
                    "centroid",
                    is_boundary_edge=torch.tensor(False),
                    is_diagonal_edge=torch.tensor(True),
                )

        load = torch.zeros((graph.num_nodes, 3), dtype=torch.float)
        load[~graph.is_support.view(-1)] = torch.tensor(
            [0.0, 0.0, -0.1], dtype=torch.float
        )
        graph.load = load

        graph.coords = torch.cat(
            [graph.pattern_coords, graph.z_coord],
            dim=1,
        )

        # Compute distance to centroid
        graph.centroid_distance = torch.linalg.norm(
            graph.pattern_coords - centroid, dim=1, keepdim=True
        )

        # Set force densities
        q = q_field * torch.ones(graph.num_edges)
        q[graph.is_boundary_edge.view(-1)] = q_boundary
        q[graph.is_diagonal_edge.view(-1)] = q_diagonal
        graph.force_density = q.unsqueeze(1)

        # Force density method
        graph = graph.fdm()

        if graph.bbox[0, 2] < 0.0 or graph.bbox[1, 2] > 1.0:
            raise InvalidSampleError("Z out of bounds:", graph.bbox[0, 2], graph.bbox[1, 2])

        # Set support
        graph.is_support = graph.is_support

        return graph

    # def is_internal_edge(self, u, v):
    #     """Check if both nodes of the edge are not on the outer boundary."""
    #     return self.is_internal_node(u) or self.is_internal_node(v)

    # def is_internal_node(self, node):
    #     """Check if a node is internal (not on the outer boundary)."""
    #     x, y = node
    #     return 0 < x < self.n_cell_x and 0 < y < self.n_cell_y

    @staticmethod
    def calculate_line(p1, p2):
        """
        Calculate the line equation (Ax + By + C = 0) from two points.

        Args:
            p1: Tuple of (x1, y1), the first point.
            p2: Tuple of (x2, y2), the second point.

        Returns:
            A tuple (A, B, C) representing the line equation.
        """
        x1, y1 = p1
        x2, y2 = p2

        # Ensure the points are not the same
        if (x1, y1) == (x2, y2):
            raise ValueError("Two points must be distinct to define a line.")

        # Line equation: (y2 - y1)x - (x2 - x1)y + (x2*y1 - x1*y2) = 0
        A = y2 - y1
        B = -(x2 - x1)
        C = x2 * y1 - x1 * y2

        return A, B, C

    @staticmethod
    def line_intersection(line1, line2):
        """
        Calculate the intersection point of two lines.

        Args:
            line1: A tuple (A1, B1, C1) for the first line equation (A1x + B1y + C1 = 0).
            line2: A tuple (A2, B2, C2) for the second line equation (A2x + B2y + C2 = 0).

        Returns:
            A tuple (x, y) representing the intersection point, or None if the lines are parallel.
        """
        A1, B1, C1 = line1
        A2, B2, C2 = line2

        # Calculate the determinant
        det = A1 * B2 - A2 * B1

        # If determinant is zero, lines are parallel or coincident
        if det == 0:
            return None

        # Use Cramer's rule to find the intersection point
        x = (B1 * C2 - B2 * C1) / det
        y = (A2 * C1 - A1 * C2) / det

        return torch.tensor([x, y])

    @staticmethod
    def quadratic_bezier(p0, p1, p2, num_points=100):
        """
        Generates a quadratic Bezier curve.

        Parameters:
            p0 (tuple): The starting point (x0, y0).
            p1 (tuple): The control point (x1, y1).
            p2 (tuple): The ending point (x2, y2).
            num_points (int): Number of points to calculate on the curve.

        Returns:
            list of tuple: Points on the Bezier curve.
        """
        t_values = torch.linspace(0, 1, num_points)
        curve = []

        for t in t_values:
            x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t**2 * p2[0]
            y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t**2 * p2[1]
            curve.append(torch.tensor([x, y]))

        return curve

    @staticmethod
    def circular_arc(p0, p1, p2, num_points=100):
        """
        Generates a section of a circle from p0 to p1, using p2 as the center.

        Parameters:
            p0 (tuple): The starting point (x0, y0).
            p1 (tuple): The ending point (x1, y1).
            p2 (tuple): The center of the circle (cx, cy).
            num_points (int): Number of points to calculate on the arc.

        Returns:
            list of tuple: Points on the circular arc.
        """
        # Compute radius from center to p0
        radius = torch.linalg.norm(torch.tensor(p0) - torch.tensor(p2))

        # Compute angles of p0 and p1 relative to the center
        angle0 = torch.arctan2(p0[1] - p2[1], p0[0] - p2[0])
        angle1 = torch.arctan2(p1[1] - p2[1], p1[0] - p2[0])

        # Ensure angles are in the correct order for a continuous arc
        if angle1 < angle0:
            angle1 += 2 * torch.pi  # Ensure counterclockwise motion

        # Generate arc points
        t_values = torch.linspace(angle0, angle1, num_points)
        arc = []

        for t in t_values:
            x = p2[0] + radius * torch.cos(t)
            y = p2[1] + radius * torch.sin(t)
            arc.append(torch.tensor([x, y]))

        return arc

    @staticmethod
    def compute_optimal_rotation(polygon_angles):
        print(torch.tensor(math.pi), len(polygon_angles))
        circle_angles = torch.linspace(0.0, 2 * math.pi, len(polygon_angles) + 1)[:-1]
        angular_differences = polygon_angles - circle_angles
        theta_shift = torch.arctan2(
            torch.sum(torch.sin(angular_differences)),
            torch.sum(torch.cos(angular_differences)),
        )
        return circle_angles + theta_shift

    @staticmethod
    def polygon_centroid(vertices):
        """
        Calculates the centroid of a polygon.

        Parameters:
            vertices (list of tuple): List of (x, y) coordinates of the polygon vertices.
                                    The polygon should be closed (first vertex == last vertex).

        Returns:
            tuple: (Cx, Cy) coordinates of the centroid.
        """

        n = len(vertices)
        A = 0  # Signed area
        Cx = 0  # Centroid x-coordinate
        Cy = 0  # Centroid y-coordinate

        for i in range(n - 1):
            x0, y0 = vertices[i]
            x1, y1 = vertices[i + 1]
            cross = x0 * y1 - x1 * y0
            A += cross
            Cx += (x0 + x1) * cross
            Cy += (y0 + y1) * cross

        A *= 0.5
        Cx /= 6 * A
        Cy /= 6 * A

        return torch.tensor([Cx, Cy])

    @staticmethod
    def sample_rectangle(width, height=None, square=False):
        """
        Randomly samples a rectangle with a minimum aspect ratio of 1:2.

        Returns:
            list of tuple: List of (x, y) coordinates of the rectangle vertices.
        """
        if square:
            height = width

        x0, y0 = -0.5 * width, -0.5 * height
        x1, y1 = 0.5 * width, 0.5 * height

        return [
            torch.tensor([x0, y0]),
            torch.tensor([x1, y0]),
            torch.tensor([x1, y1]),
            torch.tensor([x0, y1]),
        ]

    @staticmethod
    def sample_unit_circle(n, angles=None):
        """
        Randomly samples n points on a unit circle, ensuring a minimum angle between points.

        Parameters:
            n (int): Number of points to sample.
            min_angle (float): Minimum angular separation between points (in radians).

        Returns:
            list of tuple: Randomly sampled and sorted points on the unit circle.
        """
        if n == 0:
            return []

        points = [
            torch.tensor([torch.cos(angle), torch.sin(angle)]) for angle in angles
        ]
        return points

    @staticmethod
    def nd_linspace(start: torch.tensor, end: torch.tensor, num_points: int):
        t = torch.linspace(0, 1, num_points).view(-1, 1)
        return start + t * (end - start)

    @staticmethod
    def generate_random_sequence(n):
        sequence = np.array([1, 0, 1, 0])  # Start with [1, 0, 1, 0] as a numpy array

        while len(sequence) < n:
            insert_index = random.randint(0, len(sequence))  # Choose a random index
            insert_value = random.choice([0, 1])  # Randomly choose 0 or 1
            sequence = np.insert(
                sequence, insert_index, insert_value
            )  # Insert at the chosen position

        return sequence
