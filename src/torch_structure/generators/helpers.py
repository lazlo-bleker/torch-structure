
import math
import random

import numpy as np
import torch

def nd_linspace(start: torch.tensor, end: torch.tensor, num_points: int):
    t = torch.linspace(0, 1, num_points).view(-1, 1)
    return start + t * (end - start)


def sample_corner_angles(n):
    min_angle = torch.tensor(math.pi) / 8
    corner_angle_list = []
    while len(corner_angle_list) < n:
        angle = (torch.rand(1) * 2 * math.pi).item()
        if all(
            abs((angle - a + math.pi) % (2 * math.pi) - math.pi) >= min_angle
            for a in corner_angle_list
        ):
            corner_angle_list.append(angle)
    corner_angle_list.sort()
    return corner_angle_list


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


def generate_random_sequence(n):
        sequence = np.array([1, 0, 1, 0])  # Start with [1, 0, 1, 0] as a numpy array

        while len(sequence) < n:
            insert_index = random.randint(0, len(sequence))  # Choose a random index
            insert_value = random.choice([0, 1])  # Randomly choose 0 or 1
            sequence = np.insert(
                sequence, insert_index, insert_value
            )  # Insert at the chosen position

        return sequence


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

def compute_optimal_rotation(polygon_angles):
    circle_angles = torch.linspace(0.0, 2 * math.pi, len(polygon_angles) + 1)[:-1]
    angular_differences = polygon_angles - circle_angles
    theta_shift = torch.arctan2(
        torch.sum(torch.sin(angular_differences)),
        torch.sum(torch.cos(angular_differences)),
    )
    return circle_angles + theta_shift

def compute_trail_origin(angle, opening, trail_length, center_deviation_force, centroid, n_trails):
    """Returns (origin_coords, origin_load, origin_diameter) for a single dome trail."""
    if opening:
        origin_diameter = trail_length
        x = torch.cos(angle) * 0.5 * origin_diameter
        y = torch.sin(angle) * 0.5 * origin_diameter
        origin_coords = centroid + torch.tensor([x, y, 0.0])
        origin_load = torch.tensor([0.0, 0.0, -1.0])
    else:
        origin_diameter = None
        origin_coords = centroid
        x_load = torch.cos(angle) * center_deviation_force
        y_load = torch.sin(angle) * center_deviation_force
        origin_load = torch.tensor([x_load, y_load, -1.0 / n_trails])
    return origin_coords, origin_load, origin_diameter


def add_opening_ring_edges(data, n_trails, ring_force):
    """Adds ring deviation edges at the opening (node_0) of a dome."""
    for i in range(n_trails):
        data.add_edge(
            f"trail_{i}_node_0",
            f"trail_{(i + 1) % n_trails}_node_0",
            is_trail_edge=torch.tensor(False),
            force=ring_force,
        )


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

    