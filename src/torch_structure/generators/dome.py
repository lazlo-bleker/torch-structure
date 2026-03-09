import torch
import math
import numpy as np

from torch_structure.data import StructData
from torch_structure.generators.base_generator_cem import BaseGeneratorCEM

class DomeGenerator(BaseGeneratorCEM):
    def __init__(self, **overrides):
        super().__init__(**overrides)
        self.max_attempts = 100

    def validate_input(
        self, n_trails, n_rings, trail_length, center_deviation_force, opening
    ):
        if n_trails % 2 != 0:
            raise ValueError("Number of trails must be even.")

    def sample_input(
        self,
        n_trails=None,
        n_rings=None,
        trail_length=None,
        center_deviation_force=None,
        opening=False,
    ):
        if n_trails is None:
            n_trails = 2 * np.random.randint(5, 15)
        if n_rings is None:
            n_rings = np.random.randint(3, 6)
        if trail_length is None:
            trail_length = np.random.uniform(0.05, 0.2)
        if center_deviation_force is None:
            center_deviation_force = np.random.uniform(-1.0, -8.0)

        return {
            "n_trails": n_trails,
            "n_rings": n_rings,
            "trail_length": trail_length,
            "center_deviation_force": center_deviation_force,
            "opening": opening,
        }

    def generate(
        self, n_trails, n_rings, trail_length, center_deviation_force, opening
    ):
        # Initialize data object
        data = StructData(
            node_attrs=self.node_attrs,
            edge_attrs=self.edge_attrs,
            default_attrs=self.default_attrs,
        )

        # Create topology diagram
        centroid = torch.tensor([0.0, 0.0, 0.0])
        angles = torch.linspace(0, 2 * math.pi, n_trails + 1)[:-1]

        # Generate trails
        for i in range(n_trails):
            angle = angles[i]
            if opening:
                origin_diameter = trail_length
                x = torch.cos(angle) * 0.5 * origin_diameter
                y = torch.sin(angle) * 0.5 * origin_diameter
                origin_coords = centroid + torch.tensor([x, y, 0.0])
                origin_load = torch.tensor([0.0, 0.0, -1.0])
            else:
                origin_coords = centroid
                x_load = torch.cos(angle) * center_deviation_force
                y_load = torch.sin(angle) * center_deviation_force
                origin_load = torch.tensor([x_load, y_load, -1.0 / n_trails])
            self.generate_trail(
                data=data,
                origin_coords=origin_coords,
                origin_load=origin_load,
                id=i,
                n_rings=n_rings,
                trail_length=trail_length,
            )

        # Add ring deviations
        for i in range(1, n_rings):
            force_sign = torch.randint(2, (1,)) * 2 - 1
            force_magnitude = torch.rand(1) * 2 + 1
            force = force_sign * force_magnitude
            for j in range(n_trails):
                data.add_edge(
                    f"trail_{j}_node_{i}",
                    f"trail_{(j + 1) % n_trails}_node_{i}",
                    is_trail_edge=torch.tensor(False),
                    force=force,
                )

        if opening:
            ring_force = center_deviation_force * origin_diameter * 50
            for i in range(n_trails):
                data.add_edge(
                    f"trail_{i}_node_0",
                    f"trail_{(i + 1) % n_trails}_node_0",
                    is_trail_edge=torch.tensor(False),
                    force=ring_force,
                )

        # Formfinding
        data = data.mpcem()
        if not opening:
            self.fix_graph(data, n_trails)

        if self.filter(data):
            raise RuntimeError("Negative inclination detected.")

        # Scale to unit length
        radius = torch.norm(data.coords[:, 0:2], dim=1).max()
        data.coords /= radius

        # Translate to positive coordinates
        data.coords[:, 2] -= data.coords[:, 2].min()

        # Set support
        data.is_support = data.is_support

        if data.bbox[0, 2] < 0 or data.bbox[1, 2] > 3.0:
            raise RuntimeError("Z out of bounds:", data.bbox[0, 2], data.bbox[1, 2])

        return data

    def generate_trail(
        self, data, origin_coords, origin_load, id, n_rings, trail_length
    ):
        data.add_node(
            f"trail_{id}_node_0",
            coords=origin_coords,
            is_origin_node=torch.tensor(True),
            sequence=torch.tensor(0),
            load=origin_load,
        )
        for i in range(1, n_rings + 1):
            support_condition = (
                torch.tensor([True, True, True])
                if i == n_rings
                else torch.tensor([False, False, False])
            )
            data.add_node(
                f"trail_{id}_node_{i}",
                is_origin_node=torch.tensor(False),
                sequence=torch.tensor(i),
                load=torch.tensor([0.0, 0.0, -1.0]),
                support_condition=support_condition,
            )
            data.add_edge(
                f"trail_{id}_node_{i - 1}",
                f"trail_{id}_node_{i}",
                is_trail_edge=torch.tensor(True),
                length=trail_length,
                force_sign=torch.tensor(-1.0),
            )

    def fix_graph(self, data, n_trails):
        data.add_node(
            "centroid",
            coords=torch.tensor([0.0, 0.0, 0.0]),
            load=torch.tensor([0.0, 0.0, -1.0]),
        )
        merge_nodes = [f"trail_{i}_node_0" for i in range(n_trails)]
        data.merge_nodes("centroid", merge_nodes)

    def filter(self, data):
        mask = (data.is_trail_edge & data.directed_mask).view(-1)
        src, dst = data.edge_index[:, mask]
        src_centroid_distance = torch.norm(data.coords[src, 0:2], dim=1)
        dst_centroid_distance = torch.norm(data.coords[dst, 0:2], dim=1)
        invalid = dst_centroid_distance < src_centroid_distance
        return invalid.any()
