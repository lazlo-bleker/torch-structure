import torch
import math
import numpy as np

from torch_structure.data import StructData
from torch_structure.generators.base_generator import BaseGenerator


class MixedDomeGenerator(BaseGenerator):
    def __init__(self, **overrides):
        super().__init__(**overrides)
        self.max_attempts = 100
        self.node_attrs = {
            "coords": torch.empty((0, 3), dtype=torch.float),
            "load": torch.empty((0, 3), dtype=torch.float),
            "support_condition": torch.empty((0, 3), dtype=torch.long),
            "is_origin_node": torch.empty((0, 1), dtype=torch.bool),
            "sequence": torch.empty((0, 1), dtype=torch.long),
        }
        self.edge_attrs = {
            "force": torch.empty((0, 1), dtype=torch.float),
            "length": torch.empty((0, 1), dtype=torch.float),
            "is_trail_edge": torch.empty((0, 1), dtype=torch.bool),
            "force_sign": torch.empty((0, 1), dtype=torch.float),
        }
        self.default_attrs = {
            "force": torch.tensor([torch.nan]),
            "length": torch.tensor([torch.nan]),
            "coords": torch.full((3,), torch.nan),
            "load": torch.zeros(3, dtype=torch.float),
            "support_condition": torch.zeros(3, dtype=torch.bool),
            "is_origin_node": torch.tensor(0, dtype=torch.bool),
        }

    def validate_input(
        self,
        n_trails,
        n_rings,
        trail_length,
        center_deviation_force,
        opening,
        sign_flip_indices,
        center_trail_sign,
    ):
        if n_trails % 2 != 0:
            raise ValueError("Number of trails must be even.")
        if center_trail_sign not in ["tension", "compression"]:
            raise ValueError(
                "center_trail_sign must be either 'tension' or 'compression'."
            )

    def sample_input(
        self,
        n_trails=None,
        n_rings=None,
        trail_length=0.1,
        center_deviation_force=None,
        opening=None,
        sign_flip_indices=None,
        center_trail_sign=None,
    ):
        if n_trails is None:
            n_trails = 2 * np.random.randint(5, 15)
        if n_rings is None:
            n_rings = np.random.randint(4, 8)
        if sign_flip_indices is None:
            sign_flip_indices = np.random.choice(
                np.arange(2, n_rings - 1),
                size=np.random.randint(0, min(3, n_rings - 2)),
                replace=False,
            )
        if center_trail_sign is None:
            center_trail_sign = np.random.choice(["tension", "compression"])
        if center_deviation_force is None:
            center_deviation_force = np.random.uniform(1.0, 4.0)
            if center_trail_sign == "compression":
                center_deviation_force *= -1
        if opening is None:
            opening = np.random.choice([True, False], p=[0.5, 0.5])

        return {
            "n_trails": n_trails,
            "n_rings": n_rings,
            "trail_length": trail_length,
            "center_deviation_force": center_deviation_force,
            "opening": opening,
            "sign_flip_indices": sign_flip_indices,
            "center_trail_sign": center_trail_sign,
        }

    def generate(
        self,
        n_trails,
        n_rings,
        trail_length,
        center_deviation_force,
        opening,
        sign_flip_indices,
        center_trail_sign,
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

        force_signs = torch.ones(n_rings, dtype=torch.float)
        if center_trail_sign == "compression":
            force_signs *= -1
        for i in sign_flip_indices:
            force_signs[i:] *= -1

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
                force_signs=force_signs,
            )

        # Add ring deviations
        for i in range(1, n_rings):
            force_sign = torch.randint(2, (1,)) * 2 - 1
            force_magnitude = torch.rand(1) * 3 + 3
            force = force_sign * force_magnitude
            if i in sign_flip_indices:
                force = force_signs[i] * np.random.uniform(20.0, 40.0)
            for j in range(n_trails):
                data.add_edge(
                    f"trail_{j}_node_{i}",
                    f"trail_{(j + 1) % n_trails}_node_{i}",
                    is_trail_edge=torch.tensor(False),
                    force=force,
                )

        if opening:
            ring_force = (
                center_deviation_force * origin_diameter * np.random.uniform(25, 75)
            )
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

        # Scale to unit length
        radius = torch.norm(data.coords[:, 0:2], dim=1).max()
        data.coords /= radius

        # Translate to positive coordinates
        data.coords[:, 2] -= data.coords[data.is_support.view(-1), 2].min()

        # Set support
        data.is_support = data.is_support

        # Check for radial symmetry in coords
        radial_distance = torch.norm(data.coords[:, 0:2], dim=1)
        for i in range(n_rings):
            ring_radial_distance = radial_distance[i : -1 : n_rings + int(opening)]
            ring_height = data.coords[i : -1 : n_rings + int(opening), 2]
            symmetric = True
            if not torch.allclose(
                ring_radial_distance, ring_radial_distance[0], atol=1e-4
            ):
                symmetric = False
            if not torch.allclose(ring_height, ring_height[0], atol=1e-4):
                symmetric = False
            if not symmetric:
                raise ValueError("Form-found structure is not radially symmetric.")

        return data

    def generate_trail(
        self, data, origin_coords, origin_load, id, n_rings, trail_length, force_signs
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
                force_sign=torch.tensor(force_signs[i - 1]),
            )

    def fix_graph(self, data, n_trails):
        data.add_node(
            "centroid",
            coords=torch.tensor([0.0, 0.0, 0.0]),
            load=torch.tensor([0.0, 0.0, -1.0]),
        )
        merge_nodes = [f"trail_{i}_node_0" for i in range(n_trails)]
        data.merge_nodes("centroid", merge_nodes)
