import torch
import math
import numpy as np

from torch_structure.data import StructData
from torch_structure.generators.base_generator_cem import BaseGeneratorCEM
from torch_structure.generators.topology import fix_graph, build_trail
from torch_structure.generators.helpers import compute_trail_origin, add_opening_ring_edges

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
            origin_coords, origin_load, origin_diameter = compute_trail_origin(
                angle, opening, trail_length, center_deviation_force, centroid, n_trails
            )
            build_trail(
                data,
                name=f"trail_{i}",
                n_nodes=n_rings,
                origin_coords=origin_coords,
                load=origin_load,
                node_load=torch.tensor([0.0, 0.0, -1.0]),
                length=trail_length,
                force_sign=-1.0,
            )

        # Add ring deviations
        for i in range(1, n_rings):
            force = -(torch.rand(1) * 2 + 1)
            for j in range(n_trails):
                data.add_edge(
                    f"trail_{j}_node_{i}",
                    f"trail_{(j + 1) % n_trails}_node_{i}",
                    is_trail_edge=torch.tensor(False),
                    force=force,
                )

        if opening:
            ring_force = center_deviation_force * origin_diameter * 50
            add_opening_ring_edges(data, n_trails, ring_force)

        # Formfinding
        data = data.mpcem()
        if not opening:
            fix_graph(data, n_trails)

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

    def filter(self, data):
        mask = (data.is_trail_edge & data.directed_mask).view(-1)
        src, dst = data.edge_index[:, mask]
        src_centroid_distance = torch.norm(data.coords[src, 0:2], dim=1)
        dst_centroid_distance = torch.norm(data.coords[dst, 0:2], dim=1)
        invalid = dst_centroid_distance < src_centroid_distance
        return invalid.any()
