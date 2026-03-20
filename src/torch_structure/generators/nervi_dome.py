import torch
import math
import numpy as np

from torch_structure.data import StructData
from torch_structure.generators.base_generator_cem import BaseGeneratorCEM
from torch_structure.generators.topology import build_trail, fix_graph
from torch_structure.generators.helpers import compute_trail_origin, add_opening_ring_edges


class NerviDome(BaseGeneratorCEM):
    def __init__(self, **overrides):
        super().__init__(**overrides)
        self.max_attempts = 100

    def validate_input(
        self,
        n_trails,
        n_rings,
        trail_length,
        deviation_force,
        center_deviation_force,
        ring_force,
        opening_diameter,
        opening,
    ):
        if n_trails % 2 != 0:
            raise ValueError("Number of trails must be even.")

    def sample_input(
        self,
        n_trails=None,
        n_rings=None,
        trail_length=0.1,
        deviation_force=-2.0,
        center_deviation_force=-3.0,
        ring_force=-15.0,
        opening_diameter=0.2,
        opening=True,
    ):
        if n_trails is None:
            n_trails = 2 * np.random.randint(5, 15)
        if n_rings is None:
            n_rings = np.random.randint(3, 6)

        return {
            "n_trails": n_trails,
            "n_rings": n_rings,
            "trail_length": trail_length,
            "deviation_force": deviation_force,
            "center_deviation_force": center_deviation_force,
            "ring_force": ring_force,
            "opening_diameter": opening_diameter,
            "opening": opening,
        }

    def generate(
        self,
        n_trails,
        n_rings,
        trail_length,
        deviation_force,
        center_deviation_force,
        ring_force,
        opening_diameter,
        opening,
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
            origin_coords, origin_load, _ = compute_trail_origin(
                angle, opening, opening_diameter, center_deviation_force, centroid, n_trails
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
        for i in range(n_rings):
            for j in range(n_trails):
                data.add_edge(
                    f"trail_{j}_node_{i}",
                    f"trail_{(j + 1) % n_trails}_node_{i + 1}",
                    is_trail_edge=torch.tensor(False),
                    force=deviation_force,
                )

        if opening:
            add_opening_ring_edges(data, n_trails, ring_force)
        else:
            fix_graph(data, n_trails)

        # Formfinding
        data = data.mpcem()

        return data
