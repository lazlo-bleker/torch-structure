import torch
import math
import numpy as np
from collections.abc import Callable

from torch_structure.data import StructData
from torch_structure.generators.base_generator import BaseGenerator
from config import TORCH_FLOAT


def uniform_function(n_trails: int, n_rings: int, scale=1.0) -> torch.Tensor:
    return scale * torch.ones((n_trails, n_rings))


class DomeGenerator(BaseGenerator):
    def __init__(self, **overrides):
        super().__init__(**overrides)
        self.max_attempts = 100

        self.node_attrs = {
            "coords": torch.empty((0, 3), dtype=TORCH_FLOAT),
            "uv_coords": torch.empty((0, 2), dtype=torch.long),
            "load": torch.empty((0, 3), dtype=TORCH_FLOAT),
            "support_condition": torch.empty((0, 3), dtype=torch.long),
            "is_origin_node": torch.empty((0, 1), dtype=torch.bool),
            "sequence": torch.empty((0, 1), dtype=torch.long),
        }
        self.edge_attrs = {
            "force": torch.empty((0, 1), dtype=TORCH_FLOAT),
            "length": torch.empty((0, 1), dtype=TORCH_FLOAT),
            "is_trail_edge": torch.empty((0, 1), dtype=torch.bool),
            "force_sign": torch.empty((0, 1), dtype=TORCH_FLOAT),
            "active_edof": torch.empty((0, 1), dtype=torch.bool),
        }
        self.default_attrs = {
            "active_edof": torch.tensor(True, dtype=torch.bool),
            "force": torch.tensor([torch.nan]),
            "length": torch.tensor([torch.nan]),
            "coords": torch.full((3,), torch.nan),
            "uv_coords": torch.full((2,), -1),
            "load": torch.zeros(3, dtype=TORCH_FLOAT),
            "support_condition": torch.zeros(3, dtype=torch.bool),
            "is_origin_node": torch.tensor(0, dtype=torch.bool),
        }

    def validate_input(self, n_trails, **kwwargs):
        if n_trails % 2 != 0:
            raise ValueError("Number of trails must be even.")

    def sample_input(
        self,
        n_trails: int | None = None,
        n_rings: int | None = None,
        trail_length_function: Callable[[int, int], torch.Tensor] = lambda i, j: (
            uniform_function(i, j, scale=5e-2)
        ),
        deviation_force_function: Callable[[int, int], torch.Tensor] = lambda i, j: (
            uniform_function(i, j, scale=-1.0)
        ),
        center_deviation_force: float | None = None,
        opening: bool = False,
    ) -> dict:
        if n_trails is None:
            n_trails = 2 * np.random.randint(5, 15)

        if n_rings is None:
            n_rings = np.random.randint(3, 6)

        if center_deviation_force is None:
            center_deviation_force = np.random.uniform(-8.0, -1.0)

        trail_lengths_ij = trail_length_function(n_trails, n_rings - 1)
        deviation_forces_ij = deviation_force_function(n_trails, n_rings)

        return {
            "n_trails": n_trails,
            "n_rings": n_rings,
            "trail_lengths_ij": trail_lengths_ij,
            "deviation_forces_ij": deviation_forces_ij,
            "center_deviation_force": center_deviation_force,
            "opening": opening,
        }

    def generate(
        self,
        n_trails,
        n_rings,
        trail_lengths_ij,
        deviation_forces_ij,
        center_deviation_force,
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
            if opening:
                origin_diameter = trail_lengths_ij[i, 0]
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
                trail_lengths=trail_lengths_ij[i],
            )

        # Add ring deviations
        for j in range(n_rings):
            for i in range(n_trails):
                data.add_edge(
                    f"trail_{i}_node_{j}",
                    f"trail_{(i + 1) % n_trails}_node_{j}",
                    is_trail_edge=torch.tensor(False),
                    force=deviation_forces_ij[i, j],
                )

        # Formfinding
        data = data.mpcem()
        if not opening:
            self.fix_graph(data, n_trails)

        # Scale to unit length
        radius = torch.norm(data.coords[:, 0:2], dim=1).max()
        data.coords /= radius

        # Translate to positive coordinates
        data.coords[:, 2] -= data.coords[:, 2].min()

        # Set support
        data.is_support = data.is_support

        if data.bbox[0, 2] < -1e2 or data.bbox[1, 2] > 1e2:
            raise RuntimeError("Z out of bounds:", data.bbox[0, 2], data.bbox[1, 2])

        return data

    def generate_trail(
        self, data, origin_coords, origin_load, id, n_rings, trail_lengths
    ):
        data.add_node(
            f"trail_{id}_node_0",
            coords=origin_coords,
            uv_coords=torch.tensor([id, 0]),
            is_origin_node=torch.tensor(True),
            sequence=torch.tensor(0),
            load=origin_load,
        )
        for j in range(1, n_rings):
            if j == n_rings - 1:
                support_condition = torch.tensor([True, True, True])
            else:
                support_condition = torch.tensor([False, False, False])

            data.add_node(
                f"trail_{id}_node_{j}",
                uv_coords=torch.tensor([id, j]),
                is_origin_node=torch.tensor(False),
                sequence=torch.tensor(j),
                load=torch.tensor([0.0, 0.0, -1.0]),
                support_condition=support_condition,
            )
            data.add_edge(
                f"trail_{id}_node_{j - 1}",
                f"trail_{id}_node_{j}",
                is_trail_edge=torch.tensor(True),
                length=trail_lengths[j - 1],
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
