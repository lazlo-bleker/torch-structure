import torch
import math
import numpy as np
from collections.abc import Callable

from torch_structure.data import StructData
from torch_structure.generators.base_generator import BaseGenerator
from config import TORCH_FLOAT


def uniform_function(
    n_trails: int,
    n_rings: int,
    base: torch.Tensor = torch.tensor([1.0]),
) -> torch.Tensor:
    if not torch.is_tensor(base):
        base = torch.tensor(base)
    base = base.view(1, 1, -1)  # (1, 1, d)
    return base.expand(n_trails, n_rings, -1)


def circle_function(
    n_nodes: int,
    center: torch.Tensor,
    radius: float,
) -> torch.Tensor:
    t = 2 * torch.pi * torch.arange(n_nodes) / (n_nodes)
    x = radius * torch.cos(t)
    y = radius * torch.sin(t)
    z = torch.zeros_like(t)
    return torch.stack([x, y, z], dim=1) + center


class DomeAssemblyGenerator(BaseGenerator):
    def __init__(self, **overrides):
        super().__init__(**overrides)
        self.max_attempts = 100

        self.node_attrs = {
            "coords": torch.empty((0, 3), dtype=TORCH_FLOAT),
            "load": torch.empty((0, 3), dtype=TORCH_FLOAT),
            "support_condition": torch.empty((0, 3), dtype=torch.long),
            "is_origin_node": torch.empty((0, 1), dtype=torch.bool),
            "sequence": torch.empty((0, 1), dtype=torch.bool),
        }
        self.edge_attrs = {
            "force": torch.empty((0, 1), dtype=TORCH_FLOAT),
            "length": torch.empty((0, 1), dtype=TORCH_FLOAT),
            "is_trail_edge": torch.empty((0, 1), dtype=torch.bool),
            "force_sign": torch.empty((0, 1), dtype=TORCH_FLOAT),
            "active_edof": torch.empty((0, 1), dtype=torch.bool),
            "assembly_sequence": torch.empty((0, 1), dtype=torch.long),
        }
        self.default_attrs = {
            "coords": torch.full((3,), torch.nan),
            "length": torch.tensor([torch.nan]),
            "active_edof": torch.tensor(True, dtype=torch.bool),
            "assembly_sequence": torch.tensor([torch.nan]),
        }

    def validate_input(
        self,
        n_trails,
        n_rings,
        trail_lengths_ij,
        deviation_forces_ij,
        nodal_loads_ijk,
        origin_nodes_ik,
        **kwwargs,
    ):
        if n_trails < 3:
            raise ValueError("Number of trails greater than 2.")
        if n_rings < 2:
            raise ValueError("Number of trails greater than 1.")
        if trail_lengths_ij.shape != (n_trails, n_rings):
            raise ValueError("Trail lengths array does not match")
        if deviation_forces_ij.shape != (n_trails, n_rings):
            raise ValueError("Deviation forces array does not match")
        if nodal_loads_ijk.shape != (n_trails, n_rings, 3):
            raise ValueError("Node loads array does not match")
        if origin_nodes_ik.shape != (n_trails, 3):
            raise ValueError("Origin nodes array does not match")

    def sample_input(
        self,
        n_trails: int | None = None,
        n_rings: int | None = None,
        trail_length_function: Callable[[int, int], torch.Tensor] = lambda i, j: (
            uniform_function(i, j, base=5e-2)
        ),
        deviation_force_function: Callable[[int, int], torch.Tensor] = lambda i, j: (
            uniform_function(i, j, base=-1.0)
        ),
        nodal_load_function: Callable[[int, int], torch.Tensor] = lambda i, j: (
            uniform_function(i, j, base=[0.0, 0.0, -1.0])
        ),
        origin_node_function: Callable[[int], torch.Tensor] = lambda i: circle_function(
            i, center=torch.tensor([0.0, 0.0, 0.0]), radius=1.0
        ),
    ) -> dict:
        if n_trails is None:
            n_trails = 2 * np.random.randint(5, 15)

        if n_rings is None:
            n_rings = np.random.randint(3, 6)

        # Eval function to get attributes
        trail_lengths_ij = trail_length_function(n_trails, n_rings)
        deviation_forces_ij = deviation_force_function(n_trails, n_rings)
        nodal_loads_ijk = nodal_load_function(n_trails, n_rings)
        origin_nodes_ik = origin_node_function(n_trails)

        return {
            "n_trails": n_trails,
            "n_rings": n_rings,
            "trail_lengths_ij": trail_lengths_ij,
            "deviation_forces_ij": deviation_forces_ij,
            "nodal_loads_ijk": nodal_loads_ijk,
            "origin_nodes_ik": origin_nodes_ik,
        }

    def generate(
        self,
        n_trails,
        n_rings,
        trail_lengths_ij,
        deviation_forces_ij,
        nodal_loads_ijk,
        origin_nodes_ik,
    ):
        # Initialize data object
        data = StructData(
            node_attrs=self.node_attrs,
            edge_attrs=self.edge_attrs,
            default_attrs=self.default_attrs,
        )

        # Generate origin nodes
        j = 0
        for i in range(n_trails):
            origin_coords = origin_nodes_ik[i]
            data.add_node(
                f"trail_{i}_node_{j}",
                is_origin_node=torch.tensor(True),
                coords=origin_coords,
                load=nodal_loads_ijk[i, j],
                sequence=torch.tensor(j),
            )

        # Generate inner nodes
        for j in range(1, n_rings - 1):
            for i in range(n_trails):
                data.add_node(
                    f"trail_{i}_node_{j}",
                    is_origin_node=torch.tensor(False),
                    load=nodal_loads_ijk[i, j],
                    sequence=torch.tensor(j),
                )

        # Generate support nodes
        j = n_rings - 1
        for i in range(n_trails):
            data.add_node(
                f"trail_{i}_node_{j}",
                is_origin_node=torch.tensor(False),
                load=nodal_loads_ijk[i, j],
                support_condition=torch.tensor([True, True, True]),
                sequence=torch.tensor(j),
            )

        # Generate trail edges
        for j in range(n_rings - 1):
            for i in range(n_trails):
                data.add_edge(
                    f"trail_{i}_node_{j}",
                    f"trail_{i}_node_{j + 1}",
                    is_trail_edge=torch.tensor(True),
                    length=trail_lengths_ij[i, j],
                    force_sign=torch.tensor(-1.0),
                    assembly_sequence=-j,
                )

        # Generate deviation edges
        for j in range(n_rings - 1):
            for i in range(n_trails):
                i_next = (i + 1) % n_trails
                data.add_edge(
                    f"trail_{i}_node_{j}",
                    f"trail_{i_next}_node_{j}",
                    is_trail_edge=torch.tensor(False),
                    force=deviation_forces_ij[i, j],
                    assembly_sequence=-j,
                )

        data.is_support = data.is_support

        return data
