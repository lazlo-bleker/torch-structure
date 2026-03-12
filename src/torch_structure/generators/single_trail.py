import torch
import numpy as np

from torch_structure.data import StructData
from torch_structure.generators.base_generator import BaseGenerator
from config import TORCH_FLOAT


class SingleTrailGenerator(BaseGenerator):
    def __init__(self, **overrides):
        super().__init__(**overrides)
        self.max_attempts = 1

        self.node_attrs = {
            "coords": torch.empty((0, 3), dtype=TORCH_FLOAT),
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
            "coords": torch.full((3,), torch.nan),
            "load": torch.zeros(3, dtype=TORCH_FLOAT),
            "support_condition": torch.zeros(3, dtype=torch.bool),
            "is_origin_node": torch.tensor(0, dtype=torch.bool),
            "sequence": torch.tensor(-1, dtype=torch.long),
            "force": torch.tensor([torch.nan]),
            "length": torch.tensor([torch.nan]),
            "is_trail_edge": torch.tensor(True, dtype=torch.bool),
            "active_edof": torch.tensor(True, dtype=torch.bool),
            "force_sign": torch.tensor(-1.0, dtype=TORCH_FLOAT),
        }

    def sample_input(self, **kwargs):
        """
        No sampling required
        """
        return kwargs

    def validate_input(self, **kwargs):
        """
        No validation required
        """
        pass

    def generate(
        self,
        n_nodes: int,
        nodal_load: list,
        trail_element_length: float,
        origin_node_load: list,
    ) -> StructData:
        # Initialize graph
        graph = StructData(
            node_attrs=self.node_attrs,
            edge_attrs=self.edge_attrs,
            default_attrs=self.default_attrs,
        )

        # Parse data as numpy arrays
        origin_node_coords = np.zeros(3)
        nodal_load = np.array(nodal_load)
        origin_node_load = np.array(origin_node_load)
        support_condition = torch.tensor([False, False, False])

        # Add origin nodes
        graph.add_node(
            "node_0",
            coords=origin_node_coords,
            is_origin_node=torch.tensor(True),
            sequence=torch.tensor(0),
            load=nodal_load + origin_node_load,
        )
        for j in range(1, n_nodes - 1):
            # Add node
            graph.add_node(
                f"node_{j}",
                is_origin_node=torch.tensor(False),
                sequence=torch.tensor(j),
                load=nodal_load,
                support_condition=support_condition,
            )
            # Add new edge with the created node and its previous node
            graph.add_edge(
                f"node_{j - 1}",
                f"node_{j}",
                is_trail_edge=torch.tensor(True),
                length=trail_element_length,
                force_sign=torch.tensor(-1.0),
            )

        # Add node
        j = n_nodes - 1
        graph.add_node(
            f"node_{j}",
            is_origin_node=torch.tensor(False),
            sequence=torch.tensor(j),
            load=nodal_load,
            support_condition=torch.tensor([True, True, True]),
        )
        # Add new edge with the created node and its previous node
        graph.add_edge(
            f"node_{j - 1}",
            f"node_{j}",
            is_trail_edge=torch.tensor(True),
            length=trail_element_length,
            force_sign=torch.tensor(-1.0),
        )

        return graph
