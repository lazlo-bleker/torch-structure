import torch
import numpy as np

from torch_structure.data import StructData
from torch_structure.generators.base_generator import BaseGenerator


class CableStayedBridge(BaseGenerator):
    """Builds a cable-stayed bridge: a deck trail, towers with backstays, and fan cables, as CEM trail edges."""

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
        n_towers,
        n_cables,
        deck_trail_length,
        tower_trail_length,
        center_deviation_force,
        cable_deviation_force,
        tower_height,
        tower_offset,
        back_stay_offset,
        back_stay_force,
        deck_load=torch.tensor([0.0, 0.0, -1.0]),
    ):
        """Build the (not yet form-found) cable-stayed bridge graph from the given parameters.

        Returns:
            StructData: the constructed graph.
        """
        # Cmpute number of trail edges
        n_deck_trail_edges = n_towers * n_cables
        # Initialize graph to contain data
        graph = StructData(
            node_attrs=self.node_attrs,
            edge_attrs=self.edge_attrs,
            default_attrs=self.default_attrs,
        )

        # Deck
        deck_origin_coords = [
            torch.tensor([-0.5 * deck_trail_length, 0.0, 0.0]),
            torch.tensor([0.5 * deck_trail_length, 0.0, 0.0]),
        ]
        for i in range(2):
            # Add origin nodes
            graph.add_node(
                f"deck_trail_{i}_node_0",
                coords=deck_origin_coords[i],
                is_origin_node=torch.tensor(True),
                sequence=torch.tensor(0),
                load=deck_load,
            )
            for j in range(1, n_deck_trail_edges + 1):
                load = (
                    torch.zeros(3, dtype=torch.float)
                    if j == n_deck_trail_edges
                    else deck_load
                )
                support_condition = (
                    torch.tensor([True, True, True])
                    if j == n_deck_trail_edges
                    else torch.tensor([False, False, False])
                )
                # Add node
                graph.add_node(
                    f"deck_trail_{i}_node_{j}",
                    is_origin_node=torch.tensor(False),
                    sequence=torch.tensor(j),
                    load=load,
                    support_condition=support_condition,
                )
                # Add new edge with the created node and its previous node
                graph.add_edge(
                    f"deck_trail_{i}_node_{j - 1}",
                    f"deck_trail_{i}_node_{j}",
                    is_trail_edge=torch.tensor(True),
                    length=deck_trail_length,
                    force_sign=torch.tensor(-1.0),
                )

        # Add element connecting rails
        graph.add_edge(
            "deck_trail_0_node_0",
            "deck_trail_1_node_0",
            is_trail_edge=torch.tensor(False),
            force=center_deviation_force,
        )

        # Towers
        span = (n_deck_trail_edges * 2 + 1) * deck_trail_length
        tower_x = torch.linspace(-0.5 * span, 0.5 * span, 2 * n_towers + 1)[1:-1:2]
        for i in range(n_towers):
            alternate = 1 - 2 * (i % 2)
            applied_tower_offset = alternate * tower_offset
            tower_origin_coords = torch.tensor(
                [tower_x[i], applied_tower_offset, tower_height]
            )
            graph.add_node(
                f"tower_trail_{i}_node_0",
                coords=tower_origin_coords,
                is_origin_node=torch.tensor(True),
                sequence=torch.tensor(0),
            )
            for j in range(1, n_cables + 1):
                support_condition = (
                    torch.tensor([True, True, True])
                    if j == n_cables
                    else torch.tensor([False, False, False])
                )
                length = tower_height * 0.8 if j == n_cables else tower_trail_length
                graph.add_node(
                    f"tower_trail_{i}_node_{j}",
                    is_origin_node=torch.tensor(False),
                    sequence=torch.tensor(j),
                    support_condition=support_condition,
                )
                graph.add_edge(
                    f"tower_trail_{i}_node_{j - 1}",
                    f"tower_trail_{i}_node_{j}",
                    is_trail_edge=torch.tensor(True),
                    length=length,
                    force_sign=torch.tensor(-1.0),
                )
            # Backstay
            applied_back_stay_offset = alternate * (tower_offset + back_stay_offset)
            backstay_origin_coords = torch.tensor(
                [tower_x[i], applied_back_stay_offset, 0.01]
            )
            graph.add_node(
                f"backstay_trail_{i}_node_0",
                coords=backstay_origin_coords,
                is_origin_node=torch.tensor(True),
                sequence=torch.tensor(0),
            )
            graph.add_node(
                f"backstay_trail_{i}_node_1",
                is_origin_node=torch.tensor(False),
                sequence=torch.tensor(1),
                support_condition=[True, True, True],
            )
            graph.add_edge(
                f"backstay_trail_{i}_node_0",
                f"backstay_trail_{i}_node_1",
                is_trail_edge=torch.tensor(True),
                length=0.4 * tower_height,
                force_sign=torch.tensor(1.0),
            )
            graph.add_edge(
                f"tower_trail_{i}_node_0",
                f"backstay_trail_{i}_node_0",
                is_trail_edge=torch.tensor(False),
                force=back_stay_force,
            )

        # Cables
        deck_nodes = [
            f"deck_trail_0_node_{i}" for i in reversed(range(n_deck_trail_edges))
        ] + [f"deck_trail_1_node_{i}" for i in range(n_deck_trail_edges)]
        cable_nodes = [
            f"tower_trail_{i}_node_{j}"
            for i in range(n_towers)
            for reverse in [False, True]
            for j in (reversed(range(n_cables)) if reverse else range(n_cables))
        ]

        for deck_node, cable_node in zip(deck_nodes, cable_nodes):
            graph.add_edge(
                deck_node,
                cable_node,
                is_trail_edge=torch.tensor(False),
                force=cable_deviation_force,
            )

        return graph
