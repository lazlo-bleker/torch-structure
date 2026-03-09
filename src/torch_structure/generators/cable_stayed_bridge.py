import torch
import numpy as np

from torch_structure.data import StructData
from torch_structure.generators.base_generator_cem import BaseGeneratorCEM
from torch_structure.generators.topology import build_trail


class CableStayedBridge(BaseGeneratorCEM):
    def __init__(self, **overrides):
        super().__init__(**overrides)
        self.max_attempts = 100

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
            build_trail(
                graph, f"deck_trail_{i}", n_deck_trail_edges,
                origin_coords=deck_origin_coords[i],
                load=deck_load,
                force_sign=-1.0,
                length=deck_trail_length,
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
            build_trail(
                graph, f"tower_trail_{i}", n_cables,
                origin_coords=tower_origin_coords,
                force_sign=-1.0,
                length=tower_trail_length,
                last_edge_kwargs={"length": tower_height * 0.8},
            )
            # Backstay
            applied_back_stay_offset = alternate * (tower_offset + back_stay_offset)
            backstay_origin_coords = torch.tensor(
                [tower_x[i], applied_back_stay_offset, 0.01]
            )
            build_trail(
                graph, f"backstay_trail_{i}", 1,
                origin_coords=backstay_origin_coords,
                force_sign=1.0,
                length=0.4 * tower_height,
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
