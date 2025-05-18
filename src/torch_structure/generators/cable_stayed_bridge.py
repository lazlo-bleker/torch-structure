import torch

from torch_structure.data import Data


class CableStayedBridge:
    def __init__(
        self,
        n_towers,
        n_cables,
        deck_trail_length,
        tower_trail_length,
        center_deviation_force,
        cable_deviation_force,
        tower_height,
        tower_offset,
        deck_load=torch.tensor([0.0, 0.0, -1.0]),
    ):
        # Topological parameters
        self.n_towers = n_towers
        self.n_cables = n_cables
        self.n_deck_trail_edges = self.n_towers * self.n_cables

        # Metric parameters
        self.deck_trail_length = deck_trail_length
        self.tower_trail_length = tower_trail_length
        self.center_deviation_force = center_deviation_force
        self.cable_deviation_force = cable_deviation_force
        self.deck_load = deck_load
        self.tower_height = tower_height
        self.tower_offset = tower_offset

        self.generate_graph()

    def generate_graph(self):
        node_attrs = {
            "coords": torch.empty((0, 3), dtype=torch.float),
            "load": torch.empty((0, 3), dtype=torch.float),
            "support_condition": torch.empty((0, 3), dtype=torch.long),
            "is_origin_node": torch.empty((0, 1), dtype=torch.bool),
            "sequence": torch.empty((0, 1), dtype=torch.long),
        }
        edge_attrs = {
            "force": torch.empty((0, 1), dtype=torch.float),
            "length": torch.empty((0, 1), dtype=torch.float),
            "is_trail_edge": torch.empty((0, 1), dtype=torch.bool),
            "force_sign": torch.empty((0, 1), dtype=torch.float),
        }
        default_attrs = {
            "force": torch.tensor([torch.nan]),
            "length": torch.tensor([torch.nan]),
            "coords": torch.full((3,), torch.nan),
            "load": torch.zeros(3, dtype=torch.float),
            "support_condition": torch.zeros(3, dtype=torch.bool),
            "is_origin_node": torch.tensor(0, dtype=torch.bool),
        }
        self.graph = Data(
            node_attrs=node_attrs, edge_attrs=edge_attrs, default_attrs=default_attrs
        )

        # Deck
        deck_origin_coords = [
            torch.tensor([-0.5 * self.deck_trail_length, 0.0, 0.0]),
            torch.tensor([0.5 * self.deck_trail_length, 0.0, 0.0]),
        ]
        for i in range(2):
            self.graph.add_node(
                f"deck_trail_{i}_node_0",
                coords=deck_origin_coords[i],
                is_origin_node=torch.tensor(True),
                sequence=torch.tensor(0),
                load=self.deck_load,
            )
            for j in range(1, self.n_deck_trail_edges):
                support_condition = (
                    torch.tensor([True, True, True])
                    if j == self.n_deck_trail_edges - 1
                    else torch.tensor([False, False, False])
                )
                self.graph.add_node(
                    f"deck_trail_{i}_node_{j}",
                    is_origin_node=torch.tensor(False),
                    sequence=torch.tensor(j),
                    load=self.deck_load,
                    support_condition=support_condition,
                )
                self.graph.add_edge(
                    f"deck_trail_{i}_node_{j - 1}",
                    f"deck_trail_{i}_node_{j}",
                    is_trail_edge=torch.tensor(True),
                    length=self.deck_trail_length,
                    force_sign=torch.tensor(-1.0),
                )
        self.graph.add_edge(
            "deck_trail_0_node_0",
            "deck_trail_1_node_0",
            is_trail_edge=torch.tensor(False),
            force=self.center_deviation_force,
        )

        # Towers
        span = (self.n_deck_trail_edges * 2 + 1) * self.deck_trail_length
        tower_x = torch.linspace(-0.5 * span, 0.5 * span, 2 * self.n_towers + 1)[1:-1:2]
        for i in range(self.n_towers):
            tower_offset = (1 - 2 * (i % 2)) * self.tower_offset
            tower_origin_coords = torch.tensor(
                [tower_x[i], tower_offset, self.tower_height]
            )
            self.graph.add_node(
                f"tower_trail_{i}_node_0",
                coords=tower_origin_coords,
                is_origin_node=torch.tensor(True),
                sequence=torch.tensor(0),
                load=self.deck_load,  # TODO: Replace with backstay
            )
            for j in range(1, self.n_cables + 1):
                support_condition = (
                    torch.tensor([True, True, True])
                    if j == self.n_cables
                    else torch.tensor([False, False, False])
                )
                length = (
                    self.tower_height * 1.3
                    if j == self.n_cables
                    else self.tower_trail_length
                )
                self.graph.add_node(
                    f"tower_trail_{i}_node_{j}",
                    is_origin_node=torch.tensor(False),
                    sequence=torch.tensor(j),
                    support_condition=support_condition,
                )
                self.graph.add_edge(
                    f"tower_trail_{i}_node_{j - 1}",
                    f"tower_trail_{i}_node_{j}",
                    is_trail_edge=torch.tensor(True),
                    length=length,
                    force_sign=torch.tensor(-1.0),
                )

        # Cables
        deck_nodes = [
            f"deck_trail_0_node_{i}" for i in reversed(range(self.n_deck_trail_edges))
        ] + [f"deck_trail_1_node_{i}" for i in range(self.n_deck_trail_edges)]
        cable_nodes = [
            f"tower_trail_{i}_node_{j}"
            for i in range(self.n_towers)
            for reverse in [False, True]
            for j in (
                reversed(range(self.n_cables)) if reverse else range(self.n_cables)
            )
        ]

        for deck_node, cable_node in zip(deck_nodes, cable_nodes):
            self.graph.add_edge(
                deck_node,
                cable_node,
                is_trail_edge=torch.tensor(False),
                force=self.cable_deviation_force,
            )
