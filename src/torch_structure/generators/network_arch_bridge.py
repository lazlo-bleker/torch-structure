import torch
import numpy as np

from torch_structure.data import StructData
from torch_structure.generators.base_generator_cem import BaseGeneratorCEM


class NetworkArchBridge(BaseGeneratorCEM):
    def __init__(self, **overrides):
        super().__init__(**overrides)
        self.max_attempts = 1

        self.node_attrs = {**self.node_attrs,
            "target_coords": torch.empty((0, 3), dtype=torch.float),
            "bridge_side": torch.empty((0, 1), dtype=torch.long),
        }
        self.default_attrs = {**self.default_attrs,
            "optim_group": torch.tensor(-1, dtype=torch.long),
            "target_coords": torch.full((3,), torch.nan),
            "cable": torch.tensor(False, dtype=torch.bool),
        }
        self.edge_attrs = {**self.edge_attrs,
            "optim_group": torch.empty((0, 1), dtype=torch.long),
            "cable": torch.empty((0, 1), dtype=torch.bool),
        }

    def sample_input(self, **kwargs):
        return kwargs

    def validate_input(self, **kwargs):
        pass

    def generate(
        self,
        n_trail_edges,
        cable_offset,
        height,
        deck_width,
        arch_width,
        arch_force,
        deck_force,
        cable_force,
        inter_deck_force,
        inter_arch_force,
        deck_trail_length,
        arch_trail_length,
        deck_load=torch.tensor([0.0, 0.0, -10.0]),
        kinematic_cable_arrangement=True,
    ):
        n_optim_groups = 0

        graph = StructData(
            node_attrs=self.node_attrs,
            edge_attrs=self.edge_attrs,
            default_attrs=self.default_attrs,
        )

        # Deck
        deck_origin_coords = [
            torch.tensor([-0.5 * deck_trail_length, -0.5 * deck_width, 0.0]),
            torch.tensor([0.5 * deck_trail_length, -0.5 * deck_width, 0.0]),
            torch.tensor([-0.5 * deck_trail_length, 0.5 * deck_width, 0.0]),
            torch.tensor([0.5 * deck_trail_length, 0.5 * deck_width, 0.0]),
        ]
        for i in range(len(deck_origin_coords)):
            graph.add_node(
                f"deck_trail_{i}_node_0",
                coords=deck_origin_coords[i],
                is_origin_node=torch.tensor(True),
                sequence=torch.tensor(0),
                load=deck_load,
                bridge_side=i % 2,
            )
            for j in range(1, n_trail_edges + 1):
                support_condition = (
                    torch.tensor([True, True, True])
                    if j == n_trail_edges
                    else torch.tensor([False, False, False])
                )
                load = (
                    torch.zeros(3, dtype=torch.float)
                    if j == n_trail_edges
                    else deck_load
                )
                x_target = (
                    j * deck_trail_length * torch.sign(deck_origin_coords[i][0])
                )
                graph.add_node(
                    f"deck_trail_{i}_node_{j}",
                    is_origin_node=torch.tensor(False),
                    sequence=torch.tensor(j),
                    load=load,
                    support_condition=support_condition,
                    target_coords=deck_origin_coords[i]
                    + torch.tensor([x_target, 0.0, 0.0]),
                    bridge_side=i % 2,
                )
                graph.add_edge(
                    f"deck_trail_{i}_node_{j - 1}",
                    f"deck_trail_{i}_node_{j}",
                    is_trail_edge=torch.tensor(True),
                    length=deck_trail_length,
                    force_sign=torch.tensor(1.0),
                )
        graph.add_edge(
            "deck_trail_0_node_0",
            "deck_trail_1_node_0",
            is_trail_edge=torch.tensor(False),
            force=deck_force,
            optim_group=n_optim_groups,
        )
        graph.add_edge(
            "deck_trail_2_node_0",
            "deck_trail_3_node_0",
            is_trail_edge=torch.tensor(False),
            force=deck_force,
            optim_group=n_optim_groups,
        )
        n_optim_groups += 1

        # Arches
        arch_origin_coords = [
            torch.tensor(
                [-0.5 * arch_trail_length, -0.5 * arch_width, height]
            ),
            torch.tensor(
                [0.5 * arch_trail_length, -0.5 * arch_width, height]
            ),
            torch.tensor(
                [-0.5 * arch_trail_length, 0.5 * arch_width, height]
            ),
            torch.tensor(
                [0.5 * arch_trail_length, 0.5 * arch_width, height]
            ),
        ]
        for i in range(len(arch_origin_coords)):
            graph.add_node(
                f"arch_trail_{i}_node_0",
                coords=arch_origin_coords[i],
                is_origin_node=torch.tensor(True),
                sequence=torch.tensor(0),
                bridge_side=i % 2,
            )
            for j in range(1, n_trail_edges + 1):
                support_condition = (
                    torch.tensor([True, True, True])
                    if j == n_trail_edges
                    else torch.tensor([False, False, False])
                )
                x_target = (
                    j * deck_trail_length * torch.sign(deck_origin_coords[i][0])
                )
                target_coords = (
                    deck_origin_coords[i] + torch.tensor([x_target, 0.0, 0.0])
                    if j == n_trail_edges
                    else torch.full((3,), torch.nan)
                )
                graph.add_node(
                    f"arch_trail_{i}_node_{j}",
                    is_origin_node=torch.tensor(False),
                    sequence=torch.tensor(j),
                    support_condition=support_condition,
                    target_coords=target_coords,
                    bridge_side=i % 2,
                )
                graph.add_edge(
                    f"arch_trail_{i}_node_{j - 1}",
                    f"arch_trail_{i}_node_{j}",
                    is_trail_edge=torch.tensor(True),
                    length=arch_trail_length,
                    force_sign=torch.tensor(-1.0),
                    optim_group=n_optim_groups,
                )
        n_optim_groups += 1
        graph.add_edge(
            "arch_trail_0_node_0",
            "arch_trail_1_node_0",
            is_trail_edge=torch.tensor(False),
            force=arch_force,
            optim_group=n_optim_groups,
        )
        graph.add_edge(
            "arch_trail_2_node_0",
            "arch_trail_3_node_0",
            is_trail_edge=torch.tensor(False),
            force=arch_force,
            optim_group=n_optim_groups,
        )
        n_optim_groups += 1

        # Prepare node lists
        deck_nodes_0 = [
            f"deck_trail_0_node_{i}" for i in reversed(range(n_trail_edges + 1))
        ] + [f"deck_trail_1_node_{i}" for i in range(n_trail_edges + 1)]
        deck_nodes_1 = [
            f"deck_trail_2_node_{i}" for i in reversed(range(n_trail_edges + 1))
        ] + [f"deck_trail_3_node_{i}" for i in range(n_trail_edges + 1)]
        arch_nodes_0 = [
            f"arch_trail_0_node_{i}" for i in reversed(range(n_trail_edges + 1))
        ] + [f"arch_trail_1_node_{i}" for i in range(n_trail_edges + 1)]
        arch_nodes_1 = [
            f"arch_trail_2_node_{i}" for i in reversed(range(n_trail_edges + 1))
        ] + [f"arch_trail_3_node_{i}" for i in range(n_trail_edges + 1)]

        # Cables
        if kinematic_cable_arrangement:
            for deck_nodes, arch_nodes in zip(
                [deck_nodes_0, deck_nodes_1], [arch_nodes_0, arch_nodes_1]
            ):
                arch_nodes_cables_1 = arch_nodes[1 : -cable_offset : 2]
                deck_nodes_cables_1 = deck_nodes[1 + cable_offset :: 2]
                deck_nodes_cables_1[-1] = deck_nodes[-2]
                arch_nodes_cables_2 = arch_nodes[cable_offset : -1 : 2]
                deck_nodes_cables_2 = deck_nodes[0 : -cable_offset : 2]
                deck_nodes_cables_2[0] = deck_nodes[1]
                for i, (arch_node, deck_node) in enumerate(
                    zip(arch_nodes_cables_1, deck_nodes_cables_1)
                ):
                    graph.add_edge(
                        arch_node,
                        deck_node,
                        is_trail_edge=torch.tensor(False),
                        force=cable_force,
                        optim_group=n_optim_groups + i,
                        cable=True,
                    )
                for i, (arch_node, deck_node) in enumerate(
                    reversed(list(zip(arch_nodes_cables_2, deck_nodes_cables_2)))
                ):
                    graph.add_edge(
                        arch_node,
                        deck_node,
                        is_trail_edge=torch.tensor(False),
                        force=cable_force,
                        optim_group=n_optim_groups + i,
                        cable=True,
                    )
            cable_optim_groups = np.arange(n_optim_groups, n_optim_groups + i + 1)
            n_optim_groups += i + 1

        else:
            for deck_nodes, arch_nodes in zip(
                [deck_nodes_0, deck_nodes_1], [arch_nodes_0, arch_nodes_1]
            ):
                for i, (arch_node, deck_node) in enumerate(
                    zip(
                        arch_nodes[cable_offset + 1 : -1],
                        deck_nodes[1 : -cable_offset + 1],
                    )
                ):
                    graph.add_edge(
                        arch_node,
                        deck_node,
                        is_trail_edge=torch.tensor(False),
                        force=cable_force,
                        optim_group=n_optim_groups + i,
                        cable=True,
                    )
                for i, (arch_node, deck_node) in enumerate(
                    reversed(
                        list(
                            zip(
                                arch_nodes[1 : -cable_offset + 1],
                                deck_nodes[cable_offset + 1 : -1],
                            )
                        )
                    )
                ):
                    graph.add_edge(
                        arch_node,
                        deck_node,
                        is_trail_edge=torch.tensor(False),
                        force=cable_force,
                        optim_group=n_optim_groups + i,
                        cable=True,
                    )
            cable_optim_groups = np.arange(n_optim_groups, n_optim_groups + i + 1)
            n_optim_groups += i + 1

        # Deck elements
        for i, (src, dst) in enumerate(zip(deck_nodes_0[1:-1], deck_nodes_1[1:-1])):
            graph.add_edge(
                src,
                dst,
                is_trail_edge=torch.tensor(False),
                force=inter_deck_force,
                optim_group=n_optim_groups + i,
            )
        n_optim_groups += i + 1

        graph.cable_optim_groups = cable_optim_groups
        graph.n_optim_groups = n_optim_groups

        return graph
