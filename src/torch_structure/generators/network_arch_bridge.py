import torch
import numpy as np

from torch_structure.data import StructData


class NetwokrkArchBridge:
    def __init__(
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
        # Topological parameters
        self.n_trail_edges = n_trail_edges
        self.cable_offset = cable_offset

        # Metric parameters
        self.height = height
        self.deck_width = deck_width
        self.arch_width = arch_width
        self.arch_force = arch_force
        self.deck_force = deck_force
        self.cable_force = cable_force
        self.inter_deck_force = inter_deck_force
        self.inter_arch_force = inter_arch_force
        self.deck_trail_length = deck_trail_length
        self.arch_trail_length = arch_trail_length
        self.deck_load = deck_load

        # Options
        self.kinematic_cable_arrangement = kinematic_cable_arrangement

        self.n_optim_groups = 0

        self.generate_graph()

    def generate_graph(self):
        node_attrs = {
            "coords": torch.empty((0, 3), dtype=torch.float),
            "load": torch.empty((0, 3), dtype=torch.float),
            "support_condition": torch.empty((0, 3), dtype=torch.long),
            "is_origin_node": torch.empty((0, 1), dtype=torch.bool),
            "sequence": torch.empty((0, 1), dtype=torch.long),
            "target_coords": torch.empty((0, 3), dtype=torch.float),
            "bridge_side": torch.empty((0, 1), dtype=torch.long),
        }
        edge_attrs = {
            "force": torch.empty((0, 1), dtype=torch.float),
            "length": torch.empty((0, 1), dtype=torch.float),
            "is_trail_edge": torch.empty((0, 1), dtype=torch.bool),
            "force_sign": torch.empty((0, 1), dtype=torch.float),
            "optim_group": torch.empty((0, 1), dtype=torch.long),
            "cable": torch.empty((0, 1), dtype=torch.bool),
        }
        default_attrs = {
            "force": torch.tensor([torch.nan]),
            "length": torch.tensor([torch.nan]),
            "coords": torch.full((3,), torch.nan),
            "load": torch.zeros(3, dtype=torch.float),
            "support_condition": torch.zeros(3, dtype=torch.bool),
            "is_origin_node": torch.tensor(0, dtype=torch.bool),
            "optim_group": torch.tensor(-1, dtype=torch.long),
            "target_coords": torch.full((3,), torch.nan),
            "cable": torch.tensor(False, dtype=torch.bool),
        }
        self.graph = StructData(
            node_attrs=node_attrs, edge_attrs=edge_attrs, default_attrs=default_attrs
        )

        # Deck
        deck_origin_coords = [
            torch.tensor([-0.5 * self.deck_trail_length, -0.5 * self.deck_width, 0.0]),
            torch.tensor([0.5 * self.deck_trail_length, -0.5 * self.deck_width, 0.0]),
            torch.tensor([-0.5 * self.deck_trail_length, 0.5 * self.deck_width, 0.0]),
            torch.tensor([0.5 * self.deck_trail_length, 0.5 * self.deck_width, 0.0]),
        ]
        for i in range(len(deck_origin_coords)):
            self.graph.add_node(
                f"deck_trail_{i}_node_0",
                coords=deck_origin_coords[i],
                is_origin_node=torch.tensor(True),
                sequence=torch.tensor(0),
                load=self.deck_load,
                bridge_side=i % 2,
            )
            for j in range(1, self.n_trail_edges + 1):
                support_condition = (
                    torch.tensor([True, True, True])
                    if j == self.n_trail_edges
                    else torch.tensor([False, False, False])
                )
                load = (
                    torch.zeros(3, dtype=torch.float)
                    if j == self.n_trail_edges
                    else self.deck_load
                )
                x_target = (
                    j * self.deck_trail_length * torch.sign(deck_origin_coords[i][0])
                )
                self.graph.add_node(
                    f"deck_trail_{i}_node_{j}",
                    is_origin_node=torch.tensor(False),
                    sequence=torch.tensor(j),
                    load=load,
                    support_condition=support_condition,
                    target_coords=deck_origin_coords[i]
                    + torch.tensor([x_target, 0.0, 0.0]),
                    bridge_side=i % 2,
                )
                self.graph.add_edge(
                    f"deck_trail_{i}_node_{j - 1}",
                    f"deck_trail_{i}_node_{j}",
                    is_trail_edge=torch.tensor(True),
                    length=self.deck_trail_length,
                    force_sign=torch.tensor(1.0),
                )
        self.graph.add_edge(
            "deck_trail_0_node_0",
            "deck_trail_1_node_0",
            is_trail_edge=torch.tensor(False),
            force=self.deck_force,
            optim_group=self.n_optim_groups,
        )
        self.graph.add_edge(
            "deck_trail_2_node_0",
            "deck_trail_3_node_0",
            is_trail_edge=torch.tensor(False),
            force=self.deck_force,
            optim_group=self.n_optim_groups,
        )
        self.n_optim_groups += 1

        # Arches
        arch_origin_coords = [
            torch.tensor(
                [-0.5 * self.arch_trail_length, -0.5 * self.arch_width, self.height]
            ),
            torch.tensor(
                [0.5 * self.arch_trail_length, -0.5 * self.arch_width, self.height]
            ),
            torch.tensor(
                [-0.5 * self.arch_trail_length, 0.5 * self.arch_width, self.height]
            ),
            torch.tensor(
                [0.5 * self.arch_trail_length, 0.5 * self.arch_width, self.height]
            ),
        ]
        for i in range(len(arch_origin_coords)):
            self.graph.add_node(
                f"arch_trail_{i}_node_0",
                coords=arch_origin_coords[i],
                is_origin_node=torch.tensor(True),
                sequence=torch.tensor(0),
                bridge_side=i % 2,
            )
            for j in range(1, self.n_trail_edges + 1):
                support_condition = (
                    torch.tensor([True, True, True])
                    if j == self.n_trail_edges
                    else torch.tensor([False, False, False])
                )
                x_target = (
                    j * self.deck_trail_length * torch.sign(deck_origin_coords[i][0])
                )
                target_coords = (
                    deck_origin_coords[i] + torch.tensor([x_target, 0.0, 0.0])
                    if j == self.n_trail_edges
                    else torch.full((3,), torch.nan)
                )
                self.graph.add_node(
                    f"arch_trail_{i}_node_{j}",
                    is_origin_node=torch.tensor(False),
                    sequence=torch.tensor(j),
                    support_condition=support_condition,
                    target_coords=target_coords,
                    bridge_side=i % 2,
                )
                self.graph.add_edge(
                    f"arch_trail_{i}_node_{j - 1}",
                    f"arch_trail_{i}_node_{j}",
                    is_trail_edge=torch.tensor(True),
                    length=self.arch_trail_length,
                    force_sign=torch.tensor(-1.0),
                    optim_group=self.n_optim_groups,
                )
        self.n_optim_groups += 1
        self.graph.add_edge(
            "arch_trail_0_node_0",
            "arch_trail_1_node_0",
            is_trail_edge=torch.tensor(False),
            force=self.arch_force,
            optim_group=self.n_optim_groups,
        )
        self.graph.add_edge(
            "arch_trail_2_node_0",
            "arch_trail_3_node_0",
            is_trail_edge=torch.tensor(False),
            force=self.arch_force,
            optim_group=self.n_optim_groups,
        )
        self.n_optim_groups += 1

        # Prepare node lists
        deck_nodes_0 = [
            f"deck_trail_0_node_{i}" for i in reversed(range(self.n_trail_edges + 1))
        ] + [f"deck_trail_1_node_{i}" for i in range(self.n_trail_edges + 1)]
        deck_nodes_1 = [
            f"deck_trail_2_node_{i}" for i in reversed(range(self.n_trail_edges + 1))
        ] + [f"deck_trail_3_node_{i}" for i in range(self.n_trail_edges + 1)]
        arch_nodes_0 = [
            f"arch_trail_0_node_{i}" for i in reversed(range(self.n_trail_edges + 1))
        ] + [f"arch_trail_1_node_{i}" for i in range(self.n_trail_edges + 1)]
        arch_nodes_1 = [
            f"arch_trail_2_node_{i}" for i in reversed(range(self.n_trail_edges + 1))
        ] + [f"arch_trail_3_node_{i}" for i in range(self.n_trail_edges + 1)]

        # Cables
        if self.kinematic_cable_arrangement:
            for deck_nodes, arch_nodes in zip(
                [deck_nodes_0, deck_nodes_1], [arch_nodes_0, arch_nodes_1]
            ):
                arch_nodes_cables_1 = arch_nodes[1 : -self.cable_offset : 2]
                deck_nodes_cables_1 = deck_nodes[1 + self.cable_offset :: 2]
                deck_nodes_cables_1[-1] = deck_nodes[-2]
                arch_nodes_cables_2 = arch_nodes[self.cable_offset : -1 : 2]
                deck_nodes_cables_2 = deck_nodes[0 : -self.cable_offset : 2]
                deck_nodes_cables_2[0] = deck_nodes[1]
                for i, (arch_node, deck_node) in enumerate(
                    zip(arch_nodes_cables_1, deck_nodes_cables_1)
                ):
                    self.graph.add_edge(
                        arch_node,
                        deck_node,
                        is_trail_edge=torch.tensor(False),
                        force=self.cable_force,
                        optim_group=self.n_optim_groups + i,
                        cable=True,
                    )
                for i, (arch_node, deck_node) in enumerate(
                    reversed(list(zip(arch_nodes_cables_2, deck_nodes_cables_2)))
                ):
                    self.graph.add_edge(
                        arch_node,
                        deck_node,
                        is_trail_edge=torch.tensor(False),
                        force=self.cable_force,
                        optim_group=self.n_optim_groups + i,
                        cable=True,
                    )
            self.cable_optim_groups = np.arange(
                self.n_optim_groups, self.n_optim_groups + i + 1
            )
            self.n_optim_groups += i + 1

        else:
            for deck_nodes, arch_nodes in zip(
                [deck_nodes_0, deck_nodes_1], [arch_nodes_0, arch_nodes_1]
            ):
                for i, (arch_node, deck_node) in enumerate(
                    zip(
                        arch_nodes[self.cable_offset + 1 : -1],
                        deck_nodes[1 : -self.cable_offset + 1],
                    )
                ):
                    self.graph.add_edge(
                        arch_node,
                        deck_node,
                        is_trail_edge=torch.tensor(False),
                        force=self.cable_force,
                        optim_group=self.n_optim_groups + i,
                        cable=True,
                    )
                for i, (arch_node, deck_node) in enumerate(
                    reversed(
                        list(
                            zip(
                                arch_nodes[1 : -self.cable_offset + 1],
                                deck_nodes[self.cable_offset + 1 : -1],
                            )
                        )
                    )
                ):
                    self.graph.add_edge(
                        arch_node,
                        deck_node,
                        is_trail_edge=torch.tensor(False),
                        force=self.cable_force,
                        optim_group=self.n_optim_groups + i,
                        cable=True,
                    )
            self.cable_optim_groups = np.arange(
                self.n_optim_groups, self.n_optim_groups + i + 1
            )
            self.n_optim_groups += i + 1

        # Deck elements
        for i, (src, dst) in enumerate(zip(deck_nodes_0[1:-1], deck_nodes_1[1:-1])):
            self.graph.add_edge(
                src,
                dst,
                is_trail_edge=torch.tensor(False),
                force=self.inter_deck_force,
                optim_group=self.n_optim_groups + i,
            )
        self.n_optim_groups += i + 1

        # Arch elements
        # for i, (src, dst) in enumerate(zip(arch_nodes_0[2:-1], arch_nodes_1[1:-2])):
        #     self.graph.add_edge(
        #         src,
        #         dst,
        #         is_trail_edge=torch.tensor(False),
        #         force=self.inter_arch_force,
        #         # optim_group=self.n_optim_groups + i,
        #     )
        # for i, (src, dst) in enumerate(zip(arch_nodes_0[1:-2], arch_nodes_1[2:-1])):
        #     self.graph.add_edge(
        #         src,
        #         dst,
        #         is_trail_edge=torch.tensor(False),
        #         force=self.inter_arch_force,
        #         # optim_group=self.n_optim_groups + i,
        # )
        # self.arch_optim_groups = np.arange(self.n_optim_groups, self.n_optim_groups + i + 1)
        # self.n_optim_groups += i + 1
