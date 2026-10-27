import random

import numpy as np
import pytest
import torch
import torch_structure.generators as generators


def cable_stayed_bridge():
    return generators.CableStayedBridge(
        n_towers=5, n_cables=5, deck_trail_length=1.0, tower_trail_length=0.3,
        center_deviation_force=torch.tensor(-10.0), cable_deviation_force=torch.tensor(1.0),
        deck_load=torch.tensor([0.0, 0.0, -1.0]), tower_height=5.0, tower_offset=1.0,
        back_stay_offset=5.0, back_stay_force=torch.tensor(10.0),
    )()


def single_trail():
    return generators.SingleTrailGenerator(
        n_nodes=10, trail_element_length=0.1, nodal_load=[0.0, 0.0, -1.0], origin_node_load=[4.0, 0.0, 0.0],
    )()


def network_arch_bridge():
    generator = generators.NetworkArchBridge(
        n_trail_edges=6, cable_offset=1, height=3.0, deck_width=2.0, arch_width=2.0,
        arch_force=torch.tensor(-5.0), deck_force=torch.tensor(5.0), cable_force=torch.tensor(1.0),
        inter_deck_force=torch.tensor(1.0), inter_arch_force=torch.tensor(1.0),
        deck_trail_length=1.0, arch_trail_length=1.0,
    )
    generator.generate_graph()
    return generator.graph


BUILDERS = {
    "ArchSuspensionBridgeGenerator": lambda: generators.ArchSuspensionBridgeGenerator()()[0],
    "CableNetGenerator": lambda: generators.CableNetGenerator()(),
    "DomeGenerator": lambda: generators.DomeGenerator()(),
    "GridShellGenerator": lambda: generators.GridShellGenerator()(),
    "MixedDomeGenerator": lambda: generators.MixedDomeGenerator()(),
    "TrussBridgeGenerator": lambda: generators.TrussBridgeGenerator()()[0],
    "CableStayedBridge": cable_stayed_bridge,
    "SingleTrailGenerator": single_trail,
    "NetworkArchBridge": network_arch_bridge,
}


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def undirected_edge_keys(data):
    edges = data.edge_index[:, data.directed_mask.view(-1)]
    return torch.minimum(edges[0], edges[1]) * data.num_nodes + torch.maximum(edges[0], edges[1])


@pytest.mark.parametrize("name", BUILDERS)
def test_generator_builds_valid_graph(name):

    seed_everything(0)
    data = BUILDERS[name]()

    num_directed = int(data.directed_mask.sum())
    keys = undirected_edge_keys(data)

    assert data.num_nodes > 0
    assert data.num_edges == 2 * num_directed
    assert data.name.shape == (data.num_nodes,)
    assert data.edge_index.max() < data.num_nodes

    # every edge exists once, and reciprocal edges point back to each other
    assert keys.unique().numel() == keys.numel()
    assert torch.equal(data.reciprocal_edge.view(-1)[data.reciprocal_edge.view(-1)], torch.arange(data.num_edges))
    assert torch.equal(data.edge_index[:, data.reciprocal_edge.view(-1)], data.edge_index.flip(0))

    # every named node can be looked up by its name
    for node_name in data.nodes:
        data.nodes.index(node_name)


@pytest.mark.parametrize("seed", range(5))
def test_truss_bridge_has_no_duplicate_edges(seed):

    seed_everything(seed)
    data = generators.TrussBridgeGenerator()()[0]

    keys = undirected_edge_keys(data)

    assert keys.unique().numel() == keys.numel()
