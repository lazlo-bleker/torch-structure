import torch


def deviation_force_function(n_trails: int, n_rings: int) -> torch.Tensor:
    u = torch.linspace(-torch.pi, torch.pi, n_trails)
    v = torch.linspace(-torch.pi, torch.pi, n_rings)
    forces_u = 1 + 0.015 * torch.cos(2 * u)
    forces_v = torch.cos(v)
    return 0.4 * torch.outer(forces_u, forces_v)


def trail_length_function(n_trails: int, n_rings: int) -> torch.Tensor:
    u = torch.linspace(-torch.pi, torch.pi, n_trails)
    v = torch.linspace(-torch.pi, torch.pi, n_rings)
    lentghts_u = 1 + 0.05 * torch.cos(2 * u)
    lentghts_v = torch.ones_like(v)
    return 3.0 / (n_rings - 1) * torch.outer(lentghts_u, lentghts_v)


def origin_node_function(
    n_nodes: int,
) -> torch.Tensor:
    center = torch.tensor([0.0, 0.0, 0.0])
    radius = 1.0
    t = 2 * torch.pi * torch.arange(n_nodes) / (n_nodes)
    x = radius * torch.cos(t)
    y = radius * torch.sin(t)
    z = radius / 10 * torch.cos(3 * t)
    return torch.stack([x, y, z], dim=1) + center
