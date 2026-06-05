import torch
import numpy as np
from collections.abc import Callable
from collections import deque
from torch_structure.data import StructData
from torch_structure.generators.base_generator import BaseGenerator
from config import TORCH_FLOAT, DEVICE


def sample_surface(u, v, weights, phases):
    r = np.sqrt(u**2 + v**2)
    theta = np.atan2(u, v)

    coords_hat = np.zeros([u.shape[0], u.shape[1], 3])
    coords_hat[:, :, 0] = u
    coords_hat[:, :, 1] = v
    for n, weight in enumerate(weights):
        coords_hat[:, :, 2] += (
            weight * np.pow(r, n + 2) * np.cos(n + 2 * theta + phases[n])
        )

    return coords_hat


def compute_gravitational_load(coords_hat, force_per_area=1.0):
    # coords_hat: (n_u, n_v, 3)
    n_u, n_v, _ = coords_hat.shape

    # quad corners
    p00 = coords_hat[:-1, :-1]
    p10 = coords_hat[1:, :-1]
    p11 = coords_hat[1:, 1:]
    p01 = coords_hat[:-1, 1:]

    area_1 = 0.5 * torch.linalg.norm(torch.cross(p10 - p00, p11 - p00, dim=-1), dim=-1)
    area_2 = 0.5 * torch.linalg.norm(torch.cross(p11 - p00, p01 - p00, dim=-1), dim=-1)

    quad_area = area_1 + area_2

    node_area = torch.zeros((n_u, n_v), device=coords_hat.device)

    share = 0.25 * quad_area
    node_area[:-1, :-1] += share
    node_area[1:, :-1] += share
    node_area[1:, 1:] += share
    node_area[:-1, 1:] += share

    # --- compute gravitational load ---
    load = torch.zeros((n_u, n_v, 3), device=coords_hat.device)
    load[..., 2] = -force_per_area * node_area
    return load


class CapCeilingAssemblyGenerator(BaseGenerator):
    def __init__(self, **overrides):
        super().__init__(**overrides)
        self.max_attempts = 100
        self.node_attrs = {
            "coords": torch.empty((0, 3), dtype=TORCH_FLOAT, device=DEVICE),
            "coords_hat": torch.empty((0, 3), dtype=TORCH_FLOAT, device=DEVICE),
            "uv_coords": torch.empty((0, 2), dtype=TORCH_FLOAT, device=DEVICE),
            "load": torch.empty((0, 3), dtype=TORCH_FLOAT, device=DEVICE),
            "support_condition": torch.empty((0, 3), dtype=torch.long, device=DEVICE),
        }
        self.edge_attrs = {
            "force_density": torch.empty((0, 1), dtype=TORCH_FLOAT, device=DEVICE),
            "force": torch.empty((0, 1), dtype=TORCH_FLOAT, device=DEVICE),
            "length": torch.empty((0, 1), dtype=TORCH_FLOAT, device=DEVICE),
            "force_sign": torch.empty((0, 1), dtype=TORCH_FLOAT, device=DEVICE),
            "active_edof": torch.empty((0, 1), dtype=torch.bool, device=DEVICE),
            "assembly_sequence": torch.empty((0, 1), dtype=torch.long, device=DEVICE),
        }
        self.default_attrs = {
            "coords": torch.full((3,), torch.nan),
            "uv_coords": torch.full((2,), torch.nan),
            "length": torch.tensor([torch.nan]),
            "active_edof": torch.tensor(True, dtype=torch.bool),
            "assembly_sequence": torch.tensor([-1]),
        }

    def validate_input(
        *args,
        **kwwargs,
    ):
        return None

    def sample_input(
        self,
        n_u: int | None = None,
        n_v: int | None = None,
        surface_function=None,
        support_sides: list = None,
        assembly_steps=None,
        fd_init=None,
        fd_boundary_init=None,
        seed=None,
        force_per_area=1.0,
    ) -> dict:
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        if n_u is None:
            n_u = np.random.randint(5, 20)

        if n_v is None:
            n_v = np.random.randint(5, 20)

        # Handle sampling of target coords
        if surface_function is None:
            n_modes = 6
            alpha = 0.6
            decay_weights = np.exp(-alpha * n_modes)
            weights = 1.0 * decay_weights * np.random.sample(n_modes)
            phases = 2 * np.pi * np.random.sample(n_modes)
            surface_function = lambda u, v: sample_surface(u, v, weights, phases)

        u = np.linspace(-1.0, +1.0, n_u)[:, np.newaxis].repeat(n_v, axis=1)
        v = np.linspace(-1.0, +1.0, n_v)[np.newaxis, :].repeat(n_u, axis=0)
        coords_hat = torch.tensor(
            surface_function(u, v), dtype=TORCH_FLOAT, device=DEVICE
        )

        # Handle support conditions
        if support_sides is None:
            support_sides = [True, True]
            support_sides.extend(np.random.randint(0, 2, 2))
        assert len(support_sides) == 4
        support_mask = torch.zeros([n_u, n_v], dtype=bool)
        if support_sides[0]:
            support_mask[0] = True
        if support_sides[1]:
            support_mask[-1] = True
        if support_sides[2]:
            support_mask[:, 0] = True
        if support_sides[3]:
            support_mask[:, -1] = True

        load = compute_gravitational_load(coords_hat, force_per_area=force_per_area)

        if fd_init is None:
            fd_init = np.random.uniform(low=-1.5, high=-0.1, size=None)

        if fd_boundary_init is None:
            fd_boundary_init = np.random.uniform(low=-4.0, high=-0.1, size=None)

        if assembly_steps is None:
            assembly_seeds = sample_assembly_seeds(n_u - 1, n_v - 1)
            assembly_steps = populate_assembly_steps(assembly_seeds, n_u - 1, n_v - 1)

        assert assembly_steps.shape == (n_u - 1, n_v - 1)

        return {
            "n_u": n_u,
            "n_v": n_v,
            "coords_hat": coords_hat,
            "support_mask": support_mask,
            "load": load,
            "fd_init": fd_init,
            "fd_boundary_init": fd_boundary_init,
            "assembly_steps": assembly_steps,
        }

    def generate(
        self,
        n_u,
        n_v,
        coords_hat,
        support_mask,
        load,
        fd_init,
        fd_boundary_init,
        assembly_steps,
    ):
        # Initialize data object
        data = StructData(
            node_attrs=self.node_attrs,
            edge_attrs=self.edge_attrs,
            default_attrs=self.default_attrs,
        )

        # Generate nodes
        for u in range(n_u):
            for v in range(n_v):
                uv_coords = torch.tensor([u / (n_u - 1), v / (n_v - 1)])
                data.add_node(
                    data.num_nodes,
                    uv_coords=uv_coords,
                    coords=coords_hat[u, v],
                    coords_hat=coords_hat[u, v],
                    load=load[u, v],
                    support_condition=torch.tensor([True, True, True])
                    * support_mask[u, v],
                )

        # Create elements from grid of points
        node_indices = torch.arange(data.num_nodes).reshape(n_u, n_v)
        elements = -torch.ones([(n_u - 1), (n_v - 1), 4], dtype=torch.long)
        elements[:, :, 0] = node_indices[:-1, :-1]
        elements[:, :, 1] = node_indices[1:, :-1]
        elements[:, :, 2] = node_indices[1:, 1:]
        elements[:, :, 3] = node_indices[:-1, 1:]

        edges = torch.stack(
            [
                torch.stack([elements[..., 0], elements[..., 1]], dim=-1),
                torch.stack([elements[..., 1], elements[..., 2]], dim=-1),
                torch.stack([elements[..., 2], elements[..., 3]], dim=-1),
                torch.stack([elements[..., 3], elements[..., 0]], dim=-1),
            ],
            dim=0,
        )
        edges = edges.reshape(-1, 2)
        edge_vals = assembly_steps.reshape(-1).repeat(4)
        edges = torch.sort(edges, dim=1).values
        edges_unique, inv = torch.unique(edges, dim=0, return_inverse=True)

        edge_vals_min = torch.full(
            (edges_unique.size(0),), edge_vals.max() + 1, dtype=edge_vals.dtype
        )

        edge_vals_min.scatter_reduce_(0, inv, edge_vals, reduce="amin")

        # Generate edges
        for e, edge in enumerate(edges_unique):
            data.add_edge(
                int(edge[0]), int(edge[1]), assembly_sequence=edge_vals_min[e]
            )

        # Populate force densities of inner edges
        data.force_density = fd_init * torch.ones_like(data.force_density)
        # Populate force densities of boundary edges
        edge_uv = data.uv_coords[data.edge_index]
        eps = 1e-6
        u = edge_uv[..., 0]
        v = edge_uv[..., 1]
        on_u0 = torch.all(u.abs() < eps, dim=0)
        on_u1 = torch.all((u - 1.0).abs() < eps, dim=0)
        on_v0 = torch.all(v.abs() < eps, dim=0)
        on_v1 = torch.all((v - 1.0).abs() < eps, dim=0)
        is_boundary_edge = on_u0 | on_u1 | on_v0 | on_v1
        data.force_density[is_boundary_edge] = fd_boundary_init

        return data


def sample_assembly_seeds(n_u, n_v, sample_lines=True, sample_elements=True):
    assembly_seeds = torch.zeros((n_u, n_v), dtype=torch.bool)
    if sample_lines:
        while True:
            n_seeds_u = np.random.random_integers(0, 3)
            n_seeds_v = np.random.random_integers(0, 3)
            n_seeds_total = n_seeds_u + n_seeds_v
            if n_seeds_total >= 1 and n_seeds_total <= 3:
                break

        seeds_u = np.random.random_integers(0, n_u - 1, size=[n_seeds_u])
        assembly_seeds[seeds_u, :] = True
        seeds_v = np.random.random_integers(0, n_v - 1, size=[n_seeds_v])
        assembly_seeds[:, seeds_v] = True

    if sample_elements:
        n_seeds = np.random.random_integers(1, 6)
        seed_indices_u = np.random.random_integers(0, n_u - 1, size=[n_seeds])
        seed_indices_v = np.random.random_integers(0, n_v - 1, size=[n_seeds])
        assembly_seeds[seed_indices_u, seed_indices_v] = True

    return assembly_seeds


def populate_assembly_steps(assembly_seeds, n_u, n_v):
    elements_assembly = -torch.ones((n_u, n_v), dtype=torch.long)
    elements_assembly[assembly_seeds] = 0

    # queue of active cells
    queue = deque()

    for i in range(n_u):
        for j in range(n_v):
            if elements_assembly[i, j] == 0:
                queue.append((i, j))

    # BFS propagation
    while queue:
        i, j = queue.popleft()
        val = elements_assembly[i, j]

        for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            ni, nj = i + di, j + dj

            if 0 <= ni < n_u and 0 <= nj < n_v:
                if elements_assembly[ni, nj] == -1:
                    elements_assembly[ni, nj] = val + 1
                    queue.append((ni, nj))
    return elements_assembly
