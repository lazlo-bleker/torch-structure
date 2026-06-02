import torch
import numpy as np
from collections.abc import Callable

from torch_structure.data import StructData
from torch_structure.generators.base_generator import BaseGenerator
from config import TORCH_FLOAT, DEVICE

def sample_surface(u, v, weights, phases):    
    r = np.sqrt(u ** 2 + v ** 2)
    theta = np.atan2(u, v)

    coords_hat = np.zeros([u.shape[0], u.shape[1], 3])
    coords_hat[:,:,0] = u
    coords_hat[:,:,1] = v
    for n, weight in enumerate(weights):
        coords_hat[:,:,2] += weight * np.pow(r, n) * np.cos(n * theta + phases[n])

    return coords_hat

def compute_gravitational_load(coords_hat, force_per_area = -1.0):
    # coords_hat: (n_u, n_v, 3)
    n_u, n_v, _ = coords_hat.shape

    # quad corners
    p00 = coords_hat[:-1, :-1]
    p10 = coords_hat[1:,  :-1]
    p11 = coords_hat[1:,  1:]
    p01 = coords_hat[:-1, 1:]

    area_1 = 0.5 * torch.linalg.norm(
        torch.cross(p10 - p00, p11 - p00, dim=-1),
        dim=-1
    )
    area_2 = 0.5 * torch.linalg.norm(
        torch.cross(p11 - p00, p01 - p00, dim=-1),
        dim=-1
    )

    quad_area = area_1 + area_2

    node_area = torch.zeros((n_u, n_v), device=coords_hat.device)

    share = 0.25 * quad_area
    node_area[:-1, :-1] += share
    node_area[1:,  :-1] += share
    node_area[1:,  1:]  += share
    node_area[:-1, 1:]  += share

    # --- compute gravitational load ---
    load = torch.zeros((n_u, n_v, 3), device=coords_hat.device)
    load[...,2] = force_per_area * node_area
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
            "assembly_sequence": torch.tensor([0]),
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
        surface_function = None,
        support_sides : list = None,
        fd_init = None,
        fd_boundary_init = None,
        seed = None,
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
            n_modes = 4
            alpha = 0.2
            decay_weights = np.exp(-alpha * n_modes)
            weights =  decay_weights * np.random.sample(n_modes)
            phases =  2 * np.pi * np.random.sample(n_modes)
            surface_function = lambda u,v : sample_surface(u,v, weights, phases)
        
        u = np.linspace(-1.0,+1.0, n_u)[:, np.newaxis].repeat(n_v, axis=1)
        v = np.linspace(-1.0,+1.0, n_v)[np.newaxis, :].repeat(n_u, axis=0)
        coords_hat = torch.tensor(surface_function(u,v), dtype=TORCH_FLOAT, device=DEVICE)

        # Handle support conditions
        if support_sides is None:
            while True:
                support_sides = np.random.randint(0, 2, 4)
                if support_sides.sum() >= 2:
                    break
        assert len(support_sides) == 4
        support_mask = torch.zeros([n_u, n_v], dtype=bool)
        if support_sides[0]:
            support_mask[0] = True
        if support_sides[1]:
            support_mask[-1] = True
        if support_sides[2]:
            support_mask[:,0] = True
        if support_sides[3]:
            support_mask[:,-1] = True

        load = compute_gravitational_load(coords_hat)

        if fd_init is None:
            fd_init = np.random.uniform(low=-50, high=-10, size=None)

        if fd_boundary_init is None:
            fd_boundary_init = np.random.uniform(low=-200, high=-10, size=None)

        return {
            "n_u": n_u,
            "n_v": n_v,
            "coords_hat": coords_hat,
            "support_mask": support_mask,
            "load" : load,
            "fd_init" : fd_init,
            "fd_boundary_init" : fd_boundary_init,
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
    ):
        # Initialize data object
        data = StructData(
            node_attrs=self.node_attrs,
            edge_attrs=self.edge_attrs,
            default_attrs=self.default_attrs,
        )

        # Generate nodes
        for v in range(n_v):
            for u in range(n_u):
                uv_coords = torch.tensor([u/(n_u-1), v/(n_v-1)])
                data.add_node(
                    f"{u}_{v}",
                    uv_coords=uv_coords,
                    coords = coords_hat[u,v],
                    coords_hat = coords_hat[u,v],
                    load = load[u,v],
                    support_condition = torch.tensor([True, True, True]) * support_mask[u,v],
                )

        # Generate edges
        for u in range(n_u):
            for v in range(n_v):
                if u < n_u - 1:
                    data.add_edge(
                        f"{u}_{v}",
                        f"{u+1}_{v}",
                    )
                if v < n_v - 1:
                    data.add_edge(
                        f"{u}_{v}",
                        f"{u}_{v+1}",
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
