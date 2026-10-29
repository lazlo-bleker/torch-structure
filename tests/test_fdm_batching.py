import pytest
import torch
from torch_structure.data import StructData
from torch_structure.loader import DataLoader


def build_net(n, force_density=5.0, dtype=torch.float32):

    # a square cable net of n x n nodes, supported along its boundary, with raised corners
    node_attrs = {
        "coords": torch.empty((0, 3), dtype=dtype),
        "load": torch.empty((0, 3), dtype=dtype),
        "is_support": torch.empty((0, 1), dtype=torch.bool),
    }
    edge_attrs = {
        "force_density": torch.empty((0, 1), dtype=dtype),
        "force": torch.empty((0, 1), dtype=dtype),
    }
    default_attrs = {
        "load": torch.tensor([0.0, 0.5, -1.0], dtype=dtype),
        "force_density": torch.tensor([force_density], dtype=dtype),
    }
    data = StructData(node_attrs=node_attrs, edge_attrs=edge_attrs, default_attrs=default_attrs)

    def coords(x_unit_coord, y_unit_coord, is_corner):
        return torch.cat([n * x_unit_coord, n * y_unit_coord, is_corner.to(dtype)], dim=1)

    def support(is_boundary):
        return is_boundary

    data.add_grid(n, n, node_attrs={"coords": coords, "is_support": support})
    return data


def build_batch(nets):

    return next(iter(DataLoader(nets, batch_size=len(nets))))


@pytest.mark.parametrize("use_batching", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_batched_fdm_matches_single(use_batching, dtype):

    # graphs of different sizes, so that the blocks are padded
    nets = [build_net(n, force_density=n, dtype=dtype) for n in (3, 6, 4)]
    result = build_batch(nets).fdm(use_batching=use_batching)
    single = [net.fdm() for net in nets]

    assert result.coords.dtype == dtype
    assert torch.allclose(result.coords, torch.cat([s.coords for s in single]), atol=1e-5)
    assert torch.allclose(result.force, torch.cat([s.force for s in single]), atol=1e-4)
    assert torch.allclose(result.reaction_force, torch.cat([s.reaction_force for s in single]), atol=1e-4,
                          equal_nan=True)  # NaN at free nodes


def test_batched_fdm_solve_only_z():

    nets = [build_net(3), build_net(5)]
    result = build_batch(nets).fdm(solve_only_z=True)
    single = torch.cat([net.fdm(solve_only_z=True).coords for net in nets])

    assert torch.allclose(result.coords, single, atol=1e-5)


def test_batched_fdm_keeps_supports():

    nets = [build_net(3), build_net(4)]
    batch = build_batch(nets)
    result = batch.fdm()
    supports = batch.is_support.view(-1)

    assert torch.equal(result.coords[supports], batch.coords[supports])


def test_batched_fdm_gradient_matches_single():

    # the force densities of all directed edges are passed to fdm directly
    nets = [build_net(3, dtype=torch.float64), build_net(5, dtype=torch.float64)]

    def gradient(data):
        force_density = data.force_density[data.directed_mask.view(-1)].clone().requires_grad_(True)
        data.fdm(force_density=force_density).coords[:, 2].sum().backward()
        return force_density.grad

    single = torch.cat([gradient(net) for net in nets])
    batched = gradient(build_batch(nets))

    assert torch.allclose(batched, single, atol=1e-8)


def test_batched_fdm_works_with_torch_func():

    batch = build_batch([build_net(3, dtype=torch.float64), build_net(4, dtype=torch.float64)])
    force_density = batch.force_density[batch.directed_mask.view(-1)].clone()

    grad = torch.func.grad(lambda q: batch.fdm(force_density=q).coords[:, 2].sum())(force_density)

    assert grad.shape == force_density.shape
    assert torch.isfinite(grad).all()
