import numpy as np
import pytest
import torch
import torch_structure as ts
from torch_structure.data import StructData
from torch_structure.loader import DataLoader


PRECISIONS = [torch.float32, torch.float64]


def build_trail():

    return ts.generators.SingleTrailGenerator(
        n_nodes=5, trail_element_length=0.1, nodal_load=[0.0, 0.0, -1.0], origin_node_load=[4.0, 0.0, 0.0],
    )()


def build_cable(height):

    # a chain of two loaded nodes between two supports, for form-finding with FDM
    node_attrs = {
        "coords": torch.empty((0, 3), dtype=torch.float),
        "load": torch.empty((0, 3), dtype=torch.float),
        "is_support": torch.empty((0, 1), dtype=torch.bool),
    }
    edge_attrs = {
        "force_density": torch.empty((0, 1), dtype=torch.float),
        "force": torch.empty((0, 1), dtype=torch.float),
    }
    default_attrs = {"force_density": torch.tensor([1.0])}

    data = StructData(node_attrs=node_attrs, edge_attrs=edge_attrs, default_attrs=default_attrs)
    data.add_node("left", coords=[0., 0., 0.], is_support=True)
    data.add_node("right", coords=[3., 0., 0.], is_support=True)
    data.add_node("a", coords=[1., 0., 0.], load=[0., 0., -height])
    data.add_node("b", coords=[2., 0., 0.], load=[0., 0., -height])
    data.add_edge("left", "a")
    data.add_edge("a", "b")
    data.add_edge("b", "right")

    return data


def test_to_dtype():

    data = build_cable(1.0)

    converted = data.to_dtype(torch.float64)

    assert converted.coords.dtype == torch.float64
    assert converted.force_density.dtype == torch.float64
    assert converted.metadata["default_attrs"]["force_density"].dtype == torch.float64
    # integer and boolean attributes are left unchanged
    assert converted.edge_index.dtype == torch.long
    assert converted.is_support.dtype == torch.bool
    assert converted.name.dtype == torch.long
    # the original is left unchanged
    assert data.coords.dtype == torch.float32
    assert data.metadata["default_attrs"]["force_density"].dtype == torch.float32
    assert torch.equal(converted.coords, data.coords.double())

    # nodes added afterwards get the converted default values
    converted.add_node("c", coords=[4., 0., 0.])
    assert converted.coords.dtype == torch.float64


def test_to_dtype_rejects_non_floating_dtype():

    with pytest.raises(ValueError, match="floating point dtype"):
        build_cable(1.0).to_dtype(torch.long)


@pytest.mark.parametrize("dtype", PRECISIONS)
def test_set_edge_attr_casts_to_attribute_dtype(dtype):

    data = build_cable(1.0).to_dtype(dtype)
    mask = data.directed_mask.view(-1)

    forces = torch.tensor([[1.], [2.], [3.]], dtype=torch.float64, requires_grad=True)
    data.set_edge_attr("force", mask, forces)
    data.force.sum().backward()

    assert data.force.dtype == dtype
    assert torch.equal(forces.grad, torch.full((3, 1), 2., dtype=torch.float64))


@pytest.mark.parametrize("dtype", PRECISIONS)
@pytest.mark.parametrize("method", ["mpcem", "cem", "seqcem"])
def test_cem_computes_in_structure_dtype(method, dtype):

    data = build_trail().to_dtype(dtype)

    result = getattr(data, method)(max_iter=100)
    reference = getattr(build_trail().to_dtype(torch.float64), method)(max_iter=100)

    assert result.coords.dtype == dtype
    assert result.force.dtype == dtype
    assert torch.allclose(result.coords.double(), reference.coords, atol=1e-5)


@pytest.mark.parametrize("dtype", PRECISIONS)
def test_fdm_computes_in_structure_dtype(dtype):

    data = build_cable(1.0).to_dtype(dtype)

    result = data.fdm()
    reference = build_cable(1.0).to_dtype(torch.float64).fdm()

    assert result.coords.dtype == dtype
    assert torch.allclose(result.coords.double(), reference.coords, atol=1e-5)


@pytest.mark.parametrize("dtype", PRECISIONS)
def test_batched_fdm_computes_in_structure_dtype(dtype):

    data_list = [build_cable(1.0).to_dtype(dtype), build_cable(2.0).to_dtype(dtype)]
    for data in data_list:
        del data.metadata

    batch = next(iter(DataLoader(data_list, batch_size=2)))
    result = batch.fdm(use_batching=True)

    assert result.coords.dtype == dtype


@pytest.mark.parametrize("dtype", PRECISIONS)
def test_scipy_gradient_without_cast(dtype):

    # the objective takes scipy's float64 values directly, without casting them to the structure's dtype
    data = build_trail().to_dtype(dtype)
    mask = (data.is_trail_edge & data.directed_mask).view(-1)
    target = torch.tensor([0.5, 0.0, -0.5], dtype=dtype)

    @ts.utils.scipy_jacobian
    def objective(lengths, data):
        data.set_edge_attr("length", mask, lengths.unsqueeze(1))
        solved = data.mpcem(max_iter=100)
        return torch.sum((solved.coords[-1] - target) ** 2)

    loss, grad = objective(np.full(int(mask.sum()), 0.1), data)

    assert np.isfinite(loss)
    assert grad.dtype == np.float64
    assert grad.shape == (int(mask.sum()),)
    assert np.all(np.isfinite(grad)) and np.any(grad != 0)


ACCELERATORS = [
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")),
    pytest.param("mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")),
]


@pytest.mark.parametrize("device", ACCELERATORS)
def test_scipy_gradient_on_accelerator(device):

    # scipy's float64 values live on the CPU; set_edge_attr moves them to the structure's device, and the
    # gradient comes back to the CPU in float64, also on devices without float64 support (MPS)
    @ts.utils.scipy_jacobian
    def objective(lengths, data):
        mask = (data.is_trail_edge & data.directed_mask).view(-1)
        target = torch.tensor([0.5, 0.0, -0.5], device=data.coords.device)
        data.set_edge_attr("length", mask, lengths.unsqueeze(1))
        solved = data.mpcem(max_iter=100)
        return torch.sum((solved.coords[-1] - target) ** 2)

    data = build_trail().to(device)
    x = np.full(int((data.is_trail_edge & data.directed_mask).sum().item()), 0.1)

    loss, grad = objective(x, data)
    loss_cpu, grad_cpu = objective(x, build_trail())

    assert data.length.device.type == device
    assert grad.dtype == np.float64
    assert np.isclose(loss, loss_cpu, atol=1e-5)
    assert np.allclose(grad, grad_cpu, atol=1e-4)
