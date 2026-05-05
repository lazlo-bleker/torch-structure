import pytest
import torch
from obj_functions.self_supporting import supporting_loss_cache, supporting_loss_func


@pytest.mark.parametrize("cablenet_data", [0, 1, 3, 5, 7], indirect=True)
def test_with_specific_seeds_fdm(cablenet_data):
    assert cablenet_data is not None
    cablenet_data.assembly_sequence = torch.zeros(
        cablenet_data.num_edges, dtype=torch.long
    )
    cache_dict = supporting_loss_cache(cablenet_data)
    loss = supporting_loss_func(cablenet_data, **cache_dict)
    assert torch.allclose(loss, torch.zeros_like(loss))


@pytest.mark.parametrize("dome_data", [0, 1, 3, 5, 7], indirect=True)
def test_with_specific_seeds_cem(dome_data):
    assert dome_data is not None
    dome_data.assembly_sequence = torch.zeros(dome_data.num_edges, dtype=torch.long)
    cache_dict = supporting_loss_cache(dome_data)
    loss = supporting_loss_func(dome_data, **cache_dict)
    assert torch.allclose(loss, torch.zeros_like(loss))
