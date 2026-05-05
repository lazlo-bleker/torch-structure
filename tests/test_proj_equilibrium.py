import pytest
import torch
from obj_functions.self_supporting import _eval_auxiliary_forces, supporting_loss_cache


@pytest.mark.parametrize("cablenet_data", [0, 1, 3, 5, 7], indirect=True)
def test_with_specific_seeds_fdm(cablenet_data):
    assert cablenet_data is not None
    cablenet_data.assembly_sequence = torch.zeros(
        cablenet_data.num_edges, dtype=torch.long
    )
    cache_dict = supporting_loss_cache(cablenet_data)
    aux_forces_steps, internal_force_steps = _eval_auxiliary_forces(cablenet_data, **cache_dict)
    # Check 1: Forces add up to zero at every node
    loss = torch.linalg.norm(aux_forces_steps)
    assert torch.allclose(loss, torch.zeros_like(loss))
    # Check 2: Internal forces are equal to the ones used during form-finding
    assert torch.allclose(internal_force_steps.squeeze(1), cablenet_data.force[cablenet_data.directed_mask])

@pytest.mark.parametrize("dome_data", [0, 1, 3, 5, 7], indirect=True)
def test_with_specific_seeds_cem(dome_data):
    assert dome_data is not None
    dome_data.assembly_sequence = torch.zeros(dome_data.num_edges, dtype=torch.long)
    cache_dict = supporting_loss_cache(dome_data)
    aux_forces_steps, internal_force_steps = _eval_auxiliary_forces(dome_data, **cache_dict)
    # Check 1: Forces add up to zero at every node
    loss = torch.linalg.norm(aux_forces_steps)
    assert torch.allclose(loss, torch.zeros_like(loss))
    # Check 2: Internal forces are equal to the ones used during form-finding
    assert torch.allclose(internal_force_steps.squeeze(1), dome_data.force[dome_data.directed_mask])

