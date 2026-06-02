import pytest
import torch
from obj_functions.self_supporting import _eval_auxiliary_forces, supporting_loss_cache

# Tolerance for auxiliary forces (residual)
rtol_res = 1e-8
atol_res = 1e-10
# Tolerance for internal forces (solution)
rtol_x = 1e-8
atol_x = 1e-10


@pytest.mark.parametrize("cablenet_data", [0, 1, 3, 5, 7], indirect=True)
def test_with_specific_seeds_fdm(cablenet_data):
    assert cablenet_data is not None
    cablenet_data.assembly_sequence = torch.zeros(
        cablenet_data.num_edges, dtype=torch.long
    )
    cache_dict = supporting_loss_cache(cablenet_data)
    aux_forces_steps, internal_force_steps = _eval_auxiliary_forces(
        cablenet_data, **cache_dict
    )
    # Check 1: Forces add up to zero at every node
    target_values = torch.zeros_like(aux_forces_steps)
    computed_values = aux_forces_steps
    assert torch.allclose(target_values, computed_values, rtol_res, atol_res)
    # Check 2: Internal forces are equal to the ones used during form-finding
    target_values = cablenet_data.force[cablenet_data.directed_mask]
    computed_values = internal_force_steps.squeeze(1)
    assert torch.allclose(target_values, computed_values, rtol_x, atol_x)


@pytest.mark.parametrize("dome_data", [0, 1, 3, 5, 7], indirect=True)
def test_with_specific_seeds_cem(dome_data):
    assert dome_data is not None
    dome_data.assembly_sequence = torch.zeros(dome_data.num_edges, dtype=torch.long)
    cache_dict = supporting_loss_cache(dome_data)
    aux_forces_steps, internal_force_steps = _eval_auxiliary_forces(
        dome_data, **cache_dict
    )
    # Check 1: Forces add up to zero at every node
    target_values = torch.zeros_like(aux_forces_steps)
    computed_values = aux_forces_steps
    assert torch.allclose(target_values, computed_values, rtol_res, atol_res)
    # Check 2: Internal forces are equal to the ones used during form-finding
    target_values = dome_data.force[dome_data.directed_mask]
    computed_values = internal_force_steps.squeeze(1)
    assert torch.allclose(target_values, computed_values, rtol_x, atol_x)
