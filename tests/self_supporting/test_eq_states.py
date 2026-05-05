import torch
from obj_functions.self_supporting import _eval_auxiliary_forces, supporting_loss_cache
# Tolerance for auxiliary forces (residual)
rtol_res    = 1e-8
atol_res    = 1e-10

def test_equilibrium_assembly_states(struc_data):
    assert struc_data is not None
    cache_dict = supporting_loss_cache(struc_data)
    aux_forces_steps, internal_force_steps = _eval_auxiliary_forces(struc_data, **cache_dict)
    # Check 1: Forces add up to zero at every node
    target_values = torch.zeros_like(aux_forces_steps)
    computed_values = aux_forces_steps
    assert torch.allclose(target_values, computed_values, rtol_res, atol_res)

