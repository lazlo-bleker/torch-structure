import pytest
import torch
import torch_structure as ts


def build_trail():

    return ts.generators.SingleTrailGenerator(
        n_nodes=5, trail_element_length=0.1, nodal_load=[0.0, 0.0, -1.0], origin_node_load=[4.0, 0.0, 0.0],
    )()


@pytest.mark.parametrize("method", ["mpcem", "cem", "seqcem"])
def test_reactions_balance_the_loads(method):

    result = getattr(build_trail().to_dtype(torch.float64), method)(max_iter=100)
    is_support = result.is_support.view(-1)
    reaction = result.reaction_force

    assert reaction[is_support].abs().sum() > 0
    assert torch.isnan(reaction[~is_support]).all()
    assert torch.allclose(reaction[is_support].sum(0) + result.load.sum(0), torch.zeros(3, dtype=torch.float64), atol=1e-6)


def test_fdm_reactions_balance_the_loads():

    net = ts.generators.CableNetGenerator(n=5, boundary_density=4, seed=0)().to_dtype(torch.float64)
    result = net.fdm()
    is_support = result.is_support.view(-1)
    reaction = result.reaction_force

    assert reaction[is_support].abs().sum() > 0
    assert torch.isnan(reaction[~is_support]).all()
    assert torch.allclose(reaction[is_support].sum(0) + result.load.sum(0), torch.zeros(3, dtype=torch.float64), atol=1e-6)
