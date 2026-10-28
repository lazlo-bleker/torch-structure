import warnings

import pytest
import torch

from torch_structure.analysis import frame_analysis
from torch_structure.data.data import StructData


def build_data(**supports):
    """Three nodes in a row, two edges, and the given support attributes stored on the nodes."""
    node_attrs = {"coords": torch.empty((0, 3), dtype=torch.float)}
    for name, value in supports.items():
        node_attrs[name] = torch.empty((0, value.shape[1]), dtype=torch.bool)

    data = StructData(node_attrs=node_attrs)
    data.add_nodes(coords=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]), **supports)
    data.add_edges(torch.tensor([[0, 1], [1, 2]]))
    return data


PIN = torch.tensor([[True], [False], [True]])
ROLLER_XYZ = torch.tensor([[True, True, True], [False, False, False], [False, False, True]])


def test_is_support_only():
    data = build_data(is_support=PIN)

    assert torch.equal(data.support_xyz, PIN.repeat(1, 3))
    assert torch.equal(data.support_rxyz, torch.zeros(3, 3, dtype=torch.bool))
    assert torch.equal(data.support_dofs, torch.cat([PIN.repeat(1, 3), torch.zeros(3, 3, dtype=torch.bool)], dim=1))


def test_support_xyz_only():
    data = build_data(support_xyz=ROLLER_XYZ)

    assert torch.equal(data.is_support, PIN)
    assert torch.equal(data.support_xyz, ROLLER_XYZ)


def test_support_rxyz():
    rxyz = torch.tensor([[True, True, True], [False, False, False], [False, False, False]])
    data = build_data(is_support=PIN, support_rxyz=rxyz)

    assert torch.equal(data.support_dofs[:, 3:], rxyz)


def test_consistent_is_support_and_support_xyz():
    data = build_data(is_support=PIN, support_xyz=ROLLER_XYZ)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert torch.equal(data.support_dofs[:, :3], ROLLER_XYZ)


def test_inconsistent_is_support_and_support_xyz_warns():
    data = build_data(is_support=torch.tensor([[True], [True], [True]]), support_xyz=ROLLER_XYZ)

    with pytest.warns(UserWarning, match="disagree"):
        support_dofs = data.support_dofs

    assert torch.equal(support_dofs[:, :3], ROLLER_XYZ)


def test_no_supports():
    data = build_data()

    assert not hasattr(data, "is_support")
    assert not hasattr(data, "support_xyz")


@pytest.mark.parametrize("fixed_rotations", [False, True])
def test_fem_reads_support_dofs(fixed_rotations):
    node_attrs = {
        "coords": torch.empty((0, 3), dtype=torch.float),
        "load": torch.empty((0, 3), dtype=torch.float),
        "is_support": torch.empty((0, 1), dtype=torch.bool),
        "support_rxyz": torch.empty((0, 3), dtype=torch.bool),
    }
    edge_attrs = {name: torch.empty((0, 1), dtype=torch.float) for name in ["E", "A", "Iy", "Iz", "J"]}
    data = StructData(node_attrs=node_attrs, edge_attrs=edge_attrs)
    if fixed_rotations:
        support_rxyz = PIN.repeat(1, 3)
    else:
        # pinned ends; torsion of the whole beam is a mechanism, so restrain it at one end
        support_rxyz = torch.tensor([[True, False, False], [False, False, False], [False, False, False]])
    data.add_nodes(
        coords=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        load=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, -1e3], [0.0, 0.0, 0.0]]),
        is_support=PIN,
        support_rxyz=support_rxyz,
    )
    section = {"E": 210e9, "A": 1e-2, "Iy": 2e-5, "Iz": 1e-5, "J": 3e-5}
    data.add_edges(torch.tensor([[0, 1], [1, 2]]), **{k: torch.full((2, 1), v) for k, v in section.items()})

    result = data.fem()

    expected_support = torch.cat([PIN.repeat(1, 3), support_rxyz], dim=1)
    expected = frame_analysis(
        data.coords, data.directed_edge_index, data.load, expected_support, **section
    )
    assert torch.allclose(result.displacement, expected.displacement)
    assert (result.reaction_moment[0, 1].abs() > 0).item() == fixed_rotations
