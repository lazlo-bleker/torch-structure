import math

import matplotlib
import matplotlib.pyplot as plt
import pytest
import torch

from torch_structure.analysis import frame_analysis, local_axes
from torch_structure.data.data import StructData

E, G = 210e9, 80e9
A, Iy, Iz, J = 1e-2, 2e-5, 1e-5, 3e-5
SECTION = dict(E=E, A=A, Iy=Iy, Iz=Iz, J=J, G=G)


def line(n_members, length, direction=(1.0, 0.0, 0.0)):
    """Nodes and members of a straight line of ``n_members`` equal members."""
    t = torch.linspace(0, length, n_members + 1, dtype=torch.float64).unsqueeze(1)
    coords = t * torch.tensor(direction, dtype=torch.float64)
    edge_index = torch.stack([torch.arange(n_members), torch.arange(1, n_members + 1)])
    return coords, edge_index


def supports(num_nodes, fixed):
    """[N, 6] support mask from a {node: dof list} dict."""
    support = torch.zeros(num_nodes, 6, dtype=torch.bool)
    for node, dofs in fixed.items():
        support[node, dofs] = True
    return support


ALL = [0, 1, 2, 3, 4, 5]


def test_axial_bar():
    L, P = 3.0, 1e5
    coords, edge_index = line(1, L)
    load = torch.zeros(2, 3, dtype=torch.float64)
    load[1, 0] = P

    result = frame_analysis(coords, edge_index, load, supports(2, {0: ALL}), **SECTION)

    assert result.displacement[1, 0].item() == pytest.approx(P * L / (E * A))
    assert result.force.item() == pytest.approx(P)
    assert result.reaction_force[0, 0].item() == pytest.approx(-P)


def test_cantilever_bending_both_planes():
    L, P = 2.0, 1e3
    coords, edge_index = line(1, L)
    load = torch.zeros(2, 3, dtype=torch.float64)
    load[1, 1] = P  # bends about local z
    load[1, 2] = -P  # bends about local y (local z is global Z for a horizontal member)

    result = frame_analysis(coords, edge_index, load, supports(2, {0: ALL}), **SECTION)

    assert result.displacement[1, 1].item() == pytest.approx(P * L**3 / (3 * E * Iz))
    assert result.displacement[1, 2].item() == pytest.approx(-P * L**3 / (3 * E * Iy))
    assert result.rotation[1, 2].item() == pytest.approx(P * L**2 / (2 * E * Iz))
    assert result.reaction_moment[0, 1].item() == pytest.approx(-P * L)  # fixed-end moment about Y
    assert result.reaction_moment[0, 2].item() == pytest.approx(-P * L)


def test_torsion():
    L, T = 1.5, 2e3
    coords, edge_index = line(1, L)
    moment = torch.zeros(2, 3, dtype=torch.float64)
    moment[1, 0] = T

    result = frame_analysis(
        coords, edge_index, torch.zeros(2, 3, dtype=torch.float64), supports(2, {0: ALL}), moment=moment, **SECTION
    )

    assert result.rotation[1, 0].item() == pytest.approx(T * L / (G * J))
    assert result.end_forces[0, 9].item() == pytest.approx(T)


def test_simply_supported_beam():
    L, P = 4.0, 1e4
    coords, edge_index = line(2, L)
    load = torch.zeros(3, 3, dtype=torch.float64)
    load[1, 2] = -P
    # pin and roller, torsion restrained at one end
    support = supports(3, {0: [0, 1, 2, 3], 2: [1, 2]})

    result = frame_analysis(coords, edge_index, load, support, **SECTION)

    assert result.displacement[1, 2].item() == pytest.approx(-P * L**3 / (48 * E * Iy))
    assert result.reaction_force[0, 2].item() == pytest.approx(P / 2)


@pytest.mark.parametrize("end_release", [False, True])
def test_propped_cantilever(end_release):
    # a fixed end with a released member end behaves like a pinned support
    L, P = 4.0, 1e4
    coords, edge_index = line(2, L)
    load = torch.zeros(3, 3, dtype=torch.float64)
    load[1, 2] = -P
    release = torch.zeros(2, 6, dtype=torch.bool)
    if end_release:
        support = supports(3, {0: ALL, 2: ALL})
        release[1, 4] = True
    else:
        support = supports(3, {0: ALL, 2: [0, 1, 2]})

    result = frame_analysis(coords, edge_index, load, support, release=release, **SECTION)

    assert result.displacement[1, 2].item() == pytest.approx(-7 * P * L**3 / (768 * E * Iy))
    assert result.reaction_force[2, 2].item() == pytest.approx(5 * P / 16)
    assert result.end_forces[1, 10].item() == pytest.approx(0.0, abs=1e-6)
    assert result.reaction_moment[0, 1].item() == pytest.approx(-3 * P * L / 16)


def tripod():
    coords = torch.tensor(
        [[math.cos(a), math.sin(a), 0.0] for a in (0, 2 * math.pi / 3, 4 * math.pi / 3)] + [[0.0, 0.0, 2.0]],
        dtype=torch.float64,
    )
    edge_index = torch.tensor([[0, 1, 2], [3, 3, 3]])
    support = supports(4, {0: [0, 1, 2], 1: [0, 1, 2], 2: [0, 1, 2]})
    load = torch.zeros(4, 3, dtype=torch.float64)
    load[3, 2] = -9e4
    return coords, edge_index, support, load


def test_truss_without_bending_properties():
    coords, edge_index, support, load = tripod()

    result = frame_analysis(coords, edge_index, load, support, E=E, A=A)

    sin = 2.0 / math.sqrt(5.0)
    assert torch.allclose(result.force, torch.full((3, 1), -9e4 / (3 * sin), dtype=torch.float64))
    assert torch.allclose(result.rotation, torch.zeros(4, 3, dtype=torch.float64))


def test_fully_released_frame_is_truss():
    coords, edge_index, support, load = tripod()
    load[3, 0] = 1e4  # unsymmetric load, which a rigid frame would resist partly in bending

    truss = frame_analysis(coords, edge_index, load, support, E=E, A=A)
    released = frame_analysis(
        coords, edge_index, load, support, release=torch.ones(3, 6, dtype=torch.bool), **SECTION
    )
    rigid = frame_analysis(coords, edge_index, load, support, **SECTION)

    assert torch.allclose(truss.displacement, released.displacement, rtol=1e-10, atol=1e-18)
    assert torch.allclose(truss.end_forces, released.end_forces, rtol=1e-10, atol=1e-6)
    assert not torch.allclose(truss.displacement, rigid.displacement, rtol=1e-6, atol=1e-18)


def test_mixed_frame_equilibrium():
    # portal frame with pinned diagonal bracing, loaded in all directions
    coords = torch.tensor(
        [[0, 0, 0], [4, 0, 0], [0, 0, 3], [4, 0, 3], [0, 3, 0], [4, 3, 0], [0, 3, 3], [4, 3, 3]],
        dtype=torch.float64,
    )
    frame_members = [[0, 2], [1, 3], [4, 6], [5, 7], [2, 3], [6, 7], [2, 6], [3, 7]]
    braces = [[0, 3], [4, 7]]
    edge_index = torch.tensor(frame_members + braces).T
    release = torch.zeros(10, 6, dtype=torch.bool)
    release[8:] = True
    support = supports(8, {0: ALL, 1: ALL, 4: ALL, 5: ALL})
    torch.manual_seed(0)
    load = torch.randn(8, 3, dtype=torch.float64) * 1e4
    moment = torch.randn(8, 3, dtype=torch.float64) * 1e3

    result = frame_analysis(coords, edge_index, load, support, release=release, moment=moment, **SECTION)

    # reactions balance the applied forces and moments about the origin
    reaction_force = result.reaction_force
    total_force = load.sum(0) + reaction_force.sum(0)
    total_moment = (
        moment.sum(0)
        + result.reaction_moment.sum(0)
        + torch.linalg.cross(coords, load + reaction_force).sum(0)
    )
    assert torch.allclose(total_force, torch.zeros(3, dtype=torch.float64), atol=1e-6)
    assert torch.allclose(total_moment, torch.zeros(3, dtype=torch.float64), atol=1e-6)
    # braces carry no moments
    assert torch.allclose(result.end_forces[8:, [3, 4, 5, 9, 10, 11]], torch.zeros(2, 6, dtype=torch.float64))


def test_batch_matches_individual_solves():
    coords_a, edge_a = line(3, 3.0)
    coords_b, edge_b, support_b, load_b = tripod()
    load_a = torch.zeros(4, 3, dtype=torch.float64)
    load_a[3] = torch.tensor([1e3, 2e3, -3e3], dtype=torch.float64)
    support_a = supports(4, {0: ALL})

    a = frame_analysis(coords_a, edge_a, load_a, support_a, **SECTION)
    b = frame_analysis(coords_b, edge_b, load_b, support_b, **SECTION)
    both = frame_analysis(
        torch.cat([coords_a, coords_b]),
        torch.cat([edge_a, edge_b + 4], dim=1),
        torch.cat([load_a, load_b]),
        torch.cat([support_a, support_b]),
        batch=torch.tensor([0] * 4 + [1] * 4),
        **SECTION,
    )

    assert torch.allclose(both.displacement, torch.cat([a.displacement, b.displacement]))
    assert torch.allclose(both.end_forces, torch.cat([a.end_forces, b.end_forces]))


def test_roll_swaps_bending_axes():
    coords, edge_index = line(1, 2.0, direction=(0.0, 0.0, 1.0))
    load = torch.zeros(2, 3, dtype=torch.float64)
    load[1, 0] = 1e3

    plain = frame_analysis(coords, edge_index, load, supports(2, {0: ALL}), **SECTION)
    rolled = frame_analysis(coords, edge_index, load, supports(2, {0: ALL}), roll=math.pi / 2, **SECTION)

    # for a vertical member local z is global X, so a load along X bends about local y
    assert plain.displacement[1, 0].item() == pytest.approx(1e3 * 8 / (3 * E * Iy))
    assert rolled.displacement[1, 0].item() == pytest.approx(1e3 * 8 / (3 * E * Iz))


def test_local_axes_horizontal_member():
    _, rotation = local_axes(
        torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64), torch.tensor([[0.0, 2.0, 0.0]], dtype=torch.float64)
    )
    assert torch.allclose(rotation[0, 0], torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64))
    assert torch.allclose(rotation[0, 2], torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64))


def test_gradients():
    coords = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.5], [3.0, 1.0, 1.0]], dtype=torch.float64, requires_grad=True)
    area = torch.tensor([1e-2, 2e-2], dtype=torch.float64, requires_grad=True)
    edge_index = torch.tensor([[0, 1], [1, 2]])
    load = torch.tensor([[0.0, 0.0, 0.0], [1e3, 0.0, -2e3], [0.0, 1e3, -1e3]], dtype=torch.float64)
    release = torch.tensor([[False] * 6, [False, False, False, False, True, True]])

    def compliance(coords, area):
        result = frame_analysis(
            coords, edge_index, load, supports(3, {0: ALL}),
            E=1.0, A=area, Iy=1e-3, Iz=2e-3, J=1e-3, G=0.5, release=release,
        )
        return (load * result.displacement).sum()

    assert torch.autograd.gradcheck(compliance, (coords, area))


def build_frame_data():
    node_attrs = {
        "coords": torch.empty((0, 3), dtype=torch.float),
        "load": torch.empty((0, 3), dtype=torch.float),
        "is_support": torch.empty((0, 1), dtype=torch.bool),
        "support_rxyz": torch.empty((0, 3), dtype=torch.bool),
    }
    edge_attrs = {
        "E": torch.empty((0, 1), dtype=torch.float),
        "A": torch.empty((0, 1), dtype=torch.float),
        "Iy": torch.empty((0, 1), dtype=torch.float),
        "Iz": torch.empty((0, 1), dtype=torch.float),
        "release": torch.empty((0, 6), dtype=torch.bool),
    }
    data = StructData(node_attrs=node_attrs, edge_attrs=edge_attrs)
    # node 0 fixed, node 2 pinned
    is_support = torch.tensor([[True], [False], [True]])
    support_rxyz = torch.tensor([[True] * 3, [False] * 3, [False] * 3])
    load = torch.zeros(3, 3)
    load[1, 2] = -1e4
    data.add_nodes(
        coords=torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
        load=load,
        is_support=is_support,
        support_rxyz=support_rxyz,
    )
    data.add_edges(
        torch.tensor([[0, 1], [1, 2]]),
        E=torch.full((2, 1), E),
        A=torch.full((2, 1), A),
        Iy=torch.full((2, 1), Iy),
        Iz=torch.full((2, 1), Iz),
        release=torch.zeros(2, 6, dtype=torch.bool),
    )
    return data


def test_fem_mixin():
    data = build_frame_data()

    result = data.fem()

    assert result is not data
    assert result.displacement.dtype == torch.float
    assert result.displacement[1, 2].item() == pytest.approx(-7 * 1e4 * 64 / (768 * E * Iy), rel=1e-5)
    assert result.force.shape == (data.num_edges, 1)
    assert result.end_forces.shape == (data.num_edges, 12)
    # both directions of an edge carry the canonical edge's results
    assert torch.equal(result.end_forces, result.end_forces[data.reciprocal_edge.view(-1)])
    assert not hasattr(data, "displacement")


def test_fem_reactions_are_split_into_forces_and_moments():
    data = build_frame_data()

    result = data.fem()

    # the same names and shape as the reactions of form finding, so they plot as they are
    assert result.reaction_force.shape == (data.num_nodes, 3)
    assert result.reaction_moment.shape == (data.num_nodes, 3)
    assert result.reaction_force[:, 2].sum().item() == pytest.approx(1e4, rel=1e-5)
    assert result.reaction_moment[2].abs().sum().item() == 0.0  # node 2 is a pin

    matplotlib.use("Agg")
    ax = result.plot(view="xz", reactions=True)
    assert "Reaction" in [text.get_text() for text in ax.get_legend().get_texts()]
    plt.close("all")


def test_support_dofs_must_have_six_columns():
    coords, edge_index = line(1, 1.0)
    with pytest.raises(ValueError, match="support_dofs"):
        frame_analysis(coords, edge_index, torch.zeros(2, 3), torch.tensor([[True], [False]]), E=E, A=A)
