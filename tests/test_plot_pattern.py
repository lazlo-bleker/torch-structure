import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
import torch
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from torch_structure.data.data import StructData
from torch_structure.plot import classify_affine, pattern_levels


def _rotation(axis, angle):
    axis = axis / axis.norm()
    K = torch.tensor([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return torch.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


def _isometry(linear, point, slide=torch.zeros(3)):
    # x -> linear (x - point) + point + slide
    matrix = torch.eye(4)
    matrix[:3, :3] = linear
    matrix[:3, 3] = point - linear @ point + slide
    return matrix


def _structure():
    node_attrs = {"coords": torch.empty((0, 3), dtype=torch.float)}
    edge_attrs = {"force": torch.empty((0, 1), dtype=torch.float)}
    return StructData(node_attrs=node_attrs, edge_attrs=edge_attrs)


def test_classify_affine_kinds():

    z = torch.tensor([0.0, 0.0, 1.0])
    point = torch.tensor([5.0, 0.0, 0.0])
    mirror_z = torch.eye(3) - 2 * torch.outer(z, z)

    matrices = torch.stack([
        torch.eye(4),
        _isometry(torch.eye(3), torch.zeros(3), torch.tensor([2.0, 0.0, 0.0])),
        _isometry(_rotation(z, math.pi / 3), point),
        _isometry(_rotation(z, math.pi / 3), point, 0.5 * z),
        _isometry(mirror_z, torch.tensor([0.0, 0.0, 1.0])),
        _isometry(mirror_z, torch.tensor([0.0, 0.0, 1.0]), torch.tensor([1.0, 0.0, 0.0])),
        _isometry(mirror_z @ _rotation(z, math.pi / 4), point),
        _isometry(-torch.eye(3), point),
    ])

    info = classify_affine(matrices)

    assert info["kind"] == [
        "identity", "translation", "rotation", "screw", "mirror", "glide", "rotoreflection", "inversion",
    ]
    assert info["rotation"].tolist() == [False, False, True, True, False, False, True, True]
    assert info["mirror"].tolist() == [False, False, False, False, True, True, True, True]
    assert info["translation"].tolist() == [False, True, False, True, False, True, False, False]


def test_classify_affine_moving_the_origin_is_no_translation():

    # a rotation about an axis, and a mirror in a plane, that miss the origin both move the origin, but they
    # place their axis and plane there instead of sliding
    z = torch.tensor([0.0, 0.0, 1.0])
    rotation = _isometry(_rotation(z, 2 * math.pi / 3), torch.tensor([5.0, 0.0, 0.0]))
    mirror = _isometry(torch.eye(3) - 2 * torch.outer(z, z), torch.tensor([0.0, 0.0, 1.0]))

    info = classify_affine(torch.stack([rotation, mirror]))

    assert info["kind"] == ["rotation", "mirror"]
    assert torch.count_nonzero(rotation[:3, 3]) > 0
    assert torch.allclose(info["slide"], torch.zeros(2, 3, dtype=torch.float64), atol=1e-6)
    # the point of the axis and the plane closest to the origin
    assert torch.allclose(info["point"][0], torch.tensor([5.0, 0.0, 0.0], dtype=torch.float64), atol=1e-5)
    assert torch.allclose(info["point"][1], torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64), atol=1e-6)
    assert math.isclose(info["angle"][0].item(), 2 * math.pi / 3, abs_tol=1e-6)
    assert torch.allclose(info["axis"][0], torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64), atol=1e-6)


def test_classify_affine_rebuilds_random_isometries():

    torch.manual_seed(0)
    matrices = []
    for _ in range(50):
        axis = torch.randn(3)
        point = 3 * torch.randn(3)
        angle = 0.1 + 3.0 * torch.rand(1).item()
        reflection = torch.eye(3) - 2 * torch.outer(axis, axis) / axis.dot(axis)
        matrices += [
            _isometry(_rotation(axis, angle), point, 0.7 * axis),
            _isometry(reflection, point),
            _isometry(reflection @ _rotation(axis, angle), point),
        ]
    matrices = torch.stack(matrices)

    info = classify_affine(matrices)

    # the parts put back together give the matrix again
    for k, matrix in enumerate(matrices):
        linear = _rotation(info["axis"][k].float(), info["angle"][k].item())
        if info["mirror"][k]:
            linear = (torch.eye(3) - 2 * torch.outer(info["axis"][k].float(), info["axis"][k].float())) @ linear
        rebuilt = _isometry(linear, info["point"][k].float(), info["slide"][k].float())
        assert torch.allclose(rebuilt, matrix, atol=1e-4)


def test_classify_affine_scaling_is_affine():

    info = classify_affine(torch.diag(torch.tensor([2.0, 2.0, 2.0, 1.0])).unsqueeze(0))

    assert info["kind"] == ["affine"]
    assert not info["isometry"][0]


def test_pattern_levels_nested():

    data = _structure()
    array = data.create_translational_pattern(4, direction=torch.tensor([2.0, 0.0, 0.0]))
    mirror = data.create_mirror_pattern(origin=torch.tensor([0.0, 0.0, 1.0]), normal=torch.tensor([0.0, 0.0, 1.0]))
    rotation = data.create_rotational_pattern(6, origin=torch.tensor([5.0, 0.0, 0.0]))
    data.add_patterns({"nested": data.nest_patterns([array, mirror, rotation])})

    levels = pattern_levels(data, "nested")

    assert [level["kind"] for level in levels] == ["translation", "mirror", "rotation"]
    assert [level["size"] for level in levels] == [4, 2, 6]
    assert [level["place"] for level in levels] == [1, 4, 8]

    # an open array has a step between each neighbour pair, a closing rotation one back from its last copy too
    assert len(levels[0]["steps"]["kind"]) == 3
    assert len(levels[2]["steps"]["kind"]) == 6

    translation, plane, axis = (level["steps"] for level in levels)
    assert torch.allclose(translation["slide"][0], torch.tensor([2.0, 0.0, 0.0], dtype=torch.float64), atol=1e-6)
    assert torch.allclose(plane["point"][0], torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64), atol=1e-6)
    assert torch.allclose(axis["point"][0], torch.tensor([5.0, 0.0, 0.0], dtype=torch.float64), atol=1e-5)
    assert math.isclose(axis["angle"][0].item(), math.pi / 3, abs_tol=1e-5)


def test_pattern_levels_combined_levels_stay_pure():

    # combining never fuses: a rotation and a translation along its axis stay two levels, no screw, and a
    # rotation and a mirror normal to its axis stay two levels, no rotoreflection
    data = _structure()
    rotation = data.create_rotational_pattern(4)
    lift = data.create_translational_pattern(3, direction=torch.tensor([0.0, 0.0, 1.0]))
    flip = data.create_mirror_pattern(normal=torch.tensor([0.0, 0.0, 1.0]))
    data.add_patterns({"tower": data.nest_patterns([rotation, lift]), "box": data.nest_patterns([rotation, flip])})

    assert [level["kind"] for level in pattern_levels(data, "tower")] == ["rotation", "translation"]
    assert [level["kind"] for level in pattern_levels(data, "box")] == ["rotation", "mirror"]


def test_pattern_levels_hybrid_level():

    data = _structure()
    turn = data.create_rotational_pattern(24)["matrices"][1]
    lift = data.create_translational_pattern(2, direction=torch.tensor([0.0, 0.0, 1.0]))["matrices"][1]
    screw = {
        "matrices": torch.stack([torch.linalg.matrix_power(turn @ lift, k) for k in range(5)]),
        "pattern_level": torch.zeros(5, dtype=torch.long),
        "closes": torch.tensor([True, True, True, True, False]),
    }
    data.add_patterns({"screw": screw})

    level, = pattern_levels(data, "screw")

    assert level["kind"] == "screw"
    assert level["rotation"] and level["translation"] and not level["mirror"]
    assert math.isclose(level["steps"]["angle"][0].item(), math.pi / 12, abs_tol=1e-5)
    assert torch.allclose(level["steps"]["slide"][0], torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64), atol=1e-5)


def test_pattern_levels_unknown_pattern():

    data = _structure()

    with pytest.raises(ValueError, match="not registered"):
        pattern_levels(data, "missing")


def test_plot_pattern_draws_each_level():

    data = _structure()
    array = data.create_translational_pattern(4, direction=torch.tensor([2.0, 0.0, 0.0]))
    mirror_x = data.create_mirror_pattern(normal=torch.tensor([1.0, 0.0, 0.0]))
    mirror_z = data.create_mirror_pattern(origin=torch.tensor([0.0, 0.0, 1.0]), normal=torch.tensor([0.0, 0.0, 1.0]))
    data.add_patterns({"truss": data.nest_patterns([array, mirror_x, mirror_z])})
    data.add_nodes_by_pattern(pattern="truss", coords=torch.tensor([[0.0, 0.0, 0.0]]))
    data.add_edges_by_pattern_level(
        src_expansion_ids=[0], dest_expansion_ids=[0],
        src_level_positions=[[0, 0, 0]], dest_level_positions=[[1, 0, 0]],
        force=torch.full((1, 1), 1.0),
    )

    ax = data.plot(pattern=True)

    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert labels == [
        "Tension",
        "T0 translation: 4 positions, step (2, 0, 0)",
        "T1 mirror: plane x = 0",
        "T2 mirror: plane z = 1",
    ]
    # each level is drawn once, in the first copy of the mirrors around it: the 3 steps of the array, and one
    # plane per mirror
    planes, = [collection for collection in ax.collections if isinstance(collection, Poly3DCollection)]
    assert len(planes.get_facecolor()) == 2
    assert len([patch for patch in ax.patches if isinstance(patch, FancyArrowPatch)]) == 3

    # with copies, the array is drawn in all 4 copies the mirrors make of it, while every copy the outer mirror
    # makes of the plane x = 0 is the same plane, so it is still drawn once
    ax = data.plot(pattern={"copies": True})

    planes, = [collection for collection in ax.collections if isinstance(collection, Poly3DCollection)]
    assert len(planes.get_facecolor()) == 2
    assert len([patch for patch in ax.patches if isinstance(patch, FancyArrowPatch)]) == 12

    plt.close("all")


def test_plot_pattern_shows_a_glide_as_translation():

    data = _structure()
    slide = data.create_translational_pattern(2, direction=torch.tensor([1.0, 0.0, 0.0]))["matrices"][1]
    flip = data.create_mirror_pattern(origin=torch.tensor([0.0, 0.0, 0.5]), normal=torch.tensor([0.0, 0.0, 1.0]))
    glide = {
        "matrices": torch.stack([torch.linalg.matrix_power(slide @ flip["matrices"][1], k) for k in range(5)]),
        "pattern_level": torch.zeros(5, dtype=torch.long),
        "closes": torch.tensor([True, True, True, True, False]),
    }
    data.add_patterns({"warren": glide})
    data.add_nodes_by_pattern(pattern="warren", coords=torch.tensor([[0.0, 0.0, 0.0]]))

    ax = data.plot(pattern=True)

    # the inference still reads a glide, the plot shows the translation it slides by, without its plane
    assert pattern_levels(data, "warren")[0]["kind"] == "glide"
    assert [text.get_text() for text in ax.get_legend().get_texts()] == ["T0 translation: 5 positions, step (1, 0, 0)"]
    assert not any(isinstance(collection, Poly3DCollection) for collection in ax.collections)

    plt.close("all")


def test_plot_pattern_without_nodes():

    data = _structure()
    data.add_patterns({"ring": data.create_rotational_pattern(5, origin=torch.tensor([1.0, 0.0, 0.0]))})

    ax = data.plot(pattern={"show_seeds": True})

    assert [text.get_text() for text in ax.get_legend().get_texts()] == [
        "T0 rotation: 5 positions, 72° about z through (1, 0, 0)",
    ]

    plt.close("all")


def test_plot_pattern_unknown_pattern():

    data = _structure()
    data.add_patterns({"ring": data.create_rotational_pattern(5)})

    with pytest.raises(ValueError, match="not registered"):
        data.plot(pattern="missing")
    with pytest.raises(ValueError, match="not registered"):
        data.plot(pattern={"patterns": ["ring", "missing"]})

