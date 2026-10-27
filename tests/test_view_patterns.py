import pytest
import torch
from torch_structure.data.data import StructData


def test_view_patterns_without_patterns(capsys):

    data = StructData()

    data.view_patterns()

    output_expected = "No patterns registered.\n"

    assert capsys.readouterr().out == output_expected


def test_view_patterns_lists_each_pattern(capsys):

    node_attrs = {
        "coords": torch.empty((0, 3), dtype=torch.float),
        "load": torch.empty((0, 3), dtype=torch.float),
    }

    edge_attrs = {
        "force": torch.empty((0, 1), dtype=torch.float),
    }

    data = StructData(node_attrs=node_attrs, edge_attrs=edge_attrs)

    rotation = data.create_rotational_pattern(4)
    mirror = data.create_mirror_pattern()
    data.add_patterns({"rot4": rotation, "mirror": mirror})
    data.add_patterns({"d4": data.nest_patterns([mirror, rotation])}, transform_attrs=[])

    data.view_patterns()

    # the size is the number of matrices, the levels are listed innermost first
    output_expected = (
        "rot4: size=4, levels=[4 (closing)], transform_attrs=['coords']\n"
        "mirror: size=2, levels=[2 (closing)], transform_attrs=['coords']\n"
        "d4: size=8, levels=[2 (closing), 4 (closing)], transform_attrs=[]\n"
    )

    assert capsys.readouterr().out == output_expected
