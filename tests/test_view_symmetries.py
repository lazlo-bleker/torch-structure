import pytest
import torch
from torch_structure.data.data import StructData


def test_view_symmetries_without_symmetries(capsys):

    data = StructData()

    data.view_symmetries()

    output_expected = "No symmetries registered.\n"

    assert capsys.readouterr().out == output_expected


def test_view_symmetries_lists_each_symmetry(capsys):

    node_attrs = {
        "coords": torch.empty((0, 3), dtype=torch.float),
        "load": torch.empty((0, 3), dtype=torch.float),
    }

    edge_attrs = {
        "force": torch.empty((0, 1), dtype=torch.float),
    }

    data = StructData(node_attrs=node_attrs, edge_attrs=edge_attrs)

    rotation = data.create_rotational_symmetry(4)
    mirror = data.create_mirror_symmetry()
    data.add_symmetry({"rot4": rotation, "mirror": mirror}, copy_attrs=["load", "force"])
    data.add_symmetry({"d4": data.combine_symmetry(mirror, rotation)}, transform_attrs=[])

    data.view_symmetries()

    # the order is the number of matrices
    output_expected = (
        "rot4: order=4, transform_attrs=['coords'], copy_attrs=['load', 'force']\n"
        "mirror: order=2, transform_attrs=['coords'], copy_attrs=['load', 'force']\n"
        "d4: order=8, transform_attrs=[], copy_attrs=[]\n"
    )

    assert capsys.readouterr().out == output_expected
