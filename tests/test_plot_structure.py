import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
import torch
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from torch_structure.data import StructData
from torch_structure.plot import TENSION_COLOR, COMPRESSION_COLOR, plot_structure


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _truss():
    # a triangle on two supports: two compressed rafters and a tie in tension
    data = StructData(
        node_attrs={
            "coords": torch.empty((0, 3)),
            "load": torch.empty((0, 3)),
            "is_support": torch.empty((0, 1), dtype=torch.bool),
        },
        edge_attrs={
            "force": torch.empty((0, 1)),
            "material_id": torch.empty((0, 1), dtype=torch.long),
        },
    )
    data.add_node("left", coords=[0.0, 0.0, 0.0], is_support=[True])
    data.add_node("right", coords=[4.0, 0.0, 0.0], is_support=[True])
    data.add_node("top", coords=[2.0, 0.0, 1.0], load=[0.0, 0.0, -1.0])
    data.add_edge("left", "top", force=[-1.1], material_id=[0])
    data.add_edge("right", "top", force=[-1.1], material_id=[0])
    data.add_edge("left", "right", force=[1.0], material_id=[1])
    return data


def _edge_collection(ax):
    return next(c for c in ax.collections if isinstance(c, Line3DCollection))


def _texts(ax):
    return sorted(text.get_text() for text in ax.texts)


def test_each_undirected_edge_is_drawn_once_colored_by_force_sign():
    data = _truss()

    ax = plot_structure(data)

    colors = _edge_collection(ax).get_color()
    assert len(colors) == 3
    assert [tuple(c) for c in colors] == [to_rgba(COMPRESSION_COLOR)] * 2 + [to_rgba(TENSION_COLOR)]
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["Tension", "Compression", "Support"]


def test_edge_and_node_labels():
    data = _truss()

    ax = plot_structure(data, edge_labels="index", node_labels="name")
    assert _texts(ax) == ["0", "2", "4", "left", "right", "top"]

    plt.close("all")
    ax = plot_structure(data, edge_labels="force", node_labels=["A", "B", "C"])
    assert _texts(ax) == ["-1.1", "-1.1", "1", "A", "B", "C"]


def test_edge_values_per_undirected_edge():
    data = _truss()
    lengths = data.length_from_coords[data.directed_mask.view(-1)]

    ax = plot_structure(data, edge_labels=lengths, label_format="{:.2f}")

    assert _texts(ax) == ["2.24", "2.24", "4.00"]


def test_continuous_values_get_a_colorbar_and_integers_categories():
    data = _truss()

    ax = plot_structure(data, edge_color="force")
    assert len(ax.figure.axes) == 2
    assert ax.get_legend() is None or "Tension" not in [t.get_text() for t in ax.get_legend().get_texts()]

    plt.close("all")
    ax = plot_structure(data, edge_color="material_id", colorbar=True)
    assert len(ax.figure.axes) == 1
    assert {"material_id = 0", "material_id = 1"} <= {t.get_text() for t in ax.get_legend().get_texts()}


def test_constant_color_and_masks():
    data = _truss()

    ax = plot_structure(data, edge_color="black", edge_mask=data.force.view(-1) > 0, node_mask=[2], node_labels="index")

    ax.figure.canvas.draw()
    assert len(_edge_collection(ax).get_segments()) == 1
    assert _texts(ax) == ["2"]


def test_projection_view_and_loads():
    data = _truss()

    ax = plot_structure(data, view="xz", loads=True)

    assert ax.name != "3d"
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "Load" in labels


def test_reactions_point_at_their_supports():
    data = _truss()
    data.reaction_force = torch.tensor([[0.0, 0.0, 0.5], [0.0, 0.0, 0.5], [float("nan")] * 3])

    ax = plot_structure(data, view="xz", reactions=True, vector_scale=1.0)

    quiver = ax.collections[-1]
    assert [tuple(p) for p in quiver.get_offsets()] == [(0.0, -0.5), (4.0, -0.5)]
    assert "Reaction" in [t.get_text() for t in ax.get_legend().get_texts()]


def test_loads_and_reactions_share_one_scale():
    data = _truss()
    data.reaction_force = torch.tensor([[0.0, 0.0, 0.5], [0.0, 0.0, 0.5], [float("nan")] * 3])

    ax = plot_structure(data, view="xz", loads=True, reactions=True)

    load, reaction = ax.collections[-2:]
    assert reaction.V[0] == pytest.approx(-0.5 * load.V[0])


def test_pulling_reactions_start_at_their_supports():
    data = _truss()
    # supports holding a hanging structure pull it outwards and down, away from its members
    data.reaction_force = torch.tensor([[-1.0, 0.0, -0.5], [1.0, 0.0, -0.5], [float("nan")] * 3])

    ax = plot_structure(data, view="xz", reactions=True, vector_scale=1.0)

    assert [tuple(p) for p in ax.collections[-1].get_offsets()] == [(0.0, 0.0), (4.0, 0.0)]


def test_node_vectors_need_three_components():
    data = _truss()
    data.reaction_force = torch.zeros(3, 6)

    with pytest.raises(ValueError, match=r"\[num_nodes, 3\]"):
        plot_structure(data, reactions=True)


def test_overlays_on_one_axis_keep_their_legend():
    data = _truss()

    ax = plot_structure(data, ghost=data.coords + 0.5, ghost_label="Before")
    ax = plot_structure(data, ax=ax)

    assert "Before" in [t.get_text() for t in ax.get_legend().get_texts()]


def test_invalid_arguments():
    data = _truss()

    with pytest.raises(ValueError, match="entries"):
        plot_structure(data, edge_color=torch.ones(5))
    with pytest.raises(ValueError, match="no attribute"):
        plot_structure(data, node_color="missing_attribute")
    with pytest.raises(ValueError, match="3D view"):
        plot_structure(data, view="xz", pattern=True)
    with pytest.raises(ValueError, match="Unknown view"):
        plot_structure(data, view="top")


def test_structdata_plot_translates_legacy_arguments():
    data = _truss()

    with pytest.warns(FutureWarning, match="show_supports"):
        ax = data.plot(show_supports=True, show_load=True, show_edge_indices=True)

    assert _texts(ax) == ["0", "2", "4"]
    assert "Load" in [t.get_text() for t in ax.get_legend().get_texts()]
