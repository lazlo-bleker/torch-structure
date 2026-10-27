import torch
from torch_structure.data import StructData, Dataset, save
from torch_structure.loader import DataLoader


def build_structure(height):

    # a chain of two loaded nodes between two supports, for form-finding with FDM
    node_attrs = {
        "coords": torch.empty((0, 3), dtype=torch.float),
        "load": torch.empty((0, 3), dtype=torch.float),
        "is_support": torch.empty((0, 1), dtype=torch.bool),
    }
    edge_attrs = {
        "force_density": torch.empty((0, 1), dtype=torch.float),
        "force": torch.empty((0, 1), dtype=torch.float),
    }
    default_attrs = {"force_density": torch.tensor([1.0])}

    data = StructData(node_attrs=node_attrs, edge_attrs=edge_attrs, default_attrs=default_attrs)
    data.add_node("left", coords=[0., 0., 0.], is_support=True)
    data.add_node("right", coords=[3., 0., 0.], is_support=True)
    data.add_node("a", coords=[1., 0., 0.], load=[0., 0., -height])
    data.add_node("b", coords=[2., 0., 0.], load=[0., 0., -height])
    data.add_edge("left", "a")
    data.add_edge("a", "b")
    data.add_edge("b", "right")

    return data


def test_save_and_load(tmp_path):

    data_list = [build_structure(1.0), build_structure(2.0)]
    expected = [data.clone() for data in data_list]

    save(data_list, tmp_path)
    dataset = Dataset(tmp_path)

    assert len(dataset) == 2

    for loaded, original in zip(dataset, expected):
        assert isinstance(loaded, StructData)
        for key in ["edge_index", "directed_mask", "reciprocal_edge", "coords", "load", "is_support", "force_density", "name"]:
            assert torch.equal(loaded[key], original[key])


def test_dataloader_batches(tmp_path):

    save([build_structure(1.0), build_structure(2.0), build_structure(3.0)], tmp_path)
    dataset = Dataset(tmp_path)

    batches = list(DataLoader(dataset, batch_size=2))

    assert [batch.num_graphs for batch in batches] == [2, 1]
    assert torch.equal(batches[0].batch, torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]))

    # edge indices of the second graph are offset by the nodes of the first one
    edge_index_expected = torch.tensor([
        [0, 2, 2, 3, 3, 1, 4, 6, 6, 7, 7, 5],
        [2, 0, 3, 2, 1, 3, 6, 4, 7, 6, 5, 7],
    ])
    assert torch.equal(batches[0].edge_index, edge_index_expected)
    assert torch.equal(batches[0].reciprocal_edge.view(-1), torch.tensor([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]))


def test_batched_fdm_matches_single_fdm(tmp_path):

    save([build_structure(1.0), build_structure(2.0)], tmp_path)
    dataset = Dataset(tmp_path)

    single_coords = torch.cat([data.fdm().coords for data in dataset])

    batch = next(iter(DataLoader(dataset, batch_size=2)))
    batched_coords = batch.fdm(use_batching=True).coords

    assert torch.allclose(batched_coords, single_coords, atol=1e-6)
