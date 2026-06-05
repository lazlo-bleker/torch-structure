import pytest
import torch
from torch_structure.data.data import StructData

def test_add_edges():

    edge_attrs = {
        "force_densities": torch.empty((0, 1), dtype=torch.float),
    }

    data = StructData(edge_attrs=edge_attrs)

    fd = torch.tensor([[0], [3], [2]])

    new_edge_attrs = {
        "force_densities": fd,
    }

    data.add_n_empty_nodes(3)

    edge_indices = torch.tensor([[0,1,2], [1,2,0]])
    data.add_edges(edge_indices=edge_indices, **new_edge_attrs)
   


