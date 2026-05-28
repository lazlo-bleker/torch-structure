import pytest
import torch
from torch_structure.data.data import StructData
import matplotlib.pyplot as plt


# do small grid example and verify data manually?


#grid test with no custom attributes
#grid test with custom edge_attribute

#edge cases!!!

#outputs to test:
#node attrs, edge attrs, edge indices, directed mask, reciprocal edges



#???separate node funtion to add empty node?
#???name: add_n_empty_nodes()

#grid test with custom node_attribute
#@pytest.fixture
def test_grid_with_node_attributes():

    node_attrs = {
        "coords": torch.empty((0, 3), dtype=torch.float),
    }

    data = StructData(
        node_attrs=node_attrs,
    )

    def node_attr(u,v):

        return torch.hstack([u,v,torch.zeros_like(u)])


    node_attrs = {
        "coords": node_attr
    }

    data.add_grid(2,2, node_attrs=node_attrs)

    edge_index_expected = torch.tensor([[0, 2, 0, 1, 1, 3, 2, 3], [1, 3, 2, 3, 0, 2, 0, 1]], dtype=torch.long)
    directed_mask_expected = torch.tensor([True, True, True, True, False, False, False, False]).unsqueeze(-1)
    reciprocal_edge_expected = torch.tensor([4,5,6,7,0,1,2,3]).unsqueeze(-1)
    num_nodes_expected = torch.tensor(4) 
    coords_expected = torch.tensor([[0,0,0], [0,1,0], [1,0,0], [1,1,0]])

    assert torch.equal(data.reciprocal_edge, reciprocal_edge_expected)
    assert torch.equal(data.edge_index, edge_index_expected)
    assert torch.equal(data.directed_mask, directed_mask_expected)
    assert torch.equal(torch.tensor(data.num_nodes), num_nodes_expected)
    assert torch.equal(getattr(data, "coords"), coords_expected)

test_grid_with_node_attributes()


# test grid with node attrs and edge attrs
# test grid with empty kwargs -> richtige fehlermeldung?
# test grid with only edge attrs? -> richtige fehlermeldung? wobei wir da ja schon wissen wie viele nodes wir brauchen?? no, then we call add_n_empty_nodes()
# what else?


#test merge merge group based
#empty merge
#merge of two grids


#test merge attrs based
#empty merge
#merge of two grids

#test add_nodes
#test add_edges