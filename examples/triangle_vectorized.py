import torch
from torch_structure.data.data import StructData
from torch_structure.plot import Plotter

node_attrs = {
    "coords": torch.empty((0, 3), dtype=torch.float)
}

edge_attrs = {
    "force": torch.empty((0), dtype=torch.long)
}


data = StructData(
    node_attrs=node_attrs,
    edge_attrs=edge_attrs
)

def f(u_ind, v_ind):

    height = torch.ones_like(u_ind)

    return torch.hstack([u_ind,v_ind,height])


def g(x_unit_coord):

    res = x_unit_coord

    return res


node_attrs = {
    "coords": f
}


edge_attrs = {
    "force": g
}

data.add_triangular_grid(10, node_attrs=node_attrs, edge_attrs=edge_attrs)

Plotter().plot(data, show=True)