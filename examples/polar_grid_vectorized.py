import torch
from torch_structure.data.data import StructData
from torch_structure.plot import Plotter
import matplotlib.pyplot as plt
import math

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

def f(r_ind, a_ind, is_boundary):

    height = torch.ones_like(r_ind)
    angle = a_ind/5 * 2 * math.pi
    coords = torch.hstack([r_ind * torch.cos(angle), r_ind * torch.sin(angle), height])
    
    return coords

def g(r_ind, is_boundary):

    res = torch.ones_like(r_ind)

    res[is_boundary] *= 10

    return res


node_attrs = {

    "coords": f
}

edge_attrs = {

    "force": g

}


data.add_polar_grid(5, 3, node_attrs=node_attrs, edge_attrs=edge_attrs)

Plotter().plot(data, load=True)

plt.show()