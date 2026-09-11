import torch
from torch_structure.data.data import StructData
from torch_structure.plot import Plotter
N_SECTORS = 8
N_RINGS = 5

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

def f(x_unit_coord, y_unit_coord, z_unit_coord):
    return torch.hstack([x_unit_coord, y_unit_coord, z_unit_coord])

def g(z_unit_coord):
    return 1 + z_unit_coord * 15

node_attrs = {
    "coords": f
}

edge_attrs = {
    "force": g
}

data.add_cylinder(N_SECTORS, N_RINGS, node_attrs=node_attrs, edge_attrs=edge_attrs)

Plotter().plot(data, show=True)
