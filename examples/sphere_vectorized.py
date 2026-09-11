import torch
from torch_structure.data.data import StructData
from torch_structure.plot import Plotter
N_SECTORS = 10
N_RINGS = 7
NUM_EDGES = (2 * N_RINGS + 1) * N_SECTORS

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


node_attrs = {
    "coords": f
}

edge_attrs = {
    "force": torch.ones(NUM_EDGES, dtype=torch.float)
}

data.add_sphere(N_SECTORS, N_RINGS, node_attrs=node_attrs, edge_attrs=edge_attrs)

Plotter().plot(data, show=True)
