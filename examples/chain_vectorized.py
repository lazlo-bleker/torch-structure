import torch
from torch_structure.data.data import StructData
import matplotlib.pyplot as plt


node_attrs = {"coords": torch.empty((0, 3), dtype=torch.float)}
edge_attrs = {"forcea": torch.empty((0, 1), dtype=torch.long)}
data = StructData(node_attrs=node_attrs, edge_attrs=edge_attrs)

num_nodes = 10

force = 10 * torch.ones(num_nodes - 1, 1, dtype=torch.long)

def f(x_unit_coord):
    height = 3 * x_unit_coord * (1 - x_unit_coord)
    return torch.hstack([x_unit_coord, torch.zeros_like(x_unit_coord), height])

node_attrs = {"coords": f}
edge_attrs = {"forcea": force}

data.add_chain(num_nodes, node_attrs=node_attrs, edge_attrs=edge_attrs)

data.plot()
plt.show()
