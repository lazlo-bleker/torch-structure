import torch
from torch_structure.data.data import StructData
import matplotlib.pyplot as plt


node_attrs = {"coords": torch.empty((0, 3), dtype=torch.float)}
edge_attrs = {"force": torch.empty((0), dtype=torch.long)}
data = StructData(node_attrs=node_attrs, edge_attrs=edge_attrs)

def f(u):
    height = (u**2) / 10
    return torch.hstack([u, torch.zeros_like(u), height])

def g(u_coord):
    res = u_coord**2
    return res

node_attrs = {"coords": f}
edge_attrs = {"force": g}
data.add_chain(10, node_attrs=node_attrs, edge_attrs=edge_attrs)

data.plot()
plt.show()
