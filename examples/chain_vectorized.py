import torch
from torch_structure.data.data import StructData
import matplotlib.pyplot as plt


node_attrs = {"coords": torch.empty((0, 3), dtype=torch.float)}
edge_attrs = {"force": torch.empty((0), dtype=torch.long)}
data = StructData(node_attrs=node_attrs, edge_attrs=edge_attrs)

def f(u_ind):

    height = (u_ind**2)/10

    return torch.hstack([u_ind,torch.zeros_like(u_ind),height])


def g(x_unit_coord):

    res = x_unit_coord**2

    return res

node_attrs = {"coords": f}
edge_attrs = {"force": g}
data.add_chain(10, node_attrs=node_attrs, edge_attrs=edge_attrs)

data.plot()
plt.show()
