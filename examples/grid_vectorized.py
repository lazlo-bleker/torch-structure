import torch
from torch_structure.data.data import StructData
import matplotlib.pyplot as plt


node_attrs = {
    "coords": torch.empty((0, 3), dtype=torch.float),
    "load": torch.empty((0, 3), dtype=torch.float),
}
edge_attrs = {"force": torch.empty((0), dtype=torch.long)}
default_attrs = {"load": 10 * torch.ones((1, 3), dtype=torch.float)}
data = StructData(
    node_attrs=node_attrs, edge_attrs=edge_attrs, default_attrs=default_attrs
)

def f(u_ind, v_ind):

    height = (u_ind**2+v_ind**2)/10

    return torch.hstack([u_ind,v_ind,height])


def f_2(u_ind, v_ind):

    return torch.hstack([u_ind, v_ind, torch.zeros_like(u_ind)])


def g_2(u_ind, is_boundary):

    res = torch.ones_like(u_ind)
    res[is_boundary] = -10

    return res


def g(x_unit_coord, y_unit_coord):

    res = x_unit_coord**2 + y_unit_coord**2

    return res

def g_2(u_ind, is_boundary):
    res = torch.ones_like(u_ind)
    res[is_boundary] = -10
    return res

node_attrs = {"coords": f}

edge_attrs = {"force": g}

data.add_grid(7, 7, node_attrs=node_attrs, edge_attrs=edge_attrs)

data.plot(load=True)

plt.show()
