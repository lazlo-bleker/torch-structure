import torch
from torch_structure.data.data import StructData
import matplotlib.pyplot as plt


node_attrs = {
    "coords": torch.empty((0, 3), dtype=torch.float),
    "load" : torch.empty((0,3), dtype=torch.float)
}

edge_attrs = {
    "force": torch.empty((0), dtype=torch.long)
}

default_attrs = {

    "load" : 10*torch.ones((1,3), dtype=torch.float)

}

data = StructData(
    node_attrs=node_attrs,
    edge_attrs=edge_attrs,
    default_attrs=default_attrs
)

def f(u,v):

    height = (u**2+v**2)/10

    return torch.hstack([u,v,height])

def f_2(u,v):

    return torch.hstack([u, v, torch.zeros_like(u)])
    

def g_2(u_ind, is_boundary):

    res = torch.ones_like(u_ind)
    res[is_boundary] = -10

    return res


def g(u_coord, v_coord):

    res = u_coord**2 + v_coord**2
    
    return res


node_attrs = {
    "coords": f
}


edge_attrs = {
    "force": g
}

data.add_grid(7,7, node_attrs=node_attrs, edge_attrs=edge_attrs)

data.plot(load=True)

plt.show()

