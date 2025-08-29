
import torch
import torch_structure as ts
import matplotlib.pyplot as plt
import torch_geometric as pyg
import matplotlib.pyplot as plt
from matplotlib.path import Path
import numpy as np

def create_Flower_Points():

    a = 4            
    b = 1.5       
    k = 4            
    n_points = 1000  

    theta = np.linspace(0, 2 * np.pi, n_points)
    r = a + b * np.sin(k * theta)

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    return Path(np.column_stack([x, y]))


x_res = .2
y_res = .2

path = create_Flower_Points()

input_params = {
    'path': path,
    'x_res': x_res,
    'y_res': y_res
}

pneu_dome = ts.generators.PneuStructure(**input_params)

hetero_data = pneu_dome.hetero_graph  

data = hetero_data.data

force_densities = torch.full((data.num_edges,), 10.0, dtype=torch.float32)

eps = 1e-5
max_its = 500
pressure = 5

data.force_density = force_densities

C = ts.formfinding.utils.create_branch_node_matrix(data.edge_index[:, data.directed_mask.view(-1)])


def iterative_fdm(data, hetero_data, eps, max_its, C):

    norm_steps = []

    for i in range(max_its):
        
        hetero_data.hetero_graph['node'].coords = data.coords

        load = pneu_dome.calculate_loads(pressure, hetero_data)

        data.load = load

        old_coords = data.coords
        
        data.fdm(inplace=True, C=C)

        data.pattern_coords = data.coords[:, :2]
        data.z_coord = data.coords[:, 2]

        step = old_coords - data.coords

        norm_step = torch.norm(step)
        norm_steps.append(norm_step)

        if norm_step < eps:
            break


    return data


data = iterative_fdm(data, hetero_data, eps, max_its, C)

#data.plot(title="Pneu Dome", legend=False, show_load=True, load=data.load, force_scale = .2)

data.plot(lw_scale = 0.2)

plt.show()





