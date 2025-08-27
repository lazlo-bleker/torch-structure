"""
Example: Form-find a Cable Net with the Force Density Method (FDM)
"""

import torch
import torch_structure as ts
import matplotlib.pyplot as plt
import torch_geometric as pyg
import matplotlib.pyplot as plt


input_params = {
    'num_parallel_lines': 10,                     
    'num_meridians': 10,           
    'radius': 5          
}


pneu_dome = ts.generators.PneuDomeHetero(**input_params)

hetero_data = pneu_dome.hetero_graph  

data = pneu_dome.graph

force_densities = torch.full((data.num_edges,), 40.0, dtype=torch.float32)

eps = 1e-5
max_its = 100
pressure = 16

data.force_density = force_densities


def iterative_fdm(data, hetero_data, eps, max_its):

    norm_steps = []

    for i in range(max_its):
        
        hetero_data['node'].coords = data.coords

        load = pneu_dome.calculate_loads(pressure)

        data.load = load

        old_coords = data.coords

        data = data.fdm()

        data.pattern_coords = data.coords[:, :2]
        data.z_coord = data.coords[:, 2]

        print(len(load))
        print(len(old_coords))
        
        step = old_coords - data.coords

        norm_step = torch.norm(step)
        norm_steps.append(norm_step)

        print(i)
        print(norm_step.item())

        if norm_step < eps:
            break


    return data


data = iterative_fdm(data, hetero_data, eps, max_its)


#data.plot(title="Pneu Dome", legend=False, show_load=True, load=data.load, force_scale = 0.2)


data.plot()

plt.show()





