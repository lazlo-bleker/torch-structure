"""
Example: Form-find a Cable Net with the Force Density Method (FDM)
"""

import torch
import torch_structure as ts
import matplotlib.pyplot as plt
import torch_geometric as pyg
import matplotlib.pyplot as plt
from matplotlib.path import Path
import numpy as np
from scipy.optimize import minimize
from matplotlib.patches import PathPatch  # <-- Add this import

def create_Flower_Points():

    a = 4     
    b = 1.5   
    k = 6      
    n_points = 1000  

    theta = np.linspace(0, 2 * np.pi, n_points)
    r = a + b * np.sin(k * theta)

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    return Path(np.column_stack([x, y]))



def circle():

    n_points = 100  
    r = 5
    theta = np.linspace(0, 2 * np.pi -1, n_points)


    x = r * np.cos(theta)
    y = r * np.sin(theta)

    return Path(np.column_stack([x, y]))


def rectangle(n_points=100, width=5, height=5):
    perimeter = 2 * (width + height)
    
    n_bottom = int(n_points * width / perimeter)
    n_right = int(n_points * height / perimeter)
    n_top = int(n_points * width / perimeter)
    n_left = n_points - (n_bottom + n_right + n_top) 

    x_bottom = np.linspace(0, width, n_bottom, endpoint=False)
    y_bottom = np.zeros_like(x_bottom)

    y_right = np.linspace(0, height, n_right, endpoint=False)
    x_right = np.full_like(y_right, width)

    x_top = np.linspace(width, 0, n_top, endpoint=False)
    y_top = np.full_like(x_top, height)

    y_left = np.linspace(height, 0, n_left, endpoint=False)
    x_left = np.zeros_like(y_left)

    x = np.concatenate([x_bottom, x_right, x_top, x_left])
    y = np.concatenate([y_bottom, y_right, y_top, y_left])

    return Path(np.column_stack([x, y]))


x_res = .2
y_res = .2
max_its_opti = 100
eps = 1e-5
max_its_fdm = 500
pressure = 5

path = create_Flower_Points()

input_params = {
    'path': path,
    'x_res': x_res,
    'y_res': y_res
}

pneu = ts.generators.PneuStructure(**input_params)

hetero_data = pneu.hetero_graph  

data = pneu.graph

force_densities = torch.full((data.num_edges,), 10.0, dtype=torch.float64)

C = ts.formfinding.utils.create_branch_node_matrix(data.edge_index[:, data.directed_mask.view(-1)])

def iterative_fdm(data, hetero_data, eps, max_its, C):

    norm_steps = []

    for i in range(max_its):
        
        pneu.graph = data

        hetero_data['node'].coords = data.coords

        load = pneu.calculate_loads(pressure)

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


def make_callback():
    def callback(x):
        print(f"Iteration {callback.iteration:3d} | Val: {same_lengths.best_loss:.6f}")
        callback.iteration += 1
    callback.iteration = 0
    return callback


@ts.utils.scipy_jacobian  # Decorator to make torch function compatible with scipy
def same_lengths(force_densities, data, hetero_data, eps, max_its_fdm):
   
    data.force_density = force_densities

    data = iterative_fdm(data, hetero_data, eps, max_its_fdm, C)

    lengths = data.length_from_coords 

    return torch.sum((lengths - torch.mean(lengths))**2)


result = minimize(
    fun=same_lengths,
    args=(data, hetero_data, eps,  max_its_fdm),
    x0=force_densities.detach().numpy(),
    method="SLSQP",
    jac=True,
    callback=make_callback(),
    options={
        'disp': True,
        'ftol': 1e-3,
        'maxiter': max_its_opti
    }
)

optimized_forces = torch.tensor(result.x, dtype=torch.float32)

data.force_density = optimized_forces

data = iterative_fdm(data, hetero_data, eps, max_its_fdm, C)

data.plot(lw_scale = 0.1)

data.plot(title="Pneu Dome", legend=False, show_load=True, load=data.load, force_scale = 0.2, lw_scale = .07)

plt.show()


