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

def create_four_circle_outline(radius=5.0, n_points_per_arc=250):
    """
    Creates a Path representing the outline of 4 intersecting circles placed
    at the cardinal points (top, bottom, left, right).
    
    Each circle contributes one arc to the final shape.
    """
    # Circle centers
    centers = [
        ( radius, 0),   # Right
        (-radius, 0),   # Left
        (0,  radius),   # Top
        (0, -radius)    # Bottom
    ]

    outline_points = []

    # Create arcs for each of the four circles
    angles = [
        (np.pi/4, 3*np.pi/4),        # Right circle: top left arc
        (5*np.pi/4, 7*np.pi/4),      # Left circle: bottom right arc
        (3*np.pi/4, 5*np.pi/4),      # Top circle: left to right arc
        (-np.pi/4,  np.pi/4)         # Bottom circle: right to left arc
    ]

    for center, (theta_start, theta_end) in zip(centers, angles):
        theta = np.linspace(theta_start, theta_end, n_points_per_arc)
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        outline_points.append(np.column_stack([x, y]))

    # Concatenate all arc points
    outline = np.concatenate(outline_points, axis=0)

    return Path(outline)

x_res = .6
y_res = .6
eps_boundary = .6

path = create_four_circle_outline()

input_params = {
    'path': path,
    'x_res': x_res,
    'y_res': y_res,
    'eps': eps_boundary    
}

pneu_dome = ts.generators.PneuStructure(**input_params)

hetero_data = pneu_dome.hetero_graph  

data = pneu_dome.graph

force_densities = torch.full((data.num_edges,), 10.0, dtype=torch.float32)

eps = 1e-5
max_its = 1
pressure = 15

data.force_density = force_densities

C = ts.formfinding.utils.create_branch_node_matrix(data.edge_index[:, data.directed_mask.view(-1)])


def iterative_fdm(data, hetero_data, eps, max_its, C):

    norm_steps = []

    for i in range(max_its):
        
        hetero_data['node'].coords = data.coords

        load = pneu_dome.calculate_loads(pressure)

        data.load = load

        old_coords = data.coords
        
        data.fdm(inplace=True, C=C)

        data.pattern_coords = data.coords[:, :2]
        data.z_coord = data.coords[:, 2]

        step = old_coords - data.coords

        norm_step = torch.norm(step)
        norm_steps.append(norm_step)

        print(i)
        print(norm_step.item())

        if norm_step < eps:
            break


    return data


data = iterative_fdm(data, hetero_data, eps, max_its, C)

#data.plot(title="Pneu Dome", legend=False, show_load=True, load=data.load, force_scale = .2)

data.plot(lw_scale = 0.2)

plt.show()





