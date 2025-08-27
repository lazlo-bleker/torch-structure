"""
Example: Form-find a Cable Net with the Force Density Method (FDM)
"""

import torch
import torch_structure as ts
import matplotlib.pyplot as plt
import torch_geometric as pyg
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np

radial_res = 10
longitudinal_res = 10
verticalDamping = 1
def createStraightPolyline(segments: int, length: int):

    seg_length = length / segments
    polyline = []

    for i in range(segments):
        #polyline.append([0,i * seg_length, .2 * min(i+1, segments - i)])
        polyline.append([0, i * seg_length,0])


    return polyline


polyline = createStraightPolyline(longitudinal_res, 5)

input_params = {
    'polyline': polyline,                     
    'startNormal': [0,1,0],  
    'endNormal': [0,-1,0],  
    'startBinormal': [-1,0,0],
    'endBinormal': [1,0,0],             
    'radius': .25,
    'radialRes': radial_res,
    'verticalDamping': verticalDamping     
}

pneu_tube = ts.generators.PneuTube(**input_params)

hetero_data = pneu_tube.hetero_graph  

data = pneu_tube.graph



#maybe adapt high low also depending on if im in the middle of the thing or not?!
def radialforcedensitydistribution(i, j, low=0, high=1):
    x = np.linspace(0, (j-1)/j * 2 * np.pi, j)  
    #wave = (np.sin(x - np.pi/2)+1) / 2
    wave = (np.cos(x)+1) / 2

    #print(wave)    
    wave = wave * (high - low) + low  
    return np.repeat(np.tile(wave, i-1),2)

def radialforcedensitydistribution2(i, j, low=0, high=1):
    x = np.linspace(0, (j-1)/j * 2 * np.pi, j)  
    #wave = (np.sin(x - np.pi/2)+1) / 2
    wave = (np.cos(x)+1) / 2

    #print(wave)    
    wave = wave * (high - low) + low  
    return np.repeat(np.tile(wave, i),2)


force_densities = torch.full((data.num_edges,), 5.0, dtype=torch.float32)
force_densities[data.is_radial.view(-1)] = 20.0
#force_densities[data.is_radial.view(-1)] = torch.tensor(radialforcedensitydistribution(longitudinal_res-1, radial_res, 30, 100), dtype=torch.float)
force_densities[data.is_longitudinal.view(-1)] = torch.tensor(radialforcedensitydistribution2(longitudinal_res-1, radial_res, 30, 70), dtype=torch.float)

eps = 1e-5
max_its = 200
#pressure = 360
pressure = 30

ftol = 1e-3

data.force_density = force_densities
norm_steps = []

C = ts.formfinding.utils.create_branch_node_matrix(data.edge_index[:, data.directed_mask.view(-1)])

def iterative_fdm(data, hetero_data, eps, max_its):

    for i in range(max_its):
        
        hetero_data['node'].coords = data.coords

        load = pneu_tube.calculate_loads(pressure)

        data.load = load

        #data.load[data.is_top_node.view(-1)] += torch.tensor([0, 0, -5])

        old_coords = data.coords

        data.fdm(inplace=True, C=C)
        
        data.pattern_coords = data.coords[:, :2]
        data.z_coord = data.coords[:, 2]
        
        step = old_coords - data.coords

        norm_step = torch.norm(step)
        norm_steps.append(norm_step)

        if norm_step < eps:
            break

        print(i)
    return data



data = iterative_fdm(data, hetero_data, eps, max_its)
#print(data.verify_equilibrium())

data.plot(title="Pneu Tube", legend=False, show_load=False, load=data.load, force_scale = .1, lw_scale = .1)

plt.show()

iterations = range(len(norm_steps))
plt.plot(iterations, norm_steps, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Norm Step')
plt.yscale('log')
plt.grid(True)
plt.show()





