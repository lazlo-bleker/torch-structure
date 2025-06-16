"""
Example: Form-find a Cable Net with the Force Density Method (FDM)
"""

import torch
import torch_structure as ts
import matplotlib.pyplot as plt
from scipy.optimize import minimize


parallel_lines = 10
meridians = 10
radius = 5
input_params = {
    'num_parallel_lines': parallel_lines,                     
    'num_meridians': meridians,           
    'radius': radius          
}

pneu_dome = ts.generators.PneuDome(**input_params)
data = pneu_dome.graph  

force_densities = torch.full((data.num_edges,), 40.0, dtype=torch.float64)

eps = 1e-5
max_its = 100

C = ts.formfinding.utils.create_branch_node_matrix(data.edge_index[:, data.directed_mask.view(-1)])
def iterative_fdm(data, eps, max_its, parallel_lines, meridians, C):

    norm_steps = []

    for i in range(max_its):

        load = pneu_dome.calculate_loads(data, parallel_lines, meridians, 16)

        data.load = load

        old_coords = data.coords

        data.fdm(inplace=True, C=C)

        #print(data.verify_equilibrium(verbose = True))
        data.pattern_coords = data.coords[:, :2]
        data.z_coord = data.coords[:, 2]

        step = old_coords - data.coords
        norm_step = torch.norm(step)
        norm_steps.append(norm_step)

        if norm_step < eps:
            break
    
    return data


#data = iterative_fdm(data, eps, max_its, parallel_lines, meridians)



def getLengths(data, parallel_lines, meridians):

    lengths = []
    for i in range(parallel_lines-1):
        for j in range(meridians):        
            coords_center = data.nodes[str(i) + str(j)]["pattern_coords"]
            z_cord_center = data.nodes[str(i) + str(j)]["z_coord"]
            pt1 = torch.tensor([coords_center[0], coords_center[1],  z_cord_center], dtype=torch.float)

            coords_mid_up = data.nodes[str(i+1) + str(j)]["pattern_coords"]
            z_cord_mid_up = data.nodes[str(i+1) + str(j)]["z_coord"]
            pt2 = torch.tensor([coords_mid_up[0], coords_mid_up[1],  z_cord_mid_up], dtype=torch.float)


            lengths.append(torch.norm(pt1 - pt2))
    
    return lengths

@ts.utils.scipy_jacobian  # Decorator to make torch function compatible with scipy
def same_lengths(force_densities, data, eps, max_its, parallel_lines, meridians):
   
    data.force_density = force_densities

    data = iterative_fdm(data, eps, max_its, parallel_lines, meridians, C)
    lengths = data.length_from_coords[~data.is_meridian.view(-1)] # the is_meridian mask is filled with False, I took the complement for now but you should doubgle check how you create it
    return torch.mean((lengths - torch.mean(lengths))**2)

result = minimize(
    fun=same_lengths,
    args=(data, eps, max_its, parallel_lines, meridians),
    x0=force_densities.detach().numpy(),
    method="SLSQP",
    jac=True,
    options={
        'disp': True,
        'ftol': 1e-7,
        'maxiter': 100
    }
)


data = result

print(result)

# The result contains the optimized force densities,
# we still need to assign the to the data object and form-find one moer time before plotting

# data.plot()
# plt.show()



