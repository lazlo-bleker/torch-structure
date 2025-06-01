"""
Example: Form-find a Cable Net with the Force Density Method (FDM)
"""

import torch
import torch_structure as ts
import matplotlib.pyplot as plt

# --------------------------------
# 1. Create inital cable net setup
# --------------------------------
input_params = {
    'num_parallel_lines': 10,                     
    'num_meridians': 10,           
    'radius': 5          
}

# ----------------------------------
# 2. Generate the cable net geometry
# ----------------------------------
pneu_dome = ts.generators.PneuDome(**input_params)
data = pneu_dome.graph  

# ----------------------------------
# 3. Assign force densities to edges
# ----------------------------------
q = torch.full((data.num_edges,), 40.0)

# Assign higher stiffness to boundary edges
q[data.is_boundary_edge.view(-1)] = 250.0

# Set force density values on the graph
data.force_density = q.unsqueeze(1)

eps = 1e-5
max_its = 100

norm_steps = []

for i in range(max_its):

    ## apply loads
    load = pneu_dome.calculate_loads(data, 10, 10, 16)

    data.load = load


    old_coords = data.coords

    data = data.fdm()

    data.pattern_coords = data.coords[:, :2]
    data.z_coord = data.coords[:, 2]

    step = old_coords - data.coords
    norm_step = torch.norm(step)
    norm_steps.append(norm_step)

    print(i)
    print(norm_step.item())
    
    if norm_step < eps:
        break

    
data.plot(title="Pneu Dome", legend=False)
plt.show()

iterations = range(len(norm_steps))
plt.plot(iterations, norm_steps, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Norm Step')
plt.yscale('log')
plt.grid(True)
plt.show()

