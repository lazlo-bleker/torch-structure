"""
Example: Form-find a Cable Net with the Force Density Method (FDM)
"""

import torch
import torch_structure as ts
import matplotlib.pyplot as plt



parallel_lines = 5
meridians = 10
radius = 5
input_params = {
    'num_parallel_lines': parallel_lines,                     
    'num_meridians': meridians,           
    'radius': radius          
}

pneu_dome = ts.generators.PneuDome(**input_params)
data = pneu_dome.graph  

q = torch.full((data.num_edges,), 40.0)



k = 0
for j in range(meridians):

    q[k] = 40


    k += 2



#rest of edges
for i in range(parallel_lines):
    for j in range(meridians):
        
        if i  > 0:
            
            #q[k] = 10 - i
            #q[k] = 0
            k += 2
        
        
        #q[k] = 40 - 2 * i

        k += 2



data.force_density = q.unsqueeze(1)

eps = 1e-5
max_its = 100

norm_steps = []


load = pneu_dome.calculate_loads(data, parallel_lines, meridians, 15)
data.load = load

data.plot(title="Pneu Dome", legend=False, show_load=False, load=data.load, force_scale = 0.05, lw_scale = 0.09)
plt.show()
data.plot(title="Pneu Dome", legend=False, show_load=True, load=data.load, force_scale = 0.05, lw_scale = 0.09)
plt.show()

for i in range(max_its):



    load = pneu_dome.calculate_loads(data, parallel_lines, meridians, 15)

    data.load = load



    old_coords = data.coords

    data = data.fdm()

    #print(data.verify_equilibrium(verbose = True))
    data.pattern_coords = data.coords[:, :2]
    data.z_coord = data.coords[:, 2]

    step = old_coords - data.coords
    norm_step = torch.norm(step)
    norm_steps.append(norm_step)

    #print(i)
    #print(norm_step.item())

    #data.plot(title="Pneu Dome", legend=False, show_load=True, load=data.load, force_scale = 0.05, lw_scale = 0.09)
    #plt.show()
    

    if norm_step < eps:
        break

    
    
data.plot(title="Pneu Dome", legend=False, show_load=False, load=data.load, force_scale = 0.02, lw_scale = 0.09)
plt.show()

iterations = range(len(norm_steps))
plt.plot(iterations, norm_steps, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Norm Step')
plt.yscale('log')
plt.grid(True)
plt.show()

