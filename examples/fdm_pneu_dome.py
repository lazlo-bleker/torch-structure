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


## apply loads
load = pneu_dome.calculate_loads(data,10,10,10)

data.load = load


q = torch.full((data.num_edges,), 40.0)

data.force_density = q.unsqueeze(1)


data = data.fdm()


data.plot(title="Pneu Dome", legend=False)
plt.show()