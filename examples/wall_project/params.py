import torch
# Grid parameters
## Resolution
nu = 15
nv = 15
## Dimenisons
wall_height = 4.0
s = 4.0 

## Default element loads and magnitudes
default_magnitude = -0.0
default_load = torch.tensor([0.0, 0.0, -1.0])

## Reference points
nw_point = torch.tensor([0.0, 0.0, wall_height])
ne_point = torch.tensor([1.0, 0.0, wall_height])
sw_point = torch.tensor([0.0-(s - 1.)/2, 0.0, 0.0])
se_point = torch.tensor([1.0+(s  -1.)/2, 0.0, 0.0])

## Pack dict for generator
uv_input_params = {
    "nu": nu, 
    "nv": nv, 
    "default_length" : wall_height / (nv-1),
    "default_magnitude" : default_magnitude,
    "nw_point" : nw_point,
    "ne_point" : ne_point,
    "default_load" : default_load
}

## Optimizaiton parameters
omega_orthogonal = 1e0
max_iters_opt = 1000
n_shots = 50
