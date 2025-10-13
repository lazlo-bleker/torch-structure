import torch
# Grid parameters
## Resolution
nu = 10
nv = 10
## Dimenisons
wall_height = 4.0
s = 3.0

## Default element loads and magnitudes
default_magnitude = -0.
default_load = torch.tensor([0.0, 0.0, -1.0])

## Reference points
nw_point = torch.tensor([0.0, 0.0, wall_height])
ne_point = torch.tensor([1.0, 0.0, wall_height])

mw_point = torch.tensor([0.0-(s - 1.)/2, 0.0, wall_height/2])
me_point = torch.tensor([1.0+(s  -1.)/2, 0.0, wall_height/2])

sw_point = torch.tensor([0.0-(s - 1.)/2, 0.0, 0.0])
se_point = torch.tensor([1.0+(s - 1.)/2, 0.0, 0.0])

## Pack dict for generator
uv_input_params = {
    "nu": nu, 
    "nv": nv, 
    "default_length" : wall_height / (nv-1),
    "default_magnitude" : default_magnitude,
    "default_load" : default_load
}

# Optimizaiton parameters
omega_orthogonal = 1e-1
omega_load_path  = 0e-3
omega_bottom     = 0e0
max_iters_opt = 100
n_shots = 1

# Runtime profiler
enable_profiler = False