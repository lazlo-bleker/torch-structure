import torch

nu = 10
nv = 10
move = 0.5
wall_height = 4.0
default_magnitude = -0.0
omega_orthogonal = .1
nw_point = torch.tensor([0.0, 0.0, wall_height])
ne_point = torch.tensor([1.0, 0.0, wall_height])
sw_point = torch.tensor([0.0-move, 0.0, 0.0])
se_point = torch.tensor([1.0+move, 0.0, 0.0])
default_load = torch.tensor([0.0, 0.0, -1.0])

uv_input_params = {
    "nu": nu, 
    "nv": nv, 
    "default_length" : wall_height / (nv-1),
    "default_magnitude" : default_magnitude,
    "nw_point" : nw_point,
    "ne_point" : ne_point,
    "default_load" : default_load
}

verbose = False
test_solve_flag = False
max_iters_opt = 200
max_iters_cem = 100
ftol = 1e-8