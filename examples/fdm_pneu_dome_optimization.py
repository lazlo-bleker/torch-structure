

import torch
import torch_structure as ts
import matplotlib.pyplot as plt
from scipy.optimize import minimize


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
force_densities = torch.full((data.num_edges,), 40.0, dtype=torch.float64)

eps = 1e-5
max_its = 100

C = ts.formfinding.utils.create_branch_node_matrix(data.edge_index[:, data.directed_mask.view(-1)])

def iterative_fdm(data, eps, max_its, parallel_lines, meridians, C):

    norm_steps = []

    for i in range(max_its):

        load = pneu_dome.calculate_loads(data, parallel_lines, meridians, 11.5)

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
def same_lengths(force_densities, data, eps, max_its, parallel_lines, meridians):
   
    data.force_density = force_densities

    data = iterative_fdm(data, eps, max_its, parallel_lines, meridians, C)
    lengths = data.length_from_coords[data.is_meridian.view(-1)] 

    return torch.sum((lengths - torch.mean(lengths))**2)


result = minimize(
    fun=same_lengths,
    args=(data, eps, max_its, parallel_lines, meridians),
    x0=force_densities.detach().numpy(),
    method="SLSQP",
    jac=True,
    callback=make_callback(),
    options={
        'disp': True,
        'ftol': 1e-7,
        'maxiter': 200,
        'gtol' : 1e-10
    }
)

optimized_forces = torch.tensor(result.x, dtype=torch.float32)

data.force_density = optimized_forces

print(result)

data = iterative_fdm(data, eps, max_its, parallel_lines, meridians, C)

data.plot(title="Pneu Dome", legend=False, show_load=False, load=data.load, force_scale = 0.05, lw_scale = 0.09)



plt.show()



