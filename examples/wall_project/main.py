import numpy as np

import torch_structure as ts

from params import (
    uv_input_params,
    omega_orthogonal,
    max_iters_opt,
    n_shots,
)
from function_library import (
        orthogonal_intersections_normed, 
        evaluate_bottom_coords_target, 
        bottom_coords_constr_func
    )

from optimizer import Optimizer

# ------------------------------
# Create graph
# ------------------------------
# Generate bridge structure
uv_grid_generator = ts.generators.UVGridGenerator(**uv_input_params)
uv_grid_graph = uv_grid_generator()

# ------------------------------
# Initialize optimizer
# ------------------------------
# Put together design variable dicts
design_variables_dicts = {
    "origin_node_coords" : [
        "coords",    # Attr name
        "origin_nodes", # Mask keyword
        False, # Is dual edge
        False, # In log scale
    ],
    "deviation_element_forces" : [
        "force",    # Attr name
        "deviation_elements", # Mask keyword
        True, # Is dual edge
        False, # In log scale
    ],
    "trail_element_lengths" : [
        "length",    # Attr name
        "trail_elements", # Mask keyword
        True, # Is dual edge
        True, # In log scale
    ],
}

# Initialize optimizer
optimizer = Optimizer(
    graph = uv_grid_graph,
    dv_dicts = design_variables_dicts
    )

# Add objective functions
optimizer.add_obj_function("Loss/Orthogonal", omega_orthogonal, orthogonal_intersections_normed)

# Add constraints
bottom_coords_target = evaluate_bottom_coords_target()
kwargs = {
    "target_coords" : bottom_coords_target
}
n_nodes = bottom_coords_target.shape[0]
eps = 1e-5
lb = -eps * np.ones(n_nodes * 3)
ub = eps * np.ones(n_nodes * 3)
optimizer.add_constr_function("Constr/BottomCoords", bottom_coords_constr_func, lb, ub, kwargs)

# Run optimization
bottom_coords_scatter = {
    "xs" : bottom_coords_target[:,0].tolist(),
    "ys" : bottom_coords_target[:,1].tolist(),
    "zs" : bottom_coords_target[:,2].tolist(),
    "c" : "#007F00",
}
optimizer.logger.additional_scatter_plots.append(bottom_coords_scatter)
optimizer.run(max_iters_opt, n_shots)

print("f")