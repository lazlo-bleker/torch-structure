import cProfile
import numpy as np
import torch_structure as ts
from params import (
    uv_input_params,
    omega_orthogonal,
    omega_load_path,
    omega_bottom,
    max_iters_opt,
    n_shots,
    enable_profiler,
)
from function_library import (
        evaluate_top_coords_target,
        evaluate_bottom_coords_target, 
        evaluate_center_coords_target,
        bottom_coords_obj_func,
        bottom_coords_constr_func,
        center_coords_constr_func,
        support_reaction_force_constr_func,
        load_path,
        cache_rectangular,
        rectangular,
    )

from optimizer import Optimizer

if enable_profiler:
    profiler = cProfile.Profile()
    profiler.enable()

# ------------------------------
# Create graph
# ------------------------------
# Generate bridge structure
top_coords = evaluate_top_coords_target()
uv_grid_generator = ts.generators.UVGridGenerator(origin_nodes = top_coords, **uv_input_params)
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

# Load Path objective function
optimizer.add_obj_function("Loss/LoadPath", omega_load_path, load_path)

# Bottom Coords condtraint
bottom_coords_target = evaluate_bottom_coords_target()
n_nodes = bottom_coords_target.shape[0]
eps = 1e-5
# X,Z components of bottom coords as constraints
optimizer.add_constr_function(
    name = "Constr/BottomCoords", 
    fun = bottom_coords_constr_func, 
    lb = -eps * np.ones(2 * n_nodes), 
    ub =  eps * np.ones(2 * n_nodes), 
    kwargs = {"target_coords" : bottom_coords_target}
    )
## Y component of bottom coords as objective function to prevent from incompatible constraints
# optimizer.add_obj_function("Loss/Bottom", omega_bottom, bottom_coords_obj_func, kwargs={"target_coords":bottom_coords_target})

# Add reaction force constraint
eps = 1e-5
optimizer.add_constr_function(
    name = "Constr/Reaction", 
    fun = support_reaction_force_constr_func, 
    lb = -eps * np.ones(2 * n_nodes), 
    ub =  eps * np.ones(2 * n_nodes), 
    kwargs = {}
    )


# Add orthogonal intersections objective function
laplacian_kwargs = cache_rectangular(uv_grid_graph)
optimizer.add_obj_function("Loss/Orthogonal", omega_orthogonal, rectangular, kwargs=laplacian_kwargs)

# Run optimization
optimizer.run(max_iters_opt, n_shots)

if enable_profiler:
    profiler.disable()
    profiler.dump_stats("profile_output.prof")

print("f")