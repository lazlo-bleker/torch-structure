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
        orthogonal_intersections_normed, 
        evaluate_edge_pairs,
        bottom_coords_obj_func,
        evaluate_bottom_coords_target, 
        bottom_coords_constr_func,
        evaluate_center_coords_target,
        center_coords_constr_func,
        load_path,
        laplacian,
        cache_laplacian,
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

# Add objective functions
edge_pairs = evaluate_edge_pairs(uv_grid_graph)
# optimizer.add_obj_function("Loss/Orthogonal", omega_orthogonal, orthogonal_intersections_normed, kwargs={"edge_pairs":edge_pairs})
laplacian_kwargs = cache_laplacian(uv_grid_graph)
optimizer.add_obj_function("Loss/Orthogonal", omega_orthogonal, laplacian, kwargs=laplacian_kwargs)
# optimizer.add_obj_function("Loss/LoadPath", omega_load_path, load_path)

# Add bottom coords constraint
bottom_coords_target = evaluate_bottom_coords_target()
# Y component of bottom coords as objective function to prevent from incompatible constraints
optimizer.add_obj_function("Loss/Bottom", omega_bottom, bottom_coords_obj_func, kwargs={"target_coords":bottom_coords_target})
# X,Z components of bottom coords as constraints
n_nodes = bottom_coords_target.shape[0]
eps = 1e-5
optimizer.add_constr_function(
    name = "Constr/BottomCoords", 
    fun = bottom_coords_constr_func, 
    lb = -eps * np.ones(2 * n_nodes), 
    ub =  eps * np.ones(2 * n_nodes), 
    kwargs = {"target_coords" : bottom_coords_target}
    )
# Add scatter plot of constraints
optimizer.logger.additional_scatter_plots.append({
    "xs" : bottom_coords_target[:,0].tolist(),
    "ys" : bottom_coords_target[:,1].tolist(),
    "zs" : bottom_coords_target[:,2].tolist(),
    "c" : "#007F00",
    "marker" : 'o'
})

# # Add center coords constraint
# center_coords_target = evaluate_center_coords_target()
# n_nodes = center_coords_target.shape[0]
# eps = 1e-5
# optimizer.add_constr_function(
#     name = "Constr/BottomCoords", 
#     fun = center_coords_constr_func, 
#     lb = -eps * np.ones(3*n_nodes), 
#     ub = eps * np.ones(3*n_nodes), 
#     kwargs = {"target_coords" : center_coords_target}
#     )
# # Add scatter plot of constraints
# optimizer.logger.additional_scatter_plots.append({
#     "xs" : center_coords_target[:,0].tolist(),
#     "ys" : center_coords_target[:,1].tolist(),
#     "zs" : center_coords_target[:,2].tolist(),
#     "c" : "#007F00",
#     "marker" : 2
# })

# Add scatter plot of origin nodes
optimizer.logger.additional_scatter_plots.append({
    "xs" : top_coords[:,0].tolist(),
    "ys" : top_coords[:,1].tolist(),
    "zs" : top_coords[:,2].tolist(),
    "c" : "#007F00",
    "marker" : 'o'
})

# Run optimization
optimizer.run(max_iters_opt, n_shots)

if enable_profiler:
    profiler.disable()
    profiler.dump_stats("profile_output.prof")

print("f")
