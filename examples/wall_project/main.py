import torch
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import torch_structure as ts

from params import (
    uv_input_params,
    test_solve_flag,
    max_iters_cem,
    max_iters_opt,
    ftol, 
    verbose
)
from setup import setup_design_ariables, setup_obj_function_dict, setup_constraint_list, post_process_plot
from utils import apply_design_variables
from obj_func import obj_function, make_callback

# ------------------------------
# 1. Create initial bridge setup
# ------------------------------
# Generate bridge structure
uv_grid_generator = ts.generators.UVGridGenerator(**uv_input_params)
uv_grid_graph = uv_grid_generator()

# Solve before optimizing (can help debug)
if test_solve_flag:
    data = uv_grid_graph.mpcem(max_iter=max_iters_cem, verbose=verbose)
    data.plot(title="Initial Wall", legend=False)
    plt.show()

# --------------------------------
# 2. Define optimization variables
# --------------------------------
design_variables_access_dicts, design_variables_values = setup_design_ariables(uv_grid_graph)

# -------------------------------
# 3. Define optimization function
# -------------------------------
obj_function_dict = setup_obj_function_dict(uv_grid_graph)
constraint_list, display_obj_constraints = setup_constraint_list(uv_grid_graph, design_variables_access_dicts)

if test_solve_flag:
    test = obj_function(design_variables_values, uv_grid_graph, design_variables_access_dicts, obj_function_dict)

# Run optimization in stages to show intermediate results
for _ in range(1):
    # -------------------
    # 4. Run optimization
    # -------------------
    result = minimize(
        fun=obj_function,
        args=(uv_grid_graph, design_variables_access_dicts, obj_function_dict),
        x0=design_variables_values,
        method="SLSQP",
        jac=True,
        constraints=(constraint_list),
        callback=make_callback(),
        options={"disp": True, "ftol": ftol, "maxiter": max_iters_opt},
    )

    # ---------------------------------------
    # 5. Apply optimized forces and visualize
    # ---------------------------------------
    # Final run to compute the equilibrium structure with optimized forces
    data = uv_grid_graph.mpcem(max_iter=max_iters_cem, verbose=True)

    # Plot the optimized structure
    data_plot = data.plot(title="Optimized Wall", legend=False)
    post_process_plot(data_plot, display_obj_constraints)
    plt.savefig("./nozzle.png", dpi=400)
    plt.show()

    design_variables_values = result.x


print("f")