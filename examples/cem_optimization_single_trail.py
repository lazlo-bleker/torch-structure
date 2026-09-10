"""
Example: Optimize a single trail sequencce using CEM and scipy

This is a minimal example of optimization with torch_structure
"""

import torch
import torch_structure as ts
from torch_structure.data import StructData
from torch_structure.plot import Plotter
from scipy.optimize import minimize, Bounds
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# 1. Create initial setup
# ------------------------------
trail_params = {
    "n_nodes": 10,
    "trail_element_length": 0.1,
    "nodal_load": [0.0, 0.0, -1.0],
    "origin_node_load": [4.0, 0.0, 0.0],
}
coords_target = torch.tensor([1.0, 0.0, -1.0])

# Generate bridge structure
trail_generator = ts.generators.SingleTrailGenerator(**trail_params)
# This is the main data structure of TorchStructure we'll work with
trail = trail_generator()

# --------------------------------
# 2. Define optimization variables
# --------------------------------
# Select which edge forces to optimize: deviation (i.e. non-trail) edges in the directed mask
trail_element_mask = (trail.is_trail_edge & trail.directed_mask).clone()
trail_lengths = trail.length[trail_element_mask]

# Compute mask for forces in reciprocal edges
reciprocal_idx = trail.reciprocal_edge[trail_element_mask]
trail_element_mask_reciprocal = torch.zeros(
    trail.num_edges, dtype=torch.bool
).unsqueeze(1)
trail_element_mask_reciprocal[reciprocal_idx] = True

# Compute mask for support node
target_mask = trail.is_support


# -------------------------------
# 3. Define optimization function
# -------------------------------
@ts.utils.scipy_jacobian  # Decorator to make torch function compatible with scipy
def obj_func(trail_lengths, struc_data: StructData):
    trail_lengths = (
        trail_lengths.float()
    )  # Cast to 32-bit float (ToDo: add easy 64-bit support)

    # Update the force vector with optimization variables
    full_trail_lengths = struc_data.length.clone()
    full_trail_lengths[trail_element_mask] = trail_lengths
    full_trail_lengths[trail_element_mask_reciprocal] = trail_lengths
    struc_data.length = full_trail_lengths

    # Form-find new structure with MPCEM
    struc_data = struc_data.mpcem(max_iter=1000, damping_factor=0.5)

    # Compute mean square Z deviation (measure of flatness)
    coords_computed = struc_data.coords[target_mask.expand(-1, 3)]
    coords_deviation = coords_target - coords_computed
    return torch.sum(coords_deviation**2)


# Define callback for logging progress (optional)
def make_callback():
    def callback(x):
        print(f"Iteration {callback.iteration:3d} | Loss: {obj_func.best_loss:.6f}")
        callback.iteration += 1

    callback.iteration = 0
    return callback


# -------------------
# 5. Run optimization
# -------------------
# Start from uniform force values
initial_values = trail_params["trail_element_length"] * torch.ones(
    torch.sum(trail_element_mask), dtype=torch.float64
)

# Define constraint of lengths being positive
eps = 1e-1
bounds = Bounds(eps, np.inf)

# Run optimization using scipy
result = minimize(
    fun=obj_func,
    args=(trail),
    x0=initial_values.detach().numpy(),
    method="SLSQP",
    jac=True,
    callback=make_callback(),
    bounds=bounds,
    options={"disp": True, "ftol": 1e-7, "gtol": 1e-7, "maxiter": 100},
)

# ---------------------------------------
# 6. Apply optimized forces and visualize
# ---------------------------------------
# Update forces with optimized values
optimized_lengths = torch.tensor(result.x, dtype=torch.float32)
trail.force[trail_element_mask] = optimized_lengths
trail.force[trail_element_mask_reciprocal] = optimized_lengths


# Final run to compute the equilibrium structure with optimized forces
trail = trail.mpcem(max_iter=1000, verbose=True)

# Plot the optimized structure
Plotter().plot(trail, title="Optimized Trail", legend=False)
plt.show()
print("Finish")
