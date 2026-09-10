"""
Example: Optimize Cable Forces in a Cable-Stayed Bridge using CEM and scipy

This example demonstrates how to use torch_structure to optimize the force distribution
in a cable-stayed bridge so that the bridge deck is as flat as possible.
"""

import torch
import torch_structure as ts
from torch_structure.plot import Plotter
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ------------------------------
# 1. Create initial bridge setup
# ------------------------------
bridge_params = {
    "n_towers": 5,  # Number of vertical towers
    "n_cables": 5,  # Number of cables on each side of a tower
    "deck_trail_length": 1.0,  # Spacing between deck nodes
    "tower_trail_length": 0.3,  # Spacing between tower nodes
    "center_deviation_force": torch.tensor(-10.0),  # Axial force halfway along the deck
    "cable_deviation_force": torch.tensor(1.0),  # Tension in cables
    "deck_load": torch.tensor([0.0, 0.0, -1.0]),  # Downward gravity load on the deck
    "tower_height": 5.0,  # Height of each tower
    "tower_offset": 1.0,  # Horizontal offset from the deck to the tower peaks
    "back_stay_offset": 5.0,  # Horizontal offset from the tower peaks to the back stay
    "back_stay_force": torch.tensor(10.0),  # Force in back stays
}

# Generate bridge structure
bridge_generator = ts.generators.CableStayedBridge(**bridge_params)
data = (
    bridge_generator()
)  # This is the main data structure of TorchStructure we'll work with

# --------------------------------
# 2. Define optimization variables
# --------------------------------
# Select which edge forces to optimize: deviation (i.e. non-trail) edges in the directed mask
force_mask = (~data.is_trail_edge & data.directed_mask).clone()

# Remove deviation edge in the center of the deck from optimization
edge_to_exclude = "deck_trail_0_node_0-deck_trail_1_node_0"
force_mask[data.metadata["edge_name_to_index"][edge_to_exclude]] = False

# Compute mask for forces in reciprocal edges
reciprocal_force_idx = data.reciprocal_edge[force_mask]
reciprocal_force_mask = torch.zeros(data.num_edges, dtype=torch.bool).unsqueeze(1)
reciprocal_force_mask[reciprocal_force_idx] = True

# Create a mask for all deck nodes (used to check flatness)
deck_node_indices = [
    i for name, i in data.metadata["node_name_to_index"].items() if "deck_trail" in name
]
deck_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
deck_mask[deck_node_indices] = True


# -------------------------------
# 3. Define optimization function
# -------------------------------
@ts.utils.scipy_jacobian  # Decorator to make torch function compatible with scipy
def deck_flatness(optim_forces, data, force_mask, reciprocal_force_mask, deck_mask):
    optim_forces = (
        optim_forces.float()
    )  # Cast to 32-bit float (ToDo: add easy 64-bit support)

    # Update the force vector with optimization variables
    full_force = data.force.clone()
    full_force[force_mask] = optim_forces
    full_force[reciprocal_force_mask] = optim_forces
    data.force = full_force

    # Form-find new structure with MPCEM
    data = data.mpcem(max_iter=1000, damping_factor=0.5)

    # Compute mean square Z deviation (measure of flatness)
    deck_z = data.coords[deck_mask][:, 2]
    return torch.mean(deck_z**2)


# Define callback for logging progress (optional)
def make_callback():
    def callback(x):
        print(
            f"Iteration {callback.iteration:3d} | Loss: {deck_flatness.best_loss:.6f}"
        )
        callback.iteration += 1

    callback.iteration = 0
    return callback


# -------------------
# 5. Run optimization
# -------------------
# Start from uniform force values
initial_force_values = torch.ones(torch.sum(force_mask), dtype=torch.float64)

# Run optimization using scipy
result = minimize(
    fun=deck_flatness,
    args=(data, force_mask, reciprocal_force_mask, deck_mask),
    x0=initial_force_values.detach().numpy(),
    method="SLSQP",
    jac=True,
    callback=make_callback(),
    options={"disp": True, "ftol": 1e-7, "gtol": 1e-7, "maxiter": 100},
)

# ---------------------------------------
# 6. Apply optimized forces and visualize
# ---------------------------------------
# Update forces with optimized values
optimized_forces = torch.tensor(result.x, dtype=torch.float32)
data.force[force_mask] = optimized_forces
data.force[reciprocal_force_mask] = optimized_forces

# Final run to compute the equilibrium structure with optimized forces
data = data.mpcem(max_iter=1000, verbose=True)

# Plot the optimized structure
Plotter().plot(data, title="Optimized Cable-Stayed Bridge", legend=False)
plt.show()
