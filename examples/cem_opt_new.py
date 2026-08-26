"""
Example: Optimize Cable Forces in a Cable-Stayed Bridge using CEM and scipy

This example demonstrates how to use torch_structure to optimize the force distribution
in a cable-stayed bridge so that the bridge deck is as flat as possible.
"""

import torch
import torch_structure as ts
import matplotlib.pyplot as plt

from torch_structure.optimizer import (
    VariableConfig,
    ObjectiveConfig,
    LoggerConfig,
    SolverConfig,
    Optimizer,
)

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
struc_data = bridge_generator()

# --------------------------------
# 2. Define optimization variables
# --------------------------------
# Select which edge forces to optimize: deviation (i.e. non-trail) edges in the directed mask
force_mask = (~struc_data.is_trail_edge & struc_data.directed_mask).clone()
# Remove deviation edge in the center of the deck from optimization
edge_to_exclude = "deck_trail_0_node_0-deck_trail_1_node_0"
force_mask[struc_data.metadata["edge_name_to_index"][edge_to_exclude]] = False

# Setup arg to pass to optimizer
dv_config_list = [
    VariableConfig(
        "Deviation Forces",
        "force",
        force_mask,
    )
]

# -------------------------------
# 3. Define optimization function
# -------------------------------
# Create a mask for all deck nodes (used to check flatness)
deck_node_indices = [
    i
    for name, i in struc_data.metadata["node_name_to_index"].items()
    if "deck_trail" in name
]
deck_mask = torch.zeros(struc_data.num_nodes, dtype=torch.bool)
deck_mask[deck_node_indices] = True


def deck_flatness(data, deck_mask):
    # Compute mean square Z deviation (measure of flatness)
    deck_z = data.coords[deck_mask][:, 2]
    return torch.mean(deck_z**2)


# Setup arg to pass to optimizer
objectives_config_list = [
    ObjectiveConfig("Deck Flatness", deck_flatness, kwargs={"deck_mask": deck_mask})
]

# -------------------------------
# 4. Define optimization constraints
# -------------------------------
# Skip

# -------------------
# 5. Run optimization
# -------------------
data_prev = struc_data.mpcem(inplace=False)
logger_config = LoggerConfig()
solver_config = SolverConfig(solver="mpcem")

optimizer = Optimizer(
    struc_data, solver_config, dv_config_list, objectives_config_list, logger_config
)

res = optimizer.run()

# -------------------
# 6. Visualize results
# -------------------
data_post = struc_data.mpcem(inplace=False)
data_prev.plot(title="Initial Cable-Stayed Bridge", legend=False)
data_post.plot(title="Optimized Cable-Stayed Bridge", legend=False)
plt.show()
