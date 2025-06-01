"""
Example: Form-find a Cable Net with the Force Density Method (FDM)
"""

import torch
import torch_structure as ts
import matplotlib.pyplot as plt

# --------------------------------
# 1. Create inital cable net setup
# --------------------------------
input_params = {
    'n': 5,                          # Number of corner supports
    'boundary_density': 5,           # Density of grid
    'diagonals': False,              # Add diagonal edges
    'centroid_support': False,       # Support in the center
    'pattern': 'singularity',           # Grid layout pattern
    'unsupported_boundaries': False,  # Only corner supports
    'curve_boundaries': True,        # Curve the boundary edges
    'rectangle': False,              # Set corner supports in a rectangle
    'square': False,                 # Set corner supports in a square
    'circle': True,                 # Set corner supports in a circle
    'support_height': 0.6            # Elevation of supports
}

# ----------------------------------
# 2. Generate the cable net geometry
# ----------------------------------
cable_net = ts.generators.CableNet(**input_params)
data = cable_net.graph  # This is the main data structure of TorchStructure we'll work with

# ----------------------------------
# 3. Assign force densities to edges
# ----------------------------------
q = torch.full((data.num_edges,), 40.0)

# Assign higher stiffness to boundary edges
q[data.is_boundary_edge.view(-1)] = 250.0

# Assign stiffness to diagonal edges (if present)
#q[data.is_diagonal_edge.view(-1)] = 60.0

# Set force density values on the graph
data.force_density = q.unsqueeze(1)

# ------------------
# 4. Solve using FDM
# ------------------
#data = data.fdm()

# -------------------------------
# 5. Plot the resulting structure
# -------------------------------
data.plot(title="Randomized Cable Net", legend=False)
plt.show()
