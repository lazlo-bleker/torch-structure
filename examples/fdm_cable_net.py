"""
Example: Form-find a Cable Net with the Force Density Method (FDM)
"""

import torch_structure as ts
from torch_structure.plot import Plotter

# --------------------------------
# 1. Create inital cable net setup
# --------------------------------
input_params = {
    "n": 5,  # Number of corner supports
    "boundary_density": 5,  # Density of grid
    "diagonals": False,  # Add diagonal edges
    "centroid_support": False,  # Support in the center
    "pattern": "standard",  # Grid layout pattern
    "unsupported_boundaries": True,  # Only corner supports
    "curve_boundaries": True,  # Curve the boundary edges
    "rectangle": False,  # Set corner supports in a rectangle
    "square": False,  # Set corner supports in a square
    "circle": False,  # Set corner supports in a circle
    "support_height": 0.6,  # Elevation of supports
}

# ----------------------------------
# 2. Generate the cable net geometry
# ----------------------------------
cable_net_generator = ts.generators.CableNetGenerator(**input_params)
data = cable_net_generator()  # Main data object of TorchStructure we'll work with

# -------------------------------
# 3. Plot the resulting structure
# -------------------------------
Plotter().plot(data, title="Randomized Cable Net", legend=False, show=True)
