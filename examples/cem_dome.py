"""
Example: Generate a Dome using CEM

This example demonstrates how to use the DomeGenerator to create
dome structures. The generator handles form-finding internally
using CEM, so the returned structure already has 3D coordinates in equilibrium.
"""

import torch_structure as ts

# ------------------------------
# 1. Create dome
# ------------------------------
generator = ts.generators.DomeGenerator(
    n_trails=12,
    n_rings=4,
    trail_length=0.1,
    center_deviation_force=-3.0,
    opening=False,
)
data = generator()

# ------------------------------
# 2. Inspect result
# ------------------------------
print(f"Nodes: {data.num_nodes}")
print(f"Edges: {data.num_edges // 2}")

data.verify_equilibrium(verbose=True)

# ------------------------------
# 3. Visualize
# ------------------------------
data.plot(show=True, title="Dome")
