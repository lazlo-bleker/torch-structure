"""
Example: Generate a Network Arch Bridge using CEM

This example demonstrates how to use the NetworkArchBridge generator to create
a network arch bridge structure. The generator handles form-finding internally
using CEM, so the returned structure already has 3D coordinates in equilibrium.
"""

import torch_structure as ts

# ------------------------------
# 1. Create bridge
# ------------------------------
generator = ts.generators.NetworkArchBridge(
    n_trail_edges=4,
    cable_offset=2,
    height=10.0,
    deck_width=6.0,
    arch_width=5.0,
    arch_force=-50.0,
    deck_force=10.0,
    cable_force=-5.0,
    inter_deck_force=5.0,
    inter_arch_force=5.0,
    deck_trail_length=5.0,
    arch_trail_length=5.0,
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
data.plot(show=True, title="Network Arch Bridge")
