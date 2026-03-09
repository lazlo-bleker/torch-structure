"""
Example: Generate a Grid Shell using TNA

This example demonstrates how to use the GridShellGenerator to create
grid shell structures. The generator handles form-finding internally
using TNA (Thrust Network Analysis), so the returned structure already
has 3D coordinates in equilibrium.
"""

import torch_structure as ts

# ------------------------------
# 1. Create grid shell
# ------------------------------
generator = ts.generators.GridShellGenerator(
    n=4,
    pattern="standard",
    rectangle=True,
    square=True,
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
data.plot(show=True, title="Grid Shell")
