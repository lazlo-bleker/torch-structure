"""
Example: Generate a Nervi Dome using CEM

This example demonstrates how to use the NerviDome generator to create
a Nervi-style dome with diagonal deviation members connecting adjacent
trail rings. Form-finding is performed using MP-CEM.
"""

import torch_structure as ts

# ------------------------------
# 1. Create Nervi dome
# ------------------------------
generator = ts.generators.NerviDome(
    n_trails=20,
    n_rings=20,
    trail_length=0.1,
    deviation_force=-2,
    center_deviation_force=-5,
    ring_force=-15,
    opening_diameter=0.2,
    opening=True,
)

# ------------------------------
# 2. Form-finding
# ------------------------------
data = generator()

# ------------------------------
# 3. Inspect result
# ------------------------------
print(f"Nodes: {data.num_nodes}")
print(f"Edges: {data.num_edges // 2}")

data.verify_equilibrium(verbose=True)

# ------------------------------
# 4. Visualize
# ------------------------------
data.plot(show=True, title="Nervi Dome")
