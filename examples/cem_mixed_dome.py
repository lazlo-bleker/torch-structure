"""
Example: Generate a Mixed Dome using CEM

This example demonstrates how to use the MixedDomeGenerator to create
mixed dome structures with alternating tension/compression members.
The generator handles form-finding internally using MP-CEM, so the
returned structure already has 3D coordinates in equilibrium.
"""

import torch_structure as ts

# ------------------------------
# 1. Create mixed dome
# ------------------------------
generator = ts.generators.MixedDomeGenerator(
    n_trails=12,
    n_rings=5,
    trail_length=0.1,
    center_deviation_force=-3.0,
    center_trail_sign="compression",
    sign_flip_indices=[3],
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
data.plot(show=True, title="Mixed Dome")
