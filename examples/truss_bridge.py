"""
Example: Generate a Truss Bridge using FDM

This example demonstrates how to use the TrussBridgeGenerator to create
truss bridge structures. The generator handles form-finding internally
using FDM, so the returned structure already has 3D coordinates in equilibrium.
"""

import torch_structure as ts

# ------------------------------
# 1. Create truss bridge
# ------------------------------
generator = ts.generators.TrussBridgeGenerator(
    span=40.0,
    n_bays=8,
    truss_type="pratt",
    deck_truss=True,
    triangle=False,
)
data, labels = generator()

# ------------------------------
# 2. Inspect result
# ------------------------------
print(f"Typology: {labels['typology']}")
print(f"Span:     {labels['span']:.1f}m")
print(f"Nodes:    {data.num_nodes}")
print(f"Edges:    {data.num_edges // 2}")

data.verify_equilibrium(verbose=True)

# ------------------------------
# 3. Visualize
# ------------------------------
data.plot(show=True, title=f"{labels['typology']} bridge")
