"""
Example: Generate an Arch/Suspension Bridge using CEM

This example demonstrates how to use the ArchSuspensionBridgeGenerator to create
arch and suspension bridge structures. The generator handles form-finding internally
using CEM, so the returned structure already has 3D coordinates in equilibrium.
"""

import torch_structure as ts

# ------------------------------
# 1. Create bridge
# ------------------------------
generator = ts.generators.ArchSuspensionBridgeGenerator(
    span=60.0,
    n_cables=2,
    n_bays=8,
    cable_force=10.0,
)
data, labels = generator()

# ------------------------------
# 2. Inspect result
# ------------------------------
print(f"Typology:        {labels['typology']}")
print(f"N cables/arches: {labels['n_cables_or_arches']}")
print(f"Span:            {labels['span']:.1f}m")
print(f"Nodes:           {data.num_nodes}")
print(f"Edges:           {data.num_edges // 2}")

data.verify_equilibrium(verbose=True)

# ------------------------------
# 3. Visualize
# ------------------------------
data.plot(show=True, title=f"{labels['typology']} bridge")