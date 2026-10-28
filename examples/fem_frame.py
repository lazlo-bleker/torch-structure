"""
Example: Linear-elastic analysis of a braced 3D frame with rigid and pin-ended members
"""

import torch
import torch_structure as ts
import matplotlib.pyplot as plt

# ------------------------------------
# 1. Set up the attributes of the frame
# ------------------------------------
node_attrs = {
    "coords": torch.empty((0, 3), dtype=torch.float),
    "load": torch.empty((0, 3), dtype=torch.float),
    "is_support": torch.empty((0, 1), dtype=torch.bool),  # translations restrained (a pin)
    "support_rxyz": torch.empty((0, 3), dtype=torch.bool),  # rotations restrained per axis
}
edge_attrs = {
    "E": torch.empty((0, 1), dtype=torch.float),
    "A": torch.empty((0, 1), dtype=torch.float),
    "Iy": torch.empty((0, 1), dtype=torch.float),
    "Iz": torch.empty((0, 1), dtype=torch.float),
    "J": torch.empty((0, 1), dtype=torch.float),
    "release": torch.empty((0, 6), dtype=torch.bool),  # [rx_i, ry_i, rz_i, rx_j, ry_j, rz_j]
}
default_attrs = {
    "load": torch.zeros(3),
    "is_support": torch.tensor(False),
    "support_rxyz": torch.zeros(3, dtype=torch.bool),
    "E": torch.tensor([210e9]),  # steel
    "release": torch.zeros(6, dtype=torch.bool),
}
data = ts.data.StructData(node_attrs=node_attrs, edge_attrs=edge_attrs, default_attrs=default_attrs)

# ---------------------------------------
# 2. Nodes: two bays of 6 m, 4 m high, 5 m deep
# ---------------------------------------
xs, ys, h = [0.0, 6.0, 12.0], [0.0, 5.0], 4.0
base = [[x, y, 0.0] for y in ys for x in xs]
top = [[x, y, h] for y in ys for x in xs]
data.add_nodes(
    coords=torch.tensor(base + top),
    is_support=torch.tensor([[True]] * 6 + [[False]] * 6),
    support_rxyz=torch.tensor([[True] * 3] * 6 + [[False] * 3] * 6),  # fixed column bases
)

# --------------------------------------------------------
# 3. Members: rigid columns and beams, pin-ended diagonal braces
# --------------------------------------------------------
columns = [[i, i + 6] for i in range(6)]
beams = [[6, 7], [7, 8], [9, 10], [10, 11], [6, 9], [7, 10], [8, 11]]
braces = [[0, 9], [2, 11], [0, 7], [5, 10]]  # end walls and side walls

rigid = torch.tensor(columns + beams).T
n = rigid.shape[1]
data.add_edges(
    rigid,
    A=torch.full((n, 1), 5.4e-3),
    Iy=torch.full((n, 1), 8.4e-5),
    Iz=torch.full((n, 1), 6.0e-6),
    J=torch.full((n, 1), 2.0e-7),
)

bracing = torch.tensor(braces).T
n = bracing.shape[1]
data.add_edges(
    bracing,
    A=torch.full((n, 1), 1.0e-3),
    Iy=torch.full((n, 1), 1.0e-6),
    Iz=torch.full((n, 1), 1.0e-6),
    J=torch.full((n, 1), 1.0e-7),
    release=torch.ones((n, 6), dtype=torch.bool),  # pinned at both ends: a truss bar
)

# ---------------------------------
# 4. Gravity on the roof, wind along x
# ---------------------------------
load = torch.zeros_like(data.load)
load[6:, 2] = -40e3
load[[6, 9], 0] = 15e3
data.load = load

# -----------
# 5. Analysis
# -----------
result = data.fem()

print(f"max displacement: {result.displacement.norm(dim=1).max() * 1e3:.2f} mm")
print(f"sum of vertical reactions: {result.reaction[:, 2].sum() / 1e3:.1f} kN")
brace_rows = data.directed_mask.view(-1).nonzero().view(-1)[-n:]
print(f"brace axial forces: {(result.force[brace_rows].view(-1) / 1e3).tolist()} kN")

# ----------------------------------------
# 6. Plot the deformed shape, exaggerated
# ----------------------------------------
scale = 50.0
result.plot(
    coords=result.coords + scale * result.displacement,
    ghost=result.coords,  # the undeformed shape
    ghost_label="Undeformed",
    loads=True,
    title=f"Deformed shape (x{scale:.0f})",
)
plt.show()
