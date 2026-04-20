"""
Example: Equilateral triangular grid using triangle_grid.

A single triangular grid with n subdivisions per side, giving n² small triangles.
Nodes sit at origin + i*dx + j*dy for i≥0, j≥0, i+j≤n.
Edges run in three directions: dx, dy, and the cross diagonal.
"""

import torch
from torch_structure.data import StructData
from torch_structure.generators.topology import triangle_grid

n = 4

node_attrs = {"coords": torch.empty((0, 3), dtype=torch.float)}
data = StructData(node_attrs=node_attrs)

triangle_grid(data, n=n, name="tri")

data.plot(show=True, title=f"Triangle Grid  (n={n},  {n**2} triangles)")
