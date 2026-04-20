"""
Example: Regular n-gon using polygon_n.

The polygon is divided into n triangular sectors from the center.
Each sector is a triangle_grid with m subdivisions per triangle.
merge_nodes_by_coords collapses all shared spoke nodes automatically.
"""

import torch
from torch_structure.data import StructData
from torch_structure.generators.topology import polygon_n

n = 5   # number of corners
m = 3   # subdivisions per triangle

node_attrs = {"coords": torch.empty((0, 3), dtype=torch.float)}
data = StructData(node_attrs=node_attrs)

polygon_n(data, n=n, m=m)

data.plot(show=True, title=f"Regular {n}-gon  (m={m})")
