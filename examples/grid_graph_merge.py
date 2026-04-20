"""
Example: Three grids merged into a hexagon using merge_nodes_by_coords.

The grids are placed at coincident coordinates along their shared edges.
merge_nodes_by_coords finds and collapses all duplicate nodes automatically.
"""

import math
import torch
from torch_structure.data import StructData
from torch_structure.generators.topology import grid, merge_nodes_by_coords

s32 = math.sqrt(3) / 2

n = 4

node_attrs = {"coords": torch.empty((0, 3), dtype=torch.float)}
data = StructData(node_attrs=node_attrs)

grid(data, nx=n, ny=n, name="R1",
     origin=torch.tensor([n * 0.5, -n * s32, 0.0]),
     dx=torch.tensor([ 0.5,  s32, 0.0]),
     dy=torch.tensor([-0.5,  s32, 0.0]))

grid(data, nx=n, ny=n, name="R2",
     origin=torch.tensor([0.0, 0.0, 0.0]),
     dx=torch.tensor([ 0.5,  s32, 0.0]),
     dy=torch.tensor([-1.0,  0.0, 0.0]))

grid(data, nx=n, ny=n, name="R3",
     origin=torch.tensor([0.0, 0.0, 0.0]),
     dx=torch.tensor([-1.0,  0.0, 0.0]),
     dy=torch.tensor([ 0.5, -s32, 0.0]))

merge_nodes_by_coords(data)

data.plot(show=True, title="Hexagon — Merged Nodes")


#go over code again
#separate topology and geometry cooords

