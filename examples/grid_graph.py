"""
Example: Three grids welded into a hexagon with a triangular hole in the center.

Each rhombus is pushed outward along its bisector (0°, 120°, 240°) by `gap`,
opening a triangular gap instead of a shared centroid.
"""

import math
import torch
from torch_structure.data import StructData
from torch_structure.generators.topology import grid, grid_side, weld_grids

s32 = math.sqrt(3) / 2  # sin(60°)

n   = 4    # subdivisions per rhombus side
gap = 0.5  # outward offset of each rhombus (controls triangle size)

node_attrs = {"coords": torch.empty((0, 3), dtype=torch.float)}
data = StructData(node_attrs=node_attrs)

# R1: right rhombus — pushed in the 0° direction
grid(data, nx=n, ny=n, name="grid1",
     origin=torch.tensor([n * 0.5 + gap, -n * s32, 0.0]),
     dx=torch.tensor([ 0.5,  s32, 0.0]),
     dy=torch.tensor([-0.5,  s32, 0.0]))

# R2: upper-left rhombus — pushed in the 120° direction
grid(data, nx=n, ny=n, name="grid2",
     origin=torch.tensor([-0.5 * gap,  s32 * gap, 0.0]),
     dx=torch.tensor([ 0.5,  s32, 0.0]),
     dy=torch.tensor([-1.0,  0.0, 0.0]))

# R3: lower-left rhombus — pushed in the 240° direction
grid(data, nx=n, ny=n, name="grid3",
     origin=torch.tensor([-0.5 * gap, -s32 * gap, 0.0]),
     dx=torch.tensor([-1.0,  0.0, 0.0]),
     dy=torch.tensor([ 0.5, -s32, 0.0]))

# Weld R1 (y1) to R2 (y0)
weld_grids(data, grid_side(data, "grid1", "y1"), grid_side(data, "grid2", "y0"))

# Weld R2 (x0) to R3 (y0)
weld_grids(data, grid_side(data, "grid2", "x0"), grid_side(data, "grid3", "y0"))

# Weld R1 (x0: V5->inner) to R3 (x0: inner->V5) — reversed
weld_grids(data, grid_side(data, "grid1", "x0"), list(reversed(grid_side(data, "grid3", "x0"))))

data.plot(show=True, title="Hexagon with Central Triangle")
