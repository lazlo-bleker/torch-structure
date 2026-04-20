"""
Example: Irregular polygon using irregular_polygon.

Arbitrary vertices are provided in order. The centroid is computed automatically,
then the polygon is divided into triangular sectors (centroid → each edge) and
filled with triangle grids. Shared spokes are merged by coordinates.
"""

import torch
from torch_structure.data import StructData
from torch_structure.generators.topology import irregular_polygon

vertices = torch.tensor([
    [ 2.2,  0.0, 0.0],
    [ 1.2,  1.8, 0.0],
    [-0.5,  2.5, 0.0],
    [-2.0,  0.8, 0.0],
    [-1.5, -1.5, 0.0],
    [ 0.3, -2.2, 0.0],
])

m = 3

node_attrs = {"coords": torch.empty((0, 3), dtype=torch.float)}
data = StructData(node_attrs=node_attrs)

irregular_polygon(data, vertices=vertices, m=m)

data.plot(show=True, title=f"Irregular {len(vertices)}-gon  (m={m})")
