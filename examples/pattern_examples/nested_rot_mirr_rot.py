"""A rotation nested inside a mirror nested inside a rotation: squares, mirrored, then arrayed."""

import torch
import torch_structure as ts
import matplotlib.pyplot as plt

node_attrs = {"coords": torch.empty((0, 3), dtype=torch.float)}
edge_attrs = {"force": torch.empty((0, 1), dtype=torch.float)}

data = ts.data.StructData(node_attrs=node_attrs, edge_attrs=edge_attrs)

n_corners = 4
n_sectors = 3

#the inner rotation makes a square, the mirror copies it below the z = 0 plane into a box, the
#outer rotation arrays that box around a distant axis, giving n_sectors boxes in a ring
inner_rotation = data.create_rotational_pattern(n_corners)
mirror = data.create_mirror_pattern(normal=torch.tensor([0.0, 0.0, 1.0]))
outer_rotation = data.create_rotational_pattern(n_sectors, origin=torch.tensor([5.0, 0.0, 0.0]))

nested_pattern = data.nest_patterns([
    inner_rotation,
    mirror,
    outer_rotation,
])

data.add_patterns({"nested": nested_pattern})
data.view_patterns()

#one seed node, copied to n_corners * 2 * n_sectors positions
data.add_nodes_by_pattern(pattern="nested", coords=torch.tensor([[1.0, 0.0, 1.0]]))

#level 0: the sides of each square
data.add_edges_per_expansion_by_pattern(
    src_expansion_ids=[0], dest_expansion_ids=[0],
    src_expansion_positions=[0], dest_expansion_positions=[1],
    force=torch.full((1, 1), 1.0),
)

#level 1: each corner to its own mirror image, so 4 uprights per box
data.add_edges_per_expansion_by_pattern(
    src_expansion_ids=[0], dest_expansion_ids=[0],
    src_expansion_positions=[0], dest_expansion_positions=[n_corners],
    force=torch.full((1, 1), 2.0),
)

#level 2: each corner to the matching corner of the next box
data.add_edges_per_expansion_by_pattern(
    src_expansion_ids=[0], dest_expansion_ids=[0],
    src_expansion_positions=[0], dest_expansion_positions=[2 * n_corners],
    force=torch.full((1, 1), 3.0),
)

print(f"nodes: {data.num_nodes}, undirected edges: {data.num_edges // 2}")

data.plot()

plt.show()
