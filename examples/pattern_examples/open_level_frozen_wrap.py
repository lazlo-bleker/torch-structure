"""``closes=False`` leaves the wrap out of the copies; it does not forbid the wrap as an edge.

An open level means that sliding an edge along it stops at the end instead of running round to the
start. Asking for the closing edge directly is a different thing, and that edge is added: the slide
is where the rule applies, not the pair of positions.
"""

import torch
import torch_structure as ts
import matplotlib.pyplot as plt

node_attrs = {"coords": torch.empty((0, 3), dtype=torch.float)}
edge_attrs = {"force": torch.empty((0, 1), dtype=torch.float)}

data = ts.data.StructData(node_attrs=node_attrs, edge_attrs=edge_attrs)

n_corners = 6
n_levels = 3

#an open rotation, so each polygon is missing its closing side, stacked by an open array
open_rotation = data.create_rotational_pattern(n_corners, closes=False)
vertical_array = data.create_translational_pattern(n_levels, direction=torch.tensor([0.0, 0.0, 1.5]))

stacked = data.nest_patterns([open_rotation, vertical_array])

data.add_patterns({"stacked": stacked})
data.view_patterns()

data.add_nodes_by_pattern(pattern="stacked", coords=torch.tensor([[2.0, 0.0, 0.0]]))

#sliding one side around the open rotation stops after 5 of the 6 sides, on every level
data.add_edges_per_expansion_by_pattern(
    src_expansion_ids=[0], dest_expansion_ids=[0],
    src_expansion_positions=[0], dest_expansion_positions=[1],
    force=torch.full((1, 1), 1.0),
)

#the uprights
data.add_edges_per_expansion_by_pattern(
    src_expansion_ids=[0], dest_expansion_ids=[0],
    src_expansion_positions=[0], dest_expansion_positions=[n_corners],
    force=torch.full((1, 1), 2.0),
)


print(f"nodes: {data.num_nodes}, undirected edges: {data.num_edges // 2}")

data.plot()

plt.show()
