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
open_rotation = data.create_rotational_symmetry(n_corners, closes=False)
vertical_array = data.create_translation(n_levels, direction=torch.tensor([0.0, 0.0, 1.5]))

stacked = data.combine_symmetry(open_rotation, vertical_array)

data.add_symmetry({"stacked": stacked}, copy_attrs=["force"])
data.view_symmetries()

data.add_nodes_symmetrical(symmetry="stacked", coords=torch.tensor([[2.0, 0.0, 0.0]]))

#sliding one side around the open rotation stops after 5 of the 6 sides, on every level
data.add_edges_by_orbit_symmetrical(
    src_orbit_ids=[0], dest_orbit_ids=[0],
    src_orbit_positions=[0], dest_orbit_positions=[1],
    force=torch.full((1, 1), 1.0),
)

#the uprights
data.add_edges_by_orbit_symmetrical(
    src_orbit_ids=[0], dest_orbit_ids=[0],
    src_orbit_positions=[0], dest_orbit_positions=[n_corners],
    force=torch.full((1, 1), 2.0),
)

#asking for the closing side itself: it is added, and replicated up the array. sliding it around
#the open rotation drops every copy but the one given, since the rest run off the end
data.add_edges_by_orbit_symmetrical(
    src_orbit_ids=[0], dest_orbit_ids=[0],
    src_orbit_positions=[n_corners - 1], dest_orbit_positions=[0],
    force=torch.full((1, 1), 5.0),
)

print(f"nodes: {data.num_nodes}, undirected edges: {data.num_edges // 2}")

data.plot()

plt.show()
