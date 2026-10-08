"""A cylinder grown from a single seed node: a vertical array nested inside a rotation."""

import torch
import torch_structure as ts
import matplotlib.pyplot as plt

node_attrs = {"coords": torch.empty((0, 3), dtype=torch.float)}
edge_attrs = {"force": torch.empty((0, 1), dtype=torch.float)}

data = ts.data.StructData(node_attrs=node_attrs, edge_attrs=edge_attrs)

n_sectors = 12
n_rings = 5

#the array stacks the rings and stays open, the rotation closes each ring
vertical_array = data.create_translation(n_rings, direction=torch.tensor([0.0, 0.0, 1.0]))
rotation = data.create_rotational_symmetry(n_sectors, closes=True)

cylinder_symmetry = data.combine_symmetry(vertical_array, rotation)

data.add_symmetry({"cylinder": cylinder_symmetry}, copy_attrs=["force"])
data.view_symmetries()

#one seed node, copied to n_rings * n_sectors positions
data.add_nodes(symmetry="cylinder", coords=torch.tensor([[1.0, 0.0, 0.0]]))

#verticals: one seed per direction is enough, each is copied over the whole symmetry
data.add_edges_by_orbit(
    src_orbit_ids=[0], dest_orbit_ids=[0],
    src_orbit_positions=[0], dest_orbit_positions=[1],
    force=torch.full((1, 1), 1.0),
)

#rings
data.add_edges_by_orbit(
    src_orbit_ids=[0], dest_orbit_ids=[0],
    src_orbit_positions=[0], dest_orbit_positions=[n_rings],
    force=torch.full((1, 1), 2.0),
)

data.add_edges_by_orbit(
    src_orbit_ids=[0], dest_orbit_ids=[0],
    src_orbit_positions=[0], dest_orbit_positions=[n_rings+1],
    force=torch.full((1, 1), 2.0),
)

data.set_edge_attr_by_orbit(
    "force",
    src_orbit_ids=[0], dest_orbit_ids=[0],
    src_orbit_positions=[0], dest_orbit_positions=[n_rings],
    value=torch.full((1, 1), 20.0),
    replicate_levels=[False, True],
)

data.set_edge_attr_by_orbit(
    "force",
    src_orbit_ids=[0], dest_orbit_ids=[0],
    src_orbit_positions=[0], dest_orbit_positions=[1],
    value=torch.full((1, 1), 20.0),
    replicate_levels=[True, False],
)

print(f"nodes: {data.num_nodes}, undirected edges: {data.num_edges // 2}")

data.plot()

plt.show()
