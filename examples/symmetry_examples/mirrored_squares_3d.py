"""Two mirrored squares: a 4-fold rotation nested inside a mirror, braced by 4 edges."""

import torch
import torch_structure as ts
import matplotlib.pyplot as plt

node_attrs = {"coords": torch.empty((0, 3), dtype=torch.float)}
edge_attrs = {"force": torch.empty((0, 1), dtype=torch.float)}

data = ts.data.StructData(node_attrs=node_attrs, edge_attrs=edge_attrs)

n_corners = 4

#the rotation makes the square, the mirror makes the second one below the z = 0 plane
rotation = data.create_rotational_symmetry(n_corners)
mirror = data.create_mirror_symmetry(normal=torch.tensor([0.0, 0.0, 1.0]))

mirrored_square = data.combine_symmetry(rotation, mirror)

data.add_symmetry({"mirrored_square": mirrored_square}, copy_attrs=["force"])
data.view_symmetries()

#one seed node, copied to the 4 corners of each of the 2 squares
data.add_nodes_symmetrical(symmetry="mirrored_square", coords=torch.tensor([[1.0, 0.0, 1.0]]))

#the sides of the squares: 4 per square
data.add_edges_by_orbit_symmetrical(
    src_orbit_ids=[0], dest_orbit_ids=[0],
    src_orbit_positions=[0], dest_orbit_positions=[1],
    force=torch.full((1, 1), 1.0),
)

#between the squares: each corner to its own mirror image, so 4 edges
data.add_edges_by_orbit_symmetrical(
    src_orbit_ids=[0], dest_orbit_ids=[0],
    src_orbit_positions=[1], dest_orbit_positions=[4],
    force=torch.full((1, 1), 2.0),
)

print(f"nodes: {data.num_nodes}, undirected edges: {data.num_edges // 2}")

data.plot()

plt.show()
