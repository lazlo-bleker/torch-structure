"""Two mirrored squares: a 4-fold rotation nested inside a mirror, braced by 4 edges."""

import torch
import torch_structure as ts
import matplotlib.pyplot as plt

node_attrs = {"coords": torch.empty((0, 3), dtype=torch.float)}
edge_attrs = {"force": torch.empty((0, 1), dtype=torch.float)}

data = ts.data.StructData(node_attrs=node_attrs, edge_attrs=edge_attrs)


#the rotation makes the square, the mirror makes the second one below the z = 0 plane
mirror1 = data.create_mirror_symmetry(normal=torch.tensor([1.0, 0.0, 0.0]))
mirror2 = data.create_mirror_symmetry(normal=torch.tensor([0.0, 1.0, 0.0]))

double_mirror = data.combine_symmetry(mirror1, mirror2)

data.add_symmetry({"mirrored_square": double_mirror}, copy_attrs=["force"])
data.view_symmetries()

data.add_nodes(symmetry="mirrored_square", coords=torch.tensor([[1.0, 1.0, 0.0]]))



data.add_edges_by_orbit(
    src_orbit_ids=[0], dest_orbit_ids=[0],
    src_orbit_positions=[0], dest_orbit_positions=[2],
    force=torch.full((1, 1), 1.0),
)


print(f"nodes: {data.num_nodes}, undirected edges: {data.num_edges // 2}")

data.plot()

plt.show()
