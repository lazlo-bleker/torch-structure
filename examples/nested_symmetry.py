import torch
from torch_structure.data.data import StructData
import matplotlib.pyplot as plt

n = 4
k = 2
tower_radius = 1.0
radius = 3.0

node_attrs = {"coords": torch.empty((0, 3), dtype=torch.float)}
edge_attrs = {"force": torch.empty((0, 1), dtype=torch.float)}
data = StructData(node_attrs=node_attrs, edge_attrs=edge_attrs)

n_fold = data.create_rotational_symmetry(n, origin=torch.tensor([radius, 0.0, 0.0]))
k_fold = data.create_rotational_symmetry(k, origin=torch.tensor([0, 0.0, 0.0]))

n_fold_k_fold = data.combine_symmetry(n_fold, k_fold)
data.add_symmetry({"n_fold_k_fold": n_fold_k_fold}, transform_attrs=["coords"])

data.add_nodes(symmetry="n_fold_k_fold", coords=torch.stack([
    torch.tensor([tower_radius + radius, 0.0, 0.0])
]))

data.add_edges_by_orbit(
    src_orbit_ids=[0],
    dest_orbit_ids=[0],
    src_orbit_position=[0],
    dest_orbit_position=[7],
    force=torch.tensor([[1.0]]),
)

data.plot()
plt.show()
