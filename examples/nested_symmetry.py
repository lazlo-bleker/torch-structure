import torch
from torch_structure.data.data import StructData
import matplotlib.pyplot as plt

n = 5
k = 6
radius = 1.0
tower_spacing = 10.0

node_attrs = {"coords": torch.empty((0, 3), dtype=torch.float)}
edge_attrs = {"force": torch.empty((0, 1), dtype=torch.float)}
data = StructData(node_attrs=node_attrs, edge_attrs=edge_attrs)

rotation_matrices, rotation_cyclic = data.create_rotational_symmetry(n, origin=torch.tensor([tower_spacing, 0.0, 0.0]))
rotation2_matrices, rotation2_cyclic = data.create_rotational_symmetry(k, origin=torch.tensor([0, 0.0, 0.0]))
symmetry_matrices, symmetry_cyclic = data.combine_symmetry(rotation2_matrices, rotation_matrices, rotation2_cyclic, rotation_cyclic)

data.add_symmetry({"n_fold_k_fold": (symmetry_matrices, symmetry_cyclic)}, transform_attrs=["coords"])

data.add_nodes(symmetry="n_fold_k_fold", coords=torch.stack([
    torch.tensor([radius + tower_spacing, 0.0, 0.0]),
    torch.tensor([radius + tower_spacing, 0.0, 1.0])
]))

data.add_edges(
    consider_symmetry=True,
    edge_indices=torch.tensor([
        [0, n*k, 0],
        [1, n*k + 1, n*k],
    ]),
    force=torch.tensor([[1.0], [1.0],[1.0]]),
)

data.plot()
plt.show()

data.set_node_attr_with_symmetry("coords", mask = torch.tensor([False, True, False]), value = torch.tensor([[5.0, 0.0, 0.0]]))

data.plot()
plt.show()
