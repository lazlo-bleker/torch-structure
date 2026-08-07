import torch
from torch_structure.data.data import StructData
import matplotlib.pyplot as plt

n = 5
radius = 3.0          
tower_spacing = 10.0   

node_attrs = {"coords": torch.empty((0, 3), dtype=torch.float)}
edge_attrs = {"force": torch.empty((0, 1), dtype=torch.float)}
data = StructData(node_attrs=node_attrs, edge_attrs=edge_attrs)

rotation = data.create_rotational_symmetry(n, origin=torch.tensor([tower_spacing, 0.0, 0.0]))
mirror = data.create_mirror_symmetry(origin=torch.tensor([0.0, 0.0, 0]), normal=torch.tensor([1.0, 0.0, 0.0]))
symmetry = data.combine_symmetry(mirror, rotation)
group_size = symmetry.shape[0]

data.add_symmetry({"n_fold_mirror": symmetry}, transform_attrs=["coords"])

data.add_nodes(symmetry="n_fold_mirror", coords=torch.stack([
    torch.tensor([radius + tower_spacing, 0.0, 0.0])

]))

data.add_edges(
    consider_symmetry=True,
    edge_indices=torch.tensor([
        [0],
        [1],
    ]),
    force=torch.tensor([[1.0]]),
)

data.plot()
plt.show()

data.set_node_attr_with_symmetry("coords", mask = torch.tensor([False, True, False, False, False]), value = torch.tensor([[5.0, 0.0, 0.0]]))

data.plot()
plt.show()
