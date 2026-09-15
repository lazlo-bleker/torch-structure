import torch
from torch_structure.data.data import StructData
import matplotlib.pyplot as plt

n = 6  # n-fold rotational symmetry about the z-axis

node_attrs = {"coords": torch.empty((0, 3), dtype=torch.float)}
edge_attrs = {"force": torch.empty((0, 1), dtype=torch.float)}
data = StructData(node_attrs=node_attrs, edge_attrs=edge_attrs)

symmetry = data.create_rotational_symmetry(n)

data.add_symmetry({"n_fold": symmetry}, transform_attrs=["coords"])


data.add_nodes(symmetry="n_fold", coords=torch.stack([
    torch.tensor([1.3, 0.0, 0.0]),  # bottom ring
    torch.tensor([1.0, 0.0, 1.0]),  # middle ring
    torch.tensor([.3, 0.0, 2.0]),  # top ring
]))

data.add_edges(
    consider_symmetry=True,
    edge_indices=torch.tensor([
        [0, n, 0, n, 2*n],
        [n, 2 * n, 1, n + 1, 2*n + 1],
    ]),
    force=torch.tensor([[1.0], [1.0], [1.0], [1.0], [1.0]]),
)

data.add_nodes(coords=torch.tensor([[0.0, 0.0, -1.0]]))
assym_ind = data.num_nodes - 1

data.add_edges(consider_symmetry=False, edge_indices=torch.tensor([[assym_ind], [0]]), force=torch.tensor([[1.0]]))

data.plot()
plt.show()



k = 3  # k-fold rotational symmetry, k divides n so it can connect to the n-ring
symmetry_k = data.create_rotational_symmetry(k)
data.add_symmetry({"k_fold": symmetry_k}, transform_attrs=["coords"])

num_nodes_before = data.num_nodes
data.add_nodes(symmetry="k_fold", coords=torch.tensor([[0.6, 0.0, 3.0]]))

data.add_edges(
    consider_symmetry=True,
    edge_indices=torch.tensor([[num_nodes_before + 1], [num_nodes_before]]),
    force=torch.tensor([[1.0]])
)


data.add_edges(
    consider_symmetry=True,
    edge_indices=torch.tensor([[num_nodes_before - 2], [num_nodes_before]]),
    force=torch.tensor([[1.0]])
)


data.plot()
plt.show()

mask = torch.zeros(data.num_nodes, dtype=torch.bool)
mask[assym_ind] = True
new_value = data.coords[assym_ind:assym_ind + 1].clone()
new_value[:, 2] -= 10
data.set_node_attr("coords", mask, new_value)

edge_mask = (data.edge_index[0] == assym_ind) & (data.edge_index[1] == 0) & data.directed_mask.view(-1)
data.set_edge_attr("force", edge_mask, torch.tensor([[5.0]]))

data.plot()
plt.show()


#delete nodes: remove node triggers removal of the corresonding matrix? or break symmetry (with warning)?