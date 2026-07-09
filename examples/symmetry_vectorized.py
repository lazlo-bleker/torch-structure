import torch
from torch_structure.data.data import StructData
import matplotlib.pyplot as plt

n = 5  # n-fold rotational symmetry about the z-axis

node_attrs = {"coords": torch.empty((0, 3), dtype=torch.float)}
edge_attrs = {"force": torch.empty((0, 1), dtype=torch.float)}
data = StructData(node_attrs=node_attrs, edge_attrs=edge_attrs)

data.add_symmetry(n=n)

# top ring: one seed node, rotated n times about z by add_node_with_symmetry
data.add_node_with_symmetry(coords=torch.tensor([1.0, 0.0, 1.0]))  # nodes 0..n-1
# bottom ring: another seed node, rotated n times about z
data.add_node_with_symmetry(coords=torch.tensor([1.0, 0.0, 0.0]))  # nodes n..2n-1

# vertical struts: one seed edge connecting corresponding top/bottom nodes,
# replicated across every rotation by add_edge_with_symmetry
data.add_edge_with_symmetry(0, n, force=torch.tensor([10.0]))

data.plot()
plt.show()
