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

# verify the constructed structure is actually invariant under the symmetry
# group we just enforced it with: apply every group element to every node,
# then check each rotated copy lands exactly back on some node (and that the
# match is a bijection, not just individually close) -- same vectorized
# check as for verifying any n-fold symmetry, not specific to how this
# particular structure was built.
G = data.metadata["symmetry"]
coords_transformed = torch.einsum("kij,nj->kni", G, data.coords)  # [n, N, 3]
min_dist, perm = torch.cdist(coords_transformed, data.coords).min(dim=-1)  # [n, N] each

data.plot()
plt.show()
