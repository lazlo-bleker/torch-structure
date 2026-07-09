import torch
from torch_structure.data.data import StructData
import matplotlib.pyplot as plt

n = 25 # n-fold rotational symmetry about the z-axis

node_attrs = {"coords": torch.empty((0, 3), dtype=torch.float)}
data = StructData(node_attrs=node_attrs)

data.add_symmetry(n=n)

# three ring seeds (top, middle, bottom), all rotated n times about z in a
# single vectorized call. Seeds' copies are appended as contiguous n-blocks
# in seed order: top ring -> nodes 0..n-1, middle ring -> nodes n..2n-1,
# bottom ring -> nodes 2n..3n-1.
data.add_nodes_with_symmetry(coords=torch.stack([
    torch.tensor([1.3, 0.0, 0.0]),  # top ring, tapered inward
    torch.tensor([1.0, 0.0, 1.0]),  # middle ring
    torch.tensor([.3, 0.0, 2.0]),  # bottom ring, flared outward
]))

# vertical struts: two seed edges (top-to-middle, middle-to-bottom), each
# replicated across every rotation, added in a single vectorized call.
data.add_edges_with_symmetry(
    src=torch.tensor([0, n, 0, n, 2 * n, 0, n]),
    dst=torch.tensor([n, 2 * n, 1, n + 1, 2 * n + 1, n+1, 2*n+1])
)

data.plot()
plt.show()
