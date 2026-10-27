"""A shell made by two translations, arrayed 3 x 4 times by two more, and form-found with the force density method."""

import torch
import torch_structure as ts
import matplotlib.pyplot as plt

node_attrs = {
    "coords": torch.empty((0, 3), dtype=torch.float),
    "is_support": torch.empty((0, 1), dtype=torch.bool),
    "load": torch.empty((0, 3), dtype=torch.float),
}
edge_attrs = {
    "force_density": torch.empty((0, 1), dtype=torch.float),
}

data = ts.data.StructData(node_attrs=node_attrs, edge_attrs=edge_attrs)

n = 5
n_y = 3

shell_x = data.create_translational_pattern(n, direction=torch.tensor([1.0, 0.0, 0.0]))
shell_y = data.create_translational_pattern(n, direction=torch.tensor([0.0, 1.0, 0.0]))

mirror_x = data.create_mirror_pattern(origin = torch.tensor([n-1, 0.0,  0.0]), normal=torch.tensor([1.0, 0.0, 0.0]))

grid_y = data.create_translational_pattern(n_y, direction=torch.tensor([0, n-1, 0.0]))


blob_shell = data.nest_patterns([shell_x, shell_y, mirror_x, grid_y])
data.add_patterns({"blob_shell": blob_shell})

data.add_nodes_by_pattern(pattern="blob_shell", coords=torch.tensor([[0.0, 0.0, 0.0]]))

data.add_edges_by_pattern_level(
    src_expansion_ids=[0,0], dest_expansion_ids=[0,0],
    src_level_positions=[[0, 0, 0, 0], [0, 0, 0, 0]], dest_level_positions=[[1, 0, 0, 0], [0, 1, 0, 0]],
    force_density=torch.tensor([[-1.], [-1.]])
)

data.set_node_attr_by_pattern_level(
    "is_support", expansion_ids=[0], level_positions=[[0, 0, 0, 0]], value=torch.tensor([[True]]), replicate_levels=[False, True, True, True]
)

data.set_edge_attr_by_pattern_level(
    "force_density", src_expansion_ids=[0], dest_expansion_ids=[0],
    src_level_positions=[[0, 0, 0, 0]], dest_level_positions=[[0, 1, 0, 0]],
    value=torch.tensor([[-10.0]]), replicate_levels=[True, False, True, True]
)

data.fdm(inplace=True)

data.plot(show_supports=True)

plt.show()
