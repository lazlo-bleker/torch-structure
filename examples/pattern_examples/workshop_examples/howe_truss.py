

import torch
import torch_structure as ts
import matplotlib.pyplot as plt

node_attrs = {"coords": torch.empty((0, 3), dtype=torch.float)}
edge_attrs = {"force": torch.empty((0, 1), dtype=torch.float)}

data = ts.data.StructData(node_attrs=node_attrs, edge_attrs=edge_attrs)

n = 5

array_in_x_dir = data.create_translational_pattern(n, direction=torch.tensor([2.0, 0.0, 0.0]))
mirror_in_x_dir = data.create_mirror_pattern(origin=torch.tensor([0., 0., 0.]), normal=torch.tensor([1., 0., 0.]))
mirror_in_z_dir = data.create_mirror_pattern(origin=torch.tensor([0., 0., 1.]), normal=torch.tensor([0., 0., 1.]))
mirror_in_y_dir = data.create_mirror_pattern(origin=torch.tensor([0., 1., 0.]), normal=torch.tensor([0., 1., 0.]))

combined_pattern = data.nest_patterns([
    array_in_x_dir,
    mirror_in_x_dir,
    mirror_in_z_dir,
    mirror_in_y_dir,
])

data.add_patterns({"howe_truss": combined_pattern})

data.add_nodes_by_pattern(pattern="howe_truss", coords=torch.tensor([[0.0, 0.0, 0.0]]))

data.add_edges_by_pattern_level(
    src_expansion_ids=[0], dest_expansion_ids=[0],
    src_level_positions=[[0, 0, 0, 0]], dest_level_positions=[[1, 0, 0, 0]],
    force=torch.full((1, 1), 1.0),
)

data.add_edges_by_pattern_level(
    src_expansion_ids=[0], dest_expansion_ids=[0],
    src_level_positions=[[0, 0, 0, 0]], dest_level_positions=[[0, 0, 1, 0]],
    force=torch.full((1, 1), 1.0),
    replicate_levels=[True, True, False, True],
)

data.add_edges_by_pattern_level(
    src_expansion_ids=[0], dest_expansion_ids=[0],
    src_level_positions=[[1, 0, 0, 0]], dest_level_positions=[[0, 0, 1, 0]],
    force=torch.full((1, 1), 1.0),
    replicate_levels=[True, True, False, True],
)

data.add_edges_by_pattern_level(
    src_expansion_ids=[0], dest_expansion_ids=[0],
    src_level_positions=[[0, 0, 0, 0]], dest_level_positions=[[0, 0, 0, 1]],
    force=torch.full((1, 1), 1.0),
    replicate_levels=[True, True, True, False],
)

data.add_edges_by_pattern_level(
    src_expansion_ids=[0], dest_expansion_ids=[0],
    src_level_positions=[[0, 0, 0, 0]], dest_level_positions=[[1, 0, 0, 1]],
    force=torch.full((1, 1), 1.0),
)

data.plot()

data.plot(pattern=True)

plt.show()