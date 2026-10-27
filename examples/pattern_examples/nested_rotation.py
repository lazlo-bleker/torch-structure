import torch
import torch_structure as ts
import matplotlib.pyplot as plt

node_attrs = {"coords": torch.empty((0, 3), dtype=torch.float)}
edge_attrs = {"force": torch.empty((0, 1), dtype=torch.float)}

data = ts.data.StructData(node_attrs=node_attrs, edge_attrs=edge_attrs)

n = 10  
m = 5   
s = 5

rotational_pattern_n = data.create_rotational_pattern(n)
rotational_pattern_s = data.create_rotational_pattern(s, origin = torch.tensor([5.0, 0.0, 0.0]))
nested_rotational_pattern = data.nest_patterns([rotational_pattern_n, rotational_pattern_s])

data.add_patterns({"rotational_pattern": nested_rotational_pattern})
data.add_patterns({"rotational_pattern_s": rotational_pattern_s})

height = torch.arange(m).unsqueeze(1)
zeros = torch.zeros(m, 1)
ones = torch.ones(m, 1) * .5
coords = torch.cat([ones, zeros, height], dim=1)

headcoord = torch.tensor([[0.0, 0.0, m + .5]])


data.add_nodes_by_pattern(pattern="rotational_pattern", coords=coords)
data.add_nodes_by_pattern(pattern="rotational_pattern_s", coords=headcoord)

#rings
data.add_edges_per_expansion_by_pattern(
    src_expansion_ids=torch.arange(m),
    dest_expansion_ids=torch.arange(m),
    src_expansion_positions=torch.zeros(m, dtype=torch.long),
    dest_expansion_positions=torch.ones(m, dtype=torch.long),
    force=torch.full((m, 1), 2.0),
)

#normals
data.add_edges_per_expansion_by_pattern(
    src_expansion_ids=torch.arange(m - 1),
    dest_expansion_ids=torch.arange(1, m),
    src_expansion_positions=torch.zeros(m -1, dtype=torch.long),
    dest_expansion_positions=torch.zeros(m - 1, dtype=torch.long),
    force=torch.ones(m - 1, 1),
)

data.set_node_attr_by_name

data.plot()

plt.show()
