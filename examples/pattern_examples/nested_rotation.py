import torch
import torch_structure as ts
import matplotlib.pyplot as plt

node_attrs = {"coords": torch.empty((0, 3), dtype=torch.float)}
edge_attrs = {"force": torch.empty((0, 1), dtype=torch.float)}

data = ts.data.StructData(node_attrs=node_attrs, edge_attrs=edge_attrs)

n = 10  
m = 5   
s = 5

rotational_symmetry_n = data.create_rotational_symmetry(n)
rotational_symmetry_s = data.create_rotational_symmetry(s, origin = torch.tensor([5.0, 0.0, 0.0]))
nested_rotational_symmetry = data.combine_symmetry(rotational_symmetry_n, rotational_symmetry_s)

data.add_symmetry({"rotational_symmetry": nested_rotational_symmetry})
data.add_symmetry({"rotational_symmetry_s": rotational_symmetry_s})

height = torch.arange(m).unsqueeze(1)
zeros = torch.zeros(m, 1)
ones = torch.ones(m, 1) * .5
coords = torch.cat([ones, zeros, height], dim=1)

headcoord = torch.tensor([[0.0, 0.0, m + .5]])


data.add_nodes_symmetrical(symmetry="rotational_symmetry", coords=coords)
data.add_nodes_symmetrical(symmetry="rotational_symmetry_s", coords=headcoord)

#rings
data.add_edges_by_orbit_symmetrical(
    src_orbit_ids=torch.arange(m),
    dest_orbit_ids=torch.arange(m),
    src_orbit_positions=torch.zeros(m, dtype=torch.long),
    dest_orbit_positions=torch.ones(m, dtype=torch.long),
    force=torch.full((m, 1), 2.0),
)

#normals
data.add_edges_by_orbit_symmetrical(
    src_orbit_ids=torch.arange(m - 1),
    dest_orbit_ids=torch.arange(1, m),
    src_orbit_positions=torch.zeros(m -1, dtype=torch.long),
    dest_orbit_positions=torch.zeros(m - 1, dtype=torch.long),
    force=torch.ones(m - 1, 1),
)

data.set_node_attr_by_name

data.plot()

plt.show()
