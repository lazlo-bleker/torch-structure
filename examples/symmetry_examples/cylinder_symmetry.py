import torch
import torch_structure as ts
import matplotlib.pyplot as plt

node_attrs = {"coords": torch.empty((0, 3), dtype=torch.float)}
edge_attrs = {"force": torch.empty((0, 1), dtype=torch.float)}

data = ts.data.StructData(node_attrs=node_attrs, edge_attrs=edge_attrs)

n = 10  
m = 5   

rotational_symmetry = data.create_rotational_symmetry(n)

data.add_symmetry({"rotational_symmetry": rotational_symmetry})

height = torch.arange(m).unsqueeze(1)
zeros = torch.zeros(m, 1)
ones = torch.ones(m, 1)
coords = torch.cat([ones, zeros, height], dim=1)

data.add_nodes_symmetrical(symmetry="rotational_symmetry", coords=coords)

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

data.plot()

plt.show()
