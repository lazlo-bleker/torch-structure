import torch
import torch_structure as ts
import matplotlib.pyplot as plt

node_attrs = {
    "coords": torch.empty((0, 3), dtype=torch.float),
    "is_support": torch.empty((0, 1), dtype=torch.bool),
}
edge_attrs = {"force": torch.empty((0, 1), dtype=torch.float)}

data = ts.data.StructData(node_attrs=node_attrs, edge_attrs=edge_attrs)

n = 10  
m = 5   

rotational_symmetry = data.create_rotational_symmetry(n)

data.add_symmetry({"rotational_symmetry": rotational_symmetry}, copy_attrs={"is_support"})

height = torch.arange(m).unsqueeze(1)
zeros = torch.zeros(m, 1)
ones = torch.ones(m, 1)
coords = torch.cat([ones, zeros, height], dim=1)
names = [f"node{i}" for i in range(m)]

data.add_nodes(names = names, symmetry="rotational_symmetry", coords=coords)

#rings
data.add_edges_by_orbit(
    src_orbit_ids=torch.arange(m),
    dest_orbit_ids=torch.arange(m),
    src_orbit_positions=torch.zeros(m, dtype=torch.long),
    dest_orbit_positions=torch.ones(m, dtype=torch.long),
    force=torch.full((m, 1), 2.0),
)

#normals
data.add_edges_by_orbit(
    src_orbit_ids=torch.arange(m - 1),
    dest_orbit_ids=torch.arange(1, m),
    src_orbit_positions=torch.zeros(m -1, dtype=torch.long),
    dest_orbit_positions=torch.zeros(m - 1, dtype=torch.long),
    force=torch.ones(m - 1, 1),
)

#set supports
data.set_node_attr_by_name(attr = "is_support", names = ["node0"], value = torch.tensor([[True]]))
data.set_node_attr_by_orbit(attr = "is_support", orbit_ids=[0], orbit_positions=[0], value = torch.tensor([[False]]), consider_symmetry=False)


data.plot(show_supports=True)

plt.show()
