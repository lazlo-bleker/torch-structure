import torch
import torch_structure as ts
import time

# -------------------------
# 1. Build topology diagram
# -------------------------

# 0---1---2---3---4---5
# ^   |\ /|\ /|\ /|   ^
#     | X | X | X |
#     |/ \|/ \|/ \|
# 6---7---8---9---10--11
# ^   |   |   |   |   ^
#     V   V   V   V

node_attrs = {
    'coords': torch.empty((0, 3), dtype=torch.float),
    'load': torch.empty((0, 3), dtype=torch.float),
    'is_support': torch.empty((0, 1), dtype=torch.bool),
    'is_origin_node': torch.empty((0, 1), dtype=torch.bool),
    'sequence': torch.empty((0, 1), dtype=torch.long)
}
edge_attrs = {
    'force': torch.empty((0, 1), dtype=torch.float),
    'length': torch.empty((0, 1), dtype=torch.float),
    'is_trail_edge': torch.empty((0, 1), dtype=torch.bool),
    'force_sign': torch.empty((0, 1), dtype=torch.float)
}
default_attrs = {
    'force': torch.tensor([torch.nan]),
    'coords': torch.full((3,), torch.nan),
    'load': torch.zeros(3, dtype=torch.float),
    'is_support': torch.tensor(False),
    'is_origin_node': torch.tensor(False),
}
data = ts.data.StructData(node_attrs=node_attrs, edge_attrs=edge_attrs, default_attrs=default_attrs)

trail_length = torch.tensor(1.0)
force_sign = torch.tensor(-1.0)
deviation_force = torch.tensor(0.5)
center_deviation_force = torch.tensor(-5.0)

trails = [
    [2, 1, 0],
    [3, 4, 5],
    [8, 7, 6],
    [9, 10, 11]
]
origin_coords = [
    torch.tensor([2.0, 0.0, 1.0]),
    torch.tensor([3.0, 0.0, 1.0]),
    torch.tensor([2.0, 0.0, 0.0]),
    torch.tensor([3.0, 0.0, 0.0]),
]
load = torch.tensor([0.0, 0.0, -1.0])
for trail, origin_coord in zip(trails, origin_coords):
    # add nodes
    data.add_node(trail[0], sequence=0, load=load, is_origin_node=True, coords=origin_coord)
    data.add_node(trail[1], sequence=1, load=load)
    data.add_node(trail[2], sequence=2, is_support=True)

    # add trail edges
    data.add_edge(trail[0], trail[1], is_trail_edge=True, length=trail_length, force_sign=force_sign)
    data.add_edge(trail[1], trail[2], is_trail_edge=True, length=trail_length, force_sign=force_sign)

# center deviation edges
data.add_edge(2, 3, is_trail_edge=False, force=center_deviation_force)
data.add_edge(8, 9, is_trail_edge=False, force=center_deviation_force)

# other deviadation edges
direct_edges = [
    [1, 7],  # direct edges
    [2, 8],
    [3, 9],
    [4, 10],
    [1, 8],  # indirect edges
    [2, 7],
    [2, 9],
    [3, 8],
    [3, 10],
    [4, 9],
]
for src, dst in direct_edges:
    data.add_edge(src, dst, is_trail_edge=False, force=deviation_force)

# ----------------------
# 2. Solve and visualize
# ----------------------

# Node-wise CEM
t0 = time.time()
ff_graph = data.seqcem(max_iter=100, verbose=True)
t1 = time.time()
print(f'Sequential CEM took {t1 - t0:.5f} seconds')
ff_graph.verify_equilibrium(verbose=True)
ff_graph.plot(title='Sequential CEM', show=True)

# or

# Layer-wise CEM
t0 = time.time()
ff_graph = data.cem(max_iter=100, verbose=True)
t1 = time.time()
print(f'CEM took {t1 - t0:.5f} seconds')
ff_graph.verify_equilibrium(verbose=True)
ff_graph.plot(title='CEM')

# or

# Graph-wise CEM
t0 = time.time()
ff_graph = data.mpcem(max_iter=100, verbose=True, damping_factor=0.0)
t1 = time.time()
print(f'MP-CEM took {t1 - t0:.5f} seconds')
ff_graph.verify_equilibrium(verbose=True)
ff_graph.plot(title='MP-CEM')
