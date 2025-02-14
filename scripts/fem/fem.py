import matplotlib.pyplot as plt
import torch
import torch_structure as ts

truss_input = {'span': 70.39145157441635, 'n_cables': 1, 'n_bays': 9, 'midspan_height': -20.401622381577784, 'deck_width': 2.066573764101965, 'twist': 0.0, 'cable_distance': 19.419906032713087, 'deck_rise': 0.0, 'deck_force': -7.190351198421553, 'cable_force': 14.380492396843108, 'brace_force': 1.0, 'inter_cable_force': 0.0}
truss = ts.generators.Bridge(**truss_input)
data, meta_data = truss.pyg_data()

data.coords = data.x[:, 0:3]
data.load = data.x[:, 3:6]
data.support = torch.stack([data.x[:, 6].bool(), data.x[:, 6].bool(), data.x[:, 6].bool()], dim=1)
data.A = torch.ones((data.edge_index.shape[1], 1))
data.E = torch.ones((data.edge_index.shape[1], 1)) * 10000
data.u = torch.zeros_like(data.coords)

direction, length = ts.utils.edge_direction(data.edge_index, data.coords, return_length=True)
data.direction = direction
data.length = length

data.effective_stiffness = ts.utils.effective_stiffness(data.edge_index, data.E, data.A, data.length)
data.transformation_matrix = ts.utils.transformation_matrix(data.edge_index, data.coords, data.length)

fem = ts.gmp.FEM()
for i in range(10000):
    u_new, residual_force = fem(data.coords, data.load, data.u, data.support, data.effective_stiffness,
                                data.edge_index, data.length, data.A, data.E, data.direction, data.transformation_matrix)
    
    total_reaction = torch.sum(residual_force * data.support, dim=0)
    total_load = torch.sum(data.load, dim=0)
    global_residual = total_load - total_reaction
    local_residual = torch.mean(residual_force.abs() * ~data.support)
    u_diff = torch.abs(u_new - data.u)

    data.u = u_new

    if i % 100 == 0:
        print(f"displacement mean delta {u_diff.mean()}")
        print(f"global residual force: {global_residual}")
        print(f"local mean residual force: {local_residual}")
        
    if global_residual.abs().max() < 1e-3:
        break

print(f"converged after {i} iterations")
print(f"displacement mean delta {u_diff.mean()}")
print(f"global residual force: {global_residual}")
print(f"local mean residual force: {local_residual}")
print(f"total reaction: {total_reaction}")
print(f"total load: {total_load}")

ts.plot.plot_pyg_data(data.coords + data.u * 100, data.edge_index, 'fem_truss_test')
plt.show()