import matplotlib.pyplot as plt
import torch_geometric as pyg
import torch

import torch_structure as ts


# Generate topology
input = {
    'n': 5,
    'boundary_density': 5,
    'diagonals': False,
    'centroid_support': True,
    'pattern': 'opening',
    'unsupported_boundaries': True,
    'curve_boundaries': True,
    'rectangle': False,
    'square': False
}
shell = ts.generators.GridShell(**input)
shell.plot()
plt.show()

# Laplacian smoothing
data = shell.pyg_data()
undirected_edge_index =  pyg.utils.to_undirected(data.edge_index)
laplace_update = ts.gmp.LaplacianSmoothing(damping_factor=0.5)
iterations = 0
delta_norm = 1.0
while delta_norm > 1e-7:
    updated_coords = laplace_update(data.coords, undirected_edge_index)[~data.is_boundary]
    delta = data.coords[~data.is_boundary] - updated_coords
    delta_norm = torch.norm(delta)
    data.coords[~data.is_boundary] = updated_coords
    iterations += 1
print(f"Laplacian Smoothing Iterations: {iterations}")
ts.plot.plot_pyg_data(data.coords, data.edge_index, 'laplace_test')
plt.show()


# leaste squares force density
q_target = -20.0 * torch.ones(data.edge_index.shape[1])
q_target[data.is_boundary_edge] = -100.0
q_fit = ts.ff.least_squares_tna(data.coords, data.is_support, data.edge_index, q_target)

# random force density
b_vectors = ts.ff.utils.create_xy_equilibrium_space(data.coords, data.is_support, data.edge_index)
c = torch.zeros(b_vectors.shape[1])
print(f"Number of Degrees of Freedom: {b_vectors.shape[1]}")
q_random = torch.mv(b_vectors, c)

# Form-finding z with FDM
data.q = q_fit
print(q_fit)
undirected_edge_index, undirected_q =  pyg.utils.to_undirected(data.edge_index, data.q)
coordinates, force = ts.ff.fdm(data.coords, data.load, data.is_support, undirected_edge_index,
                                undirected_q, directed=False, use_batching=False, solve_only_z=False)

# Plot
ts.plot.plot_pyg_data(coordinates, undirected_edge_index, 'grid_shell_tna', force=force, load=data.load, show_load=False, show_residual_forces=False)
criterion = ts.loss.ResidualForceLoss()
loss = criterion(coordinates, data.load, data.is_support, force, undirected_edge_index)
print(f"Residual Force Loss: {loss:.6f}")
plt.show()

# plt.figure(figsize=(8, 6))
# plt.imshow(basis_vectors, cmap='viridis', aspect='auto')
# plt.colorbar(label='Value')
# plt.title('Basis Vectors Matrix')
# plt.xlabel('Column Index')
# plt.ylabel('Row Index')

# # Save the plot as an image
# plt.savefig('basis_vectors_matrix.png', dpi=300)
