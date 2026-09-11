import torch_structure as ts
import matplotlib.pyplot as plt
import torch.nn as nn
import torch

from torch_structure.data import StructData

# Parameters 
n_iters = 500
lr = 5e-1

# Define bestfit model
class BestFitModel(nn.Module):
    def __init__(self, struct_data : StructData, target_coords, w_global=.0, w_local=1.0):
        super().__init__()
        self.struct_data = struct_data
        self.loss = nn.MSELoss()
        self.targer_coords = target_coords
        self.target_laplace_coords = struct_data.laplacian_coordinates(target_coords)
        self.w_global = w_global
        self.w_local = w_local

    def forward(self, force_density):
        tmp_struct_data = struct_data.detach().clone()
        tmp_struct_data.force_density = force_density
        tmp_struct_data.fdm(inplace=True)

        # Global coordinates
        loss_global = self.w_global * self.loss(tmp_struct_data.coords, target_coords)

        # Local Laplacian coordinates
        laplace_coords = tmp_struct_data.laplacian_coordinates()
        loss_local = self.w_local * self.loss(laplace_coords, self.target_laplace_coords)

        return loss_global + loss_local

# Create instance of StructData
generator = ts.generators.CableNetGenerator(
    n=4,
    seed=1, 
    unsupported_boundaries = False, 
    q_field = -10.0,
    q_boundary = -10.0,
    q_diagonal = -10.0,
    )

# Plot initial
struct_data = generator()
ax_init = struct_data.plot(title="Initial Structure", show_supports = True)

# Start model
# TODO: Find a better way of defining a challenging target geometry
target_coords = struct_data.coords.detach().clone()
center = torch.tensor([0.0, 0.2])
dist = torch.norm(target_coords[:, :2] - center, dim=1)
amplitude = 0.2
sigma = dist.max() / 6.0
bump = amplitude * torch.exp(-(dist**2) / (2 * sigma**2))
target_coords[:, 2] += bump
center = torch.tensor([0.0, -0.2])
dist = torch.norm(target_coords[:, :2] - center, dim=1)
amplitude = -0.2
sigma = dist.max() / 6.0
bump = amplitude * torch.exp(-(dist**2) / (2 * sigma**2))
target_coords[:, 2] += bump

ax_init.scatter(target_coords[:,0], target_coords[:,1], target_coords[:,2], color="grey", s=3.0)

# Start model
model = BestFitModel(struct_data, target_coords)
params = nn.Parameter(struct_data.force_density)
optimizer = torch.optim.Adam([params], lr=lr)

# Optimize
for i in range(n_iters):
    optimizer.zero_grad()
    loss = model(params)

    loss.backward()
    optimizer.step()

    print(f"loss iter {i:04d}: \t{loss.item():4e}")


# Plot final
struct_data.force_density = params.detach().clone()
struct_data.fdm(inplace=True)
ax_final = struct_data.plot(title="Best-Fit Structure", show_supports = True)
ax_final.scatter(target_coords[:,0], target_coords[:,1], target_coords[:,2], color="grey", s=3.0)
plt.show()
print("f")