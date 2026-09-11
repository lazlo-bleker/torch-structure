import torch_structure as ts
import matplotlib.pyplot as plt
import torch.nn as nn
import torch

from torch_structure.data import StructData

# Parameters 
n_iters = 400
lr = 1e-1

# Define bestfit model
class BestFitModel(nn.Module):
    def __init__(self, struct_data : StructData, target_coords, w_global=.0, w_local=1.0):
        super().__init__()
        self.struct_data = struct_data
        self.loss = nn.MSELoss()
        self.target_coords = target_coords
        self.target_laplace_coords = struct_data.laplacian_coordinates(target_coords)
        self.w_global = w_global
        self.w_local = w_local

    def forward(self, force_density):
        tmp_struct_data = self.struct_data.detach().clone()
        tmp_struct_data.force_density[tmp_struct_data.directed_mask.squeeze(1)] = force_density
        tmp_struct_data.force_density[~tmp_struct_data.directed_mask.squeeze(1)] = force_density
        tmp_struct_data.fdm(inplace=True)

        # Global coordinates
        loss_global = self.w_global * self.loss(tmp_struct_data.coords, self.target_coords)

        # Local Laplacian coordinates
        laplace_coords = tmp_struct_data.laplacian_coordinates()
        loss_local = self.w_local * self.loss(laplace_coords, self.target_laplace_coords)

        return loss_global + loss_local

# Create instance of StructData
generator = ts.generators.CableNetGenerator(
    n=6,
    seed=1, 
    unsupported_boundaries = False, 
    q_field = -3.0,
    q_boundary = -10.0,
    q_diagonal = -5.0,
    )

struct_data = generator()

# Prescribe two handles; structural supports remain fixed automatically.
handle_centers = torch.tensor([[0.0, -0.3], [0.0, 0.0], [0.0, 0.3]], device=struct_data.coords.device)
handle_indices = torch.stack(
    [
        torch.argmin(torch.linalg.vector_norm(struct_data.coords[:, :2] - center, dim=1))
        for center in handle_centers
    ]
)
handle_displacements = torch.zeros(
    (handle_indices.numel(), struct_data.coords.size(1)),
    device=struct_data.coords.device,
)
handle_displacements[:, 2] = torch.tensor([0.2,-0.1,0.2])
target_coords = struct_data.laplacian_surface_edit(
    handle_indices,
    handle_displacements,
    handle_weight=1e4,
)

# Plot initial structure and target graph together.
ax_init = struct_data.plot(
    title="Initial Structure and Target",
    show_supports=True,
    target_coords=target_coords,
)

# Start model
model = BestFitModel(struct_data, target_coords)
params = nn.Parameter(struct_data.force_density[struct_data.directed_mask.squeeze(1)])
optimizer = torch.optim.Adam([params], lr=lr)

# Optimize
for i in range(n_iters):
    optimizer.zero_grad()
    loss = model(params)

    loss.backward()
    optimizer.step()

    print(f"loss iter {i:04d}: \t{loss.item():4e}")


# Plot final
struct_data.force_density[struct_data.directed_mask.squeeze(1)] = params.detach().clone()
struct_data.force_density[~struct_data.directed_mask.squeeze(1)] = params.detach().clone()
struct_data.fdm(inplace=True)
struct_data.plot(
    title="Best-Fit Structure and Target",
    show_supports=True,
    target_coords=target_coords,
)
plt.show()
print("f")