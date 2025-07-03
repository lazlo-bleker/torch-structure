import torch
from torch_structure.transforms.laplacian_z_noise import AddLaplacianZNoise

def test_add_laplacian_z_noise(dome_data):
    data = dome_data
    coords_before = data.coords.clone()
    transform = AddLaplacianZNoise(max_noise=0.2, n_eigenvectors=3)
    data_out = transform(data)

    # Check that coords shape is unchanged
    assert data_out.coords.shape == coords_before.shape
    # Check that z-coordinates have changed
    assert not torch.allclose(data_out.coords[:, 2], coords_before[:, 2])
    # Check that x and y coordinates are unchanged
    assert torch.allclose(data_out.coords[:, :2], coords_before[:, :2])
