import torch
from torch_geometric.transforms import BaseTransform


class AddGaussianXYZNoise(BaseTransform):
    r"""
    Applies independent Gaussian multiplicative noise to the x-, y-, and z-coordinate of each node.

    Args:
        std (float): The standard deviation of the noise coefficient.

    Example:
        >>> transform = AddGaussianXYZNoise(std=0.1)
        >>> data = transform(data)
    """

    def __init__(self, std=0.1):
        super().__init__()
        self.std = std

    def forward(self, data):
        if not hasattr(data, "coords"):
            raise AttributeError("Data object has no attribute 'coords'.")

        device = data.coords.device

        x_noise = torch.randn(data.num_nodes, device=device) * self.std
        y_noise = torch.randn(data.num_nodes, device=device) * self.std
        z_noise = torch.randn(data.num_nodes, device=device) * self.std
        data.coords[:, 0] = (1 + x_noise) * data.coords[:, 0]
        data.coords[:, 1] = (1 + y_noise) * data.coords[:, 1]
        data.coords[:, 2] = (1 + z_noise) * data.coords[:, 2]

        return data

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(max_noise={self.max_noise}, "
            f"n_eigenvectors={self.n_eigenvectors})"
        )
