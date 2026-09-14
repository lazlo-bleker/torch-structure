import torch
from torch_geometric.transforms import BaseTransform


class AddGaussianZNoise(BaseTransform):
    r"""
    Applies independent Gaussian multiplicative noise to the z-coordinate of each node.

    Args:
        std (float): The standard deviation of the noise coefficient.

    Example:
        >>> transform = AddGaussianZNoise(std=0.1)
        >>> data = transform(data)
    """

    def __init__(self, std=0.1):
        super().__init__()
        self.std = std

    def forward(self, data):
        """Apply the multiplicative Gaussian z-noise to ``data.coords`` in place."""
        if not hasattr(data, "coords"):
            raise AttributeError("Data object has no attribute 'coords'.")

        device = data.coords.device

        noise = torch.randn(data.num_nodes, device=device) * self.std
        data.coords[:, 2] = (1 + noise) * data.coords[:, 2]

        return data

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(max_noise={self.max_noise}, "
            f"n_eigenvectors={self.n_eigenvectors})"
        )
