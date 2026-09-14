import torch
import torch_geometric
from torch_geometric.transforms import BaseTransform


class AddRadialZNoise(BaseTransform):
    r"""
    Applies radial multiplicative noise to the z-coordinate of each node.

    This transform samples an n-degree polynomial, rescales it to lie in
    the range ``[-max_noise, max_noise]`` in the domain [0, 1], and
    multiplies each node's z-coordinate with this noise coefficient.

    Args:
        max_noise (float): The maximum relative deviation applied to the
            z-coordinate. A value of 0.15 means ±15% scaling.
        degree (int, optional): Degree of the polynomial. Defaults to 5.

    Example:
        >>> transform = AddRadialZNoise(max_noise=0.1)
        >>> data = transform(data)
    """

    def __init__(self, max_noise, degree=5):
        super().__init__()
        self.max_noise = max_noise
        self.degree = degree

    def forward(self, data):
        """Apply the radial polynomial z-noise to ``data.coords`` in place."""
        if not hasattr(data, "coords"):
            raise AttributeError("Data object has no attribute 'coords'.")

        device = data.coords.device

        # Determine normalized radius in the xy-plane
        radius = torch.linalg.norm(data.coords[:, :2], dim=1)  # [N]
        max_radius = radius.max()
        t = radius / max_radius if max_radius > 0 else torch.zeros_like(radius)

        # Define polynomial
        coefficients = torch.randn(self.degree + 1, device=device)
        powers = torch.vander(t, N=self.degree + 1, increasing=True)
        polynomial = powers @ coefficients

        noise = polynomial * (self.max_noise / polynomial.abs().max())
        data.coords[:, 2] = (1 + noise) * data.coords[:, 2]

        return data

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(max_noise={self.max_noise}, "
            f"degree={self.degree})"
        )
