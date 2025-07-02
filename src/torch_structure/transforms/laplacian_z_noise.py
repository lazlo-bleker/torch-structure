import torch
import torch_geometric
from torch_geometric.transforms import BaseTransform

class AddLaplacianZNoise(BaseTransform):
    r"""
    Applies Laplacian-eigenvector-modulated multiplicative noise to the
    z-coordinate of each node.

    This transform samples a linear combination of Laplacian eigenvectors, 
    rescales it to lie in the range ``[-max_noise, max_noise]``, and
    multiplies each node's z-coordinate with this noise coefficient.

    If the Laplacian positional encoding is missing or has fewer dimensions
    than requested, it will be recomputed.

    Args:
        max_noise (float): The maximum relative deviation applied to the
            z-coordinate. A value of 0.15 means ±15% scaling.
        n_eigenvectors (int, optional): Number of Laplacian eigenvectors to
            compute. Defaults to 20.

    Example:
        >>> transform = AddLaplacianZNoise(max_noise=0.1)
        >>> data = transform(data)
    """
    def __init__(self, max_noise, n_eigenvectors=20, keep_lpe=False):
        super().__init__()
        self.max_noise = max_noise
        self.n_eigenvectors = n_eigenvectors
        self.keep_lpe = keep_lpe
        self._add_lpe = torch_geometric.transforms.AddLaplacianEigenvectorPE(
                self.n_eigenvectors,
                attr_name='laplacian_pe',
                is_undirected=False
            )

    def forward(self, data):
        if not hasattr(data, "coords"):
            raise AttributeError("Data object has no attribute 'coords'.")
        
        device = data.coords.device

        # Ensure data has the specified number of eigenvectors
        compute_lpe = (not hasattr(data, "laplacian_pe")
                       or data.laplacian_pe.size(1) < self.n_eigenvectors)
        if compute_lpe:
            self._add_lpe(data).to(device)

        coefficients = torch.randn(self.n_eigenvectors, device=device)
        laplacian_sum = torch.matmul(data.laplacian_pe, coefficients)
        noise = laplacian_sum * (self.max_noise / laplacian_sum.abs().max())
        data.coords[:, 2] = (1 + noise) * data.coords[:, 2]

        if not self.keep_lpe and compute_lpe:
            del data.data.laplacian_pe
        
        return data
    
    def __repr__(self):
        return (f"{self.__class__.__name__}(max_noise={self.max_noise}, "
                f"n_eigenvectors={self.n_eigenvectors})")
