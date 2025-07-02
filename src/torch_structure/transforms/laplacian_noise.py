import torch
import torch_geometric
from torch_geometric.transforms import BaseTransform

class AddLaplacianZNoise(BaseTransform):
    r"""
    Applies Laplacian-eigenvector-modulated multiplicative noise to the
    z-coordinate of each node.

    This transform computes the mean of the Laplacian eigenvectors for each
    node, rescales it to lie in the range ``[-max_noise, max_noise]``, and
    multiplies the node's z-coordinate accordingly.

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
    def __init__(self, max_noise, n_eigenvectors=20):
        super().__init__()
        self.max_noise = max_noise
        self.n_eigenvectors = n_eigenvectors
        self._add_lpe = torch_geometric.transforms.AddLaplacianEigenvectorPE(
                self.n_eigenvectors,
                attr_name='laplacian_pe',
                is_undirected=False
            )

    def __call__(self, data):
        if not hasattr(data, "coords"):
            raise AttributeError("Data object has no attribute 'coords'.")
        
        device = data.coords.device

        # Ensure data has the specified number of eigenvectors
        compute_lpe = (not hasattr(data, "laplacian_pe")
                       or data.laplacian_pe.size(1) < self.n_eigenvectors)
        if compute_lpe:
            lpe_data = self._add_lpe(data).to(device)

        laplacian_pe_mean = torch.mean(lpe_data.laplacian_pe, dim=1)
        noise = laplacian_pe_mean * (self.max_noise / laplacian_pe_mean.abs().max())
        data.coords[:, 2] = (1 + noise) * data.coords[:, 2]
        return data
    
    def __repr__(self):
        return (f"{self.__class__.__name__}(max_noise={self.max_noise}, "
                f"n_eigenvectors={self.n_eigenvectors})")
