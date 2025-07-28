from torch_geometric.transforms import BaseTransform


class ScaleZ(BaseTransform):
    r"""
    Applies a scaling transformation along the z-axis of the coordinates.
    Args:
        scaling_factor (float): The factor by which to scale the z-coordinates.

    Example:
        >>> transform = ScaleZ(scaling_factor=2.0)
        >>> data = transform(data)
    """

    def __init__(self, scaling_factor):
        super().__init__()
        self.scaling_factor = scaling_factor

    def forward(self, data):
        if not hasattr(data, "coords"):
            raise AttributeError("Data object has no attribute 'coords'.")
        if not hasattr(data, "force_density"):
            raise AttributeError("Data object has no attribute 'force_density'.")
        if not hasattr(data, "force"):
            raise AttributeError("Data object has no attribute 'force'.")

        data.coords[:, 2] *= self.scaling_factor
        data.force_density *= 1 / self.scaling_factor
        data.length = data.length_from_coords
        data.force = data.length * data.force_density

        return data

    def __repr__(self):
        return f"{self.__class__.__name__}(scaling_factor={self.scaling_factor})"
