from torch_structure.message_passing.laplacian_smooth import Laplacian
from torch_structure.formfinding import laplacian_smoothing
from torch_structure.mixins.utils import OverrideResolveMixin


class LaplacianSmoothingMixin(OverrideResolveMixin):
    @property
    def laplacian(self):
        """Return the cached matrix-free Laplacian for this topology."""
        edge_index = self.edge_index
        cache = getattr(self, "_laplacian_cache", None)
        topology_version = (id(edge_index), edge_index._version, edge_index.device)

        if cache is None or cache[0] != topology_version:
            cache = (
                topology_version,
                Laplacian(edge_index, num_nodes=self.num_nodes),
            )
            self._laplacian_cache = cache

        return cache[1]

    def laplacian_coordinates(self, x=None):
        """Compute matrix-free uniform Laplacian coordinates ``Lx``."""
        x = self.coords if x is None else x
        return self.laplacian(x)

    def xy_laplacian_smoothing(
        self,
        inplace: bool=False,
        coords=None,
        is_fixed=None,
        edge_index=None,
        tolerance=1e-7,
        max_iter=10000,
        verbose=False,
    ):
        # Node inputs
        if coords is None:
            coords = self.coords[:, :2]  # Only use x and y coordinates
        is_fixed = self._resolve_override("is_fixed", is_fixed)

        # Edge inputs
        edge_index = self._resolve_override("edge_index", edge_index)

        # TNA computation
        new_coords, _, _ = laplacian_smoothing(
            coords,
            is_fixed,
            edge_index,
            tolerance,
            max_iter,
            verbose,
            laplacian=self.laplacian,
        )

        # Update data object
        new_data = self if inplace else self.clone()

        new_data.coords[:, :2] = new_coords

        if not inplace:
            return new_data