from torch_structure.formfinding import laplacian_smoothing

class LaplacianSmoothingMixin:
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
        if is_fixed is None:
            is_fixed = self.is_fixed

        # Edge inputs
        if edge_index is None:
            edge_index = self.edge_index

        # TNA computation
        new_coords, _, _ = laplacian_smoothing(
            coords,
            is_fixed,
            edge_index,
            tolerance,
            max_iter,
            verbose,
        )

        # Update data object
        new_data = self if inplace else self.clone()

        new_data.coords[:, :2] = new_coords

        if not inplace:
            return new_data