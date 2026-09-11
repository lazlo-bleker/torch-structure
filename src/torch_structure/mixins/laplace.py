import torch

from torch_structure.message_passing.laplace import Laplacian
from torch_structure.formfinding import laplacian_smoothing
from torch_structure.mixins.utils import OverrideResolveMixin


class LaplacianMixin(OverrideResolveMixin):
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

    def laplacian_surface_edit(
        self,
        node_indices=None,
        displacements=None,
        coords=None,
        is_support=None,
        inplace=False,
        node_coords=None,
        handle_weight=None,
    ):
        """Prescribe displacements and reconstruct the remaining surface.

        With hard handles, free nodes solve ``L @ edited_coords =
        L @ coords``. With ``handle_weight`` set, supports remain hard
        constraints and handles are added as weighted least-squares terms.
        """
        coords = self.coords if coords is None else coords
        is_support = self._resolve_override("is_support", is_support)
        if (node_indices is None) == (node_coords is None):
            raise ValueError("Provide exactly one of node_indices or node_coords")
        if displacements is None:
            raise ValueError("displacements is required")
        if node_coords is not None:
            node_coords = torch.as_tensor(node_coords, device=coords.device, dtype=coords.dtype)
            if node_coords.ndim == 1:
                node_coords = node_coords.unsqueeze(0)
            node_indices = torch.cdist(node_coords, coords).argmin(dim=1)
        node_indices = torch.as_tensor(
            node_indices, device=coords.device, dtype=torch.long
        ).view(-1)
        displacements = torch.as_tensor(
            displacements, device=coords.device, dtype=coords.dtype
        )
        if displacements.ndim == 1:
            displacements = displacements.unsqueeze(0)
        expected_shape = (node_indices.numel(), coords.size(1))
        if displacements.shape != expected_shape:
            raise ValueError(f"displacements must have shape {expected_shape}")

        support_indices = torch.where(is_support.view(-1).to(coords.device))[0]
        if torch.isin(node_indices, support_indices).any():
            raise ValueError("A prescribed handle cannot also be a supported node")

        matrix = self.laplacian.to(coords.device).dense_matrix().to(coords.dtype)
        target = coords[node_indices] + displacements
        fixed = torch.zeros(coords.size(0), dtype=torch.bool, device=coords.device)
        fixed[support_indices] = True
        variable = ~fixed
        differential_target = self.laplacian_coordinates(coords)
        rhs = differential_target[variable]
        if support_indices.numel():
            rhs = rhs - matrix[variable][:, support_indices] @ coords[support_indices]

        if handle_weight is None:
            constrained = torch.unique(torch.cat((support_indices, node_indices)))
            values = coords[constrained].clone()
            selected_positions = torch.searchsorted(constrained, node_indices)
            values[selected_positions] = target
            free = ~fixed
            free[ node_indices] = False
            edited = coords.clone()
            edited[constrained] = values
            if free.any():
                rhs_hard = differential_target[free]
                rhs_hard = rhs_hard - matrix[free][:, constrained] @ values
                edited[free] = torch.linalg.lstsq(
                    matrix[free][:, free], rhs_hard
                ).solution
        else:
            if handle_weight <= 0:
                raise ValueError("handle_weight must be positive")
            variable_indices = torch.where(variable)[0]
            handle_positions = torch.searchsorted(variable_indices, node_indices)
            penalty = torch.zeros(
                (node_indices.numel(), variable_indices.numel()),
                device=coords.device,
                dtype=coords.dtype,
            )
            penalty[torch.arange(node_indices.numel(), device=coords.device), handle_positions] = (
                handle_weight**0.5
            )
            system = torch.cat((matrix[variable][:, variable], penalty), dim=0)

            penalty_rhs = target * handle_weight**0.5
            rhs = torch.cat((rhs, penalty_rhs), dim=0)
            edited = coords.clone()
            edited[support_indices] = coords[support_indices]
            edited[variable] = torch.linalg.lstsq(system, rhs).solution

        if inplace:
            self.coords = edited
            return self
        return edited

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