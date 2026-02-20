from torch_structure.formfinding import fdm
from torch_structure.mixins.utils import OverrideResolveMixin

class FDMMixin(OverrideResolveMixin):
    def fdm(
        self,
        inplace: bool=False,
        coords=None,
        load=None,
        support=None,
        edge_index=None,
        force_density=None,
        use_batching=False,
        solve_only_z=False,
        C=None,
        coords_out=None,
        force_out=None,
    ):
        """
        Applies the Force Density Method (FDM) to this structure and updates `coords`
        and `force`.

        See [`torch_structure.formfinding.fdm`](../formfinding/fdm.md).
        """
        batch = getattr(self, "batch", None)

        # Node inputs
        coords = self._resolve_override("coords", coords)
        load = self._resolve_override("load", load)
        support = self._resolve_override("support", support)

        # Edge inputs
        edge_mask = self.directed_mask.view(-1)
        edge_index = self._resolve_override("directed_edge_index", edge_index)
        force_density = self._resolve_override("force_density", force_density, mask=edge_mask)

        # FDM computation
        new_coords, new_directed_force = fdm(
            coords,
            load,
            support,
            edge_index,
            force_density,
            batch=batch,
            use_batching=use_batching,
            directed=True,
            solve_only_z=solve_only_z,
            C=C,
        )
        new_force = self.edge_attr_to_undirected(new_directed_force, edge_mask)

        # Update data object
        new_data = self if inplace else self.clone()

        if coords_out is None:
            new_data.coords = new_coords
        else:
            new_data[coords_out] = new_coords

        if force_out is None:
            new_data.force = new_force
        else:
            new_data[force_out] = new_force

        if not inplace:
            return new_data
