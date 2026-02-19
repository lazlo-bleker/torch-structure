from torch_structure.formfinding import fdm

class FDMMixin:
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
        if coords is None:
            coords = self.coords
        if load is None:
            load = self.load
        if support is None:
            support = self.support

        # Edge inputs
        edge_mask = self.directed_mask.view(-1)
        if edge_index is None:
            edge_index = self.edge_index[:, edge_mask]
        if force_density is None:
            force_density = self.force_density[edge_mask]

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
