from torch_structure.formfinding import tna

class TNAMixin:
    def tna(
        self,
        inplace: bool=False,
        coords=None,
        is_support=None,
        load=None,
        edge_index=None,
        q_target=None,
        verbose=False,
    ):
        """
        Applies the Thrust Network Analysis (TNA) to this structure and updates `coords`
        , `force` and `force_density`.

        See [`torch_structure.formfinding.tna`](../formfinding/tna.md).
        """
        # batch = getattr(self, "batch", None) # todo: add batching support

        # Node inputs
        if coords is None:
            coords = self.coords
        if is_support is None:
            is_support = self.is_support
        if load is None:
            load = self.load

        # Edge inputs
        edge_mask = self.directed_mask.view(-1)
        if edge_index is None:
            edge_index = self.edge_index[:, edge_mask]
        if q_target is None:
            q_target = self.q_target[edge_mask]

        # TNA computation
        new_coords, new_directed_force, new_directed_force_density = tna(
            coords,
            is_support,
            load,
            edge_index,
            q_target,
            verbose=verbose,
        )
        new_force = self.edge_attr_to_undirected(new_directed_force, edge_mask)
        new_force_density = self.edge_attr_to_undirected(new_directed_force_density, edge_mask)

        # Update data object
        new_data = self if inplace else self.clone()

        new_data.coords = new_coords
        new_data.force = new_force
        new_data.force_density = new_force_density

        if not inplace:
            return new_data
