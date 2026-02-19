from torch_structure.formfinding import mpcem_algorithm, cem_algorithm, seq_cem_algorithm

class CEMMixin:
    def mpcem(
        self,
        inplace: bool=False,
        coords=None,
        load=None,
        is_support=None,
        is_origin_node=None,
        cem_edge_index=None,
        is_trail_edge=None,
        length=None,
        force_sign=None,
        force=None,
        constraint_plane=None,
        max_iter=100,
        tolerance=1e-5,
        damping_factor=0.5,
        verbose=False,
        track_history=False,
        callback=None,
    ):
        """
        Applies the Force Density Method (FDM) to this structure and updates `coords`
        and `force`.

        See [`torch_structure.formfinding.fdm`](../formfinding/fdm.md).
        """
        # Node inputs
        if coords is None:
            coords = self.coords
        if load is None:
            load = self.load
        if is_support is None:
            is_support = self.is_support
        if is_origin_node is None:
            is_origin_node = self.is_origin_node
        if constraint_plane is None and hasattr(self, "constraint_plane"):
            constraint_plane = self.constraint_plane

        # Edge inputs
        edge_mask = self.cem_edge_mask
        if cem_edge_index is None:
            cem_edge_index = self.cem_edge_index
        if is_trail_edge is None:
            is_trail_edge = self.is_trail_edge[edge_mask]
        if length is None:
            length = self.length[edge_mask]
        if force_sign is None:
            force_sign = self.force_sign[edge_mask]
        if force is None:
            force = self.force[edge_mask]

        # MP-CEM computation
        new_coords, new_semi_directed_force, new_reaction_force, new_load = mpcem_algorithm(
            coords,
            load,
            is_support,
            is_origin_node,
            cem_edge_index,
            is_trail_edge,
            length,
            force_sign,
            force,
            constraint_plane,
            max_iter=max_iter,
            tolerance=tolerance,
            damping_factor=damping_factor,
            verbose=verbose,
            track_history=track_history,
            callback=callback,
        )
        new_force = self.edge_attr_to_undirected(
            new_semi_directed_force,
            edge_mask,
            batched=track_history)

        # Update data object
        new_data = self if inplace else self.clone()

        if track_history:
            self._track_history("coords", new_coords)
            new_data.coords = new_coords[-1]
            self._track_history("load", new_load)
            new_data.load = new_load[-1]
            self._track_history("force", new_force)
            new_data.force = new_force[-1]
        else:
            new_data.coords = new_coords
            new_data.load = new_load
            new_data.force = new_force

        new_data.reaction_force = new_reaction_force

        if not inplace:
            return new_data
        
    def cem(
        self,
        inplace: bool=False,
        coords=None,
        load=None,
        is_support=None,
        is_origin_node=None,
        cem_edge_index=None,
        is_trail_edge=None,
        length=None,
        force_sign=None,
        force=None,
        sequence=None,
        max_iter=100,
        tolerance=1e-5,
        damping_factor=0.0,
        enhanced_first_iteration=False,
        verbose=False,
        track_history=False,
    ):
        # Node inputs
        if coords is None:
            coords = self.coords
        if load is None:
            load = self.load
        if is_support is None:
            is_support = self.is_support
        if is_origin_node is None:
            is_origin_node = self.is_origin_node
        if sequence is None:
            sequence = self.sequence

        # Edge inputs
        edge_mask = self.cem_edge_mask
        if cem_edge_index is None:
            cem_edge_index = self.cem_edge_index
        if is_trail_edge is None:
            is_trail_edge = self.is_trail_edge[edge_mask]
        if length is None:
            length = self.length[edge_mask]
        if force_sign is None:
            force_sign = self.force_sign[edge_mask]
        if force is None:
            force = self.force[edge_mask]

        # CEM computation
        new_coords, new_semi_directed_force, new_reaction_force = cem_algorithm(
            coords,
            load,
            is_support,
            is_origin_node,
            cem_edge_index,
            is_trail_edge,
            length,
            force_sign,
            force,
            sequence,
            max_iter=max_iter,
            tolerance=tolerance,
            damping_factor=damping_factor,
            enhanced_first_iteration=enhanced_first_iteration,
            verbose=verbose,
            track_history=track_history,
        )
        new_force = self.edge_attr_to_undirected(
            new_semi_directed_force,
            edge_mask,
            batched=track_history)

        # Update data object
        new_data = self if inplace else self.clone()

        if track_history:
            self._track_history("coords", new_coords)
            new_data.coords = new_coords[-1]
            self._track_history("force", new_force)
            new_data.force = new_force[-1]
        else:
            new_data.coords = new_coords
            new_data.force = new_force

        new_data.reaction_force = new_reaction_force

        if not inplace:
            return new_data

    def seqcem(
        self,
        inplace: bool=False,
        coords=None,
        load=None,
        is_support=None,
        is_origin_node=None,
        cem_edge_index=None,
        is_trail_edge=None,
        length=None,
        force_sign=None,
        force=None,
        sequence=None,
        max_iter=100,
        tolerance=1e-5,
        damping_factor=0.0,
        enhanced_first_iteration=False,
        verbose=False,
        track_history=False,
    ):
        # Node inputs
        if coords is None:
            coords = self.coords
        if load is None:
            load = self.load
        if is_support is None:
            is_support = self.is_support
        if is_origin_node is None:
            is_origin_node = self.is_origin_node
        if sequence is None:
            sequence = self.sequence

        # Edge inputs
        edge_mask = self.cem_edge_mask
        if cem_edge_index is None:
            cem_edge_index = self.cem_edge_index
        if is_trail_edge is None:
            is_trail_edge = self.is_trail_edge[edge_mask]
        if length is None:
            length = self.length[edge_mask]
        if force_sign is None:
            force_sign = self.force_sign[edge_mask]
        if force is None:
            force = self.force[edge_mask]

        # Sequential CEM computation
        new_coords, new_semi_directed_force, new_reaction_force = seq_cem_algorithm(
            coords,
            load,
            is_support,
            is_origin_node,
            cem_edge_index,
            is_trail_edge,
            length,
            force_sign,
            force,
            sequence,
            max_iter=max_iter,
            tolerance=tolerance,
            damping_factor=damping_factor,
            enhanced_first_iteration=enhanced_first_iteration,
            verbose=verbose,
            track_history=track_history,
        )
        new_force = self.edge_attr_to_undirected(
            new_semi_directed_force,
            edge_mask,
            batched=track_history)

        # Update data object
        new_data = self if inplace else self.clone()

        if track_history:
            self._track_history("coords", new_coords)
            new_data.coords = new_coords[-1]
            self._track_history("force", new_force)
            new_data.force = new_force[-1]
        else:
            new_data.coords = new_coords
            new_data.force = new_force

        new_data.reaction_force = new_reaction_force

        if not inplace:
            return new_data