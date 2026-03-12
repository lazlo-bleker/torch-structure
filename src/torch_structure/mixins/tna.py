from torch_structure.formfinding import tna
from torch_structure.mixins.utils import OverrideResolveMixin


class TNAMixin(OverrideResolveMixin):
    def tna(
        self,
        inplace: bool = False,
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
        coords = self._resolve_override("coords", coords)
        is_support = self._resolve_override("is_support", is_support)
        load = self._resolve_override("load", load)

        # Edge inputs
        edge_mask = self.directed_mask.view(-1)
        edge_index = self._resolve_override("directed_edge_index", edge_index)
        q_target = self._resolve_override("q_target", q_target, mask=edge_mask)

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
        new_force_density = self.edge_attr_to_undirected(
            new_directed_force_density, edge_mask
        )

        # Update data object
        new_data = self if inplace else self.clone()

        new_data.coords = new_coords
        new_data.force = new_force
        new_data.force_density = new_force_density

        if not inplace:
            return new_data
