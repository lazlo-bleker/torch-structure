from torch_structure.plot import plot_data
from torch_structure.mixins.utils import OverrideResolveMixin

class PlotMixin(OverrideResolveMixin):
    """Adds a plotting method to a data class."""

    def plot(
        self,
        coords=None,
        edge_index=None,
        is_support=None,
        force=None,
        load=None,
        **kwargs
    ):
        """Plot this structure's geometry, supports, forces, and loads.

        See [`torch_structure.plot.plot_data`](../plot/plot.md).
        """
        coords = self._resolve_override("coords", coords)
        edge_index = self._resolve_override("edge_index", edge_index)
        is_support = self._resolve_override("is_support", is_support, required=False)
        force = self._resolve_override("force", force, required=False)
        load = self._resolve_override("load", load, required=False)

        return plot_data(
            coords=coords,
            edge_index=edge_index,
            is_support=is_support,
            force=force,
            load=load,
            **kwargs
        )