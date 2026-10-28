import warnings

from torch_structure.plot import plot_structure


# keyword arguments of the earlier plot_data-based StructData.plot, and how they translate
_LEGACY_KWARGS = {
    "show_supports": "supports",
    "show_load": "loads",
    "target_coords": "ghost",
}
_DROPPED_LEGACY_KWARGS = ("target_edge_index", "target_color", "target_node_size", "support_marker_size",
                          "support_marker_offset", "edge_index", "is_support", "force")


def _translate_legacy_kwargs(kwargs):
    legacy = [key for key in kwargs if key in _LEGACY_KWARGS or key in _DROPPED_LEGACY_KWARGS
              or key in ("load", "show_edge_indices", "lw_constant", "force_scale")]
    if not legacy:
        return kwargs
    warnings.warn(
        f"StructData.plot no longer takes {', '.join(repr(k) for k in legacy)}; see plot_structure for its "
        "arguments, e.g. supports=, loads=, ghost=, edge_labels='index', edge_width=, vector_scale=.",
        FutureWarning,
        stacklevel=3,
    )
    kwargs = dict(kwargs)
    for old, new in _LEGACY_KWARGS.items():
        if old in kwargs:
            kwargs.setdefault(new, kwargs.pop(old))
    # a load tensor was only drawn with show_load
    load = kwargs.pop("load", None)
    if kwargs.get("loads") is True and load is not None and load is not True:
        kwargs["loads"] = load
    if kwargs.pop("show_edge_indices", False):
        kwargs.setdefault("edge_labels", "index")
    if kwargs.pop("lw_constant", False):
        kwargs.setdefault("edge_width", 1.5)
    if "force_scale" in kwargs:
        # the old load arrows were drawn 1.5 times as long as force_scale * load
        kwargs.setdefault("vector_scale", 1.5 * kwargs.pop("force_scale"))
    for key in _DROPPED_LEGACY_KWARGS:
        kwargs.pop(key, None)
    return kwargs


class PlotMixin:
    """Adds a plotting method to a data class."""

    def plot(self, coords=None, **kwargs):
        """Plot this structure, with its edges and nodes colored, sized and labelled by any attribute.

        For example ``data.plot(edge_color="length", node_labels="name", loads=True)``,
        ``data.plot(view="xz")`` for an elevation, or ``data.plot(pattern=True)`` to show the
        mirror planes, rotation axes and translations of its patterns. See
        [`torch_structure.plot.plot_structure`](../plot/structure.md) for all arguments.
        """
        return plot_structure(self, coords=coords, **_translate_legacy_kwargs(kwargs))
