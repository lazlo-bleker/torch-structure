from torch_structure.plot.plot import _draw, _draw_xz


class Plotter:
    """Render a structure data object as a matplotlib figure.

    The plotter is stateless: construct it once and call :meth:`plot` (3D) or
    :meth:`plot_xz` (XZ projection) with the data object to draw::

        plotter = Plotter()
        plotter.plot(data, title="My Structure", legend=False)

    ``data`` is expected to be a :class:`~torch_structure.data.StructData` (or
    any object exposing ``_resolve_override`` from
    :class:`~torch_structure.mixins.OverrideResolveMixin`). The fields
    ``coords``, ``edge_index``, ``is_support``, ``force`` and ``load`` are read
    from ``data`` by attribute name. Each can be overridden per call by passing
    a keyword argument:

    - a tensor / array is used as-is,
    - a string names a different attribute on ``data`` to read instead,
    - ``None`` (the default) reads ``data.<field>``.

    ``coords`` and ``edge_index`` are required; the other three are optional and
    resolve to ``None`` when absent from ``data``.

    Any remaining keyword arguments are forwarded to the underlying drawing
    routine (``show``, ``path``, ``title``, ``legend``, ``show_supports``,
    ``show_load``, ``force_scale`` ...).
    """

    _FIELDS = ("coords", "edge_index", "is_support", "force", "load")
    _REQUIRED = ("coords", "edge_index")

    def plot(self, data, **kwargs):
        """Draw ``data`` in 3D. See the class docstring for keyword handling."""
        fields = self._resolve_fields(data, kwargs)
        return _draw(**fields, **kwargs)

    def plot_xz(self, data, **kwargs):
        """Draw ``data`` projected onto the XZ plane."""
        fields = self._resolve_fields(data, kwargs)
        return _draw_xz(**fields, **kwargs)

    def _resolve_fields(self, data, kwargs):
        return {
            name: data._resolve_override(
                name, kwargs.pop(name, None), required=name in self._REQUIRED
            )
            for name in self._FIELDS
        }
