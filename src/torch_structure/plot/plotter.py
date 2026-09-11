from torch_structure.plot.plot import plot_3d, plot_xz


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
        """
        Draw `data` in 3D.

        Args:
            data: Structure data object to draw. Must expose
                `_resolve_override` (see the class docstring) and the
                attributes needed to resolve `coords`, `edge_index`, and
                optionally `is_support`, `force`, `load`.
            **kwargs: Field overrides (`coords`, `edge_index`, `is_support`,
                `force`, `load` — see the class docstring) plus any drawing
                option accepted by :func:`torch_structure.plot.plot.plot_3d`
                (`title`, `legend`, `show_supports`, `show_load`,
                `force_scale`, `path`, `show`, ...).

        Returns:
            None
        """
        fields = self._resolve_fields(data, kwargs)
        return plot_3d(**fields, **kwargs)

    def plot_xz(self, data, **kwargs):
        """
        Draw `data` projected onto the XZ plane.

        Args:
            data: Structure data object to draw. Must expose
                `_resolve_override` (see the class docstring) and the
                attributes needed to resolve `coords`, `edge_index`, and
                optionally `is_support`, `force`, `load`.
            **kwargs: Field overrides (`coords`, `edge_index`, `is_support`,
                `force`, `load` — see the class docstring) plus any drawing
                option accepted by :func:`torch_structure.plot.plot.plot_xz`
                (`title`, `legend`, `show_supports`, `show_load`,
                `force_scale`, `path`, `show`, `highlight_nodes`, ...).

        Returns:
            None
        """
        fields = self._resolve_fields(data, kwargs)
        return plot_xz(**fields, **kwargs)

    def _resolve_fields(self, data, kwargs):
        """
        Resolve the coords/edge_index/is_support/force/load fields for one call.

        Args:
            data: Structure data object to read fields from.
            kwargs (dict): The keyword arguments passed to :meth:`plot` /
                :meth:`plot_xz`. Any of `coords`, `edge_index`, `is_support`,
                `force`, `load` present here are popped out (mutating
                `kwargs` in place) and resolved; everything else is left for
                the caller to forward on to the drawing function.

        Returns:
            dict: Mapping from each field name in `_FIELDS` to its resolved value.
        """
        return {
            name: data._resolve_override(
                name, kwargs.pop(name, None), required=name in self._REQUIRED
            )
            for name in self._FIELDS
        }
