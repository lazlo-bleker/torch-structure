import matplotlib.pyplot as plt
import torch

from torch_structure.message_passing import ResidualForce
from torch_structure.plot.config import PLOT_CONFIG
from torch_structure.plot.plot import plot_3d, plot_xz

# ResidualForce has no parameters and no expensive init (just aggr="add"), so
# there's nothing to gain from constructing it lazily — one shared instance.
calculate_residual_force = ResidualForce()


def _as_numpy(value):
    """
    Convert a tensor field to a plain numpy array, flattened to 1D vectors
    where relevant (`view(-1)` on the tensor, mirroring `_edge_style`'s
    boolean/scalar-per-edge expectations).

    Args:
        value: The value to convert, or None.

    Returns:
        numpy.ndarray or None: None if `value` is None or not a tensor
        (covers stray non-tensor overrides such as `load=True` passed by
        mistake — treated the same as "not provided"); otherwise the tensor
        detached, moved to cpu, and converted to numpy.
    """
    if not torch.is_tensor(value):
        return None
    return value.detach().cpu().numpy()


def _flatten(array):
    """Flatten a numpy array to 1D, passing None through unchanged."""
    return None if array is None else array.reshape(-1)


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

    ``is_deck_node`` and ``highlight_nodes`` (plot_xz only) are accepted the
    same way as plain keyword arguments — read as-is, no override resolution
    against ``data``.

    Every other drawing setting (``show_supports``, ``show_load``,
    ``force_scale``, ...) comes from
    :data:`torch_structure.plot.config.PLOT_CONFIG` — change it there to
    affect every call. Only ``title`` and ``legend`` remain per-call keyword
    arguments, plus the Plotter-level ``path`` (save the figure) and ``show``
    (``plt.show()`` it) conveniences.
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
                `force`, `load`, `is_deck_node` — see the class docstring),
                `title`, `legend`, and the save/display conveniences `path`
                and `show`.

        Returns:
            None
        """
        path, show = kwargs.pop("path", None), kwargs.pop("show", False)
        plot_data = self._collect_plot_data(data, kwargs)
        plot_3d(plot_data, **kwargs)
        self._finalize(path, show)

    def plot_xz(self, data, **kwargs):
        """
        Draw `data` projected onto the XZ plane.

        Args:
            data: Structure data object to draw. Must expose
                `_resolve_override` (see the class docstring) and the
                attributes needed to resolve `coords`, `edge_index`, and
                optionally `is_support`, `force`, `load`.
            **kwargs: Field overrides (`coords`, `edge_index`, `is_support`,
                `force`, `load`, `highlight_nodes` — see the class docstring),
                `title`, `legend`, and the save/display conveniences `path`
                and `show`.

        Returns:
            None
        """
        path, show = kwargs.pop("path", None), kwargs.pop("show", False)
        plot_data = self._collect_plot_data(data, kwargs)
        plot_xz(plot_data, **kwargs)
        self._finalize(path, show)

    def _collect_plot_data(self, data, kwargs):
        """
        Resolve every drawing input for one call and package it as numpy.

        Args:
            data: Structure data object to read fields from.
            kwargs (dict): The keyword arguments passed to :meth:`plot` /
                :meth:`plot_xz` (with `path` / `show` already popped by the
                caller). `coords`, `edge_index`, `is_support`, `force`,
                `load`, `is_deck_node` and `highlight_nodes` are popped out
                (mutating `kwargs` in place) and resolved; `title` / `legend`
                are left for the caller to forward on to the drawing function.

        Returns:
            dict: `plot_data`, ready to hand to `plot_3d` / `plot_xz` —
            `coords`, `edge_index` (plain numpy arrays), `is_support`,
            `force`, `load`, `is_deck_node` (numpy arrays or None),
            `residual_force` (precomputed numpy array, or None unless both
            `force` and `load` resolved to tensors) and `highlight_nodes`
            (passed through as given, or None).
        """
        resolved = {
            name: data._resolve_override(
                name, kwargs.pop(name, None), required=name in self._REQUIRED
            )
            for name in self._FIELDS
        }
        coords, edge_index = resolved["coords"], resolved["edge_index"]
        force, load = resolved["force"], resolved["load"]
        is_deck_node = kwargs.pop("is_deck_node", None)

        residual_force = None
        if torch.is_tensor(force) and torch.is_tensor(load):
            residual_force = calculate_residual_force(
                coords, force, edge_index, load
            ).detach().cpu().numpy()

        return {
            "coords": coords.detach().cpu().numpy(),
            "edge_index": edge_index.detach().cpu().numpy(),
            "is_support": _flatten(_as_numpy(resolved["is_support"])),
            "force": _flatten(_as_numpy(force)),
            "load": _as_numpy(load),
            "residual_force": residual_force,
            "is_deck_node": _flatten(_as_numpy(is_deck_node)),
            "highlight_nodes": kwargs.pop("highlight_nodes", None),
        }

    def _finalize(self, path, show):
        """
        Save and/or display the figure `plot_3d` / `plot_xz` just drew.

        Args:
            path (str, optional): If given, saves the current figure as a PNG
                to this file path (without extension).
            show (bool): If True, displays the current figure with `plt.show()`.
        """
        if path is None and not show:
            return
        if path is not None:
            plt.gcf().savefig(
                f"{path}{PLOT_CONFIG['save']['file_format']}",
                bbox_inches=PLOT_CONFIG['save']['bbox_inches'],
            )
        if show:
            plt.show()
