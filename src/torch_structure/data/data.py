import torch_geometric as pyg
import torch
import warnings
import inspect
import copy
import json
import itertools

from torch_structure.data.view import NodeView
from torch_structure.data.utils import requires_metadata
from torch_structure.loss import ResidualForceLoss
from torch_structure.geometry import graph_edge_lengths
from torch_structure.mixins import TSMixin
from torch_geometric.utils import scatter

class StructData(TSMixin, pyg.data.Data):
    """A `torch_geometric.data.Data` subclass representing a directed, reciprocal-edge structural graph.

    Every undirected edge is stored as a pair of opposite directed edges;
    ``directed_mask`` marks one edge of each pair as the "canonical"
    direction, and ``reciprocal_edge`` maps each edge to its opposite. This
    lets edge attributes be defined once per undirected edge (via
    ``edge_attr_to_undirected``) while message passing still sees both
    directions. Node/edge/graph attributes are declared through
    ``node_attrs``/``edge_attrs``/``graph_attrs``, and nodes carry a
    ``name`` attribute enabling NetworkX-style access via
    [nodes][torch_structure.data.data.StructData.nodes] and name-based
    editing methods such as
    [add_edge][torch_structure.data.data.StructData.add_edge]/[add_edges_by_names][torch_structure.data.data.StructData.add_edges_by_names].
    """

    def __init__(self,
                 edge_index=torch.empty((2, 0), dtype=torch.long),
                 directed_mask=torch.empty((0, 1), dtype=torch.bool),
                 reciprocal_edge=torch.empty((0, 1), dtype=torch.long),
                 node_attrs={},
                 edge_attrs={},
                 graph_attrs={},
                 default_attrs={},
                 **kwargs,
    ):
        
        if "name" in node_attrs:
            
            raise ValueError(
                "'name' is always a default node attribute key and cannot be defined as a new attribute."
            )
        
        if "name" in default_attrs:
            
            raise ValueError(
                "'name' is always a default node attribute key and cannot be defined as a new default attribute."
            )

        node_attrs = {**node_attrs, "name": torch.empty(0, dtype=torch.long)}
        default_attrs = {**default_attrs, "name": torch.tensor(0, dtype=torch.long)}

        
        super().__init__(
            edge_index=edge_index,
            directed_mask=directed_mask,
            reciprocal_edge=reciprocal_edge,
            **node_attrs,
            **edge_attrs,
            **graph_attrs,
            **kwargs,
        )

        self.num_nodes = edge_index.max().item() + 1 if edge_index.numel() > 0 else 0

        # only add metadata if edge_index is empty
        if edge_index.numel() == 0:
            self.metadata = {
                "default_attrs": default_attrs,
                "node_name_to_index": {},
                "edge_name_to_index": {},
                "node_attr_list": [kwarg for kwarg in node_attrs.keys()],
                "edge_attr_list": [kwarg for kwarg in edge_attrs.keys()],
                "graph_attr_list": [kwarg for kwarg in graph_attrs.keys()],
            }

    @classmethod
    def from_rhino(cls, points, lines, tolerance=1e-6):
        """Build a [StructData][torch_structure.data.data.StructData] from Rhino ``Point3d``/``Line`` geometry.

        Coincident points (within ``tolerance``) are merged into a single
        node, and each line becomes a pair of reciprocal directed edges.

        Args:
            points (list[Rhino.Geometry.Point3d]): node coordinates.
            lines (list[Rhino.Geometry.Line]): edges, referencing points by
                their ``From``/``To`` endpoints.
            tolerance (float): coordinate rounding tolerance used to merge
                coincident points.

        Returns:
            StructData: the constructed graph, with a ``coords`` node
            attribute.
        """
        coords_list = []
        index_map = {}
        for pt in points:
            key = (
                round(pt.X / tolerance),
                round(pt.Y / tolerance),
                round(pt.Z / tolerance),
            )
            if key not in index_map:
                index_map[key] = len(coords_list)
                coords_list.append([pt.X, pt.Y, pt.Z])

        def point_key(pt):
            """Round a point's coordinates to ``tolerance`` to key it for coincidence merging."""
            return (
                round(pt.X / tolerance),
                round(pt.Y / tolerance),
                round(pt.Z / tolerance),
            )

        edges = []

        for ln in lines:
            i = index_map[point_key(ln.From)]
            j = index_map[point_key(ln.To)]
            edges.append((i, j))
            edges.append((j, i))  # Add reciprocal edge

        # Create edge_index tensor
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        coords = torch.tensor(coords_list, dtype=torch.float)

        num_edges = edge_index.size(1)

        directed_mask = torch.zeros((num_edges, 1), dtype=torch.bool)
        directed_mask[::2] = True

        # Reciprocal edge: forward i <-> i+1
        reciprocal_edge = torch.arange(num_edges).view(-1, 2)
        reciprocal_edge = reciprocal_edge[:, [1, 0]].reshape(-1, 1)

        obj = cls(
            edge_index=edge_index,
            directed_mask=directed_mask,
            reciprocal_edge=reciprocal_edge,
            coords=coords,
        )

        return obj
    
    def to_rhino(self):
        """Convert this graph's canonical (directed-mask) edges to Rhino ``Point3d``/``Line`` geometry.

        Returns:
            tuple[list[Rhino.Geometry.Point3d], list[Rhino.Geometry.Line]]:
            the node coordinates as points, and one line per canonical edge.
        """
        import Rhino.Geometry as rg

        xyz = self.coords.detach().cpu().numpy()
        src, dst = (
            self.edge_index[:, self.directed_mask.view(-1)]
            .detach()
            .cpu()
            .numpy()
        )

        points = [rg.Point3d(float(x), float(y), float(z)) for x, y, z in xyz]
        lines = [rg.Line(points[int(s)], points[int(d)]) for s, d in zip(src, dst)]

        return points, lines

    def __inc__(self, key, value, *args, **kwargs):
        if key == "reciprocal_edge":
            return self.num_edges
        elif "index" in key:
            return self.num_nodes
        else:
            return 0
        
    def verify_equilibrium(self, tolerance=1e-7, verbose=False, **kwargs):
        """Check whether this structure is in static equilibrium.

        Computes the residual-force loss (see
        [ResidualForceLoss][torch_structure.loss.residual_force.ResidualForceLoss]) and compares it
        against ``tolerance``.

        Args:
            tolerance (float): maximum residual-force loss still considered
                equilibrium.
            verbose (bool): if ``True``, print the result and loss value.
            **kwargs: overrides for ``coords``, ``load``, ``is_support``,
                ``force``, or ``edge_index``; each defaults to the
                corresponding attribute on ``self``.

        Returns:
            bool: ``True`` if the residual-force loss is below ``tolerance``.
        """
        if "coords" not in kwargs:
            kwargs["coords"] = self.coords
        if "load" not in kwargs:
            kwargs["load"] = self.load
        if "is_support" not in kwargs:
            kwargs["is_support"] = self.is_support
        if "force" not in kwargs:
            kwargs["force"] = self.force
        if "edge_index" not in kwargs:
            kwargs["edge_index"] = self.edge_index

        residual_force_loss = ResidualForceLoss()
        loss = residual_force_loss(**kwargs)
        equilibrium = True if loss < tolerance else False

        if verbose:
            print(f"Equilibrium: {equilibrium} (loss = {loss})")

        return equilibrium
    
    def edge_attr_to_undirected(self, edge_attr, mask, batched=False):
        """Expand an edge attribute defined on canonical edges to all (both-direction) edges.

        Args:
            edge_attr (torch.Tensor): values for the edges selected by
                ``mask``, of shape ``[num_masked_edges, 1]`` or, if
                ``batched``, ``[batch_size, num_masked_edges]``.
            mask (torch.Tensor [num_edges]): boolean mask selecting the
                canonical subset of edges ``edge_attr`` was computed for
                (e.g. ``directed_mask`` or ``cem_edge_mask``).
            batched (bool): if ``True``, treat ``edge_attr`` as a batch of
                per-edge value sets (e.g. an iteration history) instead of a
                single set.

        Returns:
            torch.Tensor: ``edge_attr`` broadcast to all ``num_edges``
            edges, with each edge outside ``mask`` set to the value of its
            reciprocal edge.
        """
        mask = mask.view(-1)
        device = edge_attr.device

        if batched:
            value = torch.empty(
                (edge_attr.shape[0], self.num_edges),
                dtype=edge_attr.dtype,
                device=device,
            )

            # set defined values
            value[:, mask] = edge_attr

            # set reciprocal values
            value[:, ~mask] = value[:, self.reciprocal_edge[~mask].view(-1)]

        else:
            value = torch.empty(
                (self.num_edges, *edge_attr.shape[1:]), dtype=edge_attr.dtype, device=device
            )

            # set defined values
            value[mask] = edge_attr

            # set reciprocal values
            value[~mask] = value[self.reciprocal_edge[~mask].view(-1)]

        return value
    
    def _track_history(
        self, attr_name, value
    ):  # Todo: requires attr exists in self
        history_attr_name = f"{attr_name}_history"
        current_attr = getattr(self, attr_name)

        # Set correct view for 1D tensors
        if current_attr.dim() == 2 and current_attr.shape[1] == 1:
            current_attr = current_attr.view(-1)

        # Add batch dimension if needed
        if value.dim() == current_attr.dim() - 1:
            value = value.unsqueeze(0)

        if hasattr(self, history_attr_name):
            history = getattr(self, history_attr_name)
            new_history = torch.cat([history, value], dim=0)
        else:
            new_history = torch.cat([current_attr.unsqueeze(0), value], dim=0)

        setattr(self, history_attr_name, new_history)
        
    # Properties
    @property
    def is_support(self):
        """torch.Tensor [N, 1]: boolean mask of supported (fixed) nodes.

        Falls back to ``any(support_xyz, dim=1)`` if ``is_support``
        was not set directly.
        """
        if "is_support" in self._store:
            return self["is_support"]

        if "support_xyz" in self._store:
            return torch.any(self["support_xyz"], dim=1, keepdim=True)

        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute 'is_support' or 'support_xyz'"
        )

    @property
    def support_xyz(self):
        """torch.Tensor [N, 3]: boolean mask of restrained translations per global axis.

        Falls back to ``is_support`` repeated over the three axes, i.e. a pin, if ``support_xyz``
        was not set directly.
        """
        if "support_xyz" in self._store:
            return self["support_xyz"]

        if "is_support" in self._store:
            return self["is_support"].view(-1, 1).repeat(1, 3)

        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute 'is_support' or 'support_xyz'"
        )

    @property
    def support_rxyz(self):
        """torch.Tensor [N, 3]: boolean mask of restrained rotations per global axis.

        Defaults to all rotations free if ``support_rxyz`` was not set directly.
        """
        if "support_rxyz" in self._store:
            return self["support_rxyz"]

        support_xyz = self.support_xyz
        return torch.zeros_like(support_xyz, dtype=torch.bool)

    @property
    def support_dofs(self):
        """torch.Tensor [N, 6]: restrained DOFs ``[ux, uy, uz, rx, ry, rz]`` per node.

        Concatenates [support_xyz][torch_structure.data.data.StructData.support_xyz] and
        [support_rxyz][torch_structure.data.data.StructData.support_rxyz]. Warns if both
        ``is_support`` and ``support_xyz`` are stored and disagree, in which case ``support_xyz``
        is used.
        """
        support_xyz = self.support_xyz.to(torch.bool)

        if "is_support" in self._store and "support_xyz" in self._store:
            if not torch.equal(self["is_support"].view(-1).to(torch.bool), support_xyz.any(dim=1)):
                warnings.warn(
                    "'is_support' and 'support_xyz' disagree; using 'support_xyz'.",
                    stacklevel=2,
                )

        return torch.cat([support_xyz, self.support_rxyz.to(torch.bool)], dim=1)

    @property
    def length_from_coords(self):
        """torch.Tensor [E, 1]: each edge's length, recomputed from ``coords``."""
        if hasattr(self, "coords"):
            return graph_edge_lengths(self.coords, self.edge_index)
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute 'coords'"
            )

    @property
    def bbox(self):
        """torch.Tensor [2, D]: the axis-aligned bounding box of ``coords``, as ``[min, max]``."""
        if not hasattr(self, "coords"):
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute 'coords' required for bounding box calculation."
            )
        return torch.stack(
            [self.coords.min(dim=0).values, self.coords.max(dim=0).values], dim=0
        )

    @property
    def directed_edge_index(self):
        """torch.Tensor [2, E_directed]: ``edge_index`` restricted to the canonical (``directed_mask``) edges."""
        return self.edge_index[:, self.directed_mask.view(-1)]

    @property
    def cem_edge_index(self):
        """torch.Tensor [2, E_cem]: ``edge_index`` restricted to the CEM-canonical ([cem_edge_mask][torch_structure.data.data.StructData.cem_edge_mask]) edges."""
        return self.edge_index[:, self.cem_edge_mask]

    @property
    def cem_edge_mask(self):
        """torch.Tensor [E], bool: mask selecting one direction of every edge for CEM.

        Like ``directed_mask``, but trail edges (whose force direction is
        meaningful and must be preserved) always keep their original
        direction regardless of ``directed_mask``.
        """
        return ~(self.is_trail_edge.view(-1) & ~self.directed_mask.view(-1))
        
    # Graph Editing Functionality
    @property
    @requires_metadata
    def nodes(self):
        """
        Provides networkx-style access to node attributes without caching.
        """
        return NodeView(self, self.metadata["node_name_to_index"], self.metadata["node_attr_list"])

    @requires_metadata
    def add_node(self, name: str, **kwargs):
        """
        Adds a new node.
        """
        # Check for unexpected attributes
        unexpected_attrs = set(kwargs.keys()) - set(self.metadata["node_attr_list"])
        if unexpected_attrs:
            raise ValueError(
                f"Unexpected node attributes: {unexpected_attrs}. Expected: {list(self.metadata['node_attr_list'])}"
            )

        # Check if node already exists
        if name in self.metadata["node_name_to_index"]:
            raise ValueError(f"Node '{name}' already exists!")

        self.metadata["node_name_to_index"][name] = self.num_nodes  # Add node to node_name_to_index
        self.num_nodes += 1  # Increment number of nodes
        
        # Add node attributes
        for attr in self.metadata["node_attr_list"]:
            if attr in kwargs:
                value = kwargs[attr]

                # Cast non-tensor attributes to tensor
                if not isinstance(value, torch.Tensor):
                    value = torch.tensor(value, dtype=getattr(self, attr).dtype)
                    # warnings.warn(f"Node '{name}' attribute '{attr}' was automatically cast to a torch tensor.", UserWarning)

            # If attribute is not provided, set it to default value
            elif attr in self.metadata["default_attrs"]:
                value = self.metadata["default_attrs"][attr]

            # If attribute has no default value, set it to zero
            else:
                attr_shape = getattr(self, attr).shape[1:]
                value = getattr(self, attr).new_zeros(attr_shape)
                # warnings.warn(f"Node '{name}' attribute '{attr}' initialized to zeros (no default value provided).", UserWarning)

            if value.dim() == 0:
                value = value.unsqueeze(0)

            setattr(
                self,
                attr,
                torch.cat([getattr(self, attr), value.unsqueeze(0)], dim=0),
            )

    @requires_metadata
    def add_n_empty_nodes(self, n, names = None):
        """
        Adds n new nodes initialized to their default values or to zero if there exists no default value.

        Args:
            n (int): number of nodes to add. ``n`` must be at least 1.

        Raises:
            ValueError: if ``n`` is less than or equal to zero.
        """

        if n <= 0:
            raise ValueError("Invalid number of nodes provided. n must at least be 1.")

        self.num_nodes += n

        if names is not None:
            if len(names) != n:
                raise ValueError(f"Number {len(names)} of names {names} does not match the given number {n} of new nodes to add.")

            encoded_names = torch.tensor([encode(name) for name in names], dtype=torch.long)

        # Add default node attributes
        for attr in self.metadata["node_attr_list"]:

            if names is not None and attr == "name":
                value = encoded_names
            else:
                value = self._resolve_attr_value(attr, {}, n)

            setattr(
                self,
                attr,
                torch.cat([getattr(self, attr), value], dim=0),
            )


    def _prepare_node_kwargs(self, names, kwargs):

        if not kwargs:
            raise ValueError("No node attributes provided for new nodes. At least one attribute must be provided.")

        # Check for unexpected attributes
        unexpected_attrs = set(kwargs.keys()) - set(self.metadata["node_attr_list"])
        if unexpected_attrs:
            raise ValueError(
                f"Unexpected node attributes: {unexpected_attrs}. Expected: {list(self.metadata['node_attr_list'])}"
            )

        bookkeeping_attrs = {"expansion_id", "expansion_position", "pattern_id"} & set(kwargs.keys())
        if bookkeeping_attrs:
            raise ValueError(f"{sorted(bookkeeping_attrs)} is pattern bookkeeping and cannot be set for new nodes.")

        num_given_nodes = len(next(iter(kwargs.values())))

        if num_given_nodes == 0:
            raise ValueError("No node attribute values provided. There must be at least one attribute value")

        mismatched = {attr: len(value) for attr, value in kwargs.items() if len(value) != num_given_nodes}
        if mismatched:
            raise ValueError(
                f"All node attributes must have the same length ({num_given_nodes}), "
                f"but got mismatched lengths: {mismatched}."
            )

        if names:
            if len(names) != num_given_nodes:
                raise ValueError(f"Number {len(names)} of names {names} does not match the length {num_given_nodes} of the given attributes {kwargs}")

            kwargs["name"] = torch.tensor([encode(name) for name in names], dtype=torch.long)

        return num_given_nodes


    @requires_metadata
    def add_nodes(self, names = None, **kwargs):
        """
        Adds new nodes with the given attribute values. The new nodes are outside any
        pattern; use ``add_nodes_by_pattern`` to copy them according to a registered
        pattern.

        Args:
            names (list[str], optional): node name for each new node. Encoded and
                stored as the ``name`` attribute.
            **kwargs (dict[str, torch.Tensor]): mapping of registered node attribute
                names to tensors. The first dimension of each tensor must equal the
                number of nodes to add. Attributes omitted from ``kwargs`` are
                initialized to their default values or to zero if no default value
                exists.

        Raises:
            ValueError: if no keyword arguments are provided. If you want to add nodes
                without any attributes, use ``add_n_empty_nodes()``.
            ValueError: if the attribute tensors are empty or have different lengths.
            ValueError: if ``names`` does not have the same length as the attribute tensors.
            ValueError: if any provided attribute name is not a registered node attribute.
            ValueError: if any provided attribute name is pattern bookkeeping
                (``expansion_id``, ``expansion_position``, ``pattern_id``).
        """

        num_new_nodes = self._prepare_node_kwargs(names, kwargs)
        self.num_nodes += num_new_nodes

        # Add node attributes
        for attr in self.metadata["node_attr_list"]:

            value = self._resolve_attr_value(attr, kwargs, num_new_nodes)

            setattr(
                self,
                attr,
                torch.cat([getattr(self, attr), value], dim=0),
            )


    @requires_metadata
    def add_nodes_by_pattern(self, pattern: str, names = None, merge_attr: str = "coords", **kwargs):
        """
        Adds new nodes with the given attribute values and copies each of them according to
        a registered pattern. Copies that land on the same spot, e.g. the mirror copy of a
        seed on the mirror plane, or the rotated copies of a seed on the rotation axis, are
        merged into one node, see ``merge_attr``.

        Args:
            pattern (str): name of the registered pattern to apply. Each provided node
                (seed) is copied according to the pattern, together with all of its
                attributes. Attributes registered as ``transform_attrs`` are geometrically
                transformed, every other attribute is copied verbatim. Each seed together
                with its copies gets a new expansion id attribute (global), counting up from the
                highest existing one. Additionally, they get an expansion position attribute
                (local) counting up from 0, in the order they are generated by the pattern.
            names (list[str], optional): node name for each seed. Encoded and stored as the
                ``name`` attribute; each seed's name is copied verbatim to all of its
                pattern copies.
            merge_attr (str, optional): the transform attribute that decides which of the new
                nodes land on the same spot. They are merged with ``merge_by_attribute``, with its
                tolerance and restricted to the new nodes, so they are never merged with
                nodes that existed before. Copies that would be merged must agree in every other
                attribute (``error_on_mismatch`` of ``merge``). Within an expansion, the copy with
                the lowest expansion position keeps the node; the other positions are recorded in the
                ``expansion_holes`` graph attribute and resolve to that node wherever an expansion position
                is used, e.g. when edges are copied. Merging renumbers the nodes, see ``merge``.
                Nothing is merged if it is ``None``, or if the attribute is not given, or if it is
                the default ``"coords"`` but not a transform attribute of the pattern. Defaults to
                ``"coords"``.
            **kwargs (dict[str, torch.Tensor]): mapping of registered node attribute
                names to tensors. The first dimension of each tensor must equal the
                number of seeds. Attributes omitted from ``kwargs`` are initialized to
                their default values or to zero if no default value exists.

        Raises:
            ValueError: if ``pattern`` is not a registered pattern.
            ValueError: if no keyword arguments are provided.
            ValueError: if the attribute tensors are empty or have different lengths.
            ValueError: if ``names`` does not have the same length as the attribute tensors.
            ValueError: if any provided attribute name is not a registered node attribute.
            ValueError: if any provided attribute name is pattern bookkeeping
                (``expansion_id``, ``expansion_position``, ``pattern_id``).
            ValueError: if ``merge_attr`` is given, but is not a transform attribute of the
                pattern.
            ValueError: if copies of a seed that would be merged differ in another attribute, e.g.
                a seed on a mirror plane with a load pointing across it, or if copies of different
                seeds land on the same spot. Both are raised by ``merge``; the new nodes stay added,
                unmerged.
        """

        if pattern not in self.metadata.get("name_to_pattern", {}):
            raise ValueError(f"Pattern '{pattern}' is not registered. Please register the pattern first using add_patterns().")

        num_seed_nodes = self._prepare_node_kwargs(names, kwargs)

        pattern_id = self.metadata["name_to_pattern"][pattern]
        matrices = self.pattern_matrices[self.pattern_matrix_id == pattern_id]
        pattern_size = matrices.shape[0]
        transform_attrs = self.metadata["pattern_transform_attrs"][pattern_id]

        if merge_attr is not None and merge_attr != "coords" and merge_attr not in transform_attrs:
            raise ValueError(
                f"merge_attr '{merge_attr}' is not a transform attribute of pattern '{pattern}' "
                f"({transform_attrs}). Pass a transform attribute, or merge_attr=None to merge nothing."
            )

        num_new_nodes = num_seed_nodes * pattern_size

        expansion_holes = getattr(self, "expansion_holes", torch.empty((0, 3), dtype=torch.long))
        used_expansion_ids = torch.cat([self.expansion_id.view(-1), expansion_holes[:, 0]])
        base_expansion_id = int(used_expansion_ids.max().item()) + 1 if used_expansion_ids.numel() > 0 else 0

        new_values = {
            "expansion_id": (base_expansion_id + torch.arange(num_seed_nodes)).repeat_interleave(pattern_size).unsqueeze(1),
            "expansion_position": torch.arange(pattern_size).repeat(num_seed_nodes).unsqueeze(1),
            "pattern_id": torch.full((num_new_nodes, 1), pattern_id, dtype=torch.long),
        }

        for attr in self.metadata["node_attr_list"]:
            if attr in new_values:
                continue

            value = self._resolve_attr_value(attr, kwargs, num_new_nodes)

            if attr in kwargs:
                if attr in transform_attrs:
                    value = self._apply_affine(matrices, value)
                    value = value.reshape(num_new_nodes, *value.shape[2:])
                else:
                    value = self._expand_value_over_pattern(value, num_seed_nodes, pattern_size)

            new_values[attr] = value

        first_new_node = self.num_nodes
        self.num_nodes += num_new_nodes

        for attr, value in new_values.items():
            setattr(self, attr, torch.cat([getattr(self, attr), value], dim=0))

        if merge_attr is not None and merge_attr in transform_attrs and merge_attr in kwargs:
            new_nodes = torch.zeros(self.num_nodes, dtype=torch.bool)
            new_nodes[first_new_node:] = True
            self.merge_by_attribute(merge_attr, mask=new_nodes, error_on_mismatch=True)


    def _expansion_canonical_members(self, nodes, pattern_size):
        # the node at every expansion position of each node's expansion; a merged copy's position gives its twin's node
        lookup, rows = self._build_expansion_lookup(self.expansion_id.view(-1)[nodes])
        members = lookup[rows, :pattern_size]
        if torch.any(members == -1):
            raise ValueError("Some expansion(s) are not intact; their nodes may have been merged or removed.")

        return members, self.expansion_position.view(-1)[nodes]


    def _infer_level_sizes(self, pattern_id):

        in_pattern = self.pattern_matrix_id == pattern_id
        levels = self.pattern_level[in_pattern]
        counts = torch.bincount(levels)
        boundaries = torch.cumsum(counts, dim=0)

        closes = getattr(self, "pattern_closes", None)
        closes = torch.ones_like(levels, dtype=torch.bool) if closes is None else closes[in_pattern]

        level_sizes, level_closes = [], []
        prev = 1
        for boundary in boundaries.tolist():
            level_sizes.append(boundary // prev)
            level_closes.append(bool(closes[boundary - 1]))
            prev = boundary

        return level_sizes, level_closes


    def _swept_levels(self, replicate_levels, num_levels, pattern_id):

        if replicate_levels is None:
            return torch.ones(num_levels, dtype=torch.bool)

        swept = torch.as_tensor(replicate_levels, dtype=torch.bool).view(-1)
        if swept.shape[0] != num_levels:
            raise ValueError(
                f"replicate_levels must have one entry per pattern level ({num_levels} for "
                f"pattern '{self._pattern_names()[pattern_id]}'), got {swept.shape[0]}."
            )

        return swept


    def _swept_copies(self, level_sizes, swept):
        # one row per copy over the swept levels: its position on each swept level, innermost first, and what those
        # positions add to the expansion position. Without a swept level there is one copy, which adds nothing
        sizes = torch.as_tensor(level_sizes, dtype=torch.long)
        swept_sizes = sizes[swept]
        digits = self._level_digits(torch.arange(int(swept_sizes.prod())), swept_sizes)
        return digits, (digits * self._level_places(sizes)[swept]).sum(dim=1)


    def _replicated_positions(self, positions, pattern_id, replicate_levels=None):

        level_sizes, _ = self._infer_level_sizes(pattern_id)
        swept = self._swept_levels(replicate_levels, len(level_sizes), pattern_id)
        place = self._level_places(level_sizes)

        # a frozen level keeps the given position's own digit
        base = (self._level_digits(positions, level_sizes)[:, ~swept] * place[~swept]).sum(dim=1)
        _, offsets = self._swept_copies(level_sizes, swept)

        # for each given position, every expansion position that shares its digits on the frozen levels
        return base.unsqueeze(1) + offsets.unsqueeze(0)


    def _expand_edge_indices_by_pattern(self, src, dst, pattern_id, replicate_levels=None):

        expansion_id = self.expansion_id.view(-1)
        expansion_position = self.expansion_position.view(-1)
        lookup, rows = self._build_expansion_lookup(torch.cat([expansion_id[src], expansion_id[dst]]))
        src_rows, dst_rows = rows[:src.shape[0]], rows[src.shape[0]:]

        level_sizes, level_closes = self._infer_level_sizes(pattern_id)
        sizes_t = torch.tensor(level_sizes, dtype=torch.long)
        closes_t = torch.tensor(level_closes, dtype=torch.bool)
        place = self._level_places(level_sizes)

        src_idx = self._level_digits(expansion_position[src], level_sizes)
        dst_idx = self._level_digits(expansion_position[dst], level_sizes)
        signed_deltas = dst_idx - src_idx
        deltas = torch.where(closes_t, signed_deltas % sizes_t, signed_deltas)

        swept = self._swept_levels(replicate_levels, len(level_sizes), pattern_id)
        frozen = ~swept

        # a frozen level keeps the given edge's own two positions
        base_src = (src_idx[:, frozen] * place[frozen]).sum(dim=1)
        base_dst = (dst_idx[:, frozen] * place[frozen]).sum(dim=1)

        swept_sizes = sizes_t[swept]
        swept_place = place[swept]
        swept_closes = closes_t[swept]

        digits, src_swept = self._swept_copies(level_sizes, swept)
        idx = torch.arange(digits.shape[0])

        raw_dst_digits = digits.unsqueeze(0) + deltas[:, swept].unsqueeze(1)
        dst_digits = raw_dst_digits % swept_sizes.view(1, 1, -1)
        dst_swept = (dst_digits * swept_place.view(1, 1, -1)).sum(dim=2)

        in_level = (raw_dst_digits >= 0) & (raw_dst_digits < swept_sizes.view(1, 1, -1))
        exists = (in_level | swept_closes.view(1, 1, -1)).all(dim=2)

        # the nodes at each copy's two expansion positions; a merged copy's position gives its twin's node
        src_all = lookup[src_rows.unsqueeze(1), base_src.unsqueeze(1) + src_swept.unsqueeze(0)]
        dst_all = lookup[dst_rows.unsqueeze(1), base_dst.unsqueeze(1) + dst_swept]

        if torch.any(exists & ((src_all == -1) | (dst_all == -1))):
            raise ValueError("Some expansion(s) are not intact; their nodes may have been merged or removed.")

        # a copy whose two ends were merged into one node is no edge, e.g. the copy of an edge across a mirror plane
        # at a node on the plane; a given loop keeps its copies
        exists &= (src_all != dst_all) | (src == dst).unsqueeze(1)

        pair_min = torch.minimum(src_all, dst_all)
        pair_max = torch.maximum(src_all, dst_all)
        keys = pair_min * self.num_nodes + pair_max
        keys = torch.where(exists, keys, -1 - idx.unsqueeze(0))

        order = torch.argsort(keys, dim=-1, stable=True)
        sorted_keys = torch.gather(keys, 1, order)
        first_in_group = torch.cat(
            [torch.ones(sorted_keys.shape[0], 1, dtype=torch.bool), sorted_keys[:, 1:] != sorted_keys[:, :-1]],
            dim=1,
        )
        keep = torch.zeros_like(keys, dtype=torch.bool)
        keep.scatter_(1, order, first_in_group)
        keep &= exists

        # index of the input edge each copy was made from
        seed_edge = torch.arange(src.shape[0]).unsqueeze(1).expand_as(keep)[keep]

        return src_all[keep], dst_all[keep], seed_edge
    

    def _expand_edges_by_pattern(self, edge_indices, kwargs, replicate_levels=None):
        # kwargs holds tensors, as returned by _check_new_edges

        src, dst = edge_indices[0], edge_indices[1]

        pattern_id = self._single_pattern(src, dst, outside_hint="Add them via add_edges().")
        if pattern_id is None:
            return edge_indices, kwargs

        src_expansions, dst_expansions, seed_edge = self._expand_edge_indices_by_pattern(src, dst, pattern_id, replicate_levels)

        edge_indices = torch.stack([src_expansions, dst_expansions], dim=0)
        kwargs_expanded = {attr: value[seed_edge] for attr, value in kwargs.items()}

        return edge_indices, kwargs_expanded


    def _require_pattern(self):
        if not hasattr(self, "pattern_id"):
            raise ValueError("No pattern is registered. Register one with add_patterns() first.")


    def _pattern_names(self):
        # the name of every registered pattern, by its pattern id
        return {pattern_id: name for name, pattern_id in self.metadata["name_to_pattern"].items()}


    def _unique_pattern(self, pattern_ids, what):
        # the one pattern id in pattern_ids, or None if it is empty; the given nodes, edges or expansions may not span
        # several patterns, as one call works on one pattern
        pattern_ids = torch.unique(pattern_ids)
        if pattern_ids.numel() > 1:
            names = self._pattern_names()
            raise ValueError(
                f"The given {what} belong to several registered patterns "
                f"{[names[pattern_id] for pattern_id in pattern_ids.tolist()]}. Use a separate call per pattern."
            )

        return int(pattern_ids[0]) if pattern_ids.numel() > 0 else None


    def _single_pattern(self, src, dst=None, outside_hint=""):
        """
        Returns the id of the one registered pattern that all given nodes belong to, or
        ``None`` if no nodes are given. For edges, ``src`` and ``dst`` are their endpoints;
        for nodes, ``dst`` is left out.
        """
        self._require_pattern()

        node_pattern_id = self.pattern_id.view(-1)
        src_pattern_id = node_pattern_id[src]
        dst_pattern_id = src_pattern_id if dst is None else node_pattern_id[dst]

        outside = (src_pattern_id == -1) | (dst_pattern_id == -1)
        if torch.any(outside):
            if dst is None:
                raise ValueError(f"Node(s) {src[outside].tolist()} are outside any pattern. {outside_hint}")
            bad_pairs = list(zip(src[outside].tolist(), dst[outside].tolist()))
            raise ValueError(f"Edge(s) {bad_pairs} connect a node outside any pattern. {outside_hint}")

        mismatched = src_pattern_id != dst_pattern_id
        if torch.any(mismatched):
            bad_pairs = list(zip(src[mismatched].tolist(), dst[mismatched].tolist()))
            raise ValueError(
                f"Edge(s) {bad_pairs} connect two different registered patterns; src and dst must "
                "belong to the same registered pattern."
            )

        return self._unique_pattern(src_pattern_id, "nodes" if dst is None else "edges")


    def _check_new_edges(self, edge_indices, kwargs):
        # returns kwargs with every value as a tensor

        #check for edge indices that are out of bound
        if torch.any(edge_indices >= self.num_nodes) or torch.any(edge_indices < 0):
            raise ValueError(f"Unexpected edge indices: {edge_indices[(edge_indices >= self.num_nodes) | (edge_indices < 0)]}, that do not correspond to existing nodes. Edge indices are expected to be between 0 and {self.num_nodes - 1}. First add respective nodes.")

        # Check for unexpected attributes
        unexpected_attrs = set(kwargs.keys()) - set(self.metadata["edge_attr_list"])
        if unexpected_attrs:
            raise ValueError(
                f"Unexpected edge attributes: {unexpected_attrs}. Expected: {list(self.metadata['edge_attr_list'])}"
            )

        # count the rows the way _resolve_attr_value later appends them, where a 0-dim value is one row
        num_edges = edge_indices.shape[1]
        tensors, mismatched = {}, {}
        for attr, value in kwargs.items():
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value, dtype=getattr(self, attr).dtype)
            tensors[attr] = value

            num_rows = value.shape[0] if value.dim() > 0 else 1
            if num_rows != num_edges:
                mismatched[attr] = num_rows

        if mismatched:
            raise ValueError(
                f"All edge attributes must have one row per edge ({num_edges}), "
                f"but got mismatched lengths: {mismatched}."
            )

        return tensors


    def _undirected_edge_keys(self, edges):
        # every edge may exist only once, in either direction; u -> v and v -> u get the same key
        return torch.minimum(edges[0], edges[1]) * self.num_nodes + torch.maximum(edges[0], edges[1])


    def _append_edges(self, edge_indices, kwargs, twice_reason):

        unique_keys, key_counts = torch.unique(self._undirected_edge_keys(edge_indices), return_counts=True)

        added_twice = unique_keys[key_counts > 1]
        if added_twice.numel() > 0:
            raise ValueError(
                f"Edge(s) {[divmod(key, self.num_nodes) for key in added_twice.tolist()]} would be added more than "
                f"once: {twice_reason}"
            )

        existing = unique_keys[torch.isin(unique_keys, self._undirected_edge_keys(self.edge_index))]
        if existing.numel() > 0:
            raise ValueError(f"Edge(s) {[divmod(key, self.num_nodes) for key in existing.tolist()]} already exist.")

        num_new_edges = edge_indices.shape[1]
        
        # Update directed mask and reciprocal edge
        self.directed_mask = torch.cat(
            [self.directed_mask, torch.ones(num_new_edges, dtype=torch.bool).unsqueeze(1), torch.zeros(num_new_edges, dtype=torch.bool).unsqueeze(1)], 
            dim=0
        )

        edges = torch.arange(self.num_edges, self.num_edges + num_new_edges).unsqueeze(1)
        reciprocal_edges = torch.arange(self.num_edges + num_new_edges, self.num_edges + num_new_edges * 2).unsqueeze(1)
        
        self.reciprocal_edge = torch.cat(
            [self.reciprocal_edge, reciprocal_edges, edges],
            dim=0
        )

        # Add edges and reciprocal edges to edge_index
        self.edge_index = torch.cat([self.edge_index, edge_indices, edge_indices.flip(0)], dim=1)

        # Add edge attributes
        for attr in self.metadata["edge_attr_list"]:

            value = self._resolve_attr_value(attr, kwargs, num_new_edges)

            setattr(
                self,
                attr,
                torch.cat(
                    [getattr(self, attr), value, value],
                    dim=0,
                ),
            )


    @requires_metadata
    def add_edges(self, edge_indices, **kwargs):
        """
        Adds new edges with the given attribute values. Only the given edges are added, even if
        their nodes belong to a registered pattern; use ``add_edges_by_pattern`` to also add
        their pattern copies.

        Args:
            edge_indices (torch.Tensor): source and destination node
                indices for each of the ``E`` edges to add. Shape [2, E]. All indices must refer to
                existing nodes.
            **kwargs (dict[str, torch.Tensor]): mapping of registered edge attribute
                names to tensors. The first dimension of each tensor must equal ``E``.
                Attributes omitted from ``kwargs`` are initialized to their default
                values or to zero if no default value exists.

        Raises:
            ValueError: if any index in ``edge_indices`` is out of range.
            ValueError: if any provided attribute name is not a registered edge
                attribute.
            ValueError: if the first dimension of an attribute tensor does not equal ``E``.
            ValueError: if an edge already exists, in either direction.
            ValueError: if an edge would be added more than once, i.e. it is given twice.
        """

        kwargs = self._check_new_edges(edge_indices, kwargs)

        self._append_edges(edge_indices, kwargs, twice_reason="they are given twice.")


    @requires_metadata
    def add_edges_by_pattern(self, edge_indices, replicate_levels = None, **kwargs):
        """
        Adds new edges with the given attribute values and copies each edge, together with its
        attributes, over the whole pattern of its source and destination nodes: every element
        of the pattern is applied to both endpoints, on every level at once. Copies that land
        on an edge already produced are added only once, so in a mirrored rotation an edge
        joining a node to its own mirror image gives one edge per rotational position rather
        than two. On a level that does not close (see ``create_translational_pattern`` and the
        ``closes`` argument of ``create_rotational_pattern``), the copy that would run from the
        last position of that level back to the first is left out, so an open level yields one
        copy less than a closing one. Where ``add_nodes_by_pattern`` merged coinciding copies
        into one node, copies of the edge attach to that node, and a copy whose two ends were
        merged into the same node, e.g. the copy of an edge across a mirror plane at a node on
        the plane, is left out. The given edge itself is always added.

        Args:
            edge_indices (torch.Tensor): source and destination node
                indices for each of the ``E`` edges to add. Shape [2, E]. All indices must refer to
                existing nodes, and all of these nodes must belong to the same registered pattern.
            replicate_levels (list[bool], optional): one entry per nesting level of the
                pattern, innermost first, selecting the levels each edge is replicated on. A
                level left out keeps the given edge's own two positions on that level, so no
                copies are made along it. Any subset works, so a level can be frozen between two
                replicated ones. Defaults to ``None``, which replicates on every level. This
                argument must match between ``add_edges_by_pattern`` and
                ``set_edge_attr_by_pattern`` for the same edge family, since it decides which
                edges count as copies of each other.
            **kwargs (dict[str, torch.Tensor]): mapping of registered edge attribute
                names to tensors. The first dimension of each tensor must equal ``E``.
                Attributes omitted from ``kwargs`` are initialized to their default
                values or to zero if no default value exists.

        Raises:
            ValueError: if any index in ``edge_indices`` is out of range.
            ValueError: if any provided attribute name is not a registered edge
                attribute.
            ValueError: if the first dimension of an attribute tensor does not equal ``E``.
            ValueError: if no pattern is registered.
            ValueError: if an edge connects a node outside any pattern.
            ValueError: if an edge connects nodes from two different registered patterns.
            ValueError: if the edges belong to more than one registered pattern.
            ValueError: if ``replicate_levels`` does not have one entry per nesting level of
                the pattern.
            ValueError: if an edge already exists, in either direction.
            ValueError: if an edge would be added more than once, i.e. it is given twice or
                the given edges are pattern copies of each other.
        """

        kwargs = self._check_new_edges(edge_indices, kwargs)

        edge_indices, kwargs = self._expand_edges_by_pattern(edge_indices, kwargs, replicate_levels)

        self._append_edges(
            edge_indices, kwargs,
            twice_reason="they are given twice, or they are pattern copies of each other (give only one edge per edge "
            "family).",
        )


    def _build_expansion_lookup(self, expansion_ids):
        # the node at every expansion position of the given expansions, -1 where there is none, with one row per distinct expansion
        # id; also returns the row of each given expansion id
        expansion_id = self.expansion_id.view(-1)
        expansion_position = self.expansion_position.view(-1)

        expansion_holes = getattr(self, "expansion_holes", torch.empty((0, 3), dtype=torch.long))
        hole_expansion, hole_position, twin_position = expansion_holes.unbind(dim=1)

        expansions, rows = torch.unique(expansion_ids, return_inverse=True)

        all_expansion_ids = torch.cat([expansion_id, hole_expansion])
        num_expansions = int(all_expansion_ids.max().item()) + 1 if all_expansion_ids.numel() > 0 else 0
        # the row of every expansion id, shifted by one: slot 0 is expansion id -1, the nodes outside any pattern, which get no
        # row, like expansion ids that are not in use
        known = (expansions >= 0) & (expansions < num_expansions)
        row_of_expansion = torch.full((num_expansions + 1,), -1, dtype=torch.long)
        row_of_expansion[expansions[known] + 1] = torch.where(known)[0]

        node_row = row_of_expansion[expansion_id + 1]
        nodes = torch.where(node_row != -1)[0]
        # wide enough for every position of the largest of their patterns, so merged positions at an expansion's end fit too
        pattern_sizes = torch.bincount(self.pattern_matrix_id)
        num_positions = int(pattern_sizes[self.pattern_id.view(-1)[nodes]].max().item()) if nodes.numel() > 0 else 0

        lookup = torch.full((expansions.numel(), num_positions), -1, dtype=torch.long)
        lookup[node_row[nodes], expansion_position[nodes]] = nodes
        # a merged copy's position resolves to its twin's node; an expansion without nodes has nothing to resolve to
        hole_row = row_of_expansion[hole_expansion + 1]
        resolved = (hole_row != -1) & (hole_position < num_positions) & (twin_position < num_positions)
        lookup[hole_row[resolved], hole_position[resolved]] = lookup[hole_row[resolved], twin_position[resolved]]
        return lookup, rows

    def _node_indices_from_expansion(self, ends):
        # ends holds (expansion ids, expansion positions) for every end of an edge, or for the nodes; one lookup serves all of
        # them. Returns the node indices of every end
        ends = [
            (torch.as_tensor(ids, dtype=torch.long), torch.as_tensor(positions, dtype=torch.long)) for ids, positions in ends
        ]
        lookup, rows = self._build_expansion_lookup(torch.cat([ids.reshape(-1) for ids, _ in ends]))

        indices_per_end = []
        for (ids, positions), end_rows in zip(ends, rows.split([ids.numel() for ids, _ in ends])):
            end_rows = end_rows.view(ids.shape)
            in_bounds = (ids >= 0) & (positions >= 0) & (positions < lookup.shape[1])
            indices = torch.full_like(ids, -1)
            indices[in_bounds] = lookup[end_rows[in_bounds], positions[in_bounds]]
            if torch.any(indices == -1):
                bad = torch.where(indices == -1)[0].tolist()
                raise ValueError(f"No node found for (expansion_id, expansion_position) pairs at query index {bad}.")
            indices_per_end.append(indices)

        return indices_per_end


    def _node_indices_from_expansion_by_pattern(self, ends, replicate_levels, outside_hint, target):
        # _node_indices_from_expansion for the ends of edges, or for nodes, passed on to target, which replicates over their
        # one pattern: an end given at a merged copy is refused if target would act on other copies from the node it
        # was merged into
        indices = self._node_indices_from_expansion(ends)

        pattern_id = self._single_pattern(*indices, outside_hint=outside_hint)
        if pattern_id is not None:
            positions = [torch.as_tensor(positions, dtype=torch.long).view(-1) for _, positions in ends]
            self._check_merged_ends(positions, indices, pattern_id, replicate_levels, target)

        return indices


    def add_edges_per_expansion(self, src_expansion_ids, dest_expansion_ids, src_expansion_positions, dest_expansion_positions, **kwargs):
        """
        Resolves ``(expansion_id, expansion_position)`` pairs to node indices, then calls
        ``add_edges``.

        Args:
            src_expansion_ids (list[int]): expansion id for each new edge's source.
            dest_expansion_ids (list[int]): expansion id for each new edge's destination.
            src_expansion_positions (list[int]): position in the expansion for each new edge's source.
            dest_expansion_positions (list[int]): position in the expansion for each new edge's destination.
            **kwargs (dict[str, torch.Tensor]): see ``add_edges``.

        Raises:
            AttributeError: if no pattern is registered.
            ValueError: if an ``(expansion_id, expansion_position)`` pair does not match any node.
            ValueError: any error raised by ``add_edges`` for the resolved edge indices.
        """
        src_indices, dest_indices = self._node_indices_from_expansion(
            [(src_expansion_ids, src_expansion_positions), (dest_expansion_ids, dest_expansion_positions)]
        )
        edge_indices = torch.stack([src_indices, dest_indices], dim=0)

        self.add_edges(edge_indices=edge_indices, **kwargs)


    def add_edges_per_expansion_by_pattern(self, src_expansion_ids, dest_expansion_ids, src_expansion_positions, dest_expansion_positions, replicate_levels = None, **kwargs):
        """
        Resolves ``(expansion_id, expansion_position)`` pairs to node indices, then calls
        ``add_edges_by_pattern``.

        Args:
            src_expansion_ids (list[int]): expansion id for each new edge's source.
            dest_expansion_ids (list[int]): expansion id for each new edge's destination.
            src_expansion_positions (list[int]): position in the expansion for each new edge's source.
            dest_expansion_positions (list[int]): position in the expansion for each new edge's destination.
            replicate_levels (list[bool], optional): see ``add_edges_by_pattern``.
            **kwargs (dict[str, torch.Tensor]): see ``add_edges_by_pattern``.

        Raises:
            AttributeError: if no pattern is registered.
            ValueError: if an ``(expansion_id, expansion_position)`` pair does not match any node.
            ValueError: if an end of an edge is a copy that was merged into another one, by
                ``add_nodes_by_pattern`` or ``merge``, and ``add_edges_by_pattern``, which reads the
                edge from the node it was merged into, would copy a different edge from there. Give
                such an edge at a copy whose ends were not merged.
            ValueError: any error raised by ``add_edges_by_pattern`` for the resolved edge indices.
        """
        src_indices, dest_indices = self._node_indices_from_expansion_by_pattern(
            [(src_expansion_ids, src_expansion_positions), (dest_expansion_ids, dest_expansion_positions)], replicate_levels,
            outside_hint="Add them via add_edges().", target="add_edges_by_pattern",
        )
        edge_indices = torch.stack([src_indices, dest_indices], dim=0)

        self.add_edges_by_pattern(edge_indices=edge_indices, replicate_levels=replicate_levels, **kwargs)


    @staticmethod
    def _level_places(level_sizes):
        # what one step on each level of a pattern adds to the expansion position, innermost level first: 1, n0, n0 * n1, ...
        sizes = torch.as_tensor(level_sizes, dtype=torch.long)
        places = torch.ones_like(sizes)
        places[1:] = torch.cumprod(sizes[:-1], dim=0)
        return places


    def _expansion_positions_from_levels(self, arg_name, level_positions, level_sizes, num_rows, pattern_name):
        # the level positions are the digits of the expansion position, innermost level first
        try:
            level_positions = torch.as_tensor(level_positions)
        except (ValueError, TypeError):
            raise ValueError(f"{arg_name} must have one row of level positions per expansion id, all of the same length.") from None

        if level_positions.is_floating_point() or level_positions.is_complex() or level_positions.dtype == torch.bool:
            raise ValueError(f"{arg_name} must hold integers, got {level_positions.dtype}.")

        num_levels = len(level_sizes)
        if level_positions.shape != (num_rows, num_levels):
            raise ValueError(
                f"{arg_name} must have shape [{num_rows}, {num_levels}]: one row per expansion id, with one position per "
                f"level of pattern '{pattern_name}' (level sizes {level_sizes}, innermost first), got shape "
                f"{list(level_positions.shape)}."
            )

        level_positions = level_positions.long()
        outside = (level_positions < 0) | (level_positions >= torch.tensor(level_sizes, dtype=torch.long))
        if torch.any(outside):
            row, level = torch.nonzero(outside)[0].tolist()
            raise ValueError(
                f"{arg_name}[{row}] = {level_positions[row].tolist()} is outside pattern '{pattern_name}': level "
                f"{level} has {level_sizes[level]} positions (0 to {level_sizes[level] - 1}), got "
                f"{level_positions[row, level].item()}."
            )

        return (level_positions * self._level_places(level_sizes)).sum(dim=1)


    def _level_digits(self, positions, level_sizes):
        # the position on every level of the pattern, innermost first, i.e. the digits of the expansion position
        sizes = torch.as_tensor(level_sizes, dtype=torch.long)
        return (positions.unsqueeze(1) // self._level_places(sizes)) % sizes


    def _check_merged_ends(self, positions, indices, pattern_id, replicate_levels, target):
        # positions and indices hold the given expansion positions and the nodes of both ends of every edge, or of every
        # node. A merged copy's position resolves to its twin's node, and target reads from the nodes' own expansion
        # positions. Refuse a row that reads differently from there than from the given positions: on a replicated
        # level the step between its two ends must stay the same, on a frozen level the ends themselves. A node is both
        # ends of a step that never changes, so only its frozen levels count
        src_positions, dest_positions = positions[0], positions[-1]
        src_indices, dest_indices = indices[0], indices[-1]

        level_sizes, level_closes = self._infer_level_sizes(pattern_id)
        sizes = torch.tensor(level_sizes, dtype=torch.long)
        closes = torch.tensor(level_closes, dtype=torch.bool)
        swept = self._swept_levels(replicate_levels, len(level_sizes), pattern_id)

        expansion_position = self.expansion_position.view(-1)
        node_src_positions, node_dest_positions = expansion_position[src_indices], expansion_position[dest_indices]
        given_src = self._level_digits(src_positions, level_sizes)
        given_dest = self._level_digits(dest_positions, level_sizes)
        node_src = self._level_digits(node_src_positions, level_sizes)
        node_dest = self._level_digits(node_dest_positions, level_sizes)

        step_change = (given_dest - given_src) - (node_dest - node_src)
        same_step = torch.where(closes, step_change % sizes == 0, step_change == 0)
        same_ends = (given_src == node_src) & (given_dest == node_dest)

        differs = torch.where(~torch.where(swept, same_step, same_ends).all(dim=1))[0]
        if differs.numel() == 0:
            return

        row = differs[0].item()
        if len(positions) == 1:
            raise ValueError(
                f"Row {row} is given at expansion position {src_positions[row].item()} (level positions "
                f"{given_src[row].tolist()}), but that copy was merged into the one at expansion position "
                f"{node_src_positions[row].item()} (level positions {node_src[row].tolist()}), which differs on a "
                f"level that replicate_levels freezes, so {target} would update other copies. Give it at a copy that "
                "was not merged."
            )

        raise ValueError(
            f"Edge {row} is given from expansion position {src_positions[row].item()} to {dest_positions[row].item()} "
            f"(level positions {given_src[row].tolist()} to {given_dest[row].tolist()}), but an end of it was merged "
            f"into another copy, so its nodes sit at expansion positions {node_src_positions[row].item()} and "
            f"{node_dest_positions[row].item()} (level positions {node_src[row].tolist()} and "
            f"{node_dest[row].tolist()}), from where {target} would read a different edge. Give the edge at a copy "
            "whose ends were not merged."
        )


    def _expansion_positions_by_level(self, ends):
        # ends holds, for every end of an edge or for the nodes, (name of its expansion ids, expansion ids, name of its level
        # positions, level positions). The expansions must exist and belong to one pattern, whose level sizes turn each
        # end's level positions into expansion positions. Returns (expansion ids, expansion positions) for every end
        self._require_pattern()

        expansion_ids = [torch.as_tensor(ids, dtype=torch.long).view(-1) for _, ids, _, _ in ends]
        if len({ids.shape[0] for ids in expansion_ids}) > 1:
            raise ValueError(
                f"{' and '.join(name for name, _, _, _ in ends)} must have one entry per edge, got "
                f"{' and '.join(str(ids.shape[0]) for ids in expansion_ids)}."
            )

        num_rows = expansion_ids[0].shape[0]
        if num_rows == 0:
            return [(ids, ids.clone()) for ids in expansion_ids]

        # the pattern of every expansion, taken from its nodes; the level sizes depend on it
        expansion_id = self.expansion_id.view(-1)
        tracked = expansion_id != -1
        num_expansions = int(expansion_id.max().item()) + 1 if expansion_id.numel() > 0 else 0
        expansion_pattern = torch.full((num_expansions,), -1, dtype=torch.long)
        expansion_pattern[expansion_id[tracked]] = self.pattern_id.view(-1)[tracked]

        all_expansion_ids = torch.cat(expansion_ids)
        known = (all_expansion_ids >= 0) & (all_expansion_ids < num_expansions)
        pattern_ids = torch.full_like(all_expansion_ids, -1)
        pattern_ids[known] = expansion_pattern[all_expansion_ids[known]]
        if torch.any(pattern_ids == -1):
            raise ValueError(f"No expansion found with id(s) {all_expansion_ids[pattern_ids == -1].unique().tolist()}.")

        pattern_id = self._unique_pattern(pattern_ids, "expansions")
        level_sizes, _ = self._infer_level_sizes(pattern_id)
        pattern_name = self._pattern_names()[pattern_id]

        return [
            (ids, self._expansion_positions_from_levels(level_name, levels, level_sizes, num_rows, pattern_name))
            for ids, (_, _, level_name, levels) in zip(expansion_ids, ends)
        ]


    def add_edges_by_pattern_level(self, src_expansion_ids, dest_expansion_ids, src_level_positions, dest_level_positions, replicate_levels = None, **kwargs):
        """
        Resolves each new edge's two endpoints, given by expansion id and one position per pattern
        level, to expansion positions, then calls ``add_edges_per_expansion_by_pattern``. The level
        positions of a node say which copy it is on every nesting level of the pattern,
        innermost first: for an array nested inside a rotation, ``[0, 5]`` is the first node of
        the array in the sixth rotated copy. They replace the single expansion position that
        ``add_edges_per_expansion_by_pattern`` takes, ``0 + 5 * (size of the array)`` here. Level
        positions of a copy that was merged into another one, by ``add_nodes_by_pattern`` or
        ``merge``, address the node it was merged into.

        Args:
            src_expansion_ids (list[int]): expansion id for each new edge's source.
            dest_expansion_ids (list[int]): expansion id for each new edge's destination.
            src_level_positions (list[list[int]] | torch.Tensor): for each new edge's source, its
                position on every level of the pattern, innermost first. Shape [E, L] for E new
                edges and a pattern with L levels; the position on a level must be smaller than
                the size of that level.
            dest_level_positions (list[list[int]] | torch.Tensor): the same for each new edge's
                destination.
            replicate_levels (list[bool], optional): see ``add_edges_by_pattern``.
            **kwargs (dict[str, torch.Tensor]): see ``add_edges_by_pattern``.

        Raises:
            ValueError: if no pattern is registered.
            ValueError: if ``src_expansion_ids`` and ``dest_expansion_ids`` do not have the same length.
            ValueError: if an expansion id does not match any expansion.
            ValueError: if the given expansions belong to more than one registered pattern.
            ValueError: if ``src_level_positions`` or ``dest_level_positions`` does not hold
                integers, or does not have one row per edge with one position per level of the
                pattern.
            ValueError: if a level position is negative or not smaller than the size of its level.
            ValueError: any error raised by ``add_edges_per_expansion_by_pattern`` for the resolved
                expansion positions, e.g. if a level position matches no node, if ``replicate_levels``
                does not have one entry per level of the pattern, or if an end of an edge is a copy
                that was merged into another one, from whose node ``add_edges_by_pattern`` would
                copy a different edge.
        """
        (src_expansion_ids, src_positions), (dest_expansion_ids, dest_positions) = self._expansion_positions_by_level([
            ("src_expansion_ids", src_expansion_ids, "src_level_positions", src_level_positions),
            ("dest_expansion_ids", dest_expansion_ids, "dest_level_positions", dest_level_positions),
        ])

        self.add_edges_per_expansion_by_pattern(
            src_expansion_ids, dest_expansion_ids, src_positions, dest_positions, replicate_levels=replicate_levels, **kwargs
        )


    def delete_edges(self, mask: torch.Tensor):
        """
        Deletes edges where ``mask`` is ``True``.

        Args:
            mask (torch.Tensor): boolean tensor of shape ``[num_edges]``. Edges
                where the mask is ``True`` are removed.
        """
        keep = ~mask.view(-1)
        old_to_new = torch.full((keep.shape[0],), -1, dtype=torch.long)
        old_to_new[keep] = torch.arange(keep.sum())

        self.reciprocal_edge = old_to_new[self.reciprocal_edge[keep]]
        self.edge_index = self.edge_index[:, keep]
        self.directed_mask = self.directed_mask[keep]

        for attr in self.metadata["edge_attr_list"]:
            setattr(self, attr, getattr(self, attr)[keep])


    def get_node_index_from_name(self, name):

        return (self.name == encode(name)).nonzero(as_tuple=True)[0]


    @requires_metadata
    def add_edge(self, src: str, dst: str, name=None, **kwargs):
        """
        Adds a new edge from `src` to `dst`.
        """
        # Check for unexpected attributes
        unexpected_attrs = set(kwargs.keys()) - set(self.metadata["edge_attr_list"])
        if unexpected_attrs:
            raise ValueError(
                f"Unexpected edge attributes: {unexpected_attrs}. Expected: {list(self.metadata['edge_attr_list'])}"
            )

        # Add nodes if they do not exist
        if src not in self.metadata["node_name_to_index"]:
            self.add_node(src)
        if dst not in self.metadata["node_name_to_index"]:
            self.add_node(dst)

        # Add edge to edge_name_to_index
        main_name = f"{src}-{dst}" if name is None else name
        self.metadata["edge_name_to_index"][main_name] = self.num_edges
        reciprocal_name = f"{dst}-{src}" if name is None else f"{name}_reciprocal"
        self.metadata["edge_name_to_index"][reciprocal_name] = self.num_edges + 1

        # Update directed mask and reciprocal edge
        self.directed_mask = torch.cat(
            [self.directed_mask, torch.tensor([[True], [False]])], dim=0
        )
        self.reciprocal_edge = torch.cat(
            [
                self.reciprocal_edge,
                torch.tensor([[self.num_edges + 1], [self.num_edges]]),
            ],
            dim=0,
        )

        # Add edge to edge_index
        src_index, dst_index = (
            self.metadata["node_name_to_index"][src],
            self.metadata["node_name_to_index"][dst],
        )
        new_edge = torch.tensor([[src_index, dst_index], [dst_index, src_index]])
        self.edge_index = torch.cat([self.edge_index, new_edge], dim=1)

        # Add edge attributes
        for attr in self.metadata["edge_attr_list"]:
            if attr in kwargs:
                value = kwargs[attr]

                # Cast non-tensor attributes to tensor
                if not isinstance(value, torch.Tensor):
                    value = torch.tensor(value, dtype=getattr(self, attr).dtype)
                    # warnings.warn(f"Edge '({src}, {dst})' attribute '{attr}' was automatically cast to a tensor.", UserWarning)

            # If attribute is not provided, set it to default value
            elif attr in self.metadata["default_attrs"]:
                value = self.metadata["default_attrs"][attr]

            # If attribute has no default value, set it to zero
            else:
                attr_shape = getattr(self, attr).shape[1:]
                value = getattr(self, attr).new_zeros(attr_shape)
                # warnings.warn(f"Edge '({src}, {dst})' attribute '{attr}' initialized to zeros (no default value provided).", UserWarning)

            if value.dim() == 0:
                value = value.unsqueeze(0)

            setattr(
                self,
                attr,
                torch.cat(
                    [getattr(self, attr), value.unsqueeze(0), value.unsqueeze(0)],
                    dim=0,
                ),
            )

    @requires_metadata
    def successors(self, node_name):
        # Todo: implement networkx-style successors
        raise NotImplementedError

    neighbors = successors

    @requires_metadata
    def predecessors(self, node_name):
        # Todo: implement networkx-style predecessors
        raise NotImplementedError


    def _merge_candidates(self, merged_nodes, group_of_node, unique_groups):
        # the merged nodes that may survive their group: in a group with nodes of an expansion only those, so that the expansion
        # keeps a node at every position. Nodes of two expansions cannot be merged into one
        if not hasattr(self, "expansion_id"):
            return torch.ones_like(merged_nodes, dtype=torch.bool)

        num_groups = unique_groups.shape[0]
        expansion = self.expansion_id.view(-1)[merged_nodes]
        in_expansion = expansion != -1

        # scatter fills a group without expansion nodes with 0, which the count tells apart from expansion 0
        expansion_nodes_per_group = scatter(in_expansion.long(), group_of_node, dim=0, dim_size=num_groups, reduce="sum")
        lowest_expansion = scatter(expansion[in_expansion], group_of_node[in_expansion], dim=0, dim_size=num_groups, reduce="min")
        highest_expansion = scatter(expansion[in_expansion], group_of_node[in_expansion], dim=0, dim_size=num_groups, reduce="max")

        several_expansions = lowest_expansion != highest_expansion
        if torch.any(several_expansions):
            members = torch.where((group_of_node == torch.where(several_expansions)[0][0]) & in_expansion)[0]
            position = self.expansion_position.view(-1)[merged_nodes[members]]
            copies = [f"expansion {o}, position {p}" for o, p in zip(expansion[members].tolist(), position.tolist())]
            message = f"{', '.join(copies[:-1])} and {copies[-1]} would be merged into one node, but a node of one expansion cannot be merged into a node of another."
            raise ValueError(message[0].upper() + message[1:])

        return in_expansion | (expansion_nodes_per_group[group_of_node] == 0)


    def _record_merged_expansion_positions(self, nodes, survivors):
        # records each removed node of an expansion in expansion_holes, so that its expansion position resolves to its survivor, a node
        # of the same expansion (see _merge_candidates). Rows that pointed at a removed position now point at its survivor
        if not hasattr(self, "expansion_id"):
            return

        expansion_id = self.expansion_id.view(-1)
        expansion_position = self.expansion_position.view(-1)

        removed = (nodes != survivors) & (expansion_id[nodes] != -1)
        nodes, survivors = nodes[removed], survivors[removed]
        if nodes.numel() == 0:
            return

        expansion = expansion_id[nodes]
        removed_position = expansion_position[nodes]
        survivor_position = expansion_position[survivors]

        expansion_holes = getattr(self, "expansion_holes", torch.empty((0, 3), dtype=torch.long)).clone()

        # one key per (expansion, position); every expansion position is below the size of the largest pattern
        width = int(torch.bincount(self.pattern_matrix_id).max().item())
        removed_keys = expansion * width + removed_position
        order = torch.argsort(removed_keys)
        sorted_keys = removed_keys[order]
        twin_keys = expansion_holes[:, 0] * width + expansion_holes[:, 2]
        found = torch.searchsorted(sorted_keys, twin_keys).clamp(max=sorted_keys.numel() - 1)
        follows = sorted_keys[found] == twin_keys
        expansion_holes[follows, 2] = survivor_position[order][found[follows]]

        self.expansion_holes = torch.cat([expansion_holes, torch.stack([expansion, removed_position, survivor_position], dim=1)], dim=0)


    @staticmethod
    def _tolerance(values):
        # up to which difference two float values of an attribute count as the same: 1e-5 relative to the largest
        # absolute value, as float rounding errors grow with the values, but at least 1e-5
        return 1e-5 * max(1., values.abs().max().item()) if values.numel() > 0 else 1e-5


    @staticmethod
    def _rows_differ(a, b, tolerance):
        # which rows of a differ from the same rows of b: floats by more than tolerance, everything else at all
        a = a.flatten(1) if a.dim() > 1 else a.unsqueeze(1)
        b = b.flatten(1) if b.dim() > 1 else b.unsqueeze(1)
        if a.is_floating_point():
            return ((a - b).abs() > tolerance).any(dim=1)
        return (a != b).any(dim=1)


    def _differing_attrs(self, attrs, removed, kept):
        # every attribute in which some removed row differs from the row kept in its place, with the first such row
        differing = {}
        for attr in attrs:
            value = getattr(self, attr)
            tolerance = self._tolerance(value) if value.is_floating_point() else 0
            rows = torch.where(self._rows_differ(value[removed], value[kept], tolerance))[0]
            if rows.numel() > 0:
                differing[attr] = rows[0].item()

        return differing


    def merge(self, merge_group_id = None, node_priority_attribute: str = None, edge_priority_attribute: str = None, error_on_mismatch = False):
        """
        Merges nodes into groups and updates edges accordingly.

        Each node is assigned a group id via merge_group_id. Within each group
        the surviving node is the one with the lowest original index, or, if provided, the one with
        the highest value of ``node_priority_attribute`` (ties broken by
        lowest index).

        With a registered pattern, a group may hold nodes of at most one expansion, besides any nodes
        outside any pattern, and a node of that expansion always survives: the rule above picks the
        survivor among the expansion's nodes only. The expansion positions of the expansion's removed nodes
        are recorded in the ``expansion_holes`` graph attribute and resolve to the survivor wherever
        an expansion position is used, as for the copies ``add_nodes_by_pattern`` merges.

        Merged nodes appear first in the new node ordering, followed
        by the unmerged nodes in their original relative order.

        After merging, edges between nodes that now map to the same merged node are
        dropped. When multiple edges connect the same pair of new nodes, the one with
        the lowest original index is kept, or, if provided, the one with the highest value of
        ``edge_priority_attribute`` (ties again broken by lowest index).

        If ``merge_group_id`` is not passed, the method falls back to a
        ``merge_group_id`` node attribute on the graph.

        Args:
            merge_group_id (torch.Tensor, optional): shape [N] integer tensor
                assigning each node to a merge group. Use ``-1`` to leave a node
                unmerged.
            node_priority_attribute (str, optional): name of a scalar node attribute
                (float or int) used to select the surviving node in case of merge conflict.
                The node with the highest value is kept.
            edge_priority_attribute (str, optional): name of a scalar edge attribute
                (float or int) used to select the surviving edge in case of merge conflict. The edge with the highest value is kept.
            error_on_mismatch (bool, optional): raise an error instead of the warning if, without
                ``node_priority_attribute``, merged nodes differ in another attribute, or if, without
                ``edge_priority_attribute``, edges that now connect the same pair of nodes differ. The
                error is raised before anything is changed. Defaults to ``False``.

        Raises:
            ValueError: if no ``merge_group_id`` is passed and no ``merge_group_id``
                node attribute exists on the graph.
            ValueError: if ``node_priority_attribute`` or ``edge_priority_attribute``
                is not a registered attribute or has an unsupported dtype.
            ValueError: if a group holds nodes of more than one expansion.
            ValueError: if ``error_on_mismatch`` is set and merged nodes or edges differ, see
                ``error_on_mismatch``.

        Warns:
            UserWarning: if ``node_priority_attribute`` ranks a node outside any pattern above
                all nodes of the expansion in its group; a node of the expansion survives anyway.
            UserWarning: if no ``node_priority_attribute`` is given, ``error_on_mismatch`` is not set
                and merged nodes differ in another attribute, since only the surviving node's values
                are kept. Float values count as the same up to 1e-5 times the attribute's largest
                absolute value, but at least up to 1e-5. Pattern bookkeeping is not compared.
            UserWarning: if no ``edge_priority_attribute`` is given, ``error_on_mismatch`` is not set
                and edges that now connect the same pair of nodes differ in an attribute, since only
                the surviving edge's values are kept.
        """

        if merge_group_id is None:
            if "merge_group_id" in self.metadata["node_attr_list"]:
                merge_group_id = getattr(self, "merge_group_id").view(-1)

            else:
                raise ValueError(f"No merge_group_id given and no merge_group_id attribute on the graph found. Either pass merge group ids or create them as a node attribute on the graph.")


        if edge_priority_attribute is not None:

            if edge_priority_attribute not in self.metadata["edge_attr_list"]:
                raise ValueError(f"edge_priority_attribute '{edge_priority_attribute}' not found in edge_attr_list: {self.metadata['edge_attr_list']}.")

            edge_priority = getattr(self, edge_priority_attribute)

            if edge_priority.dim() > 1 and edge_priority.shape[1] > 1:
                raise ValueError(f"edge_priority_attribute '{edge_priority_attribute}' must be scalar (shape (n,) or (n,1)), got {edge_priority.shape}.")
            if edge_priority.dtype == torch.bool or edge_priority.is_complex():
                raise ValueError(f"edge_priority_attribute '{edge_priority_attribute}' must be float or int, got {edge_priority.dtype}.")

        old_num_nodes = self.num_nodes

        merged_nodes_mask = merge_group_id != -1
        merged_nodes = torch.where(merged_nodes_mask)[0]
        valid_merge_groups = merge_group_id[merged_nodes_mask]

        # only process groups that actually have members (group IDs may not be contiguous). Each merged node's group is
        # numbered by its rank among them, so that scatter needs one slot per group instead of one per possible id
        unique_groups, group_of_node = torch.unique(valid_merge_groups, return_inverse=True)
        num_unique_groups = unique_groups.shape[0]

        # the nodes that may survive their group; with a pattern, an expansion's nodes win over nodes outside any pattern
        candidate = self._merge_candidates(merged_nodes, group_of_node, unique_groups)

        #nodes get merged into node with highest value of given node_priority_attribute
        if node_priority_attribute is not None:

            if node_priority_attribute in self.metadata["node_attr_list"]:

                priority_attribute = getattr(self, node_priority_attribute)

                if priority_attribute.dim() > 1 and priority_attribute.shape[1] > 1:
                    raise ValueError(f"node_priority_attribute '{node_priority_attribute}' must be scalar (shape (n,) or (n,1)), got {priority_attribute.shape}.")
                if priority_attribute.dtype == torch.bool or priority_attribute.is_complex():
                    raise ValueError(f"node_priority_attribute '{node_priority_attribute}' must be float or int, got {priority_attribute.dtype}.")

                attr_values = priority_attribute.view(-1)[merged_nodes]
                candidate_values = attr_values[candidate]
                candidate_groups = group_of_node[candidate]
                max_vals = scatter(candidate_values, candidate_groups, dim=0, dim_size=num_unique_groups, reduce="max")
                is_priority = candidate_values == max_vals[candidate_groups]
                priority_node_index = scatter(merged_nodes[candidate][is_priority], candidate_groups[is_priority], dim=0, dim_size=num_unique_groups, reduce="min")

                # a node outside any pattern that outranks every node of the expansion in its group loses anyway
                overridden = scatter(attr_values, group_of_node, dim=0, dim_size=num_unique_groups, reduce="max") > max_vals
                if torch.any(overridden):
                    group = torch.where(overridden)[0][0]
                    outside = torch.where((group_of_node == group) & ~candidate)[0]
                    strongest = merged_nodes[outside[attr_values[outside].argmax()]].item()
                    kept = priority_node_index[group].item()
                    more = f" (and in {int(overridden.sum()) - 1} more merge group(s))" if overridden.sum() > 1 else ""
                    warnings.warn(
                        f"Node {strongest}, outside any pattern, has the highest '{node_priority_attribute}' among the "
                        f"nodes merged into expansion {self.expansion_id.view(-1)[kept].item()}, position "
                        f"{self.expansion_position.view(-1)[kept].item()}{more}, but the expansion's node is kept so that the "
                        "expansion stays intact.",
                        UserWarning,
                    )

            else:
                raise ValueError(f"node_priority_attribute '{node_priority_attribute}' not found in node_attr_list: {self.metadata['node_attr_list']}.")

        else:
            priority_node_index = scatter(merged_nodes[candidate], group_of_node[candidate], dim=0, dim_size=num_unique_groups, reduce="min")

            survivors = priority_node_index[group_of_node]
            removed = merged_nodes != survivors
            removed_nodes, kept_nodes = merged_nodes[removed], survivors[removed]
            unchecked = {"expansion_id", "expansion_position", "pattern_id"}
            differing = self._differing_attrs(
                [attr for attr in self.metadata["node_attr_list"] if attr not in unchecked], removed_nodes, kept_nodes
            )

            if differing and error_on_mismatch:
                attr, row = next(iter(differing.items()))
                kept_node, removed_node = kept_nodes[row].item(), removed_nodes[row].item()
                expansion_id = getattr(self, "expansion_id", None)
                if expansion_id is not None and expansion_id.view(-1)[removed_node] != -1:
                    expansion_position = self.expansion_position.view(-1)
                    pair = (
                        f"In expansion {expansion_id.view(-1)[removed_node].item()}, positions "
                        f"{expansion_position[kept_node].item()} and {expansion_position[removed_node].item()}"
                    )
                else:
                    pair = f"Nodes {kept_node} and {removed_node}"
                raise ValueError(f"{pair} would be merged into one node, but their '{attr}' differs between them.")

            if differing:
                warnings.warn(
                    f"Merged nodes differ in {list(differing)}; each group keeps the values of its surviving node. "
                    "Pass node_priority_attribute to choose which node survives.",
                    UserWarning,
                )

        # the edges after merging are worked out before anything changes, so that their checks can still refuse the merge
        new_num_nodes = num_unique_groups + int((~merged_nodes_mask).sum())

        # build node-index remapping: the merged groups first, in group order, then the unmerged nodes
        new_node_index = torch.full((old_num_nodes,), -1, dtype=torch.long)

        new_node_index[merged_nodes_mask] = group_of_node
        new_node_index[~merged_nodes_mask] = num_unique_groups + torch.arange((~merged_nodes_mask).sum())

        # remap edges
        src, dst = self.edge_index[0], self.edge_index[1]
        new_src = new_node_index[src]
        new_dst = new_node_index[dst]

        # drop intra-group edges
        inter_mask = (new_src != new_dst)
        new_src = new_src[inter_mask]
        new_dst = new_dst[inter_mask]

        # deduplicate: keep lowest original edge index per (new_src, new_dst) pair
        original_edge_indices = torch.where(inter_mask)[0]

        # encode each (new_src, new_dst) pair as one key, and number the distinct pairs, so that scatter needs one slot per
        # pair instead of one per possible key
        edge_keys = new_src * new_num_nodes + new_dst
        unique_keys, pair_of_edge = torch.unique(edge_keys, return_inverse=True)
        num_pairs = unique_keys.shape[0]

        if edge_priority_attribute is not None:
            attr_values = edge_priority.view(-1)[original_edge_indices]
            max_vals = scatter(attr_values, pair_of_edge, dim=0, dim_size=num_pairs, reduce="max")
            is_priority = attr_values == max_vals[pair_of_edge]
            priority_edge = scatter(original_edge_indices[is_priority], pair_of_edge[is_priority], dim=0, dim_size=num_pairs, reduce="min")

        else:
            priority_edge = scatter(original_edge_indices, pair_of_edge, dim=0, dim_size=num_pairs, reduce="min")

            kept = priority_edge[pair_of_edge]
            removed = original_edge_indices != kept
            removed_edges, kept_edges = original_edge_indices[removed], kept[removed]
            differing = self._differing_attrs(self.metadata["edge_attr_list"], removed_edges, kept_edges)

            if differing and error_on_mismatch:
                attr, row = next(iter(differing.items()))
                kept_pair = tuple(self.edge_index[:, kept_edges[row]].tolist())
                removed_pair = tuple(self.edge_index[:, removed_edges[row]].tolist())
                raise ValueError(
                    f"Edges {kept_pair} and {removed_pair} would be merged into one edge, but their '{attr}' differs "
                    "between them."
                )

            if differing:
                warnings.warn(
                    f"Merged edges differ in {list(differing)}; each pair of nodes keeps the values of its surviving edge. "
                    "Pass edge_priority_attribute to choose which edge survives.",
                    UserWarning,
                )

        self._record_merged_expansion_positions(merged_nodes, priority_node_index[group_of_node])

        #set new node attributes
        for attr in self.metadata["node_attr_list"]:

            value = getattr(self, attr)
            merged_nodes = value[priority_node_index]
            non_merged_nodes = value[~merged_nodes_mask]

            setattr(
                self,
                attr,
                torch.cat([merged_nodes, non_merged_nodes], dim=0),
            )
    
        self.num_nodes = new_num_nodes


        # clean up data structure if no edges remain after filtering
        if num_pairs == 0:

            self.edge_index = self.edge_index[:, :0]
            self.directed_mask = self.directed_mask[:0]
            self.reciprocal_edge = self.reciprocal_edge[:0]

            for attr in self.metadata["edge_attr_list"]:
                setattr(self, attr, getattr(self, attr)[:0])

            return

        #decode edge keys
        self.edge_index = torch.stack([unique_keys // self.num_nodes, unique_keys % self.num_nodes], dim=0)


        #set new edge attributes
        for attr in self.metadata["edge_attr_list"]:
            
            setattr(
                self,
                attr,
                getattr(self, attr)[priority_edge]
            )

        #set new directed edge mask and reciprocal edges
        old_num_edges = inter_mask.shape[0]
        old_to_new_edge = torch.full((old_num_edges,), -1, dtype=torch.long)
        old_to_new_edge[priority_edge] = torch.arange(priority_edge.numel())

        self.directed_mask = self.directed_mask[priority_edge]
        self.reciprocal_edge = old_to_new_edge[self.reciprocal_edge[priority_edge]]

        return  


    @requires_metadata
    def merge_nodes(self, name: str, nodes: list[str]):
        """
        Merges multiple nodes into a single node. The node inherits the set of edges connected to the
        merged nodes. Edges between merged nodes are removed.
        """
        # Check if node exists
        if name not in self.metadata["node_name_to_index"]:
            raise ValueError(f"Node '{name}' does not exist!")

        main_node_index = self.metadata["node_name_to_index"][name]

        for node in nodes:
            if node not in self.metadata["node_name_to_index"]:
                raise ValueError(f"Node '{node}' does not exist!")

            node_index = self.metadata["node_name_to_index"][node]

            # replace all occurence of node_index with main_node_index
            self.edge_index = torch.where(
                self.edge_index == node_index,
                main_node_index,
                self.edge_index,
            )

        for node in nodes:
            del_node_index = self.metadata["node_name_to_index"][node]

            # remove node from all node attributes
            for attr in self.metadata["node_attr_list"]:
                values = getattr(self, attr)
                values = torch.cat(
                    [values[:del_node_index], values[del_node_index + 1 :]], dim=0
                )
                setattr(self, attr, values)
            # remove node from node_name_to_index
            del self.metadata["node_name_to_index"][node]

            # decrement number of nodes
            self.num_nodes -= 1

            # update node_name_to_index
            for node_name, node_index in self.metadata["node_name_to_index"].items():
                if node_index > del_node_index:
                    self.metadata["node_name_to_index"][node_name] = node_index - 1

            # update edge_index
            self.edge_index = torch.where(
                self.edge_index > del_node_index,
                self.edge_index - 1,
                self.edge_index,
            )

    def __repr__(self):
        attrs = {k: v for k, v in self._store.items() if k != "metadata"}
        attr_str = ", ".join(f"{k}={v.shape if isinstance(v, torch.Tensor) else v}" for k, v in attrs.items())
        return f"{self.__class__.__name__}({attr_str})"


    @staticmethod
    def _shifted_grid_groups(values, eps):
        # rounds every column to a multiple of eps, once as is and once shifted by half a step, and groups two rows if
        # they land in the same cell for at least one combination of shifted and unshifted columns. The edges of the two
        # grids are half a step apart, so rows that practically coincide are split by at most one of them per column
        scaled = values.double() / eps
        num_rows, num_columns = scaled.shape

        cells = []
        for shift in itertools.product((0., 0.5), repeat=num_columns):
            _, cell = torch.unique(torch.round(scaled + torch.tensor(shift, dtype=torch.float64)), dim=0, return_inverse=True)
            cells.append(cell)

        # rows that share a cell in any grid belong together: hand the smallest row index on until nothing changes
        group = torch.arange(num_rows)
        while True:
            previous = group
            for cell in cells:
                group = scatter(group, cell, dim=0, dim_size=int(cell.max().item()) + 1, reduce="min")[cell]
            if torch.equal(group, previous):
                return group


    def merge_by_attribute(self, attr: str, f = None, mask = None, node_priority_attribute: str = None, edge_priority_attribute: str = None, error_on_mismatch = False):
        """
        Merges nodes that share the same value of a node attribute.

        Nodes are grouped by the value of ``attr``, and nodes whose value no other node shares
        are left unmerged. Float values count as the same if they differ by less than the
        attribute's tolerance in every column: 1e-5 times its largest absolute value over all
        nodes, but at least 1e-5, the same tolerance ``merge`` compares merged nodes with. Each
        column is rounded to a multiple of the tolerance, once as is and once shifted by half a
        step, and two nodes are grouped if they land in the same cell for at least one
        combination of shifted and unshifted columns, so a value right between two multiples
        cannot split nodes that practically coincide. The number of roundings doubles with every
        column of the attribute (8 for coordinates). Groups that share a node are joined. For
        boolean attributes, only ``True`` nodes are merged together, ``False`` nodes are left as
        independent unmerged nodes.

        Args:
            attr (str): name of a registered node attribute to group by. Must be a boolean, int, or float attribute.
            f (callable, optional): optional transformation applied to the attribute
                tensor before grouping. Useful for grouping by a derived quantity
                without storing it as a separate attribute. The tolerance then follows the values
                ``f`` returns.
            mask (torch.Tensor, optional): boolean tensor of shape ``[num_nodes]`` selecting the
                nodes that may be merged; the other nodes are left untouched. Defaults to
                ``None``, which considers every node.
            node_priority_attribute (str, optional): see ``merge``.
            edge_priority_attribute (str, optional): see ``merge``.
            error_on_mismatch (bool, optional): see ``merge``. ``attr`` itself never triggers it,
                since grouped values agree within the tolerance. Defaults to ``False``.

        Raises:
            ValueError: if ``attr`` is not a registered node attribute.
            ValueError: if the attribute dtype is not boolean, int, or float.
            ValueError: any error raised by ``merge`` for the resulting groups, e.g. if nodes of
                two expansions share a value, or if ``error_on_mismatch`` is set and merged nodes
                differ in another attribute.
        """

        if attr not in self.metadata["node_attr_list"]:
            raise ValueError(f"Unexpected node attribute '{attr}' for merging. Expected an attribute in: {list(self.metadata['node_attr_list'])}")

        value = getattr(self, attr)

        if f is not None:
            value = f(value)

        nodes = torch.arange(self.num_nodes) if mask is None else torch.where(mask.view(-1))[0]
        if nodes.numel() == 0:
            return

        # taken over all nodes, as merge does when it compares the merged nodes
        tolerance = self._tolerance(value) if value.is_floating_point() else 0
        value = value[nodes].reshape(nodes.shape[0], -1)

        if value.dtype in (torch.float32, torch.float64):
            group = self._shifted_grid_groups(value, tolerance)

        elif value.dtype == torch.bool:
            # every True node joins group 0, every False node gets a group of its own
            group = torch.where(value.all(dim=1), 0, torch.arange(1, nodes.shape[0] + 1))

        elif value.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
            _, group = torch.unique(value, dim=0, return_inverse=True)

        else:
            raise ValueError(f"Unexpected data-type. Merge based on attributes requires a torch boolean, float or int.")

        # a group of one has nothing to merge, so its node is left alone
        grouped = torch.bincount(group)[group] > 1
        merge_group_id = torch.full((self.num_nodes,), -1, dtype=torch.long)
        merge_group_id[nodes[grouped]] = group[grouped]

        if torch.any(grouped):
            self.merge(
                merge_group_id,
                node_priority_attribute=node_priority_attribute,
                edge_priority_attribute=edge_priority_attribute,
                error_on_mismatch=error_on_mismatch,
            )
    

    def _create_topology(self, num_nodes, num_edges, edge_indices, node_attrs: dict = {}, edge_attrs: dict = {}, default_node_attrs: dict = {}, default_edge_attrs: dict = {}):

        #save old_num_nodes to later shift edge indices 
        old_num_nodes = self.num_nodes

        if not node_attrs:
            self.add_n_empty_nodes(num_nodes)


        #resolve node kwargs and add to graph
        else:
            
            kwargs_nodes = self._construct_kwargs(node_attrs, num_nodes, default_node_attrs)
            self.add_nodes(**kwargs_nodes)


        #resolve edge kwargs and add to graph        
        kwargs_edges = self._construct_kwargs(edge_attrs, num_edges, default_edge_attrs)
        self.add_edges(edge_indices + old_num_nodes, **kwargs_edges) 

    

    def _construct_kwargs(self, attrs, num_new_entries, default_attrs):

        kwargs = {}

        for attr in attrs:
                
            f = attrs[attr]

            if isinstance(f, torch.Tensor):

                if f.dim() == 0:
                    f = f.unsqueeze(0)

                if f.size(0) == 1:

                    kwargs[attr] = f.repeat(num_new_entries, 1)

                elif f.size(0) == num_new_entries:

                    kwargs[attr] = f

                else:
                    raise ValueError(f"Node/Edge attribute '{attr}' must have one or '{num_new_entries}' rows, but has '{f.size(0)}' rows (Pass a node/edge attribute with one row to have a constant value over all newly added nodes/edges).")


            elif callable(f):

                f_kwargs = resolve_attrs(f, default_attrs, kwargs)

                kwargs[attr] = f(**f_kwargs)
            
            else:

                raise ValueError(f"Node/Edge attribute '{attr}' must be a tensor or a callable")
            

        return kwargs



    def _resolve_attr_value(self, attr, kwargs, num_rows):

        if attr in kwargs:
            value = kwargs[attr]

            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value, dtype=getattr(self, attr).dtype)

        elif attr in self.metadata["default_attrs"]:
            default = self.metadata["default_attrs"][attr]
            attr_shape = getattr(self, attr).shape[1:]
            value = default.expand(attr_shape).unsqueeze(0).repeat(num_rows, *([1] * len(attr_shape)))

        else:
            attr_shape = getattr(self, attr).shape[1:]
            value = getattr(self, attr).new_zeros(attr_shape).repeat(num_rows, 1)

        if value.dim() == 0:
            value = value.unsqueeze(0)

        return value


    @staticmethod
    def _expand_value_over_pattern(value, num_seeds, pattern_size):
        value = value.unsqueeze(1).expand(num_seeds, pattern_size, *value.shape[1:]).clone()
        return value.reshape(num_seeds * pattern_size, *value.shape[2:])


    @staticmethod
    def _apply_affine(matrices, values):
        ones = values.new_ones(*values.shape[:-1], 1)
        values_h = torch.cat([values, ones], dim=-1)
        return torch.einsum("kij,sj->ski", matrices, values_h)[..., :3]


    @staticmethod
    def _solve_affine(matrices, values):
        """Inverse of _apply_affine, paired one-to-one instead of broadcast: solves matrices[i] @ x[i] = values[i]."""
        ones = values.new_ones(*values.shape[:-1], 1)
        values_h = torch.cat([values, ones], dim=-1)
        return torch.linalg.solve(matrices, values_h)[..., :3]


    def _get_edge_unit_coords(self, x, y, edge_indices):

        x_src = x[edge_indices[0]]
        y_src= y[edge_indices[0]]
        x_dest = x[edge_indices[1]]
        y_dest = y[edge_indices[1]]
        x_edge = (x_src + x_dest) / 2
        y_edge = (y_src + y_dest) / 2

        return x_edge, y_edge


    @staticmethod
    def _check_vector(name, vector, nonzero=False):
        # the pattern matrices use the default float dtype, so other dtypes would fail later
        if not isinstance(vector, torch.Tensor):
            raise ValueError(f"{name} must be a torch.Tensor of shape [3], got {type(vector).__name__}.")

        if vector.shape != (3,) or vector.dtype != torch.get_default_dtype():
            raise ValueError(
                f"{name} must have shape [3] and dtype {torch.get_default_dtype()}, "
                f"got shape {list(vector.shape)} and dtype {vector.dtype}."
            )

        if nonzero and not torch.any(vector != 0):
            raise ValueError(f"{name} must not be the zero vector.")


    @staticmethod
    def _closing_flags(n, closes):
        flags = torch.ones(n, dtype=torch.bool)
        flags[n - 1] = closes
        return flags


    def create_rotational_pattern(self, n, origin = torch.tensor([0.0, 0.0, 0.0]), rotation_axis = torch.tensor([0.0, 0.0, 1.0]), closes: bool = True) -> dict[str, torch.Tensor]:
        """
        Builds an n-fold rotational pattern: ``n`` affine matrices rotating evenly by
        ``2*pi/n`` about ``rotation_axis``, through ``origin``.

        Args:
            n (int): number of rotational positions. Must be at least 1.
            origin (torch.Tensor): a point the rotation axis passes through. Shape [3].
            rotation_axis (torch.Tensor): the axis to rotate about. Shape [3].
            closes (bool): whether the positions close back onto the first one, i.e. whether an
                edge between two neighbouring positions is also created between the last and the
                first position. Defaults to ``True``, since the ``n`` rotations cover the full
                turn; pass ``False`` to leave the ring open, e.g. for a fan or a partial arc.

        Returns:
            ``{"matrices": torch.Tensor, "pattern_level": torch.Tensor, "closes": torch.Tensor}``
                -- ``n`` affine matrices of shape [n, 4, 4], a ``pattern_level`` of shape [n]
                (all zeros, since this is a single, unnested level), and ``closes`` of shape [n]
                (all ``True`` except the last entry, which is ``closes``). Pass this dict to
                ``add_patterns`` directly, or nest it inside another pattern via
                ``nest_patterns``.

        Raises:
            ValueError: if ``n`` is less than 1.
            ValueError: if ``origin`` or ``rotation_axis`` is not a tensor of shape [3] with the
                default float dtype.
            ValueError: if ``rotation_axis`` is the zero vector.
        """

        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}.")

        self._check_vector("origin", origin)
        self._check_vector("rotation_axis", rotation_axis, nonzero=True)

        axis = rotation_axis / rotation_axis.norm()
        ax, ay, az = axis[0], axis[1], axis[2]
        K = torch.tensor([
            [0.0, -az, ay],
            [az, 0.0, -ax],
            [-ay, ax, 0.0],
        ])

        angles = torch.arange(n, dtype=torch.float) * (2 * torch.pi / n)
        cos, sin = torch.cos(angles), torch.sin(angles)

        R = cos.view(-1, 1, 1) * torch.eye(3) + sin.view(-1, 1, 1) * K + (1 - cos).view(-1, 1, 1) * torch.outer(axis, axis)
        
        affine = torch.eye(4).expand(n, 4, 4).clone()
        affine[:, :3, :3] = R
        affine[:, :3, 3] = origin - R @ origin

        return {
            "matrices": affine,
            "pattern_level": torch.zeros(n, dtype=torch.long),
            "closes": self._closing_flags(n, closes),
        }


    def create_translational_pattern(self, n, direction = torch.tensor([1.0, 0.0, 0.0]), scaling = None, closes: bool = False) -> dict[str, torch.Tensor]:
        """
        Builds a translation operation: ``n`` affine matrices translating along ``direction``,
        i.e. an array of ``n`` positions. Without ``scaling`` the positions are evenly spaced,
        one ``direction`` apart; with ``scaling`` each step is scaled individually.

        Args:
            n (int): number of positions in the array, the seed included. Must be at least 1.
            direction (torch.Tensor): the step from one position to the next, used as given and
                not normalized. Shape [3].
            scaling (torch.Tensor, optional): one factor per step, so shape [n-1]: the step from
                position ``k`` to position ``k+1`` is ``scaling[k] * direction``. Defaults to
                ``None``, which spaces the positions evenly, as if every factor were 1.
            closes (bool): whether the array closes back onto its first position, i.e. whether an
                edge between two neighbouring positions is also created between the last and the
                first position. Defaults to ``False``, since an array of translated copies is
                normally open; pass ``True`` for a periodic array.

        Returns:
            ``{"matrices": torch.Tensor, "pattern_level": torch.Tensor, "closes": torch.Tensor}``
                -- ``n`` affine matrices of shape [n, 4, 4], a ``pattern_level`` of shape [n]
                (all zeros, since this is a single, unnested level), and ``closes`` of shape [n]
                (all ``True`` except the last entry, which is ``closes``). Pass this dict to
                ``add_patterns`` directly, or nest it inside another pattern via
                ``nest_patterns``.

        Raises:
            ValueError: if ``n`` is less than 1.
            ValueError: if ``direction`` is not a tensor of shape [3] with the default float
                dtype.
            ValueError: if ``direction`` is the zero vector.
            ValueError: if ``scaling`` is given and is not a tensor of shape [n-1].
        """

        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}.")

        self._check_vector("direction", direction, nonzero=True)

        if scaling is None:
            steps = torch.arange(n, dtype=direction.dtype)
        else:
            if not isinstance(scaling, torch.Tensor):
                raise ValueError(
                    f"scaling must be a torch.Tensor of shape [{n - 1}], got {type(scaling).__name__}."
                )

            if scaling.shape != (n - 1,):
                raise ValueError(
                    f"scaling must have shape [{n - 1}], one factor per step between the {n} "
                    f"positions, got shape {list(scaling.shape)}."
                )

            steps = torch.zeros(n, dtype=direction.dtype)
            steps[1:] = torch.cumsum(scaling.to(direction.dtype), dim=0)

        affine = torch.eye(4).expand(n, 4, 4).clone()
        affine[:, :3, 3] = steps.unsqueeze(1) * direction

        return {
            "matrices": affine,
            "pattern_level": torch.zeros(n, dtype=torch.long),
            "closes": self._closing_flags(n, closes),
        }


    def create_mirror_pattern(self, origin = torch.tensor([0.0, 0.0, 0.0]), normal = torch.tensor([1.0, 0.0, 0.0])) -> dict[str, torch.Tensor]:
        """
        Builds a mirror pattern with respect to the plane through ``origin`` with normal
        ``normal``.

        Args:
            origin (torch.Tensor): a point the mirror plane passes through. Shape [3].
            normal (torch.Tensor): the mirror plane's normal vector. Shape [3].

        Returns:
            ``{"matrices": torch.Tensor, "pattern_level": torch.Tensor, "closes": torch.Tensor}``
                -- 2 affine matrices of shape [2, 4, 4], a ``pattern_level`` of shape [2] (all
                zeros, since this is a single, unnested level), and ``closes`` of shape [2] (all
                ``True``: a mirror has only two positions, so closing it and leaving it open give
                the same single edge between them). Pass this dict to ``add_patterns`` directly,
                or nest it inside another pattern via ``nest_patterns``.

        Raises:
            ValueError: if ``origin`` or ``normal`` is not a tensor of shape [3] with the
                default float dtype.
            ValueError: if ``normal`` is the zero vector.
        """

        self._check_vector("origin", origin)
        self._check_vector("normal", normal, nonzero=True)

        normal = normal / normal.norm()
        M = torch.eye(3) - 2 * torch.outer(normal, normal)

        affine = torch.eye(4).expand(2, 4, 4).clone()
        affine[1, :3, :3] = M
        affine[1, :3, 3] = origin - M @ origin

        return {
            "matrices": affine,
            "pattern_level": torch.zeros(2, dtype=torch.long),
            "closes": torch.ones(2, dtype=torch.bool),
        }


    def nest_patterns(self, patterns: list) -> dict[str, torch.Tensor]:
        """
        Nests patterns inside each other, innermost first, producing every combination of
        them.

        Args:
            patterns (list[dict]): the patterns to nest, innermost first, each as returned
                by ``create_rotational_pattern``, ``create_translational_pattern``,
                ``create_mirror_pattern``, or a previous ``nest_patterns`` call.

        Returns:
            ``{"matrices": torch.Tensor, "pattern_level": torch.Tensor, "closes": torch.Tensor}``
                -- the nested affine matrices, shape ``[size, 4, 4]`` with ``size`` the product
                of the patterns' sizes, and the nested ``pattern_level`` and ``closes``, shape
                ``[size]`` each. Every level keeps its own closing behaviour, so nesting an open
                pattern inside another open one stays open in both directions.

        Raises:
            ValueError: if ``patterns`` is empty.
        """
        if len(patterns) == 0:
            raise ValueError("nest_patterns() needs at least one pattern.")

        nested = patterns[0]
        for outer in patterns[1:]:
            nested = self._nest_pattern_pair(nested, outer)
        return nested


    def _nest_pattern_pair(self, pattern_a, pattern_b):
        # nests the inner pattern_a inside the outer pattern_b

        affine = pattern_b["matrices"].unsqueeze(1) @ pattern_a["matrices"].unsqueeze(0)

        level_a, level_b = pattern_a["pattern_level"], pattern_b["pattern_level"]
        size_a, size_b = level_a.shape[0], level_b.shape[0]
        num_levels_a = int(level_a.max().item()) + 1

        combined_level = (level_b + num_levels_a).repeat_interleave(size_a)
        combined_level[:size_a] = level_a

        closes_a = pattern_a.get("closes", torch.ones(size_a, dtype=torch.bool))
        closes_b = pattern_b.get("closes", torch.ones(size_b, dtype=torch.bool))

        combined_closes = torch.ones(size_a * size_b, dtype=torch.bool)
        combined_closes[:size_a] = closes_a
        open_b = torch.where(~closes_b)[0]
        combined_closes[(open_b + 1) * size_a - 1] = False

        return {
            "matrices": affine.reshape(-1, 4, 4),
            "pattern_level": combined_level,
            "closes": combined_closes,
        }


    def _nodes_to_set(self, attr, mask, value):

        if attr not in self.metadata["node_attr_list"]:
            raise ValueError(f"Unexpected node attribute '{attr}'. Expected an attribute in: {list(self.metadata['node_attr_list'])}")

        if attr in ("expansion_id", "expansion_position", "pattern_id"):
            raise ValueError(f"'{attr}' is pattern bookkeeping and cannot be set.")

        mask = mask.view(-1)

        if mask.shape[0] != self.num_nodes:
            raise ValueError(f"mask must have {self.num_nodes} entries, but has {mask.shape[0]}.")

        nodes = torch.where(mask)[0]

        if value.shape[0] != nodes.shape[0]:
            raise ValueError(f"value must have one row per masked node ({nodes.shape[0]}), but has {value.shape[0]}.")

        return nodes


    def set_node_attr(self, attr: str, mask: torch.Tensor, value: torch.Tensor):
        """
        Sets a node attribute for the nodes selected by ``mask``. Only the selected nodes are
        updated, even if they belong to a registered pattern; use ``set_node_attr_by_pattern``
        to propagate the update to the rest of their expansions.

        Args:
            attr (str): name of the registered node attribute to set.
            mask (torch.Tensor): boolean tensor of shape ``[num_nodes]``. Selects the nodes
                to set.
            value (torch.Tensor): one row per ``True`` entry in ``mask``, in the same order.

        Raises:
            ValueError: if ``attr`` is not a registered node attribute, or is pattern
                bookkeeping (``expansion_id``, ``expansion_position``, ``pattern_id``).
            ValueError: if ``mask`` does not have ``num_nodes`` entries.
            ValueError: if ``value`` does not have one row per ``True`` entry in ``mask``.
        """

        nodes = self._nodes_to_set(attr, mask, value)

        current = getattr(self, attr)
        current[nodes] = value
        setattr(self, attr, current)


    def set_node_attr_by_pattern(self, attr: str, mask: torch.Tensor, value: torch.Tensor, replicate_levels = None):
        """
        Sets a node attribute for the nodes selected by ``mask`` and propagates the update to
        all pattern copies of each selected node (the rest of its expansion). Attributes registered
        as ``transform_attrs`` are geometrically transformed per copy, every other attribute is
        copied verbatim.

        Args:
            attr (str): name of the registered node attribute to set.
            mask (torch.Tensor): boolean tensor of shape ``[num_nodes]``. Selects the nodes
                to set.
            value (torch.Tensor): one row per ``True`` entry in ``mask``, in the same order.
            replicate_levels (list[bool], optional): one entry per nesting level of the
                pattern, innermost first, selecting the levels the update is propagated on. A
                level left out keeps the selected node's own position on that level, so the
                copies along it are not updated. Any subset works, so a level can be frozen
                between two replicated ones. Defaults to ``None``, which propagates on every
                level, i.e. to the whole expansion.

        Raises:
            ValueError: if ``attr`` is not a registered node attribute, or is pattern
                bookkeeping (``expansion_id``, ``expansion_position``, ``pattern_id``).
            ValueError: if ``mask`` does not have ``num_nodes`` entries.
            ValueError: if ``value`` does not have one row per ``True`` entry in ``mask``.
            ValueError: if no pattern is registered.
            ValueError: if a selected node is outside any pattern.
            ValueError: if the selected nodes belong to more than one registered pattern.
            ValueError: if two selected nodes belong to the same expansion.
            ValueError: if a selected node's expansion is no longer intact, e.g. because some of
                its nodes were merged or removed.
            ValueError: if ``replicate_levels`` does not have one entry per nesting level of
                the pattern.
            ValueError: if a node that several copies were merged into (see the ``merge_attr``
                argument of ``add_nodes_by_pattern``) would get different values through them,
                e.g. a node on a mirror plane moved off the plane.
        """

        nodes = self._nodes_to_set(attr, mask, value)

        pattern_id = self._single_pattern(nodes, outside_hint="Set them via set_node_attr().")
        if pattern_id is None:
            return

        if torch.any(torch.bincount(self.expansion_id.view(-1)[nodes]) > 1):
            raise ValueError("Multiple selected nodes belong to the same expansion. Select only one node per expansion.")

        matrices = self.pattern_matrices[self.pattern_matrix_id == pattern_id]
        pattern_size = matrices.shape[0]

        members, idx = self._expansion_canonical_members(nodes, pattern_size)

        # the expansion positions each update is propagated to, all of them unless replicate_levels
        # freezes a level
        positions = self._replicated_positions(idx, pattern_id, replicate_levels)
        members = torch.gather(members, 1, positions)

        if attr in self.metadata["pattern_transform_attrs"][pattern_id]:
            effective_seed = self._solve_affine(matrices[idx], value)
            new_values = self._apply_affine(matrices, effective_seed)
            new_values = new_values[torch.arange(new_values.shape[0]).unsqueeze(1), positions]
            new_values = new_values.reshape(-1, *new_values.shape[2:])

        else:
            new_values = self._expand_value_over_pattern(value, value.shape[0], positions.shape[1])

        # a node that several copies were merged into is reached through each of their expansion positions; it is
        # written once, so the values reaching it must agree
        rows = members.reshape(-1)
        order = torch.argsort(rows, stable=True)
        rows, new_values = rows[order], new_values[order]
        row_positions = positions.reshape(-1)[order]

        current = getattr(self, attr)
        tolerance = self._tolerance(torch.cat([current.reshape(-1), new_values.reshape(-1)])) if new_values.is_floating_point() else 0

        shared = rows[1:] == rows[:-1]
        differs = self._rows_differ(new_values[1:], new_values[:-1], tolerance)

        conflicts = torch.where(shared & differs)[0]
        if conflicts.numel() > 0:
            node = rows[conflicts[0] + 1]
            raise ValueError(
                f"In expansion {self.expansion_id.view(-1)[node].item()}, positions {row_positions[rows == node].tolist()} "
                f"were merged into one node, e.g. a node on a mirror plane or rotation axis, but the new '{attr}' "
                "differs between them. Set it via set_node_attr()."
            )

        first = torch.ones_like(rows, dtype=torch.bool)
        first[1:] = ~shared

        current[rows[first]] = new_values[first]
        setattr(self, attr, current)

    def _mask_and_reorder(self, indices: torch.Tensor, value: torch.Tensor, num_rows: int):
        if value.shape[0] != indices.shape[0]:
            raise ValueError(f"value must have one row per entry ({indices.shape[0]}), but has {value.shape[0]}.")

        order = torch.argsort(indices)
        indices = indices[order]
        value = value[order]

        mask = torch.zeros(num_rows, dtype=torch.bool)
        mask[indices] = True
        return mask, value


    def set_node_attr_by_name(self, attr: str, names: list, value: torch.Tensor):
        """
        Resolves node names to indices, then calls ``set_node_attr``. Each name selects
        every node that carries it, and every matching node is set to the given value.
        Pattern copies created by ``add_nodes_by_pattern`` share their seed's name, so a
        name selects a whole expansion in this case.

        Args:
            attr (str): name of the registered node attribute to set.
            names (list[str]): node name for each row of ``value``.
            value (torch.Tensor): one row per entry in ``names``.

        Raises:
            ValueError: if ``names`` contains duplicates.
            ValueError: if ``value`` does not have one row per entry in ``names``.
            ValueError: if a name in ``names`` does not match any node.
            ValueError: any error raised by ``set_node_attr`` for the resolved nodes.
        """
        if len(set(names)) != len(names):
            raise ValueError(f"names must not contain duplicates, got {names}.")

        if value.shape[0] != len(names):
            raise ValueError(f"value must have one row per name ({len(names)}), but has {value.shape[0]}.")

        indices, counts = [], []
        for name in names:
            nodes = self.get_node_index_from_name(name)
            if nodes.numel() == 0:
                raise ValueError(f"No node found with name '{name}'.")

            indices.extend(nodes.tolist())
            counts.append(nodes.numel())

        indices = torch.tensor(indices, dtype=torch.long)
        value = value.repeat_interleave(torch.tensor(counts, dtype=torch.long), dim=0)
        mask, value = self._mask_and_reorder(indices, value, self.num_nodes)

        self.set_node_attr(attr, mask, value)

    def set_node_attr_per_expansion(self, attr: str, expansion_ids: list, expansion_positions: list, value: torch.Tensor):
        """
        Resolves ``(expansion_id, expansion_position)`` pairs to node indices, then calls
        ``set_node_attr``.

        Args:
            attr (str): name of the registered node attribute to set.
            expansion_ids (list[int]): expansion id for each row of ``value`` as given by ``add_nodes_by_pattern``.
            expansion_positions (list[int]): position in the expansion for each row of ``value`` as given by ``add_nodes_by_pattern``.
            value (torch.Tensor): one row per entry in ``expansion_ids``/``expansion_positions``.

        Raises:
            AttributeError: if no pattern is registered.
            ValueError: if an ``(expansion_id, expansion_position)`` pair does not match any node.
            ValueError: if ``value`` does not have one row per entry in ``expansion_ids``/``expansion_positions``.
            ValueError: any error raised by ``set_node_attr`` for the resolved nodes.
        """
        (indices,) = self._node_indices_from_expansion([(expansion_ids, expansion_positions)])
        mask, value = self._mask_and_reorder(indices, value, self.num_nodes)

        self.set_node_attr(attr, mask, value)


    def set_node_attr_per_expansion_by_pattern(self, attr: str, expansion_ids: list, expansion_positions: list, value: torch.Tensor, replicate_levels = None):
        """
        Resolves ``(expansion_id, expansion_position)`` pairs to node indices, then calls
        ``set_node_attr_by_pattern``.

        Args:
            attr (str): name of the registered node attribute to set.
            expansion_ids (list[int]): expansion id for each row of ``value`` as given by ``add_nodes_by_pattern``.
            expansion_positions (list[int]): position in the expansion for each row of ``value`` as given by ``add_nodes_by_pattern``.
            value (torch.Tensor): one row per entry in ``expansion_ids``/``expansion_positions``.
            replicate_levels (list[bool], optional): see ``set_node_attr_by_pattern``.

        Raises:
            AttributeError: if no pattern is registered.
            ValueError: if an ``(expansion_id, expansion_position)`` pair does not match any node.
            ValueError: if a position is a copy that was merged into another one, by
                ``add_nodes_by_pattern`` or ``merge``, whose position differs on a level that
                ``replicate_levels`` freezes, so that ``set_node_attr_by_pattern``, which reads the
                node it was merged into, would update other copies. Give such a node at a copy that
                was not merged.
            ValueError: if ``value`` does not have one row per entry in ``expansion_ids``/``expansion_positions``.
            ValueError: any error raised by ``set_node_attr_by_pattern`` for the resolved nodes.
        """
        (indices,) = self._node_indices_from_expansion_by_pattern(
            [(expansion_ids, expansion_positions)], replicate_levels,
            outside_hint="Set them via set_node_attr().", target="set_node_attr_by_pattern",
        )
        mask, value = self._mask_and_reorder(indices, value, self.num_nodes)

        self.set_node_attr_by_pattern(attr, mask, value, replicate_levels)


    def set_node_attr_by_pattern_level(self, attr: str, expansion_ids: list, level_positions, value: torch.Tensor, replicate_levels = None):
        """
        Resolves each node, given by expansion id and one position per pattern level, to its expansion
        position, then calls ``set_node_attr_per_expansion_by_pattern``. The level positions of a node
        say which copy it is on every nesting level of the pattern, innermost first, see
        ``add_edges_by_pattern_level``. Level positions of a copy that was merged into another
        one, by ``add_nodes_by_pattern`` or ``merge``, address the node it was merged into.

        Args:
            attr (str): name of the registered node attribute to set.
            expansion_ids (list[int]): expansion id for each row of ``value``.
            level_positions (list[list[int]] | torch.Tensor): for each row of ``value``, the node's
                position on every level of the pattern, innermost first. Shape [N, L] for N rows
                and a pattern with L levels; the position on a level must be smaller than the size
                of that level.
            value (torch.Tensor): one row per entry in ``expansion_ids``.
            replicate_levels (list[bool], optional): see ``set_node_attr_by_pattern``.

        Raises:
            ValueError: if no pattern is registered.
            ValueError: if an expansion id does not match any expansion.
            ValueError: if the given expansions belong to more than one registered pattern.
            ValueError: if ``level_positions`` does not hold integers, or does not have one row per
                expansion id with one position per level of the pattern.
            ValueError: if a level position is negative or not smaller than the size of its level.
            ValueError: any error raised by ``set_node_attr_per_expansion_by_pattern`` for the resolved
                expansion positions, e.g. if a level position matches no node, if ``value`` does not have
                one row per entry in ``expansion_ids``, or if a level position addresses a copy that was
                merged into another one whose position differs on a level that ``replicate_levels``
                freezes.
        """
        ((expansion_ids, positions),) = self._expansion_positions_by_level([
            ("expansion_ids", expansion_ids, "level_positions", level_positions),
        ])

        self.set_node_attr_per_expansion_by_pattern(attr, expansion_ids, positions, value, replicate_levels=replicate_levels)

    def _edge_indices_for_pairs(self, src_indices: torch.Tensor, dest_indices: torch.Tensor, missing_message: str = None):
        all_keys = self.edge_index[0] * self.num_nodes + self.edge_index[1]
        query_keys = src_indices * self.num_nodes + dest_indices

        sorted_keys, sort_idx = torch.sort(all_keys)
        pos = torch.searchsorted(sorted_keys, query_keys)
        # a key beyond the largest edge key, e.g. on a graph without edges, has no edge
        in_range = pos < sorted_keys.numel()
        found = torch.zeros_like(in_range)
        found[in_range] = sorted_keys[pos[in_range]] == query_keys[in_range]
        if not torch.all(found):
            bad = torch.where(~found)[0].tolist()
            raise ValueError(missing_message or f"No edge found for (src, dest) pairs at query index {bad}.")

        return sort_idx[pos]

    def set_edge_attr_per_expansion(self, attr: str, src_expansion_ids: list, dest_expansion_ids: list, src_expansion_positions: list, dest_expansion_positions: list, value: torch.Tensor):
        """
        Resolves the ``(expansion_id, expansion_position)`` pairs of each edge's endpoints to edge
        indices, then calls ``set_edge_attr``.

        Args:
            attr (str): name of the registered edge attribute to set.
            src_expansion_ids (list[int]): expansion id for each edge's source as given by ``add_nodes_by_pattern``.
            dest_expansion_ids (list[int]): expansion id for each edge's destination as given by ``add_nodes_by_pattern``.
            src_expansion_positions (list[int]): position in the expansion for each edge's source as given by ``add_nodes_by_pattern``.
            dest_expansion_positions (list[int]): position in the expansion for each edge's destination as given by ``add_nodes_by_pattern``.
            value (torch.Tensor): one row per entry in ``src_expansion_ids``/``dest_expansion_ids``.

        Raises:
            AttributeError: if no pattern is registered.
            ValueError: if an ``(expansion_id, expansion_position)`` pair does not match any node.
            ValueError: if no edge exists between a resolved source/destination node pair.
            ValueError: if ``value`` does not have one row per entry in ``src_expansion_ids``/``dest_expansion_ids``.
            ValueError: any error raised by ``set_edge_attr`` for the resolved edges.
        """
        src_indices, dest_indices = self._node_indices_from_expansion(
            [(src_expansion_ids, src_expansion_positions), (dest_expansion_ids, dest_expansion_positions)]
        )
        edges = self._edge_indices_for_pairs(src_indices, dest_indices)
        mask, value = self._mask_and_reorder(edges, value, self.edge_index.shape[1])

        self.set_edge_attr(attr, mask, value)


    def set_edge_attr_per_expansion_by_pattern(self, attr: str, src_expansion_ids: list, dest_expansion_ids: list, src_expansion_positions: list, dest_expansion_positions: list, value: torch.Tensor, replicate_levels = None):
        """
        Resolves the ``(expansion_id, expansion_position)`` pairs of each edge's endpoints to edge
        indices, then calls ``set_edge_attr_by_pattern``.

        Args:
            attr (str): name of the registered edge attribute to set.
            src_expansion_ids (list[int]): expansion id for each edge's source as given by ``add_nodes_by_pattern``.
            dest_expansion_ids (list[int]): expansion id for each edge's destination as given by ``add_nodes_by_pattern``.
            src_expansion_positions (list[int]): position in the expansion for each edge's source as given by ``add_nodes_by_pattern``.
            dest_expansion_positions (list[int]): position in the expansion for each edge's destination as given by ``add_nodes_by_pattern``.
            value (torch.Tensor): one row per entry in ``src_expansion_ids``/``dest_expansion_ids``.
            replicate_levels (list[bool], optional): see ``set_edge_attr_by_pattern``.

        Raises:
            AttributeError: if no pattern is registered.
            ValueError: if an ``(expansion_id, expansion_position)`` pair does not match any node.
            ValueError: if an end of an edge is a copy that was merged into another one, by
                ``add_nodes_by_pattern`` or ``merge``, and ``set_edge_attr_by_pattern``, which reads
                the edge from the node it was merged into, would set a different edge family. Give
                such an edge at a copy whose ends were not merged.
            ValueError: if no edge exists between a resolved source/destination node pair.
            ValueError: if ``value`` does not have one row per entry in ``src_expansion_ids``/``dest_expansion_ids``.
            ValueError: any error raised by ``set_edge_attr_by_pattern`` for the resolved edges.
        """
        src_indices, dest_indices = self._node_indices_from_expansion_by_pattern(
            [(src_expansion_ids, src_expansion_positions), (dest_expansion_ids, dest_expansion_positions)], replicate_levels,
            outside_hint="Set them via set_edge_attr().", target="set_edge_attr_by_pattern",
        )
        edges = self._edge_indices_for_pairs(src_indices, dest_indices)
        mask, value = self._mask_and_reorder(edges, value, self.edge_index.shape[1])

        self.set_edge_attr_by_pattern(attr, mask, value, replicate_levels)


    def set_edge_attr_by_pattern_level(self, attr: str, src_expansion_ids: list, dest_expansion_ids: list, src_level_positions, dest_level_positions, value: torch.Tensor, replicate_levels = None):
        """
        Resolves each edge's two endpoints, given by expansion id and one position per pattern level,
        to expansion positions, then calls ``set_edge_attr_per_expansion_by_pattern``. The level positions
        of a node say which copy it is on every nesting level of the pattern, innermost first, see
        ``add_edges_by_pattern_level``. Level positions of a copy that was merged into another
        one, by ``add_nodes_by_pattern`` or ``merge``, address the node it was merged into.

        Args:
            attr (str): name of the registered edge attribute to set.
            src_expansion_ids (list[int]): expansion id for each edge's source.
            dest_expansion_ids (list[int]): expansion id for each edge's destination.
            src_level_positions (list[list[int]] | torch.Tensor): for each edge's source, its
                position on every level of the pattern, innermost first. Shape [E, L] for E edges
                and a pattern with L levels; the position on a level must be smaller than the size
                of that level.
            dest_level_positions (list[list[int]] | torch.Tensor): the same for each edge's
                destination.
            value (torch.Tensor): one row per entry in ``src_expansion_ids``/``dest_expansion_ids``.
            replicate_levels (list[bool], optional): see ``set_edge_attr_by_pattern``.

        Raises:
            ValueError: if no pattern is registered.
            ValueError: if ``src_expansion_ids`` and ``dest_expansion_ids`` do not have the same length.
            ValueError: if an expansion id does not match any expansion.
            ValueError: if the given expansions belong to more than one registered pattern.
            ValueError: if ``src_level_positions`` or ``dest_level_positions`` does not hold
                integers, or does not have one row per edge with one position per level of the
                pattern.
            ValueError: if a level position is negative or not smaller than the size of its level.
            ValueError: any error raised by ``set_edge_attr_per_expansion_by_pattern`` for the resolved
                expansion positions, e.g. if a level position matches no node, if no edge exists between
                the resolved nodes, if ``value`` does not have one row per edge, or if an end of an
                edge is a copy that was merged into another one, from whose node
                ``set_edge_attr_by_pattern`` would set a different edge family.
        """
        (src_expansion_ids, src_positions), (dest_expansion_ids, dest_positions) = self._expansion_positions_by_level([
            ("src_expansion_ids", src_expansion_ids, "src_level_positions", src_level_positions),
            ("dest_expansion_ids", dest_expansion_ids, "dest_level_positions", dest_level_positions),
        ])

        self.set_edge_attr_per_expansion_by_pattern(
            attr, src_expansion_ids, dest_expansion_ids, src_positions, dest_positions, value, replicate_levels=replicate_levels
        )


    def _modify_edge_attr_and_reciprocal(self, attr: str, edge_index: torch.Tensor, value: torch.Tensor):

        if value.shape[0] != edge_index.shape[0]:
            raise ValueError(f"edge_index and value must have the same length, got {edge_index.shape[0]} and {value.shape[0]}.")

        current = getattr(self, attr)
        current[edge_index] = value
        current[self.reciprocal_edge[edge_index].view(-1)] = value
        setattr(self, attr, current)


    def _forward_edges(self, edges):
        # the forward row of every given edge, whether its forward or its reciprocal row is given
        return torch.where(self.directed_mask.view(-1)[edges], edges, self.reciprocal_edge.view(-1)[edges])


    def _edges_to_set(self, attr, mask, value):

        if attr not in self.metadata["edge_attr_list"]:
            raise ValueError(f"Unexpected edge attribute '{attr}'. Expected an attribute in: {list(self.metadata['edge_attr_list'])}")

        mask = mask.view(-1)

        num_edges = self.edge_index.shape[1]
        if mask.shape[0] != num_edges:
            raise ValueError(f"mask must have {num_edges} entries, but has {mask.shape[0]}.")

        edges = torch.where(mask)[0]

        if value.shape[0] != edges.shape[0]:
            raise ValueError(f"value must have one row per masked edge ({edges.shape[0]}), got {value.shape[0]}.")

        edges = self._forward_edges(edges)

        edges_unique, edge_counts = torch.unique(edges, return_counts=True)
        duplicated = edges_unique[edge_counts > 1]
        if duplicated.numel() > 0:
            bad_pairs = [(int(self.edge_index[0, e]), int(self.edge_index[1, e])) for e in duplicated.tolist()]
            raise ValueError(
                f"mask selects both the forward and reciprocal row of the same edge(s): {bad_pairs}."
            )

        return edges


    def set_edge_attr(self, attr: str, mask: torch.Tensor, value: torch.Tensor):
        """
        Sets an edge attribute for the edges selected by ``mask``. Only the selected edges (and
        their reciprocal edges) are updated, even if their nodes belong to a registered pattern;
        use ``set_edge_attr_by_pattern`` to copy the value to the rest of their edge families.

        Args:
            attr (str): name of the registered edge attribute to set.
            mask (torch.Tensor): boolean tensor of shape ``[num_edges]``. Selects the edges
                to set; for each undirected edge, its forward or reciprocal may be selected
                interchangeably, but not both.
            value (torch.Tensor): one row per ``True`` entry in ``mask``.

        Raises:
            ValueError: if ``attr`` is not a registered edge attribute.
            ValueError: if ``mask`` does not have ``num_edges`` entries.
            ValueError: if ``value`` does not have one row per ``True`` entry in ``mask``.
            ValueError: if ``mask`` selects both the forward and reciprocal row of the
                same edge.
        """

        edges = self._edges_to_set(attr, mask, value)

        self._modify_edge_attr_and_reciprocal(attr, edges, value)


    def set_edge_attr_by_pattern(
        self,
        attr: str,
        mask: torch.Tensor,
        value: torch.Tensor,
        replicate_levels = None,
    ):
        """
        Sets an edge attribute for the edges selected by ``mask`` and copies the value of each
        selected edge verbatim to the rest of its edge family, i.e. to exactly the same copies
        that ``add_edges_by_pattern`` creates for it, so a level that does not close is left
        open here too.

        Args:
            attr (str): name of the registered edge attribute to set.
            mask (torch.Tensor): boolean tensor of shape ``[num_edges]``. Selects the edges
                to set; for each undirected edge, its forward or reciprocal may be selected
                interchangeably, but not both.
            value (torch.Tensor): one row per ``True`` entry in ``mask``.
            replicate_levels (list[bool], optional): one entry per nesting level of the
                pattern, innermost first, selecting the levels each edge is replicated on. A
                level left out keeps the given edge's own two positions on that level, so no
                copies are made along it. Any subset works, so a level can be frozen between two
                replicated ones. Defaults to ``None``, which replicates on every level. This
                argument must match between ``add_edges_by_pattern`` and
                ``set_edge_attr_by_pattern`` for the same edge family, since it decides which
                edges count as copies of each other.

        Raises:
            ValueError: if ``attr`` is not a registered edge attribute.
            ValueError: if ``mask`` does not have ``num_edges`` entries.
            ValueError: if ``value`` does not have one row per ``True`` entry in ``mask``.
            ValueError: if ``mask`` selects both the forward and reciprocal row of the
                same edge.
            ValueError: if no pattern is registered.
            ValueError: if a selected edge connects a node outside any pattern.
            ValueError: if a selected edge connects nodes from two different registered
                patterns.
            ValueError: if the selected edges belong to more than one registered pattern.
            ValueError: if ``replicate_levels`` does not have one entry per nesting level of
                the pattern.
            ValueError: if a pattern copy of a selected edge does not exist.
            ValueError: if two selected edges are pattern copies of each other but were given
                different values.
        """

        edges = self._edges_to_set(attr, mask, value)

        src, dst = self.edge_index[0, edges], self.edge_index[1, edges]

        pattern_id = self._single_pattern(src, dst, outside_hint="Set them via set_edge_attr().")
        if pattern_id is None:
            return

        sib_src, sib_dst, seed_edge = self._expand_edge_indices_by_pattern(src, dst, pattern_id, replicate_levels)
        sib_value = value[seed_edge]

        sib_edges = self._edge_indices_for_pairs(
            sib_src, sib_dst, missing_message="No edge found for a pattern copy of a selected edge."
        )

        sib_forward = self._forward_edges(sib_edges)
        order = torch.argsort(sib_forward)
        same_edge = sib_forward[order][1:] == sib_forward[order][:-1]
        sorted_value = sib_value[order].reshape(order.shape[0], -1)
        different_value = torch.any(sorted_value[1:] != sorted_value[:-1], dim=1)
        if torch.any(same_edge & different_value):
            raise ValueError(
                "Multiple selected edges are pattern copies of each other but were given "
                "different values. Select one edge per edge family, or give them the same value."
            )

        self._modify_edge_attr_and_reciprocal(attr, sib_edges, sib_value)

    def add_patterns(self, patterns: dict, transform_attrs: list = None):
        """
        Registers one or more patterns on the graph, so ``add_nodes_by_pattern``,
        ``add_edges_by_pattern``, ``set_node_attr_by_pattern``, and ``set_edge_attr_by_pattern``
        can propagate updates across a pattern.
        All patterns passed in one call share the same ``transform_attrs``; register
        patterns in separate calls to give them different ones.

        Args:
            patterns (dict[str, dict]): mapping of pattern name to a pattern dict, as
                returned by ``create_rotational_pattern``, ``create_translational_pattern``,
                ``create_mirror_pattern``, or ``nest_patterns``. Each name must not already be
                registered. A dict without a ``closes`` entry is taken to close on every level.
            transform_attrs (list[str], optional): node attributes that should be
                geometrically transformed (rotated/mirrored, via the full affine matrix)
                per copy. Defaults to ``["coords"]`` when omitted (pass ``[]``
                explicitly for no transform attributes at all). Every other node attribute,
                and every edge attribute, is copied verbatim to every copy of a node or edge.

        Raises:
            ValueError: if ``transform_attrs`` contains pattern bookkeeping (``expansion_id``,
                ``expansion_position``, ``pattern_id``).
            ValueError: if ``transform_attrs`` contains a registered edge attribute name
                (edge attributes are always copied verbatim).
            ValueError: if a name in ``patterns`` is already registered.
            ValueError: if a pattern's ``pattern_level`` or ``closes`` does not have one entry per
                matrix.
            ValueError: if a pattern's ``closes`` is ``False`` anywhere other than at the last
                matrix of a level.
        """

        transform_attrs = transform_attrs if transform_attrs is not None else ["coords"]

        bookkeeping = {"expansion_id", "expansion_position", "pattern_id"}
        invalid_attrs = bookkeeping & set(transform_attrs)
        if invalid_attrs:
            raise ValueError(
                f"{sorted(invalid_attrs)} is pattern bookkeeping and cannot be registered as a "
                "transform attribute."
            )

        invalid_transform_attrs = set(transform_attrs) & set(self.metadata["edge_attr_list"])
        if invalid_transform_attrs:
            raise ValueError(
                f"{sorted(invalid_transform_attrs)} is a registered edge attribute and cannot be "
                "registered as a transform attribute; edge attributes are always copied verbatim."
            )

        is_first_pattern = "name_to_pattern" not in self.metadata
        name_to_pattern = self.metadata.setdefault("name_to_pattern", {})

        if is_first_pattern:

            self.metadata["graph_attr_list"].append("pattern_matrices")
            self.metadata["graph_attr_list"].append("pattern_matrix_id")
            self.metadata["graph_attr_list"].append("pattern_level")
            self.metadata["graph_attr_list"].append("pattern_closes")
            self.metadata["graph_attr_list"].append("expansion_holes")
            self.metadata["node_attr_list"].append("expansion_id")
            self.metadata["node_attr_list"].append("expansion_position")
            self.metadata["node_attr_list"].append("pattern_id")

            self.metadata["default_attrs"]["expansion_id"] = torch.tensor(-1, dtype=torch.long)
            self.metadata["default_attrs"]["expansion_position"] = torch.tensor(-1, dtype=torch.long)
            self.metadata["default_attrs"]["pattern_id"] = torch.tensor(-1, dtype=torch.long)

            self.pattern_matrices = torch.empty((0, 4, 4))
            self.pattern_matrix_id = torch.empty((0,), dtype=torch.long)
            self.pattern_level = torch.empty((0,), dtype=torch.long)
            self.pattern_closes = torch.empty((0,), dtype=torch.bool)
            # one row (expansion id, expansion position, twin's expansion position) per copy merged by add_nodes_by_pattern
            self.expansion_holes = torch.empty((0, 3), dtype=torch.long)
            self.expansion_id = torch.full((self.num_nodes, 1), -1, dtype=torch.long)
            self.expansion_position = torch.full((self.num_nodes, 1), -1, dtype=torch.long)
            self.pattern_id = torch.full((self.num_nodes, 1), -1, dtype=torch.long)

            self.metadata["pattern_transform_attrs"] = {}

        for name, pattern in patterns.items():

            if name in name_to_pattern:
                raise ValueError(f"Pattern '{name}' already exists.")

            pattern_matrices = pattern["matrices"]
            pattern_level = pattern["pattern_level"]

            m = pattern_matrices.shape[0]
            if pattern_level.shape[0] != m:
                raise ValueError(
                    f"pattern_level for '{name}' must have one entry per matrix ({m}), got {pattern_level.shape[0]}."
                )

            pattern_closes = pattern.get("closes", torch.ones(m, dtype=torch.bool)).to(torch.bool)
            if pattern_closes.shape[0] != m:
                raise ValueError(
                    f"closes for '{name}' must have one entry per matrix ({m}), got {pattern_closes.shape[0]}."
                )

            boundaries = torch.cumsum(torch.bincount(pattern_level.to(torch.long)), dim=0)
            on_boundary = torch.zeros(m, dtype=torch.bool)
            on_boundary[boundaries - 1] = True
            misplaced = ~pattern_closes & ~on_boundary
            if torch.any(misplaced):
                raise ValueError(
                    f"closes for '{name}' may only be False at the last matrix of a pattern level "
                    f"(indices {(boundaries - 1).tolist()}), but is False at "
                    f"{torch.where(misplaced)[0].tolist()}."
                )

            pattern_id = int(self.pattern_matrix_id.max().item()) + 1 if self.pattern_matrix_id.numel() > 0 else 0

            self.pattern_matrices = torch.cat([self.pattern_matrices, pattern_matrices], dim=0)
            self.pattern_matrix_id = torch.cat(
                [self.pattern_matrix_id, torch.full((m,), pattern_id, dtype=torch.long)], dim=0
            )
            self.pattern_level = torch.cat([self.pattern_level, pattern_level.to(torch.long)], dim=0)
            self.pattern_closes = torch.cat([self.pattern_closes, pattern_closes], dim=0)

            name_to_pattern[name] = pattern_id
            self.metadata["pattern_transform_attrs"][pattern_id] = list(transform_attrs)

    def view_patterns(self):
        """
        Prints for every registered pattern its name, its total size, the size and closing
        behaviour of each of its nesting levels (innermost first), and its transform attributes.
        """
        if "name_to_pattern" not in self.metadata:
            print("No patterns registered.")
            return

        for name, pattern_id in self.metadata["name_to_pattern"].items():
            size = int((self.pattern_matrix_id == pattern_id).sum())
            level_sizes, level_closes = self._infer_level_sizes(pattern_id)
            levels = ", ".join(
                f"{level_size} ({'closing' if closes else 'open'})"
                for level_size, closes in zip(level_sizes, level_closes)
            )
            transform_attrs = self.metadata["pattern_transform_attrs"][pattern_id]
            print(f"{name}: size={size}, levels=[{levels}], transform_attrs={transform_attrs}")


    def add_chain(self, n: int, node_attrs: dict = {}, edge_attrs: dict = {}):

        """
        Adds a chain of ``N = n`` nodes to the graph.
        The chain has ``E = n - 1`` edges, connecting nodes sequentially: ``0 → 1 → ... → n-1``.

        When a value in ``node_attrs`` or ``edge_attrs`` is a callable, it is invoked
        with the subset of node or edge attributes whose names match its parameter names.
        Available inputs are the default chain attributes listed below as well as any
        other attributes already defined earlier in the same ``node_attrs`` or
        ``edge_attrs`` dict. The following default attributes are available as callable
        arguments:

        Node defaults:
            - ``u_ind`` (torch.Tensor [N, 1]): index of each node (``0`` to ``n-1``).
            - ``v_ind`` (torch.Tensor [N, 1]): zero (chain has no second axis).
            - ``x_unit_coord`` (torch.Tensor [N, 1]): ``x`` coordinate in unit interval.
            - ``y_unit_coord`` (torch.Tensor [N, 1]): zero.
            - ``is_boundary`` (torch.Tensor [N, 1], bool): ``True`` for the first and last node.

        Edge defaults:
            - ``u_ind`` (torch.Tensor [E, 1]): index of the source node.
            - ``v_ind`` (torch.Tensor [E, 1]): zero.
            - ``x_unit_coord`` (torch.Tensor [E, 1]): mean unit ``x`` coordinate of source and
              destination node.
            - ``y_unit_coord`` (torch.Tensor [E, 1]): zero.

        Args:
            n (int): number of nodes.
            node_attrs (dict[str, torch.Tensor | callable]): mapping of registered
                node attribute names to either a tensor of shape ``[N, *]`` or
                ``[1, *]`` (broadcast over all nodes), or a callable whose
                parameter names are resolved from the node defaults listed above as well as any
                other node attributes already defined earlier.
            edge_attrs (dict[str, torch.Tensor | callable]): mapping of registered
                edge attribute names to either a tensor of shape ``[E, *]`` or
                ``[1, *]`` (broadcast over all edges), or a callable whose
                parameter names are resolved from the edge defaults listed above or any
                other edge attributes already defined earlier.
        """

        #create default node attributes
        u_node_ind = torch.arange(n).unsqueeze(1)
        v_node_ind = torch.zeros(n).unsqueeze(1)

        x_node_unit_coord = u_node_ind / (n-1)
        y_node_unit_coord = v_node_ind

        is_boundary_node = (u_node_ind == 0) | (u_node_ind == n-1)
        default_node_attrs = {"u_ind": u_node_ind, "v_ind": v_node_ind, "x_unit_coord": x_node_unit_coord, "y_unit_coord": y_node_unit_coord, "is_boundary": is_boundary_node}

        #create edge indices
        row = torch.arange(n-1)
        col = torch.arange(1, n)
        edge_indices = torch.stack([row, col], dim=0)

        #create default edge attributes
        x_edge_unit_coord, y_edge_unit_coord = self._get_edge_unit_coords(x_node_unit_coord, y_node_unit_coord, edge_indices)


        u_edge_ind = u_node_ind[edge_indices[0]]
        v_edge_ind = v_node_ind[edge_indices[0]]

        default_edge_attrs = {"u_ind": u_edge_ind, "v_ind": v_edge_ind, "x_unit_coord": x_edge_unit_coord, "y_unit_coord": y_edge_unit_coord}

        self._create_topology(num_nodes = n, num_edges = n-1, edge_indices=edge_indices, node_attrs=node_attrs, edge_attrs=edge_attrs, default_node_attrs=default_node_attrs, default_edge_attrs=default_edge_attrs)

    

    def add_triangular_grid(self, n, node_attrs: dict = {}, edge_attrs: dict = {}):

        """
        Adds a triangular grid of ``N = n(n+1)/2`` nodes to the graph.
        The grid has ``E = 3n(n-1)/2`` edges: ``n(n-1)/2`` horizontal, ``n(n-1)/2`` vertical,
        and ``n(n-1)/2`` diagonal.

        When a value in ``node_attrs`` or ``edge_attrs`` is a callable, it is invoked
        with the subset of node or edge attributes whose names match its parameter names.
        Available inputs are the default grid attributes listed below as well as any
        other attributes already defined earlier in the same ``node_attrs`` or
        ``edge_attrs`` dict. The following default attributes are available as callable
        arguments:

        Node defaults:
            - ``u_ind`` (torch.Tensor [N, 1]): row index of each node (``0`` to ``n-1``).
            - ``v_ind`` (torch.Tensor [N, 1]): local column index within each row (``0`` to ``n-1-u_ind``).
            - ``x_unit_coord`` (torch.Tensor [N, 1]): ``x`` coordinate in unit isosceles triangle.
            - ``y_unit_coord`` (torch.Tensor [N, 1]): ``y`` coordinate in unit isosceles triangle.
            - ``is_boundary`` (torch.Tensor [N, 1], bool): ``True`` for nodes on the boundary of the triangle.
            - ``is_corner`` (torch.Tensor [N, 1], bool): ``True`` for nodes at the corners of the
              triangle.

        Edge defaults:
            - ``u_ind`` (torch.Tensor [E, 1]): row index of the source node.
            - ``v_ind`` (torch.Tensor [E, 1]): column index of the source node.
            - ``x_unit_coord`` (torch.Tensor [E, 1]): mean unit ``x`` coordinate of source and
              destination node.
            - ``y_unit_coord`` (torch.Tensor [E, 1]): mean unit ``y`` coordinate of source and
              destination node.
            - ``edge_direction`` (torch.Tensor [E, 1]): ``0`` for horizontal edges connecting
              ``(u_ind, v_ind) → (u_ind, v_ind+1)``, ``1`` for vertical edges connecting
              ``(u_ind, v_ind) → (u_ind+1, v_ind)``, ``2`` for diagonal edges connecting
              ``(u_ind, v_ind+1) → (u_ind+1, v_ind)``.
            - ``is_boundary`` (torch.Tensor [E, 1], bool): ``True`` when both endpoint nodes are
              boundary nodes.

        Args:
            n (int): number of nodes along each side of the triangle. Must be at least 2.
            node_attrs (dict[str, torch.Tensor | callable]): mapping of registered
                node attribute names to either a tensor of shape ``[N, *]`` or
                ``[1, *]`` (broadcast over all nodes), or a callable whose
                parameter names are resolved from the node defaults listed above as well as any
                other node attributes already defined earlier.
            edge_attrs (dict[str, torch.Tensor | callable]): mapping of registered
                edge attribute names to either a tensor of shape ``[E, *]`` or
                ``[1, *]`` (broadcast over all edges), or a callable whose
                parameter names are resolved from the edge defaults listed above or any
                other edge attributes already defined earlier.

        Raises:
            ValueError: if ``n`` is less than 2.
        """

        if n <= 1:
            raise ValueError(f"n must be greater than 1, got {n}.")

        #create default node attributes
        u_node_ind, v_node_ind = torch.triu_indices(n, n)

        v_node_ind = (v_node_ind - u_node_ind).unsqueeze(1)
        u_node_ind = u_node_ind.unsqueeze(1)

        x_node_unit_coord = u_node_ind / (n-1)
        y_node_unit_coord = v_node_ind / (n-1)
        
        is_boundary_node = (u_node_ind == 0) | (u_node_ind == n-1) | (v_node_ind == 0) | (v_node_ind == n-1) | (u_node_ind + v_node_ind == n-1)
        is_corner_node = ((u_node_ind == 0) | (u_node_ind == n - 1)) & ((v_node_ind == 0) | (v_node_ind == n - 1))
        default_node_attrs = {"u_ind": u_node_ind, "v_ind": v_node_ind, "x_unit_coord": x_node_unit_coord, "y_unit_coord": y_node_unit_coord , "is_boundary": is_boundary_node, "is_corner": is_corner_node}

        #create edge indices
        v_idx = torch.arange(n - 1)
        start_v = v_idx * n - v_idx * (v_idx - 1) // 2  
        row_sizes = torch.arange(n - 1, 0, -1)           

        cumsum = torch.cumsum(row_sizes, 0)
        group_starts = cumsum - row_sizes
        offset = torch.arange(cumsum[-1]) - torch.repeat_interleave(group_starts, row_sizes)

        start_rep = torch.repeat_interleave(start_v, row_sizes)
        v_rep = torch.repeat_interleave(v_idx, row_sizes)

        # horizontal: (u, v) → (u, v+1)
        row_h = start_rep + offset
        col_h = row_h + 1

        # vertical: (u, v) → (u+1, v)
        row_v = start_rep + offset
        col_v = row_v + (n - v_rep)

        # diagonal: (u, v+1) → (u+1, v)
        row_d = start_rep + offset + 1
        col_d = row_d + (n - v_rep - 1)

        row = torch.cat([row_h, row_v, row_d])
        col = torch.cat([col_h, col_v, col_d])

        edge_indices = torch.stack([row, col], dim=0)

        #create default edge attributes
        x_edge_unit_coord, y_edge_unit_coord = self._get_edge_unit_coords(x_node_unit_coord, y_node_unit_coord, edge_indices)
         
        u_edge_ind = u_node_ind[edge_indices[0]]
        v_edge_ind = v_node_ind[edge_indices[0]]
        
        is_boundary_edge = is_boundary_node[edge_indices[0]] & is_boundary_node[edge_indices[1]]

        edge_direction = torch.cat([torch.zeros(row_h.shape[0]), torch.ones(row_v.shape[0]), torch.full((row_d.shape[0],), 2.0)], dim=0).unsqueeze(1)

        default_edge_attrs = {"u_ind": u_edge_ind, "v_ind": v_edge_ind, "x_unit_coord": x_edge_unit_coord, "y_unit_coord": y_edge_unit_coord, "edge_direction": edge_direction,"is_boundary": is_boundary_edge}

        num_nodes = n * (n + 1) // 2
        num_edges = 3 * n * (n-1) // 2

        self._create_topology(num_nodes = num_nodes, num_edges = num_edges, edge_indices=edge_indices, node_attrs=node_attrs, edge_attrs=edge_attrs, default_node_attrs=default_node_attrs, default_edge_attrs=default_edge_attrs)


    def add_polar_grid(self, n_sectors, n_rings, node_attrs: dict = {}, edge_attrs: dict = {}):

        """
        Adds a polar grid of ``N = n_rings × n_sectors + 1`` nodes to the graph. The grid has ``E = 2 × n_rings × n_sectors`` edges:
        ``n_rings × n_sectors`` radial and ``n_rings × n_sectors`` angular.

        When a value in ``node_attrs`` or ``edge_attrs`` is a callable, it is invoked
        with the subset of node or edge attributes whose names match its parameter names.
        Available inputs are the default grid attributes listed below as well as any
        other attributes already defined earlier in the same ``node_attrs`` or
        ``edge_attrs`` dict. The following default attributes are available as callable
        arguments:

        Node defaults:
            - ``a_ind`` (torch.Tensor [N, 1]): angular index of each node (``0`` to ``n_sectors-1``).
              The center node has ``a_ind == 0``.
            - ``r_ind`` (torch.Tensor [N, 1]): radial index of each node (``0`` for the center,
              ``1`` to ``n_rings`` for the rings).
            - ``x_unit_coord`` (torch.Tensor [N, 1]): ``x`` coordinate (cartesian) in unit disc.
            - ``y_unit_coord`` (torch.Tensor [N, 1]): ``y`` coordinate (cartesian) in unit disc.
            - ``is_boundary`` (torch.Tensor [N, 1], bool): ``True`` for nodes on the outermost ring.

        Edge defaults:
            - ``r_ind`` (torch.Tensor [E, 1]): radial index of the source node.
            - ``a_ind`` (torch.Tensor [E, 1]): angular index of the source node. For radial
              edges from the center node, this is instead the sector (angular index) the
              edge points into, since they all share the center node as their source.
            - ``x_unit_coord`` (torch.Tensor [E, 1]): mean unit ``x`` coordinate of source and
              destination node.
            - ``y_unit_coord`` (torch.Tensor [E, 1]): mean unit ``y`` coordinate of source and
              destination node.
            - ``edge_direction`` (torch.Tensor [E, 1]): ``0`` for radial edges, ``1`` for angular edges.
            - ``is_boundary`` (torch.Tensor [E, 1], bool): ``True`` when both endpoint nodes are
              boundary nodes.

        Args:
            n_sectors (int): number of angular sectors. Must be at least 3, so that each ring
                is closed and every angular edge connects a different pair of nodes.
            n_rings (int): number of concentric rings (excluding the center node). Must be at
                least 2.
            node_attrs (dict[str, torch.Tensor | callable]): mapping of registered
                node attribute names to either a tensor of shape ``[N, *]`` or
                ``[1, *]`` (broadcast over all nodes), or a callable whose
                parameter names are resolved from the node defaults listed above as well as any
                other node attributes already defined earlier.
            edge_attrs (dict[str, torch.Tensor | callable]): mapping of registered
                edge attribute names to either a tensor of shape ``[E, *]`` or
                ``[1, *]`` (broadcast over all edges), or a callable whose
                parameter names are resolved from the edge defaults listed above or any
                other edge attributes already defined earlier.

        Raises:
            ValueError: if ``n_rings`` is less than 2.
            ValueError: if ``n_sectors`` is less than 3.
        """


        if n_rings <= 1:
            raise ValueError(f"n_rings must be greater than 1, got {n_rings}.")
        if n_sectors <= 2:
            raise ValueError(f"n_sectors must be greater than 2, got {n_sectors}.")
        
        r_node_ind = torch.cat([torch.tensor([0]), torch.arange(1, n_rings+1).repeat(n_sectors)]).unsqueeze(1)
        a_node_ind = torch.cat([torch.tensor([0]), torch.arange(0, n_sectors).repeat_interleave(n_rings)]).unsqueeze(1)
        r_node_ind = torch.cat([torch.tensor([0]), torch.arange(1, n_rings+1).repeat(n_sectors)]).unsqueeze(1)
        
        angle = a_node_ind / n_sectors * 2 * torch.pi

        x_node_unit_coord = r_node_ind / n_rings * torch.cos(angle)
        y_node_unit_coord = r_node_ind / n_rings * torch.sin(angle)

        is_boundary_node = (r_node_ind == n_rings)
        default_node_attrs = {"a_ind": a_node_ind, "r_ind": r_node_ind, "x_unit_coord": x_node_unit_coord, "y_unit_coord": y_node_unit_coord, "is_boundary": is_boundary_node}

        #create edge indices
        # radial edges: center → first ring of each sector
        a_center = torch.arange(n_sectors)
        row_r_center = torch.zeros(n_sectors, dtype=torch.long)
        col_r_center = a_center * n_rings + 1

        # radial edges: ring r → ring r+1 within each sector
        a_radial = torch.arange(n_sectors).repeat_interleave(n_rings - 1)
        r_radial = torch.arange(1, n_rings).repeat(n_sectors)
        row_r_inner = a_radial * n_rings + r_radial
        col_r_inner = a_radial * n_rings + r_radial + 1
        row_r = torch.cat([row_r_center, row_r_inner])
        col_r = torch.cat([col_r_center, col_r_inner])

        # angular edges: (r, a) → (r, (a+1) % n_sectors)
        a_angular = torch.arange(n_sectors).repeat_interleave(n_rings)
        r_angular = torch.arange(1, n_rings + 1).repeat(n_sectors)
        row_a = a_angular * n_rings + r_angular
        col_a = (a_angular + 1) % n_sectors * n_rings + r_angular

        row = torch.cat([row_r, row_a])
        col = torch.cat([col_r, col_a])
        edge_indices = torch.stack([row, col], dim=0)

        #create default edge attributes
        x_edge_unit_coord, y_edge_unit_coord = self._get_edge_unit_coords(x_node_unit_coord, y_node_unit_coord, edge_indices)

        r_edge_ind = r_node_ind[edge_indices[0]]
        # a_ind is the source node's angular index, except for radial edges
        # from the center: the center node has no angular position of its
        # own (a_ind == 0 for all sectors), so those edges would otherwise
        # all collapse to the same a_ind. Use the sector each spoke points
        # into instead, so every radial edge gets a unique (r_ind, a_ind).
        a_edge_ind = torch.cat([a_center, a_radial, a_angular]).unsqueeze(1)

        is_boundary_edge = is_boundary_node[edge_indices[0]] & is_boundary_node[edge_indices[1]]
        edge_direction = torch.cat([torch.zeros(row_r.shape[0]), torch.ones(row_a.shape[0])]).unsqueeze(1)

        default_edge_attrs = {"a_ind": a_edge_ind, "r_ind": r_edge_ind, "x_unit_coord": x_edge_unit_coord, "y_unit_coord": y_edge_unit_coord, "edge_direction": edge_direction, "is_boundary": is_boundary_edge}
        
        num_nodes = n_rings * n_sectors + 1
        num_edges = 2 * n_rings * n_sectors

        self._create_topology(num_nodes = num_nodes, num_edges=num_edges, edge_indices=edge_indices, node_attrs=node_attrs, edge_attrs=edge_attrs, default_node_attrs=default_node_attrs, default_edge_attrs=default_edge_attrs)


    def add_grid(self, n: int, m: int, node_attrs: dict = {}, edge_attrs: dict = {}):

        """
        Adds a rectangular grid of ``N = n × m`` nodes to the graph.
        The grid has ``E = n(m-1) + m(n-1)`` edges.

        When a value in ``node_attrs`` or ``edge_attrs`` is a callable, it is invoked
        with the subset of node or edge attributes whose names match its parameter names.
        Available inputs are the default grid attributes listed below as well as any
        other attributes already defined earlier in the same ``node_attrs`` or
        ``edge_attrs`` dict. The following default attributes are available as callable
        arguments:

        Node defaults:
            - ``u_ind`` (torch.Tensor [N, 1]): row index of each node (``0`` to ``n-1``).
            - ``v_ind`` (torch.Tensor [N, 1]): column index of each node (``0`` to ``m-1``).
            - ``x_unit_coord`` (torch.Tensor [N, 1]): ``x`` coordinate in unit square.
            - ``y_unit_coord`` (torch.Tensor [N, 1]): ``y`` coordinate in unit square.
            - ``is_boundary`` (torch.Tensor [N, 1], bool): ``True`` for nodes on
              the boundary of the grid.
            - ``is_corner`` (torch.Tensor [N, 1], bool): ``True`` for nodes at the
              four corners of the grid.

        Edge defaults:
            - ``u_ind`` (torch.Tensor [E, 1]): row index of the source node.
            - ``v_ind`` (torch.Tensor [E, 1]): column index of the source node.
            - ``x_unit_coord`` (torch.Tensor [E, 1]): mean unit ``x`` coordinate of source and
              destination node.
            - ``y_unit_coord`` (torch.Tensor [E, 1]): mean unit ``y`` coordinate of source and
              destination node.
            - ``edge_direction`` (torch.Tensor [E, 1]): ``0`` for horizontal edges connecting
              ``(u_ind, v_ind) → (u_ind, v_ind+1)``, ``1`` for vertical edges connecting
              ``(u_ind, v_ind) → (u_ind+1, v_ind)``.
            - ``is_boundary`` (torch.Tensor [E, 1], bool): ``True`` when both
              endpoint nodes are boundary nodes.

        Args:
            n (int): number of rows.
            m (int): number of columns.
            node_attrs (dict[str, torch.Tensor | callable]): mapping of registered
                node attribute names to either a tensor of shape ``[N, *]`` or
                ``[1, *]`` (broadcast over all nodes), or a callable whose
                parameter names are resolved from the node defaults listed above as well as any
                other node attributes already defined earlier.
            edge_attrs (dict[str, torch.Tensor | callable]): mapping of registered
                edge attribute names to either a tensor of shape ``[E, *]`` or
                ``[1, *]`` (broadcast over all edges), or a callable whose
                parameter names are resolved from the edge defaults listed above or any
                other edge attributes already defined earlier.
        """

        #create default node attributes
        u_node_ind = torch.arange(n).repeat_interleave(m).unsqueeze(1)
        v_node_ind = torch.arange(m).repeat(n).unsqueeze(1)

        x_node_unit_coord = u_node_ind / (n-1)
        y_node_unit_coord = v_node_ind / (m-1)
        
        is_boundary_node = (u_node_ind == 0) | (u_node_ind == n-1) | (v_node_ind == 0) | (v_node_ind == m-1)
        is_corner_node = ((u_node_ind == 0) | (u_node_ind == n - 1)) & ((v_node_ind == 0) | (v_node_ind == m - 1))
        default_node_attrs = {"u_ind": u_node_ind, "v_ind": v_node_ind, "x_unit_coord": x_node_unit_coord, "y_unit_coord": y_node_unit_coord, "is_boundary": is_boundary_node, "is_corner": is_corner_node}


        #create edge indices  
        u_horizontal = torch.arange(n).repeat_interleave(m - 1)
        v_horizontal = torch.arange(m - 1).repeat(n)
        row_h = u_horizontal * m + v_horizontal
        col_h = u_horizontal * m + (v_horizontal + 1)

        u_vertical = torch.arange(n - 1).repeat_interleave(m)
        v_vertical = torch.arange(m).repeat(n - 1)
        row_v = u_vertical * m + v_vertical
        col_v = (u_vertical + 1) * m + v_vertical

        row = torch.cat([row_h, row_v])
        col = torch.cat([col_h, col_v])
        edge_indices = torch.stack([row, col], dim=0)

        #create default edge attributes
        x_edge_unit_coord, y_edge_unit_coord = self._get_edge_unit_coords(x_node_unit_coord, y_node_unit_coord, edge_indices)

        u_edge_ind = u_node_ind[edge_indices[0]]
        v_edge_ind = v_node_ind[edge_indices[0]]
        is_boundary_edge = is_boundary_node[edge_indices[0]] & is_boundary_node[edge_indices[1]]
        edge_direction = torch.cat([torch.zeros(row_h.shape[0]), torch.ones(row_v.shape[0])], dim=0).unsqueeze(1)

        default_edge_attrs = {"u_ind": u_edge_ind, "v_ind": v_edge_ind, "x_unit_coord": x_edge_unit_coord, "y_unit_coord": y_edge_unit_coord, "edge_direction": edge_direction,"is_boundary": is_boundary_edge}

        num_nodes = n * m
        num_edges = (n-1)*m + n * (m-1)

        self._create_topology(num_nodes = num_nodes, num_edges = num_edges, edge_indices=edge_indices, node_attrs=node_attrs, edge_attrs=edge_attrs, default_node_attrs=default_node_attrs, default_edge_attrs=default_edge_attrs)


    def name_nodes(self, names: list, mask: torch.Tensor):
        """Assign names to the nodes selected by ``mask``, in order.

        Args:
            names (list[str]): names to assign, one per ``True`` entry in
                ``mask``; each must encode to at most 8 bytes (see
                [encode][torch_structure.data.data.encode]).
            mask (torch.Tensor [N], bool): mask selecting which nodes to
                name.
        """
        if len(names) != mask.sum().item():

            raise ValueError(f"Length of names does not match the number of 'True' values in the mask.")

        encoded_names = torch.empty(0, dtype=torch.long)

        for i in range(len(names)):

            encoded_name = encode(names[i])
            encoded_names = torch.cat([encoded_names, torch.tensor([encoded_name], dtype=torch.long)])

        self.name[mask] = encoded_names

    def add_cylinder(self, n_sectors: int, n_rings: int, node_attrs: dict = {}, edge_attrs: dict = {}):

        """
        Adds a cylindrical grid of ``N = n_sectors × n_rings`` nodes to the graph. The grid has
        ``E = n_sectors × (2 × n_rings - 1)`` edges: ``n_sectors × (n_rings - 1)`` axial and
        ``n_sectors × n_rings`` angular.

        When a value in ``node_attrs`` or ``edge_attrs`` is a callable, it is invoked
        with the subset of node or edge attributes whose names match its parameter names.
        Available inputs are the default grid attributes listed below as well as any
        other attributes already defined earlier in the same ``node_attrs`` or
        ``edge_attrs`` dict. The following default attributes are available as callable
        arguments:

        Node defaults:
            - ``a_ind`` (torch.Tensor [N, 1]): angular index of each node (``0`` to ``n_sectors-1``).
            - ``r_ind`` (torch.Tensor [N, 1]): axial index of each node (``0`` to ``n_rings-1``).
            - ``x_unit_coord`` (torch.Tensor [N, 1]): ``x`` coordinate (cartesian) in unit disc.
            - ``y_unit_coord`` (torch.Tensor [N, 1]): ``y`` coordinate (cartesian) in unit disc.
            - ``z_unit_coord`` (torch.Tensor [N, 1]): ``z`` coordinate in unit interval (``0`` to ``1``).
            - ``is_boundary`` (torch.Tensor [N, 1], bool): ``True`` for nodes on the first or last ring.

        Edge defaults:
            - ``a_ind`` (torch.Tensor [E, 1]): angular index of the source node.
            - ``r_ind`` (torch.Tensor [E, 1]): axial index of the source node.
            - ``x_unit_coord`` (torch.Tensor [E, 1]): mean unit ``x`` coordinate of source and
              destination node.
            - ``y_unit_coord`` (torch.Tensor [E, 1]): mean unit ``y`` coordinate of source and
              destination node.
            - ``z_unit_coord`` (torch.Tensor [E, 1]): mean unit ``z`` coordinate of source and
              destination node.
            - ``edge_direction`` (torch.Tensor [E, 1]): ``0`` for axial edges connecting
              ``(a_ind, r_ind) → (a_ind, r_ind+1)``, ``1`` for angular edges connecting
              ``(a_ind, r_ind) → ((a_ind+1) % n_sectors, r_ind)``.
            - ``is_boundary`` (torch.Tensor [E, 1], bool): ``True`` when both endpoint nodes are
              boundary nodes.

        Args:
            n_sectors (int): number of angular sectors.
            n_rings (int): number of rings along the axis.
            node_attrs (dict[str, torch.Tensor | callable]): mapping of registered
                node attribute names to either a tensor of shape ``[N, *]`` or
                ``[1, *]`` (broadcast over all nodes), or a callable whose
                parameter names are resolved from the node defaults listed above as well as any
                other node attributes already defined earlier.
            edge_attrs (dict[str, torch.Tensor | callable]): mapping of registered
                edge attribute names to either a tensor of shape ``[E, *]`` or
                ``[1, *]`` (broadcast over all edges), or a callable whose
                parameter names are resolved from the edge defaults listed above or any
                other edge attributes already defined earlier.
        """

        a_node_ind = torch.arange(n_sectors).repeat_interleave(n_rings).unsqueeze(1)
        r_node_ind = torch.arange(n_rings).repeat(n_sectors).unsqueeze(1)

        angle = a_node_ind / n_sectors * 2 * torch.pi

        x_node_unit_coord = torch.cos(angle)
        y_node_unit_coord = torch.sin(angle)
        z_node_unit_coord = r_node_ind / (n_rings - 1)

        is_boundary_node = (r_node_ind == n_rings-1) | (r_node_ind == 0)

        default_node_attrs = {"a_ind": a_node_ind, "r_ind": r_node_ind, "x_unit_coord": x_node_unit_coord, "y_unit_coord": y_node_unit_coord, "z_unit_coord": z_node_unit_coord, "is_boundary": is_boundary_node}

        # axial edges: (r, a) → (r+1, a)
        a_axial = torch.arange(n_sectors).repeat_interleave(n_rings - 1)
        r_axial = torch.arange(n_rings - 1).repeat(n_sectors)
        row_ax = a_axial * n_rings + r_axial
        col_ax = a_axial * n_rings + r_axial + 1

        # angular edges: (r, a) → (r, (a+1) % n_sectors)
        a_angular = torch.arange(n_sectors).repeat_interleave(n_rings)
        r_angular = torch.arange(n_rings).repeat(n_sectors)
        row_ang = a_angular * n_rings + r_angular
        col_ang = (a_angular + 1) % n_sectors * n_rings + r_angular

        row = torch.cat([row_ax, row_ang])
        col = torch.cat([col_ax, col_ang])
        edge_indices = torch.stack([row, col], dim=0)

        # default edge attributes
        x_edge_unit_coord, y_edge_unit_coord = self._get_edge_unit_coords(x_node_unit_coord, y_node_unit_coord, edge_indices)
        z_edge_unit_coord = (z_node_unit_coord[edge_indices[0]] + z_node_unit_coord[edge_indices[1]]) / 2

        a_edge_ind = a_node_ind[edge_indices[0]]
        r_edge_ind = r_node_ind[edge_indices[0]]
        is_boundary_edge = is_boundary_node[edge_indices[0]] & is_boundary_node[edge_indices[1]]
        edge_direction = torch.cat([torch.zeros(row_ax.shape[0]), torch.ones(row_ang.shape[0])]).unsqueeze(1)

        default_edge_attrs = {"a_ind": a_edge_ind, "r_ind": r_edge_ind, "x_unit_coord": x_edge_unit_coord, "y_unit_coord": y_edge_unit_coord, "z_unit_coord": z_edge_unit_coord, "edge_direction": edge_direction, "is_boundary": is_boundary_edge}

        num_nodes = n_rings * n_sectors
        num_edges = (n_rings - 1) * n_sectors + n_rings * n_sectors

        self._create_topology(num_nodes=num_nodes, num_edges=num_edges, edge_indices=edge_indices, node_attrs=node_attrs, edge_attrs=edge_attrs, default_node_attrs=default_node_attrs, default_edge_attrs=default_edge_attrs)


    def add_sphere(self, n_sectors: int, n_rings: int, node_attrs: dict = {}, edge_attrs: dict = {}):

        """
        Adds a spherical grid of ``N = n_sectors × n_rings + 2`` nodes to the graph: a north pole,
        ``n_rings`` intermediate rings of ``n_sectors`` nodes each, and a south pole. The grid has
        ``E = n_sectors × (2 × n_rings + 1)`` edges: ``n_sectors × (n_rings + 1)`` meridional and
        ``n_sectors × n_rings`` angular.

        When a value in ``node_attrs`` or ``edge_attrs`` is a callable, it is invoked
        with the subset of node or edge attributes whose names match its parameter names.
        Available inputs are the default grid attributes listed below as well as any
        other attributes already defined earlier in the same ``node_attrs`` or
        ``edge_attrs`` dict. The following default attributes are available as callable
        arguments:

        Node defaults:
            - ``a_ind`` (torch.Tensor [N, 1]): angular index (``0`` to ``n_sectors-1`` for
              intermediate nodes, ``0`` for both poles).
            - ``r_ind`` (torch.Tensor [N, 1]): ring index (``0`` for north pole, ``1`` to
              ``n_rings`` for intermediate rings, ``n_rings+1`` for south pole).
            - ``x_unit_coord`` (torch.Tensor [N, 1]): ``x`` coordinate on the unit sphere.
            - ``y_unit_coord`` (torch.Tensor [N, 1]): ``y`` coordinate on the unit sphere.
            - ``z_unit_coord`` (torch.Tensor [N, 1]): ``z`` coordinate on the unit sphere (``1`` at
              the north pole, ``-1`` at the south pole).
            - ``is_boundary`` (torch.Tensor [N, 1], bool): ``True`` for the north and south pole nodes.

        Edge defaults:
            - ``a_ind`` (torch.Tensor [E, 1]): angular index of the source node.
            - ``r_ind`` (torch.Tensor [E, 1]): ring index of the source node.
            - ``x_unit_coord`` (torch.Tensor [E, 1]): mean unit ``x`` coordinate of source and
              destination node.
            - ``y_unit_coord`` (torch.Tensor [E, 1]): mean unit ``y`` coordinate of source and
              destination node.
            - ``z_unit_coord`` (torch.Tensor [E, 1]): mean unit ``z`` coordinate of source and
              destination node.
            - ``edge_direction`` (torch.Tensor [E, 1]): ``0`` for meridional edges, ``1`` for angular
              edges connecting ``(a_ind, r_ind) → ((a_ind+1) % n_sectors, r_ind)``.
            - ``is_boundary`` (torch.Tensor [E, 1], bool): ``True`` when both endpoint nodes are
              boundary nodes.

        Args:
            n_sectors (int): number of angular sectors (meridians).
            n_rings (int): number of intermediate horizontal rings (excluding poles).
            node_attrs (dict[str, torch.Tensor | callable]): mapping of registered
                node attribute names to either a tensor of shape ``[N, *]`` or
                ``[1, *]`` (broadcast over all nodes), or a callable whose
                parameter names are resolved from the node defaults listed above as well as any
                other node attributes already defined earlier.
            edge_attrs (dict[str, torch.Tensor | callable]): mapping of registered
                edge attribute names to either a tensor of shape ``[E, *]`` or
                ``[1, *]`` (broadcast over all edges), or a callable whose
                parameter names are resolved from the edge defaults listed above or any
                other edge attributes already defined earlier.
        """

        a_node_ind = torch.cat([torch.tensor([0]), torch.arange(n_sectors).repeat_interleave(n_rings), torch.tensor([0])]).unsqueeze(1)
        r_node_ind = torch.cat([torch.tensor([0]), torch.arange(1, n_rings + 1).repeat(n_sectors), torch.tensor([n_rings + 1])]).unsqueeze(1)

        theta = a_node_ind / n_sectors * 2 * torch.pi
        phi = r_node_ind / (n_rings + 1) * torch.pi

        x_node_unit_coord = torch.sin(phi) * torch.cos(theta)
        y_node_unit_coord = torch.sin(phi) * torch.sin(theta)
        z_node_unit_coord = torch.cos(phi)

        is_boundary_node = (r_node_ind == 0) | (r_node_ind == n_rings + 1)

        default_node_attrs = {"a_ind": a_node_ind, "r_ind": r_node_ind, "x_unit_coord": x_node_unit_coord, "y_unit_coord": y_node_unit_coord, "z_unit_coord": z_node_unit_coord, "is_boundary": is_boundary_node}

        # meridional edges: north pole → first intermediate ring
        row_north = torch.zeros(n_sectors, dtype=torch.long)
        col_north = torch.arange(n_sectors) * n_rings + 1

        # meridional edges: (a, r) → (a, r+1) among intermediate rings
        a_inner = torch.arange(n_sectors).repeat_interleave(n_rings - 1)
        r_inner = torch.arange(1, n_rings).repeat(n_sectors)
        row_inner = a_inner * n_rings + r_inner
        col_inner = a_inner * n_rings + r_inner + 1

        # meridional edges: last intermediate ring → south pole
        row_south = torch.arange(n_sectors) * n_rings + n_rings
        col_south = torch.full((n_sectors,), n_sectors * n_rings + 1, dtype=torch.long)

        row_mer = torch.cat([row_north, row_inner, row_south])
        col_mer = torch.cat([col_north, col_inner, col_south])

        # angular edges among intermediate rings: (a, r) → ((a+1) % n_sectors, r)
        a_angular = torch.arange(n_sectors).repeat_interleave(n_rings)
        r_angular = torch.arange(1, n_rings + 1).repeat(n_sectors)
        row_ang = a_angular * n_rings + r_angular
        col_ang = (a_angular + 1) % n_sectors * n_rings + r_angular

        row = torch.cat([row_mer, row_ang])
        col = torch.cat([col_mer, col_ang])
        edge_indices = torch.stack([row, col], dim=0)

        # default edge attributes
        x_edge_unit_coord, y_edge_unit_coord = self._get_edge_unit_coords(x_node_unit_coord, y_node_unit_coord, edge_indices)
        z_edge_unit_coord = (z_node_unit_coord[edge_indices[0]] + z_node_unit_coord[edge_indices[1]]) / 2

        a_edge_ind = a_node_ind[edge_indices[0]]
        r_edge_ind = r_node_ind[edge_indices[0]]
        is_boundary_edge = is_boundary_node[edge_indices[0]] & is_boundary_node[edge_indices[1]]
        edge_direction = torch.cat([torch.zeros(row_mer.shape[0]), torch.ones(row_ang.shape[0])]).unsqueeze(1)

        default_edge_attrs = {"a_ind": a_edge_ind, "r_ind": r_edge_ind, "x_unit_coord": x_edge_unit_coord, "y_unit_coord": y_edge_unit_coord, "z_unit_coord": z_edge_unit_coord, "edge_direction": edge_direction, "is_boundary": is_boundary_edge}

        num_nodes = n_sectors * n_rings + 2
        num_edges = n_sectors * (2 * n_rings + 1)

        self._create_topology(num_nodes=num_nodes, num_edges=num_edges, edge_indices=edge_indices, node_attrs=node_attrs, edge_attrs=edge_attrs, default_node_attrs=default_node_attrs, default_edge_attrs=default_edge_attrs)


def resolve_attrs(f, default_attrs, custom_attrs):
    """Resolve the keyword arguments of ``f`` from custom and default attribute dicts.

    For each parameter of ``f``, prefers a matching key in ``custom_attrs``
    over one in ``default_attrs``. Used to call the callables accepted by
    generator methods like
    [StructData.add_polar_grid][torch_structure.data.data.StructData.add_polar_grid], whose
    parameters are resolved by name from the available node/edge
    attributes.

    Args:
        f (Callable): the function whose parameters should be resolved.
        default_attrs (dict): fallback attribute values, keyed by name.
        custom_attrs (dict): attribute values that take precedence, keyed by
            name.

    Returns:
        dict: keyword arguments for calling ``f``.

    Raises:
        ValueError: if a parameter of ``f`` is found in neither dict.
    """
    resolve_attrs = {}

    sig = inspect.signature(f)

    for param_name in sig.parameters:
        if param_name in custom_attrs:
            resolve_attrs[param_name] = custom_attrs[param_name]
        elif param_name in default_attrs:
            resolve_attrs[param_name] = default_attrs[param_name]
        else:
            raise ValueError(f"Attribute '{param_name}' required for function '{f.__name__}' could not be found in the custom attributes or '{default_attrs}'. If '{param_name}' is in the custom_attributes, make sure it is defined before the function that requires it. Custom attributes are either node_attrs or edge_attrs.")

    return resolve_attrs


def encode(string: str):
    """Encode a string of at most 8 UTF-8 bytes into an integer, for storage as a node/edge name tensor.

    Args:
        string (str): the string to encode; must encode to at most 8 bytes.

    Returns:
        int: the big-endian integer representation of the encoded bytes.

    Raises:
        ValueError: if ``string`` encodes to more than 8 bytes.
    """
    encoded = string.encode("utf-8")
    if len(encoded) > 8:
        raise ValueError(f"String '{string}' encodes to {len(encoded)} bytes, which exceeds the 8-byte limit.")

    return int.from_bytes(encoded, "big")

def decode(integer: int):
    """Decode an integer produced by [encode][torch_structure.data.data.encode] back into a string."""
    return integer.to_bytes(8, "big").decode("utf-8").lstrip("\x00")

    