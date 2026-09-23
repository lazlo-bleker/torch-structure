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
from torch_scatter import scatter

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
                (self.num_edges, 1), dtype=edge_attr.dtype, device=device
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

        Falls back to ``any(support_condition, dim=1)`` if ``is_support``
        was not set directly.
        """
        if "is_support" in self._store:
            return self["is_support"]

        # If 'is_support' is missing, fall back to support_condition if available
        elif "support_condition" in self._store:
            return torch.any(self.support_condition, dim=1, keepdim=True)  # ToDO: check per dim

        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute 'is_support' or 'support_condition'"
            )
        
    @property
    def support(self):
        """torch.Tensor [N, 1]: alias for [is_support][torch_structure.data.data.StructData.is_support]."""
        return self.is_support  ### REMOVE LATER TEMP

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


    @requires_metadata
    def add_nodes(self, names = None, symmetry: str = None, **kwargs):
        """
        Adds new nodes with the given attribute values.

        Args:
            names (list[str], optional): node name for each new node. Encoded and
                stored as the ``name`` attribute; if ``symmetry`` is also given, each
                node's name is copied verbatim to all of its symmetry replicas.
            symmetry (str, optional): name of a registered symmetry to apply. If given,
                each provided node is replicated accordingly.
            **kwargs (dict[str, torch.Tensor]): mapping of registered node attribute
                names to tensors. The first dimension of each tensor must equal the
                number of nodes to add. Attributes omitted from ``kwargs`` are
                initialized to their default values or to zero if no default value
                exists.

        Raises:
            ValueError: if no keyword arguments are provided. If you want to add nodes without any attributes use add_n_empty_nodes()
            ValueError: if any provided attribute name is not a registered node attribute.
        """

        if not kwargs:
            raise ValueError("No node attributes provided for new nodes. At least one attribute must be provided.")

        if symmetry is not None:
            if symmetry not in self.metadata["name_to_symmetry"]:
                raise ValueError(f"Symmetry '{symmetry}' is not registered. Please register the symmetry first using add_symmetry().")

            symmetry_id = self.metadata["name_to_symmetry"][symmetry]
            group_matrices = self.symmetry_matrices[self.symmetry_matrix_id == symmetry_id]
            group_size = group_matrices.shape[0]
            transform_attrs = self.metadata["symmetry_transform_attrs"][symmetry_id]

        else:
            group_size = 1

        num_seed_nodes = len(next(iter(kwargs.values())))

        if num_seed_nodes == 0:
            raise ValueError("No node attribute values provided. There must be at least one attribute value")

        mismatched = {attr: len(value) for attr, value in kwargs.items() if len(value) != num_seed_nodes}
        if mismatched:
            raise ValueError(
                f"All node attributes must have the same length ({num_seed_nodes}), "
                f"but got mismatched lengths: {mismatched}."
            )

        if names:
            if len(names) != num_seed_nodes:
                raise ValueError(f"Number {len(names)} of names {names} does not match the length {num_seed_nodes} of the given attributes {kwargs}")

            kwargs["name"] = torch.tensor([encode(name) for name in names], dtype=torch.long)

        # Check for unexpected attributes
        unexpected_attrs = set(kwargs.keys()) - set(self.metadata["node_attr_list"])
        if unexpected_attrs:
            raise ValueError(
                f"Unexpected node attributes: {unexpected_attrs}. Expected: {list(self.metadata['node_attr_list'])}"
            )

        num_new_nodes = num_seed_nodes * group_size
        self.num_nodes += num_new_nodes 

        if symmetry is not None:
            base_orbit_id = int(self.orbit_id.max().item()) + 1 if self.orbit_id.numel() > 0 else 0
            new_orbit_id = (base_orbit_id + torch.arange(num_seed_nodes)).repeat_interleave(group_size).unsqueeze(1)
            new_orbit_position = torch.arange(group_size).repeat(num_seed_nodes).unsqueeze(1)
            new_symmetry_id = torch.full((num_new_nodes, 1), symmetry_id, dtype=torch.long)

            self.orbit_id = torch.cat([self.orbit_id, new_orbit_id], dim=0)
            self.orbit_position = torch.cat([self.orbit_position, new_orbit_position], dim=0)
            self.symmetry_id = torch.cat([self.symmetry_id, new_symmetry_id], dim=0)

        # Add node attributes
        for attr in self.metadata["node_attr_list"]:
            if symmetry is not None and attr in ("orbit_id", "orbit_position", "symmetry_id"):
                continue

            value = self._resolve_attr_value(attr, kwargs, num_new_nodes)

            if symmetry is not None and attr in kwargs:
                if attr in transform_attrs:
                    value = self._apply_affine(group_matrices, value)
                    value = value.reshape(num_new_nodes, *value.shape[2:])
                else:
                    value = self._expand_value_over_symmetry(value, num_seed_nodes, group_size)

            setattr(
                self,
                attr,
                torch.cat([getattr(self, attr), value], dim=0),
            )

    def _orbit_canonical_members(self, nodes, full_group_size):
        orbit_id = self.orbit_id.view(-1)
        orbit_position = self.orbit_position.view(-1)
        num_nodes = nodes.shape[0]

        idx = orbit_position[nodes]
        base = nodes - idx
        if torch.any(base < 0) or torch.any(base + full_group_size > orbit_id.numel()):
            raise ValueError(
                "Orbit of some node(s) does not fit within the current node range; "
                "its nodes may have been merged or removed."
            )

        members = base.unsqueeze(1) + torch.arange(full_group_size, device=nodes.device) 
        group = orbit_id[nodes].unsqueeze(1)
        expected_index = torch.arange(full_group_size, device=nodes.device, dtype=orbit_position.dtype).unsqueeze(0).expand(num_nodes, full_group_size)
        if not torch.all(orbit_id[members] == group) or not torch.equal(orbit_position[members], expected_index):
            raise ValueError(
                "Some orbit(s) are not contiguous or intact; their nodes may have been merged or removed."
            )

        return members, idx


    def _infer_level_sizes(self, group_id):
       
        levels = self.symmetry_level[self.symmetry_matrix_id == group_id]
        counts = torch.bincount(levels)
        boundaries = torch.cumsum(counts, dim=0)

        level_sizes = []
        prev = 1
        for boundary in boundaries.tolist():
            level_sizes.append(boundary // prev)
            prev = boundary

        return level_sizes


    def _expand_edge_indices_symmetrically(self, src, dst, group_id):
    
        orbit_position = self.orbit_position.view(-1)
        src_base = src - orbit_position[src]
        dst_base = dst - orbit_position[dst]

        level_sizes = self._infer_level_sizes(group_id)
        num_levels = len(level_sizes)
        sizes_t = torch.tensor(level_sizes, dtype=torch.long)

        place_full = torch.ones(num_levels, dtype=torch.long)
        for level in range(1, num_levels):
            place_full[level] = place_full[level - 1] * level_sizes[level - 1]

        def decode_batch(p):
            p = p.clone()
            digits = []
            for size in level_sizes:
                digits.append(p % size)
                p //= size
            return torch.stack(digits, dim=1) if digits else p.new_zeros((p.shape[0], 0))

        src_idx = decode_batch(orbit_position[src])
        dst_idx = decode_batch(orbit_position[dst])
        deltas = (dst_idx - src_idx) % sizes_t if num_levels else src_idx

        diffs = src_idx != dst_idx
        level_arange = torch.arange(num_levels, dtype=torch.long).unsqueeze(0)
        max_level = (diffs * level_arange).amax(dim=1) if num_levels else torch.zeros_like(src)

        src_out, dst_out = [], []
        counts = torch.zeros(src.shape[0], dtype=torch.long)

        for ml in torch.unique(max_level).tolist():
            bucket = torch.where(max_level == ml)[0]

            frozen_src = src_idx[bucket, :ml]
            frozen_deltas = deltas[bucket, :ml]
            frozen_sizes = sizes_t[:ml]
            frozen_dst = (frozen_src + frozen_deltas) % frozen_sizes if ml else frozen_src

            base_src = src_base[bucket] + (frozen_src * place_full[:ml]).sum(dim=1)
            base_dst = dst_base[bucket] + (frozen_dst * place_full[:ml]).sum(dim=1)

            swept_sizes = level_sizes[ml:]
            enc_place = place_full[ml:]
            num_swept = len(swept_sizes)

            total = 1
            for size in swept_sizes:
                total *= size
            idx = torch.arange(total)

            iter_place = torch.ones(num_swept, dtype=torch.long)
            for level in range(num_swept - 2, -1, -1):
                iter_place[level] = iter_place[level + 1] * swept_sizes[level + 1]
            swept_sizes_t = torch.tensor(swept_sizes, dtype=torch.long)

            digits = (idx.unsqueeze(1) // iter_place.unsqueeze(0)) % swept_sizes_t.unsqueeze(0)
            src_swept = (digits * enc_place.unsqueeze(0)).sum(dim=1)

            swept_deltas = deltas[bucket, ml:]
            dst_digits = (digits.unsqueeze(0) + swept_deltas.unsqueeze(1)) % swept_sizes_t.view(1, 1, -1)
            dst_swept = (dst_digits * enc_place.view(1, 1, -1)).sum(dim=2)

            src_all = base_src.unsqueeze(1) + src_swept.unsqueeze(0)
            dst_all = base_dst.unsqueeze(1) + dst_swept

            pair_min = torch.minimum(src_all, dst_all)
            pair_max = torch.maximum(src_all, dst_all)
            keys = pair_min * self.num_nodes + pair_max

            order = torch.argsort(keys, dim=-1, stable=True)
            sorted_keys = torch.gather(keys, 1, order)
            first_in_group = torch.cat(
                [torch.ones(sorted_keys.shape[0], 1, dtype=torch.bool), sorted_keys[:, 1:] != sorted_keys[:, :-1]],
                dim=1,
            )
            keep = torch.zeros_like(keys, dtype=torch.bool)
            keep.scatter_(1, order, first_in_group)

            src_out.append(src_all[keep])
            dst_out.append(dst_all[keep])
            counts[bucket] = keep.sum(dim=1)

        return torch.cat(src_out), torch.cat(dst_out), counts
    

    def _expand_edges_symmetrically(self, edge_indices, kwargs):

        src, dst = edge_indices[0], edge_indices[1]

        symmetry_id = self.symmetry_id.view(-1)
        src_symmetry_id, dst_symmetry_id = symmetry_id[src], symmetry_id[dst]

        for attr, value in kwargs.items():
            if not isinstance(value, torch.Tensor):
                kwargs[attr] = torch.tensor(value, dtype=getattr(self, attr).dtype)

        symmetric = (src_symmetry_id != -1) & (dst_symmetry_id != -1)

        mismatched = symmetric & (src_symmetry_id != dst_symmetry_id)
        if torch.any(mismatched):
            bad = torch.where(mismatched)[0].tolist()
            raise ValueError(
                f"Edge(s) at index {bad} connect two different registered symmetries; "
                "add_edges(consider_symmetry=True) only supports src and dst from the same "
                "registered symmetry."
            )

        #add the non-symmetric edges to the new edge list and kwargs
        src_expanded = [src[~symmetric]]
        dst_expanded = [dst[~symmetric]]
        kwargs_expanded = {attr: [value[~symmetric]] for attr, value in kwargs.items()}

        sym_idx = torch.where(symmetric)[0]
        sym_group_ids = src_symmetry_id[sym_idx]
        for group_id in torch.unique(sym_group_ids).tolist():
            group_edge_idx = sym_idx[sym_group_ids == group_id]

            src_orbits, dst_orbits, counts = self._expand_edge_indices_symmetrically(
                src[group_edge_idx], dst[group_edge_idx], group_id
            )
            src_expanded.append(src_orbits)
            dst_expanded.append(dst_orbits)

            for attr, value in kwargs.items():
                kwargs_expanded[attr].append(value[group_edge_idx].repeat_interleave(counts, dim=0))

        edge_indices = torch.stack([torch.cat(src_expanded), torch.cat(dst_expanded)], dim=0)
        kwargs_expanded = {attr: torch.cat(values, dim=0) for attr, values in kwargs_expanded.items()}

        return edge_indices, kwargs_expanded    


    @requires_metadata
    def add_edges(self, edge_indices, consider_symmetry: bool = True, **kwargs):
        """
        Adds new edges.

        Args:
            edge_indices (torch.Tensor): source and destination node
                indices for each of the ``E`` edges to add. Shape [2, E]. All indices must refer to
                existing nodes.
            consider_symmetry (bool): if ``True`` (default) and the graph has a registered
                symmetry, each edge is replicated according to the the symmetry of the source and destination nodes. If ``False``, only
                the given edge are added.
            **kwargs (dict[str, torch.Tensor]): mapping of registered edge attribute
                names to tensors. The first dimension of each tensor must equal ``E``.
                Attributes omitted from ``kwargs`` are initialized to their default
                values or to zero if no default value exists.

        Raises:
            ValueError: if any index in ``edge_indices`` is out of range.
            ValueError: if any provided attribute name is not a registered edge
                attribute.
        """

        #check for edge indices that are out of bound
        if torch.any(edge_indices >= self.num_nodes) or torch.any(edge_indices < 0):
            raise ValueError(f"Unexpected edge indices: {edge_indices[edge_indices >= self.num_nodes | (edge_indices < 0)]}, that do not correspond to existing nodes. Edge indices are expected to be between 0 and {self.num_nodes - 1}. First add respective nodes.")

        # Check for unexpected attributes
        unexpected_attrs = set(kwargs.keys()) - set(self.metadata["edge_attr_list"])
        if unexpected_attrs:
            raise ValueError(
                f"Unexpected edge attributes: {unexpected_attrs}. Expected: {list(self.metadata['edge_attr_list'])}"
            )

        if consider_symmetry and hasattr(self, "symmetry_id"):
            edge_indices, kwargs = self._expand_edges_symmetrically(edge_indices, kwargs)
        
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
       

    def add_edges_by_node_names(self, src_names, dest_names, consider_symmetry: bool = True, **kwargs):
        """
        Resolves node names to indices, then calls ``add_edges``.

        Args:
            src_names (list[str]): node name for each new edge's source endpoint.
            dest_names (list[str]): node name for each new edge's destination endpoint,
                aligned with ``src_names``.
            consider_symmetry (bool): see ``add_edges``.
            **kwargs (dict[str, torch.Tensor]): edge attribute values, see ``add_edges``.

        Raises:
            IndexError: if a name in ``src_names`` or ``dest_names`` does not match any node.
            ValueError: any error raised by ``add_edges`` for the resolved edge indices.
        """

        src_indices = torch.tensor([self.get_node_index_from_name(src)[0] for src in src_names])
        dest_indices = torch.tensor([self.get_node_index_from_name(dest)[0] for dest in dest_names])
        edge_indices = torch.stack([src_indices, dest_indices], dim=0)

        self.add_edges(edge_indices=edge_indices, consider_symmetry=consider_symmetry, **kwargs)


    def _build_orbit_lookup(self):
        orbit_id = self.orbit_id.view(-1)
        orbit_position = self.orbit_position.view(-1)

        lookup = torch.full((int(orbit_id.max()) + 1, int(orbit_position.max()) + 1), -1, dtype=torch.long)
        lookup[orbit_id, orbit_position] = torch.arange(orbit_id.shape[0])
        return lookup

    def _node_indices_from_orbit(self, lookup, orbit_ids, orbit_positions):
        ids = torch.as_tensor(orbit_ids, dtype=torch.long)
        positions = torch.as_tensor(orbit_positions, dtype=torch.long)
        in_bounds = (ids >= 0) & (ids < lookup.shape[0]) & (positions >= 0) & (positions < lookup.shape[1])
        indices = torch.full_like(ids, -1)
        indices[in_bounds] = lookup[ids[in_bounds], positions[in_bounds]]
        if torch.any(indices == -1):
            bad = torch.where(indices == -1)[0].tolist()
            raise ValueError(f"No node found for (orbit_id, orbit_position) pairs at query index {bad}.")
        return indices

    def add_edges_by_orbit(self, src_orbit_ids, dest_orbit_ids, src_orbit_positions, dest_orbit_positions, consider_symmetry: bool = True, **kwargs):
        """
        Resolves ``(orbit_id, orbit_position)`` pairs to node indices, then calls ``add_edges``.

        Args:
            src_orbit_ids (list[int]): orbit id for each new edge's source endpoint.
            dest_orbit_ids (list[int]): orbit id for each new edge's destination endpoint.
            src_orbit_positions (list[int]): orbit position for each new edge's source
                endpoint, aligned with ``src_orbit_ids``.
            dest_orbit_positions (list[int]): orbit position for each new edge's destination
                endpoint, aligned with ``dest_orbit_ids``.
            consider_symmetry (bool): see ``add_edges``.
            **kwargs (dict[str, torch.Tensor]): edge attribute values, see ``add_edges``.

        Raises:
            ValueError: if an ``(orbit_id, orbit_position)`` pair does not match any node.
            ValueError: any error raised by ``add_edges`` for the resolved edge indices.
        """
        lookup = self._build_orbit_lookup()
        src_indices = self._node_indices_from_orbit(lookup, src_orbit_ids, src_orbit_positions)
        dest_indices = self._node_indices_from_orbit(lookup, dest_orbit_ids, dest_orbit_positions)
        edge_indices = torch.stack([src_indices, dest_indices], dim=0)

        self.add_edges(edge_indices=edge_indices, consider_symmetry=consider_symmetry, **kwargs)


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

    
    #should warning be given always when nodes with different attrs are merged? 
    def merge(self, merge_group_id = None, node_priority_attribute: str = None, edge_priority_attribute: str = None):
        """
        Merges nodes into groups and updates edges accordingly.

        Each node is assigned a group id via merge_group_id. Within each group
        the surviving node is the one with the lowest original index, or, if provided, the one with
        the highest value of ``node_priority_attribute`` (ties broken by
        lowest index). 
        
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

        Raises:
            ValueError: if no ``merge_group_id`` is passed and no ``merge_group_id``
                node attribute exists on the graph.
            ValueError: if ``node_priority_attribute`` or ``edge_priority_attribute``
                is not a registered attribute or has an unsupported dtype.
        """
        
        if merge_group_id is None:
            if "merge_group_id" in self.metadata["node_attr_list"]:
                merge_group_id = getattr(self, "merge_group_id").view(-1)

            else:
                raise ValueError(f"No merge_group_id given and no merge_group_id attribute on the graph found. Either pass merge group ids or create them as a node attribute on the graph.")


        old_num_nodes = self.num_nodes

        merged_nodes_mask = merge_group_id != -1
        valid_merge_groups = merge_group_id[merged_nodes_mask]

        node_indices = torch.arange(self.num_nodes)

        # only process groups that actually have members (group IDs may not be contiguous)
        unique_groups = torch.unique(valid_merge_groups)
        num_unique_groups = unique_groups.shape[0]

        #nodes get merged into node with highest value of given node_priority_attribute  
        if node_priority_attribute is not None:

            if node_priority_attribute in self.metadata["node_attr_list"]:

                priority_attribute = getattr(self, node_priority_attribute)

                if priority_attribute.dim() > 1 and priority_attribute.shape[1] > 1:
                    raise ValueError(f"node_priority_attribute '{node_priority_attribute}' must be scalar (shape (n,) or (n,1)), got {priority_attribute.shape}.")
                if priority_attribute.dtype == torch.bool or priority_attribute.is_complex():
                    raise ValueError(f"node_priority_attribute '{node_priority_attribute}' must be float or int, got {priority_attribute.dtype}.")

                attr_values = priority_attribute.view(-1)[merged_nodes_mask]
                max_vals = scatter(attr_values, valid_merge_groups, dim=0, reduce="max")
                is_priority = attr_values == max_vals[valid_merge_groups]  
                priority_node_index = scatter(node_indices[merged_nodes_mask][is_priority], valid_merge_groups[is_priority], dim=0, reduce="min")
            
            else:
                raise ValueError(f"node_priority_attribute '{node_priority_attribute}' not found in node_attr_list: {self.metadata['node_attr_list']}.")

        else:
            priority_node_index = scatter(node_indices[merged_nodes_mask], valid_merge_groups, dim=0, reduce="min")

        priority_node_index = priority_node_index[unique_groups]

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
    
        self.num_nodes = num_unique_groups + int((~merged_nodes_mask).sum())


        # map original group id to contiguous new index
        group_to_new_idx = torch.full((valid_merge_groups.max().item() + 1,), -1, dtype=torch.long)
        group_to_new_idx[unique_groups] = torch.arange(num_unique_groups)

        # build node-index remapping
        new_node_index = torch.full((old_num_nodes,), -1, dtype=torch.long)

        new_node_index[merged_nodes_mask] = group_to_new_idx[valid_merge_groups]
        new_node_index[~merged_nodes_mask] = num_unique_groups + torch.arange((~merged_nodes_mask).sum())

        # remap edges
        src, dst = self.edge_index[0], self.edge_index[1]
        new_src = new_node_index[src]
        new_dst = new_node_index[dst]

        # drop intra-group edges
        inter_mask = (new_src != new_dst)
        new_src = new_src[inter_mask]
        new_dst = new_dst[inter_mask]

        # clean up data structure if no edges remain after filtering
        if new_src.numel() == 0:

            self.edge_index = self.edge_index[:, :0]
            self.directed_mask = self.directed_mask[:0]
            self.reciprocal_edge = self.reciprocal_edge[:0]

            for attr in self.metadata["edge_attr_list"]:
            
                setattr(self, attr, getattr(self, attr)[:0])

            return 


        # deduplicate: keep lowest original edge index per (new_src, new_dst) pair
        original_edge_indices = torch.where(inter_mask)[0]

        #encode edge key indices so scatter can be applied
        edge_keys = new_src * self.num_nodes + new_dst
        unique_keys = torch.unique(edge_keys)

        if edge_priority_attribute is not None:

            if edge_priority_attribute not in self.metadata["edge_attr_list"]:
                raise ValueError(f"edge_priority_attribute '{edge_priority_attribute}' not found in edge_attr_list: {self.metadata['edge_attr_list']}.")

            priority_attribute = getattr(self, edge_priority_attribute)

            if priority_attribute.dim() > 1 and priority_attribute.shape[1] > 1:
                raise ValueError(f"edge_priority_attribute '{edge_priority_attribute}' must be scalar (shape (n,) or (n,1)), got {priority_attribute.shape}.")
            if priority_attribute.dtype == torch.bool or priority_attribute.is_complex():
                raise ValueError(f"edge_priority_attribute '{edge_priority_attribute}' must be float or int, got {priority_attribute.dtype}.")

            attr_values = priority_attribute.view(-1)[original_edge_indices]
            max_vals = scatter(attr_values, edge_keys, dim=0, reduce="max")
            is_priority = attr_values == max_vals[edge_keys]
            priority_edge = scatter(original_edge_indices[is_priority], edge_keys[is_priority], dim=0, reduce="min")

        else:
            priority_edge = scatter(original_edge_indices, edge_keys, dim=0, reduce="min")

        priority_edge = priority_edge[unique_keys]

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


    def merge_based_on_attributes(self, attr: str, eps = 10e-4, f = None):
        """
        Merges nodes that share the same value of a node attribute.

        Nodes are grouped by the value of ``attr``. For float attributes, values are rounded to
        the nearest multiple of ``eps`` before comparison. For boolean attributes,
        only ``True`` nodes are merged together, ``False`` nodes are left as
        independent unmerged nodes.

        Args:
            attr (str): name of a registered node attribute to group by. Must be a boolean, int, or float attribute.
            eps (float, optional): rounding tolerance applied before comparing float
                attribute values. Defaults to ``10e-4``.
            f (callable, optional): optional transformation applied to the attribute
                tensor before grouping. Useful for grouping by a derived quantity
                without storing it as a separate attribute.

        Raises:
            ValueError: if ``attr`` is not a registered node attribute.
            ValueError: if the attribute dtype is not boolean, int, or float.
        """

        if attr not in self.metadata["node_attr_list"]:
            raise ValueError(f"Unexpected node attribute '{attr}' for merging. Expected an attribute in: {list(self.metadata['node_attr_list'])}")

        value = getattr(self, attr)

        if f is not None:
            value = f(value)

        if value.dtype in (torch.float32, torch.float64):            
            value = torch.round(value / eps)

        elif value.dtype == torch.bool:
            group_id = torch.zeros_like(value)
            group_id[~value] = -1

        elif value.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError(f"Unexpected data-type. Merge based on attributes requires a torch boolean, float or int.")


        _, group_id = torch.unique(value, dim = 0, return_inverse=True)

        self.merge(group_id)

        return
    

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
    def _expand_value_over_symmetry(value, num_seeds, symmetry_size):
        value = value.unsqueeze(1).expand(num_seeds, symmetry_size, *value.shape[1:]).clone()
        return value.reshape(num_seeds * symmetry_size, *value.shape[2:])


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


    def create_rotational_symmetry(self, n, origin = torch.tensor([0.0, 0.0, 0.0]), rotation_axis = torch.tensor([0.0, 0.0, 1.0])):
        """
        Builds an n-fold rotational symmetry: ``n`` affine matrices rotating evenly by
        ``2*pi/n`` about ``rotation_axis``, through ``origin``.

        Args:
            n (int): number of rotational positions. Must be at least 1.
            origin (torch.Tensor): a point the rotation axis passes through. Shape [3].
            rotation_axis (torch.Tensor): the axis to rotate about. Shape [3]

        Returns:
            dict: ``{"matrices": torch.Tensor, "symmetry_level": torch.Tensor}`` -- ``n``
            affine matrices of shape [n, 4, 4], and a ``symmetry_level`` of shape [n]
            (all zeros, since this is a single, unnested level). Pass this dict to
            ``add_symmetry`` directly, or nest it inside another symmetry via
            ``combine_symmetry``.

        Raises:
            ValueError: if ``n`` is less than 1.
        """

        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}.")

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

        return {"matrices": affine, "symmetry_level": torch.zeros(n, dtype=torch.long)}


    def create_mirror_symmetry(self, origin = torch.tensor([0.0, 0.0, 0.0]), normal = torch.tensor([1.0, 0.0, 0.0])):
        """
        Builds a mirror symmetry with respect to the plane through ``origin`` with normal ``normal``.

        Args:
            origin (torch.Tensor): a point the mirror plane passes through. Shape [3].
            normal (torch.Tensor): the mirror plane's normal vector. Shape [3].

        Returns:
            dict: ``{"matrices": torch.Tensor, "symmetry_level": torch.Tensor}`` -- 2
            affine matrices of shape [2, 4, 4] (identity, then the reflection), and a
            ``symmetry_level`` of shape [2] (all zeros, since this is a single, unnested
            level). Pass this dict to ``add_symmetry`` directly, or nest it inside another
            symmetry via ``combine_symmetry``.
        """

        normal = normal / normal.norm()
        M = torch.eye(3) - 2 * torch.outer(normal, normal)

        affine = torch.eye(4).expand(2, 4, 4).clone()
        affine[1, :3, :3] = M
        affine[1, :3, 3] = origin - M @ origin

        return {"matrices": affine, "symmetry_level": torch.zeros(2, dtype=torch.long)}


    def combine_symmetry(self, symmetry_a, symmetry_b):
        """
        Nests ``symmetry_b`` inside ``symmetry_a``, producing every combination of the
        two.

        Args:
            symmetry_a (dict): the inner symmetry, as returned by ``create_rotational_symmetry``,
                ``create_mirror_symmetry``, or a previous ``combine_symmetry`` call.
            symmetry_b (dict): the outer symmetry to nest around ``symmetry_a``, in the
                same dict form.

        Returns:
            dict: ``{"matrices": torch.Tensor, "symmetry_level": torch.Tensor}`` -- the
            combined affine matrices, shape ``[size_a * size_b, 4, 4]``, and a combined
            ``symmetry_level`` that appends ``symmetry_b``'s levels (shifted) on top of
            ``symmetry_a``'s own levels, one level deeper per nesting.
        """

        affine = symmetry_b["matrices"].unsqueeze(1) @ symmetry_a["matrices"].unsqueeze(0)

        level_a, level_b = symmetry_a["symmetry_level"], symmetry_b["symmetry_level"]
        size_a = level_a.shape[0]
        num_levels_a = int(level_a.max().item()) + 1

        combined_level = (level_b + num_levels_a).repeat_interleave(size_a)
        combined_level[:size_a] = level_a

        return {"matrices": affine.reshape(-1, 4, 4), "symmetry_level": combined_level}


    def set_node_attr(self, attr: str, mask: torch.Tensor, value: torch.Tensor, consider_symmetry: bool = True):
        """
        Sets a node attribute for the nodes selected by ``mask``.

        Args:
            attr (str): name of the registered node attribute to set.
            mask (torch.Tensor): boolean tensor of shape ``[num_nodes]``. Selects the nodes
                to set.
            value (torch.Tensor): one row per ``True`` entry in ``mask``, in the same order.
            consider_symmetry (bool): if ``True`` (default) and the graph has a registered
                symmetry, the update also propagates to every symmetry sibling of each selected node, not just the nodes selected by ``mask``. Attributes
                registered as ``transform_attrs`` are geometrically transformed per sibling,
                attributes registered as ``copy_attrs`` are copied verbatim. If ``False``, only
                the selected nodes are updated.

        Raises:
            ValueError: if ``attr`` is not a registered node attribute, or is symmetry
                bookkeeping (``orbit_id``, ``orbit_position``, ``symmetry_id``).
            ValueError: if ``mask`` does not have ``num_nodes`` entries.
            ValueError: if ``value`` does not have one row per ``True`` entry in ``mask``.
            ValueError: if multiple selected nodes belong to the same symmetry orbit.
            ValueError: if a selected node's symmetry group has ``attr`` classified as
                neither a transform nor a copy attribute.
        """

        if attr not in self.metadata["node_attr_list"]:
            raise ValueError(f"Unexpected node attribute '{attr}'. Expected an attribute in: {list(self.metadata['node_attr_list'])}")

        if attr in ("orbit_id", "orbit_position", "symmetry_id"):
            raise ValueError(f"'{attr}' is symmetry bookkeeping and cannot be set via set_node_attr().")

        mask = mask.view(-1)

        if mask.shape[0] != self.num_nodes:
            raise ValueError(f"mask must have {self.num_nodes} entries, but has {mask.shape[0]}.")

        nodes = torch.where(mask)[0]

        if value.shape[0] != nodes.shape[0]:
            raise ValueError(f"value must have one row per masked node ({nodes.shape[0]}), but has {value.shape[0]}.")

        current = getattr(self, attr)
        current[nodes] = value

        if not consider_symmetry or not hasattr(self, "symmetry_id"):
            setattr(self, attr, current)
            return

        orbit_id = self.orbit_id.view(-1)
        symmetry_id = self.symmetry_id.view(-1)

        node_orbit_ids = orbit_id[nodes]
        tracked = node_orbit_ids != -1

        counts = torch.bincount(node_orbit_ids[tracked])
        if torch.any(counts > 1):
            raise ValueError(
                "Multiple nodes selected by mask belong to the same orbit(s). "
                "Only one node per orbit may be set per call in order to avoid conflicts when applying symmetries to the graph."
            )

        tracked_nodes = nodes[tracked]
        tracked_values = value[tracked]
        tracked_group_ids = symmetry_id[tracked_nodes]

        for group_id in torch.unique(tracked_group_ids).tolist():
            group_mask = tracked_group_ids == group_id
            group_nodes = tracked_nodes[group_mask]
            group_values = tracked_values[group_mask]

            group_matrices = self.symmetry_matrices[self.symmetry_matrix_id == group_id]
            group_size = group_matrices.shape[0]

            members, idx = self._orbit_canonical_members(group_nodes, group_size)
            member_rows = members.reshape(-1)

            if attr in self.metadata["symmetry_transform_attrs"][group_id]:
                effective_seed = self._solve_affine(group_matrices[idx], group_values)
                new_values = self._apply_affine(group_matrices, effective_seed)
                new_values = new_values.reshape(-1, *new_values.shape[2:])
            elif attr in self.metadata["symmetry_copy_attrs"][group_id]:
                new_values = self._expand_value_over_symmetry(group_values, group_values.shape[0], group_size)
            else:
                raise ValueError(
                    f"Attribute '{attr}' is not classified as a transform or copy attribute for "
                    f"symmetry group {group_id}. Register it via add_symmetry(transform_attrs=..., "
                    "copy_attrs=...)."
                )

            current[member_rows] = new_values

        setattr(self, attr, current)

    def _mask_and_reorder(self, indices: torch.Tensor, value: torch.Tensor, num_rows: int):
        order = torch.argsort(indices)
        indices = indices[order]
        value = value[order]

        mask = torch.zeros(num_rows, dtype=torch.bool)
        mask[indices] = True
        return mask, value


    def set_node_attr_by_name(self, attr: str, names: list, value: torch.Tensor, consider_symmetry: bool = True):
        """
        Resolves node names to indices, then calls ``set_node_attr``.

        Args:
            attr (str): name of the registered node attribute to set.
            names (list[str]): node name for each row of ``value``.
            value (torch.Tensor): one row per entry in ``names``.
            consider_symmetry (bool): see ``set_node_attr``.

        Raises:
            IndexError: if a name in ``names`` does not match any node.
            ValueError: any error raised by ``set_node_attr`` for the resolved nodes.
        """
        indices = torch.tensor([self.get_node_index_from_name(name)[0] for name in names], dtype=torch.long)
        mask, value = self._mask_and_reorder(indices, value, self.num_nodes)

        self.set_node_attr(attr, mask, value, consider_symmetry)


    def set_node_attr_by_orbit(self, attr: str, orbit_ids: list, orbit_positions: list, value: torch.Tensor, consider_symmetry: bool = True):
        """
        Resolves ``(orbit_id, orbit_position)`` pairs to node indices, then calls ``set_node_attr``.

        Args:
            attr (str): name of the registered node attribute to set.
            orbit_ids (list[int]): orbit id for each row of ``value``.
            orbit_positions (list[int]): position inside the orbit for each row of ``value``.
            value (torch.Tensor): one row per entry in ``orbit_ids``/``orbit_positions``.
            consider_symmetry (bool): see ``set_node_attr``.

        Raises:
            ValueError: if an ``(orbit_id, orbit_position)`` pair does not match any node.
            ValueError: any error raised by ``set_node_attr`` for the resolved nodes.
        """
        lookup = self._build_orbit_lookup()
        indices = self._node_indices_from_orbit(lookup, orbit_ids, orbit_positions)
        mask, value = self._mask_and_reorder(indices, value, self.num_nodes)

        self.set_node_attr(attr, mask, value, consider_symmetry)

    def _edge_indices_for_pairs(self, src_indices: torch.Tensor, dest_indices: torch.Tensor):
        all_keys = self.edge_index[0] * self.num_nodes + self.edge_index[1]
        query_keys = src_indices * self.num_nodes + dest_indices

        sorted_keys, sort_idx = torch.sort(all_keys)
        pos = torch.searchsorted(sorted_keys, query_keys).clamp(max=sorted_keys.numel() - 1)
        found = sorted_keys[pos] == query_keys
        if not torch.all(found):
            bad = torch.where(~found)[0].tolist()
            raise ValueError(f"No edge found for (src, dest) pairs at query index {bad}.")

        return sort_idx[pos]

    def set_edge_attr_by_orbit(self, attr: str, src_orbit_ids: list, dest_orbit_ids: list, src_orbit_positions: list, dest_orbit_positions: list, value: torch.Tensor, consider_symmetry: bool = True):
        """
        Resolves the ``(orbit_id, orbit_position)`` pairs of each edge's endpoints to edge indices, then calls ``set_edge_attr``.

        Args:
            attr (str): name of the registered edge attribute to set.
            src_orbit_ids (list[int]): orbit id for each edge's source endpoint.
            dest_orbit_ids (list[int]): orbit id for each edge's destination endpoint.
            src_orbit_positions (list[int]): position inside the orbit for each edge's source
                endpoint.
            dest_orbit_positions (list[int]): position inside the orbit for each edge's destination
                endpoint.
            value (torch.Tensor): one row per edge.
            consider_symmetry (bool): see ``set_edge_attr``.

        Raises:
            ValueError: if an ``(orbit_id, orbit_position)`` pair does not match any node.
            ValueError: if no edge exists between a resolved source/destination node pair.
            ValueError: any error raised by ``set_edge_attr`` for the resolved edges.
        """
        lookup = self._build_orbit_lookup()
        src_indices = self._node_indices_from_orbit(lookup, src_orbit_ids, src_orbit_positions)
        dest_indices = self._node_indices_from_orbit(lookup, dest_orbit_ids, dest_orbit_positions)
        edges = self._edge_indices_for_pairs(src_indices, dest_indices)
        mask, value = self._mask_and_reorder(edges, value, self.edge_index.shape[1])

        self.set_edge_attr(attr, mask, value, consider_symmetry)


    def _modify_edge_attr_and_reciprocal(self, attr: str, edge_index: torch.Tensor, value: torch.Tensor):

        if value.shape[0] != edge_index.shape[0]:
            raise ValueError(f"edge_index and value must have the same length, got {edge_index.shape[0]} and {value.shape[0]}.")

        current = getattr(self, attr)
        current[edge_index] = value
        current[self.reciprocal_edge[edge_index].view(-1)] = value
        setattr(self, attr, current)


    def set_edge_attr(
        self,
        attr: str,
        mask: torch.Tensor,
        value: torch.Tensor,
        consider_symmetry: bool = True,
    ):
        """
        Sets an edge attribute for the edges selected by ``mask``.

        Args:
            attr (str): name of the registered edge attribute to set.
            mask (torch.Tensor): boolean tensor of shape ``[num_edges]``. Selects the edges
                to set; for each undirected edge, its forward or reciprocal may be selected
                interchangeably, but not both.
            value (torch.Tensor): one row per ``True`` entry in ``mask``.
            consider_symmetry (bool): if ``True`` (default) and the graph has a registered
                symmetry, the update also propagates to every symmetry sibling of each
                selected edge, not just the edges selected by ``mask`` -- the value is
                copied verbatim to every sibling. This requires ``attr`` to be registered
                as an edge copy attribute for that symmetry, via
                ``add_symmetry(edge_copy_attrs=...)``. If ``False``, only the selected
                edges (and their reciprocal edges) are updated.

        Raises:
            ValueError: if ``attr`` is not a registered edge attribute.
            ValueError: if ``value`` does not have one row per ``True`` entry in ``mask``.
            ValueError: if ``mask`` selects both the forward and reciprocal row of the
                same edge.
            ValueError: if selected edges span more than one registered symmetry, no
                sibling edge exists for a symmetry orbit implied by the selection, or
                ``attr`` is not registered as an edge copy attribute for a selected
                edge's symmetry.
        """

        if attr not in self.metadata["edge_attr_list"]:
            raise ValueError(f"Unexpected edge attribute '{attr}'. Expected an attribute in: {list(self.metadata['edge_attr_list'])}")

        mask = mask.view(-1)
        edges = torch.where(mask)[0]

        if value.shape[0] != edges.shape[0]:
            raise ValueError(f"value must have one row per masked edge ({edges.shape[0]}), got {value.shape[0]}.")

        directed = self.directed_mask.view(-1)
        edges = torch.where(directed[edges], edges, self.reciprocal_edge.view(-1)[edges])

        edges_unique, edge_counts = torch.unique(edges, return_counts=True)
        duplicated = edges_unique[edge_counts > 1]
        if duplicated.numel() > 0:
            bad_pairs = [(int(self.edge_index[0, e]), int(self.edge_index[1, e])) for e in duplicated.tolist()]
            raise ValueError(
                f"mask selects both the forward and reciprocal row of the same edge(s): {bad_pairs}."
            )

        self._modify_edge_attr_and_reciprocal(attr, edges, value)

        if not consider_symmetry or not hasattr(self, "symmetry_id"):
            return

        orbit_id = self.orbit_id.view(-1)
        symmetry_id = self.symmetry_id.view(-1)

        src, dst = self.edge_index[0, edges], self.edge_index[1, edges]
        src_group, dst_group = symmetry_id[src], symmetry_id[dst]

        tracked = (src_group != -1) & (dst_group != -1)
        src, dst, value, src_group, dst_group = (
            src[tracked], dst[tracked], value[tracked], src_group[tracked], dst_group[tracked]
        )

        num_orbits = int(self.orbit_id.max().item()) + 1
        counts = torch.bincount(orbit_id[src] * num_orbits + orbit_id[dst])
        if torch.any(counts > 1):
            warnings.warn(
                "Multiple edges selected by mask belong to the same symmetry orbit pair; "
                "the last write for each orbit pair will take effect."
            )

        mismatched = src_group != dst_group
        if torch.any(mismatched):
            bad = torch.where(mismatched)[0].tolist()
            raise ValueError(
                f"Tracked edge(s) at index {bad} connect two different registered symmetries; "
                "set_edge_attr only supports src and dst from the same registered symmetry."
            )

        all_keys = self.edge_index[0] * self.num_nodes + self.edge_index[1]
        sorted_keys, sort_idx = torch.sort(all_keys)

        for group_id in torch.unique(src_group).tolist():
            if attr not in self.metadata["symmetry_copy_attrs"][group_id]:
                raise ValueError(
                    f"Attribute '{attr}' is not registered as a copy attribute for symmetry "
                    f"group {group_id}. Register it via add_symmetry(edge_copy_attrs=...)."
                )

            group_idx = torch.where(src_group == group_id)[0]

            sib_src, sib_dst, counts = self._expand_edge_indices_symmetrically(
                src[group_idx], dst[group_idx], group_id
            )
            sib_value = value[group_idx].repeat_interleave(counts, dim=0)

            keys = sib_src * self.num_nodes + sib_dst
            pos = torch.searchsorted(sorted_keys, keys).clamp(max=sorted_keys.numel() - 1)
            if not torch.all(sorted_keys[pos] == keys):
                raise ValueError("No edge found for a sibling node pair implied by the symmetry orbit.")

            self._modify_edge_attr_and_reciprocal(attr, sort_idx[pos], sib_value)

    def add_symmetry(self, symmetries: dict, transform_attrs: list = None, copy_attrs: list = None, edge_copy_attrs: list = None):
        """
        Registers one or more symmetries on the graph, so ``add_nodes``, ``add_edges``,
        ``set_node_attr``, and ``set_edge_attr`` can propagate updates across a symmetry.
        Args:
            symmetries (dict[str, dict]): mapping of symmetry name to a symmetry dict, as
                returned by ``create_rotational_symmetry``, ``create_mirror_symmetry``, or
                ``combine_symmetry``. Each name must not already be registered.
            transform_attrs (list[str], optional): node attributes that should be
                geometrically transformed (rotated/mirrored, via the full affine matrix)
                per orbit member. Defaults to ``["coords"]`` when omitted (pass ``[]``
                explicitly for no transform attributes at all).
            copy_attrs (list[str], optional): node attributes that should be copied
                verbatim to every member of the same symmetry orbit, without any transform.
            edge_copy_attrs (list[str], optional): edge attributes that should be copied
                verbatim to every symmetry sibling edge. Edges have no transform category --
                only copying is supported.

        Raises:
            ValueError: if ``transform_attrs``/``copy_attrs`` contains symmetry bookkeeping
                (``orbit_id``, ``orbit_position``, ``symmetry_id``), or a registered edge
                attribute name (edges only support ``edge_copy_attrs``).
            ValueError: if a name in ``symmetries`` is already registered.
            ValueError: if a symmetry's ``symmetry_level`` does not have one entry per matrix.
        """

        transform_attrs = transform_attrs if transform_attrs is not None else ["coords"]
        copy_attrs = copy_attrs or []
        edge_copy_attrs = edge_copy_attrs or []

        bookkeeping = {"orbit_id", "orbit_position", "symmetry_id"}
        invalid_attrs = bookkeeping & (set(transform_attrs) | set(copy_attrs))
        if invalid_attrs:
            raise ValueError(
                f"{sorted(invalid_attrs)} is symmetry bookkeeping and cannot be registered as a "
                "transform or copy attribute."
            )

        invalid_transform_attrs = set(transform_attrs) & set(self.metadata["edge_attr_list"])
        if invalid_transform_attrs:
            raise ValueError(
                f"{sorted(invalid_transform_attrs)} is a registered edge attribute and cannot be "
                "registered as a transform attribute; edges only support copy_attrs, via edge_copy_attrs=..."
            )

        is_first_symmetry = "name_to_symmetry" not in self.metadata
        name_to_symmetry = self.metadata.setdefault("name_to_symmetry", {})

        if is_first_symmetry:

            self.metadata["graph_attr_list"].append("symmetry_matrices")
            self.metadata["graph_attr_list"].append("symmetry_matrix_id")
            self.metadata["graph_attr_list"].append("symmetry_level")
            self.metadata["node_attr_list"].append("orbit_id")
            self.metadata["node_attr_list"].append("orbit_position")
            self.metadata["node_attr_list"].append("symmetry_id")

            self.metadata["default_attrs"]["orbit_id"] = torch.tensor(-1, dtype=torch.long)
            self.metadata["default_attrs"]["orbit_position"] = torch.tensor(-1, dtype=torch.long)
            self.metadata["default_attrs"]["symmetry_id"] = torch.tensor(-1, dtype=torch.long)

            self.symmetry_matrices = torch.empty((0, 4, 4))
            self.symmetry_matrix_id = torch.empty((0,), dtype=torch.long)
            self.symmetry_level = torch.empty((0,), dtype=torch.long)
            self.orbit_id = torch.full((self.num_nodes, 1), -1, dtype=torch.long)
            self.orbit_position = torch.full((self.num_nodes, 1), -1, dtype=torch.long)
            self.symmetry_id = torch.full((self.num_nodes, 1), -1, dtype=torch.long)

            self.metadata["symmetry_transform_attrs"] = {}
            self.metadata["symmetry_copy_attrs"] = {}

        for name, symmetry in symmetries.items():

            if name in name_to_symmetry:
                raise ValueError(f"Symmetry '{name}' already exists.")

            symmetry_matrices = symmetry["matrices"]
            symmetry_level = symmetry["symmetry_level"]

            m = symmetry_matrices.shape[0]
            if symmetry_level.shape[0] != m:
                raise ValueError(
                    f"symmetry_level for '{name}' must have one entry per matrix ({m}), got {symmetry_level.shape[0]}."
                )
            group_id = int(self.symmetry_matrix_id.max().item()) + 1 if self.symmetry_matrix_id.numel() > 0 else 0

            self.symmetry_matrices = torch.cat([self.symmetry_matrices, symmetry_matrices], dim=0)
            self.symmetry_matrix_id = torch.cat(
                [self.symmetry_matrix_id, torch.full((m,), group_id, dtype=torch.long)], dim=0
            )
            self.symmetry_level = torch.cat([self.symmetry_level, symmetry_level.to(torch.long)], dim=0)

            name_to_symmetry[name] = group_id
            self.metadata["symmetry_transform_attrs"][group_id] = transform_attrs
            self.metadata["symmetry_copy_attrs"][group_id] = copy_attrs + edge_copy_attrs

    def view_symmetries(self):
        """
        Prints every registered symmetry's name, order, transform
        attributes, and copy attributes.
        """
        if "name_to_symmetry" not in self.metadata:
            print("No symmetries registered.")
            return

        for name, group_id in self.metadata["name_to_symmetry"].items():
            order = int((self.symmetry_matrix_id == group_id).sum())
            transform_attrs = self.metadata["symmetry_transform_attrs"][group_id]
            copy_attrs = self.metadata["symmetry_copy_attrs"][group_id]
            print(f"{name}: order={order}, transform_attrs={transform_attrs}, copy_attrs={copy_attrs}")


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
            n (int): number of nodes along each side of the triangle.
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
            n_rings (int): number of concentric rings (excluding the center node).
            n_sectors (int): number of angular sectors.
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

    