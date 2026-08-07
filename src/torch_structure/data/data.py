import torch_geometric as pyg
import torch
import warnings
import inspect
import copy
import json

from torch_structure.data.view import NodeView
from torch_structure.data.utils import requires_metadata
from torch_structure.loss import ResidualForceLoss
from torch_structure.geometry import graph_edge_lengths
from torch_structure.mixins import TSMixin
from torch_scatter import scatter

class StructData(TSMixin, pyg.data.Data):
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
        return self.is_support  ### REMOVE LATER TEMP
    
    @property
    def length_from_coords(self):
        if hasattr(self, "coords"):
            return graph_edge_lengths(self.coords, self.edge_index)
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute 'coords'"
            )

    @property
    def bbox(self):
        if not hasattr(self, "coords"):
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute 'coords' required for bounding box calculation."
            )
        return torch.stack(
            [self.coords.min(dim=0).values, self.coords.max(dim=0).values], dim=0
        )

    @property
    def directed_edge_index(self):
        return self.edge_index[:, self.directed_mask.view(-1)]
    
    @property
    def cem_edge_index(self):
        return self.edge_index[:, self.cem_edge_mask]
    
    @property
    def cem_edge_mask(self):
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
            **kwargs (dict[str, torch.Tensor]): mapping of registered node attribute
                names to tensors. The first dimension of each tensor must equal the
                number of nodes to add. Attributes omitted from ``kwargs`` are
                initialized to their default values or to zero if no default value
                exists.

        Raises:
            ValueError: if no keyword arguments are provided. If you want to add nodes without any attributes use add_n_empty_nodes()
            ValueError: if any provided attribute name is not a registered node
                attribute.
        """

        if not kwargs:
            raise ValueError("No node attributes provided for new nodes. At least one attribute must be provided.")

        if symmetry is not None:
            if symmetry not in self.metadata["name_to_symmetry"]:
                raise ValueError(f"Symmetry '{symmetry}' is not registered. Please register the symmetry first using add_symmetry().")

            group_id = self.metadata["name_to_symmetry"][symmetry]
            group_matrices = self.symmetry_matrices[self.symmetry_matrix_ind == group_id]
            group_size = group_matrices.shape[0]
            transform_attrs = self.metadata["symmetry_transform_attrs"][group_id]

        else:
            group_size = 1

        num_seed_nodes = len(next(iter(kwargs.values())))

        if num_seed_nodes == 0:
            raise ValueError("No node attribute values provided. There must be at least one attribute value")

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
            new_orbit_index = torch.arange(group_size).repeat(num_seed_nodes).unsqueeze(1)
            new_symmetry_id = torch.full((num_new_nodes, 1), group_id, dtype=torch.long)
            new_cyclic_subgroup = torch.zeros((num_new_nodes, 1), dtype=torch.long)

            self.orbit_id = torch.cat([self.orbit_id, new_orbit_id], dim=0)
            self.orbit_index = torch.cat([self.orbit_index, new_orbit_index], dim=0)
            self.symmetry_id = torch.cat([self.symmetry_id, new_symmetry_id], dim=0)
            self.cyclic_subgroup = torch.cat([self.cyclic_subgroup, new_cyclic_subgroup], dim=0)

        # Add node attributes
        for attr in self.metadata["node_attr_list"]:
            if symmetry is not None and attr in ("orbit_id", "orbit_index", "symmetry_id", "cyclic_subgroup"):
                continue

            from_kwargs = attr in kwargs
            value = self._resolve_attr_value(attr, kwargs, num_new_nodes)

            if symmetry is not None and from_kwargs:
                if attr in transform_attrs:
                    value = self._apply_affine(group_matrices, value)
                    value = value.reshape(num_new_nodes, *value.shape[2:])
                else:
                    value = self._tile_over_group(value, num_seed_nodes, group_size)

            setattr(
                self,
                attr,
                torch.cat([getattr(self, attr), value], dim=0),
            )


    def _orbit_members(self, nodes, full_group_size, stride):
        orbit_id = self.orbit_id.view(-1)
        orbit_index = self.orbit_index.view(-1)
        num_nodes = nodes.shape[0]

        idx = orbit_index[nodes]
        base = nodes - idx
        if torch.any(base < 0) or torch.any(base + full_group_size > orbit_id.numel()):
            raise ValueError(
                "Orbit of some node(s) does not fit within the current node range; "
                "its nodes may have been merged or removed."
            )

        members = base.unsqueeze(1) + torch.arange(full_group_size, device=nodes.device)  # [S, full_group_size]
        group = orbit_id[nodes].unsqueeze(1)
        expected_index = torch.arange(full_group_size, device=nodes.device, dtype=orbit_index.dtype).unsqueeze(0).expand(num_nodes, full_group_size)
        if not torch.all(orbit_id[members] == group) or not torch.equal(orbit_index[members], expected_index):
            raise ValueError(
                "Some orbit(s) are not contiguous or intact; their nodes may have been merged or removed."
            )

        gather_idx = (idx.unsqueeze(1) + torch.arange(full_group_size, device=nodes.device)) % full_group_size
        rolled = torch.gather(members, 1, gather_idx)
        return rolled[:, ::stride]


    def _resolve_node_symmetry_group_size(self, nodes, role):
        symmetry_id = self.symmetry_id.view(-1)
        group_id = symmetry_id[nodes]

        untracked = group_id == -1
        if torch.any(untracked):
            raise ValueError(
                f"{role} node(s) {nodes[untracked].tolist()} have no symmetry orbit; "
                "they must have been added via add_nodes(symmetry=...)."
            )

        if not torch.all(group_id == group_id[0]):
            raise ValueError(f"All {role.lower()} nodes in a single add_edges() call must belong to the same symmetry group.")

        group_sizes = torch.bincount(self.symmetry_matrix_ind)
        return int(group_sizes[group_id[0]])


    @staticmethod
    def _resolve_output_group_size(src_group_size, dst_group_size, symmetry_order):
        if symmetry_order is None:
        
            bigger, smaller = max(src_group_size, dst_group_size), min(src_group_size, dst_group_size)
            if bigger % smaller != 0:
                raise ValueError(
                    f"Source and destination symmetry groups have sizes {src_group_size} and {dst_group_size}, "
                    "which must evenly divide each other."
                )
            return smaller

        if symmetry_order < 1:
            raise ValueError(f"symmetry_order must be at least 1, got {symmetry_order}.")
        if src_group_size % symmetry_order != 0 or dst_group_size % symmetry_order != 0:
            raise ValueError(
                f"symmetry_order={symmetry_order} must evenly divide both the source ({src_group_size}) "
                f"and destination ({dst_group_size}) symmetry group sizes."
            )
        return symmetry_order


    def _expand_edges_for_symmetry(self, edge_indices, symmetry_order, kwargs):
        if not isinstance(edge_indices, torch.Tensor):
            edge_indices = torch.tensor(edge_indices, dtype=torch.long)

        src, dst = edge_indices[0], edge_indices[1]
        num_seed_edges = src.shape[0]

        src_group_size = self._resolve_node_symmetry_group_size(src, "Source")
        dst_group_size = self._resolve_node_symmetry_group_size(dst, "Destination")

        output_group_size = self._resolve_output_group_size(src_group_size, dst_group_size, symmetry_order)
        src_stride = src_group_size // output_group_size
        dst_stride = dst_group_size // output_group_size

        src_orbits = self._orbit_members(src, src_group_size, src_stride)  
        dst_orbits = self._orbit_members(dst, dst_group_size, dst_stride)  

        edge_indices = torch.stack([src_orbits.reshape(-1), dst_orbits.reshape(-1)], dim=0)

        symmetrized_kwargs = {}
        for attr, value in kwargs.items():
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value, dtype=getattr(self, attr).dtype)

            symmetrized_kwargs[attr] = self._tile_over_group(value, num_seed_edges, output_group_size)

        return edge_indices, symmetrized_kwargs


    @requires_metadata
    def add_edges(self, edge_indices, consider_symmetry: bool = True, symmetry_order: int = None, **kwargs):
        """
        Adds new edges.

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
        """

        if consider_symmetry and hasattr(self, "symmetry_id"):
            edge_indices, kwargs = self._expand_edges_for_symmetry(edge_indices, symmetry_order, kwargs)

        if torch.any(edge_indices >= self.num_nodes) or torch.any(edge_indices < 0):
            raise ValueError(f"Unexpected edge indices: {edge_indices[edge_indices >= self.num_nodes | (edge_indices < 0)]}, that do not correspond to existing nodes. Edge indices are expected to be between 0 and {self.num_nodes - 1}. First add respective nodes.")

        num_new_edges = edge_indices.shape[1] 

        # Check for unexpected attributes
        unexpected_attrs = set(kwargs.keys()) - set(self.metadata["edge_attr_list"])
        if unexpected_attrs:
            raise ValueError(
                f"Unexpected edge attributes: {unexpected_attrs}. Expected: {list(self.metadata['edge_attr_list'])}"
            )
        
        # Update directed mask and reciprocal edge
        self.directed_mask = torch.cat(

            [self.directed_mask, torch.ones(num_new_edges, dtype=torch.bool).unsqueeze(1), torch.zeros(num_new_edges, dtype=torch.bool).unsqueeze(1)], 

        dim=0)
        
        edges = torch.arange(self.num_edges, self.num_edges + num_new_edges).unsqueeze(1)
        reciprocal_edges = torch.arange(self.num_edges + num_new_edges, self.num_edges + num_new_edges * 2).unsqueeze(1)
        
        self.reciprocal_edge = torch.cat(
            [self.reciprocal_edge, reciprocal_edges, edges],
            dim=0
        )

        # Add edge to edge_index
        self.edge_index = torch.cat([
            self.edge_index,
            edge_indices,
            edge_indices.flip(0)
        ], dim=1)

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
       

    def add_edges_by_names(self, src_names, dest_names, **kwargs):

        src_indices = torch.tensor([self.get_node_index_from_name(src) for src in src_names])
        dest_indices = torch.tensor([self.get_node_index_from_name(dest) for dest in dest_names])
        edge_indices = torch.stack([src_indices, dest_indices], dim=0)

        self.add_edges(edge_indices=edge_indices, **kwargs)

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

        return int((self.name == encode(name)).nonzero(as_tuple=True)[0])


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

                f_kwargs = resolve_attrs(f, default_attrs, attrs)

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
    def _tile_over_group(value, num_seeds, group_size):
        value = value.unsqueeze(1).expand(num_seeds, group_size, *value.shape[1:]).clone()
        return value.reshape(num_seeds * group_size, *value.shape[2:])


    @staticmethod
    def _apply_affine(matrices, values):
        ones = values.new_ones(*values.shape[:-1], 1)
        values_h = torch.cat([values, ones], dim=-1)
        return torch.einsum("kij,sj->ski", matrices, values_h)[..., :3]


    def _get_edge_unit_coords(self, x, y, edge_indices):

        x_src = x[edge_indices[0]]
        y_src= y[edge_indices[0]]
        x_dest = x[edge_indices[1]]
        y_dest = y[edge_indices[1]]
        x_edge = (x_src + x_dest) / 2
        y_edge = (y_src + y_dest) / 2

        return x_edge, y_edge


    def create_rotational_symmetry(self, n, origin = torch.tensor([0.0, 0.0, 0.0]), rotation_axis = torch.tensor([0.0, 0.0, 1.0])):

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
        
        return affine


    def create_mirror_symmetry(self, origin = torch.tensor([0.0, 0.0, 0.0]), normal = torch.tensor([1.0, 0.0, 0.0])):

        normal = normal / normal.norm()
        M = torch.eye(3) - 2 * torch.outer(normal, normal)

        affine = torch.eye(4).expand(2, 4, 4).clone()
        affine[1, :3, :3] = M
        affine[1, :3, 3] = origin - M @ origin

        return affine


    def combine_symmetry(self, symmetry_a, symmetry_b):

        affine = symmetry_a.unsqueeze(1) @ symmetry_b.unsqueeze(0)

        return affine.reshape(-1, 4, 4)


    def set_node_attr_with_symmetry(self, attr: str, mask: torch.Tensor, value: torch.Tensor):
        if attr not in self.metadata["node_attr_list"]:
            raise ValueError(f"Unexpected node attribute '{attr}'. Expected an attribute in: {list(self.metadata['node_attr_list'])}")

        if attr in ("orbit_id", "orbit_index", "symmetry_id", "cyclic_subgroup"):
            raise ValueError(f"'{attr}' is symmetry bookkeeping and cannot be set via set_node_attr_with_symmetry().")

        if not hasattr(self, "symmetry_id"):
            raise ValueError("Graph has no symmetry information. Use add_symmetry() to add symmetry information before setting node attributes with symmetry.")

        mask = mask.view(-1)
        nodes = torch.where(mask)[0]

        if value.shape[0] != nodes.shape[0]:
            raise ValueError(f"value must have one row per masked node ({nodes.shape[0]}), got {value.shape[0]}.")

        orbit_id = self.orbit_id.view(-1)
        symmetry_id = self.symmetry_id.view(-1)

        node_orbit_ids = orbit_id[nodes]
        tracked = node_orbit_ids != -1

        counts = torch.bincount(node_orbit_ids[tracked])
        if torch.any(counts > 1):
            raise ValueError(
                f"Multiple nodes selected by mask belong to the same orbit(s)"
                f"Only one node per orbit may be set per call in order to avoid conflicts when applying symmetries to the graph."
            )

        current = getattr(self, attr).clone()
        current[nodes] = value

        tracked_nodes = nodes[tracked]
        tracked_values = value[tracked]
        tracked_group_ids = symmetry_id[tracked_nodes]

        for group_id in torch.unique(tracked_group_ids).tolist():
            group_mask = tracked_group_ids == group_id
            group_nodes = tracked_nodes[group_mask]
            group_values = tracked_values[group_mask]

            group_matrices = self.symmetry_matrices[self.symmetry_matrix_ind == group_id]
            group_size = group_matrices.shape[0]

            member_rows = self._orbit_members(group_nodes, group_size, 1).reshape(-1)

            if attr in self.metadata["symmetry_transform_attrs"][group_id]:
                new_values = self._apply_affine(group_matrices, group_values)
                new_values = new_values.reshape(-1, *new_values.shape[2:])
            elif attr in self.metadata["symmetry_copy_attrs"][group_id]:
                new_values = self._tile_over_group(group_values, group_values.shape[0], group_size)
            else:
                raise ValueError(
                    f"Attribute '{attr}' is not classified as a transform or copy attribute for "
                    f"symmetry group {group_id}. Register it via add_symmetry(transform_attrs=..., "
                    "copy_attrs=...)."
                )

            current[member_rows] = new_values

        setattr(self, attr, current)


    def set_node_attr_with_symmetry1(self, attr: str, value: torch.Tensor):

        if attr not in self.metadata["node_attr_list"]:
            raise ValueError(f"Unexpected node attribute '{attr}'. Expected an attribute in: {list(self.metadata['node_attr_list'])}")

        if attr in ("orbit_id", "orbit_index", "symmetry_id", "cyclic_subgroup"):
            raise ValueError(f"'{attr}' is symmetry bookkeeping and cannot be set via set_node_attr_with_symmetry().")

        if not hasattr(self, "symmetry_id"):
            raise ValueError("Graph has no symmetry information. Use add_symmetry() to add symmetry information before setting node attributes with symmetry.")

        if value.shape[0] != self.num_nodes:
            raise ValueError(f"Value tensor must have shape [num_nodes, *], got {value.shape}.")

        value = value.clone()
        old_value = getattr(self, attr)
        changed_indices = torch.where(torch.any((value != old_value).reshape(self.num_nodes, -1), dim=1))[0]

        orbit_id = self.orbit_id.view(-1)
        symmetry_id = self.symmetry_id.view(-1)

        symmetric_changed = changed_indices[orbit_id[changed_indices] != -1]
        symmetric_orbit_ids = orbit_id[symmetric_changed]

        for touched_orbit_id in torch.unique(symmetric_orbit_ids).tolist():
            node = symmetric_changed[symmetric_orbit_ids == touched_orbit_id]

            if node.numel() > 1:
                raise ValueError(
                    f"Multiple nodes {node.tolist()} in orbit {touched_orbit_id} were "
                    f"changed at once for attribute '{attr}'; only one member of an orbit may be "
                    "changed per call."
                )

            node_idx = int(node)
            group_id = int(symmetry_id[node_idx])
            group_matrices = self.symmetry_matrices[self.symmetry_matrix_ind == group_id]
            group_size = group_matrices.shape[0]
     
            member_rows = self._orbit_members(node, group_size, 1).view(-1)
            new_val = value[node_idx]

            if attr in self.metadata["symmetry_transform_attrs"][group_id]:
                value[member_rows] = self._apply_affine(group_matrices, new_val.unsqueeze(0)).squeeze(0)
            elif attr in self.metadata["symmetry_copy_attrs"][group_id]:
                value[member_rows] = new_val.unsqueeze(0).expand(group_size, *new_val.shape).clone()
            else:
                raise ValueError(
                    f"Attribute '{attr}' is not classified as a transform or copy attribute for "
                    f"symmetry group {group_id}. Register it via add_symmetry(transform_attrs=..., "
                    "copy_attrs=...)."
                )

        setattr(self, attr, value)


    def set_edge_attr_with_symmetry(self, attr: str, mask: torch.Tensor, value: torch.Tensor, symmetry_order: int = None):
        if attr not in self.metadata["edge_attr_list"]:
            raise ValueError(f"Unexpected edge attribute '{attr}'. Expected an attribute in: {list(self.metadata['edge_attr_list'])}")

        mask = mask.view(-1)
        edges = torch.where(mask)[0]

        if value.shape[0] != edges.shape[0]:
            raise ValueError(f"value must have one row per masked edge ({edges.shape[0]}), got {value.shape[0]}.")

        if torch.any(~self.directed_mask.view(-1)[edges]):
            raise ValueError("mask may only select directed (forward) edge rows; the reciprocal row is updated automatically.")

        current = getattr(self, attr).clone()

        def write(rows, vals):
            current[rows] = vals
            current[self.reciprocal_edge[rows].view(-1)] = vals

        write(edges, value)

        if not hasattr(self, "symmetry_id"):
            setattr(self, attr, current)
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
            raise ValueError(
                "Multiple edges selected by mask belong to the same symmetry orbit pair; "
                "only one edge per orbit pair may be set per call in order to avoid conflicts "
                "when applying symmetries to the graph."
            )
        
        all_keys = self.edge_index[0] * self.num_nodes + self.edge_index[1]
        sorted_keys, sort_idx = torch.sort(all_keys)

        group_sizes = torch.bincount(self.symmetry_matrix_ind)
        num_groups = group_sizes.numel()
        combo_keys = src_group * num_groups + dst_group

        for combo in torch.unique(combo_keys).tolist():
            c = combo_keys == combo
            c_src, c_dst, c_value = src[c], dst[c], value[c]
            src_size = int(group_sizes[int(src_group[c][0])])
            dst_size = int(group_sizes[int(dst_group[c][0])])
            group_size = self._resolve_output_group_size(src_size, dst_size, symmetry_order)

            sib_src = self._orbit_members(c_src, src_size, src_size // group_size).reshape(-1)
            sib_dst = self._orbit_members(c_dst, dst_size, dst_size // group_size).reshape(-1)
            sib_value = self._tile_over_group(c_value, c_value.shape[0], group_size)

            keys = sib_src * self.num_nodes + sib_dst
            pos = torch.searchsorted(sorted_keys, keys).clamp(max=sorted_keys.numel() - 1)
            if not torch.all(sorted_keys[pos] == keys):
                raise ValueError("No edge found for a sibling node pair implied by the symmetry orbit.")

            write(sort_idx[pos], sib_value)

        setattr(self, attr, current)


    def set_edge_attr_with_symmetry1(self, attr: str, src: int, dst: int, value, symmetry_order: int = None):

        if attr not in self.metadata["edge_attr_list"]:
            raise ValueError(f"Unexpected edge attribute '{attr}'. Expected an attribute in: {list(self.metadata['edge_attr_list'])}")

        src, dst = int(src), int(dst)
        value = torch.as_tensor(value, dtype=getattr(self, attr).dtype)

        src_group_id = int(self.symmetry_id.view(-1)[src]) if hasattr(self, "symmetry_id") else -1
        dst_group_id = int(self.symmetry_id.view(-1)[dst]) if hasattr(self, "symmetry_id") else -1

        if src_group_id == -1 or dst_group_id == -1:
            # Not part of any orbit (on either end): just this one edge, no
            # siblings to propagate to.
            sibling_pairs = [(src, dst)]
        else:
            group_sizes = torch.bincount(self.symmetry_matrix_ind)
            src_group_size = int(group_sizes[src_group_id])
            dst_group_size = int(group_sizes[dst_group_id])
            group_size = self._resolve_output_group_size(src_group_size, dst_group_size, symmetry_order)

            src_orbit = self._orbit_members(torch.tensor([src]), src_group_size, src_group_size // group_size).view(-1)
            dst_orbit = self._orbit_members(torch.tensor([dst]), dst_group_size, dst_group_size // group_size).view(-1)
            sibling_pairs = list(zip(src_orbit.tolist(), dst_orbit.tolist()))

        # Edge attributes are always copied (never rotated), matching how
        # add_edges(consider_symmetry=True) expands them. Each logical edge is
        # stored as both a forward and a reciprocal row, so match either order.
        updated = False
        for s, d in sibling_pairs:
            mask = ((self.edge_index[0] == s) & (self.edge_index[1] == d)) | \
                   ((self.edge_index[0] == d) & (self.edge_index[1] == s))
            if torch.any(mask):
                getattr(self, attr)[mask] = value
                updated = True

        if not updated:
            raise ValueError(f"No edge between nodes {src} and {dst} (or its symmetry orbit) was found.")


    def add_symmetry(self, symmetries: dict, transform_attrs: list = None, copy_attrs: list = None):

        transform_attrs = transform_attrs or []
        copy_attrs = copy_attrs or []

        is_first_symmetry = "name_to_symmetry" not in self.metadata
        name_to_symmetry = self.metadata.setdefault("name_to_symmetry", {})

        if is_first_symmetry:

            self.metadata["graph_attr_list"].append("symmetry_matrices")
            self.metadata["graph_attr_list"].append("symmetry_matrix_ind")
            self.metadata["node_attr_list"].append("orbit_id")
            self.metadata["node_attr_list"].append("orbit_index")
            self.metadata["node_attr_list"].append("symmetry_id")
            self.metadata["node_attr_list"].append("cyclic_subgroup")

            self.metadata["default_attrs"]["orbit_id"] = torch.tensor(-1, dtype=torch.long)
            self.metadata["default_attrs"]["orbit_index"] = torch.tensor(-1, dtype=torch.long)
            self.metadata["default_attrs"]["symmetry_id"] = torch.tensor(-1, dtype=torch.long)
            self.metadata["default_attrs"]["cyclic_subgroup"] = torch.tensor(-1, dtype=torch.long)

            self.symmetry_matrices = torch.empty((0, 4, 4))
            self.symmetry_matrix_ind = torch.empty((0,), dtype=torch.long)
            self.orbit_id = torch.full((self.num_nodes, 1), -1, dtype=torch.long)
            self.orbit_index = torch.full((self.num_nodes, 1), -1, dtype=torch.long)
            self.symmetry_id = torch.full((self.num_nodes, 1), -1, dtype=torch.long)
            self.cyclic_subgroup = torch.full((self.num_nodes, 1), -1, dtype=torch.long)

            self.metadata["symmetry_transform_attrs"] = {}
            self.metadata["symmetry_copy_attrs"] = {}

        for name, symmetry_matrices in symmetries.items():

            if name in name_to_symmetry:
                raise ValueError(f"Symmetry '{name}' already exists.")

            m = symmetry_matrices.shape[0]
            group_id = int(self.symmetry_matrix_ind.max().item()) + 1 if self.symmetry_matrix_ind.numel() > 0 else 0

            self.symmetry_matrices = torch.cat([self.symmetry_matrices, symmetry_matrices], dim=0)
            self.symmetry_matrix_ind = torch.cat(
                [self.symmetry_matrix_ind, torch.full((m,), group_id, dtype=torch.long)], dim=0
            )

            name_to_symmetry[name] = group_id
            self.metadata["symmetry_transform_attrs"][group_id] = transform_attrs
            self.metadata["symmetry_copy_attrs"][group_id] = copy_attrs


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
            - ``edge_direction`` (torch.Tensor [E]): ``0`` for horizontal edges connecting
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

        edge_direction = torch.cat([torch.zeros(row_h.shape[0]), torch.ones(row_v.shape[0]), torch.full((row_d.shape[0],), 2.0)], dim=0)

        default_edge_attrs = {"u_ind": u_edge_ind, "v_ind": v_edge_ind, "x_unit_coord": x_edge_unit_coord, "y_unit_coord": y_edge_unit_coord, "edge_direction": edge_direction,"is_boundary": is_boundary_edge}

        num_nodes = n * (n + 1) // 2
        num_edges = 3 * n * (n-1) // 2

        self._create_topology(num_nodes = num_nodes, num_edges = num_edges, edge_indices=edge_indices, node_attrs=node_attrs, edge_attrs=edge_attrs, default_node_attrs=default_node_attrs, default_edge_attrs=default_edge_attrs)


    def add_polar_grid(self, n_rings, n_sectors, node_attrs: dict = {}, edge_attrs: dict = {}):

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
            - ``r_ind`` (torch.Tensor [N, 1]): radial index of each node (``0`` for the center,
              ``1`` to ``n_rings`` for the rings).
            - ``a_ind`` (torch.Tensor [N, 1]): angular index of each node (``0`` to ``n_sectors-1``).
              The center node has ``a_ind == 0``.
            - ``x_unit_coord`` (torch.Tensor [N, 1]): ``x`` coordinate (cartesian) in unit disc. 
            - ``y_unit_coord`` (torch.Tensor [N, 1]): ``y`` coordinate (cartesian) in unit disc.
            - ``is_boundary`` (torch.Tensor [N, 1], bool): ``True`` for nodes on the outermost ring.

        Edge defaults:
            - ``r_ind`` (torch.Tensor [E, 1]): radial index of the source node.
            - ``a_ind`` (torch.Tensor [E, 1]): angular index of the source node.
            - ``x_unit_coord`` (torch.Tensor [E, 1]): mean unit ``x`` coordinate of source and
              destination node.
            - ``y_unit_coord`` (torch.Tensor [E, 1]): mean unit ``y`` coordinate of source and
              destination node.
            - ``edge_direction`` (torch.Tensor [E]): ``0`` for radial edges, ``1`` for angular edges.
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

        r_node_ind = torch.cat([torch.tensor([0]), torch.arange(1, n_rings+1).repeat(n_sectors)]).unsqueeze(1)
        a_node_ind = torch.cat([torch.tensor([0]), torch.arange(0, n_sectors).repeat_interleave(n_rings)]).unsqueeze(1)
        
        angle = a_node_ind / n_sectors * 2 * torch.pi

        x_node_unit_coord = r_node_ind / n_rings * torch.cos(angle)
        y_node_unit_coord = r_node_ind / n_rings * torch.sin(angle)

        is_boundary_node = (r_node_ind == n_rings)
        default_node_attrs = {"r_ind": r_node_ind, "a_ind": a_node_ind, "x_unit_coord": x_node_unit_coord, "y_unit_coord": y_node_unit_coord, "is_boundary": is_boundary_node}

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
        a_edge_ind = a_node_ind[edge_indices[0]]

        is_boundary_edge = is_boundary_node[edge_indices[0]] & is_boundary_node[edge_indices[1]]
        edge_direction = torch.cat([torch.zeros(row_r.shape[0]), torch.ones(row_a.shape[0])])

        default_edge_attrs = {"r_ind": r_edge_ind, "a_ind": a_edge_ind, "x_unit_coord": x_edge_unit_coord, "y_unit_coord": y_edge_unit_coord, "edge_direction": edge_direction, "is_boundary": is_boundary_edge}
        
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
            - ``edge_direction`` (torch.Tensor [E]): ``0`` for horizontal edges connecting
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
        edge_direction = torch.cat([torch.zeros(row_h.shape[0]), torch.ones(row_v.shape[0])], dim=0)

        default_edge_attrs = {"u_ind": u_edge_ind, "v_ind": v_edge_ind, "x_unit_coord": x_edge_unit_coord, "y_unit_coord": y_edge_unit_coord, "edge_direction": edge_direction,"is_boundary": is_boundary_edge}

        num_nodes = n * m
        num_edges = (n-1)*m + n * (m-1)

        self._create_topology(num_nodes = num_nodes, num_edges = num_edges, edge_indices=edge_indices, node_attrs=node_attrs, edge_attrs=edge_attrs, default_node_attrs=default_node_attrs, default_edge_attrs=default_edge_attrs)


    def name_nodes(self, names: list, mask: torch.Tensor):

        if len(names) != mask.sum().item():

            raise ValueError(f"Length of names does not match the number of 'True' values in the mask.")

        encoded_names = torch.empty(0, dtype=torch.long)

        for i in range(len(names)):

            encoded_name = encode(names[i])
            encoded_names = torch.cat([encoded_names, torch.tensor([encoded_name], dtype=torch.long)])

        self.name[mask] = encoded_names


def resolve_attrs(f, default_attrs, custom_attrs):

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

    encoded = string.encode("utf-8")
    if len(encoded) > 8:
        raise ValueError(f"String '{string}' encodes to {len(encoded)} bytes, which exceeds the 8-byte limit.")

    return int.from_bytes(encoded, "big")

def decode(integer: int):

    return integer.to_bytes(8, "big").decode("utf-8").lstrip("\x00")

    