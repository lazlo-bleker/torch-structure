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
    

    #???add empty nodes as extra function?
    @requires_metadata
    def add_nodes(self, **kwargs):
        """
        Adds new nodes
        """

        if not kwargs:
            raise ValueError("No node attributes provided for new nodes. At least one attribute must be provided.")
        


        #also check if all kwargs have same length?

        num_new_nodes = len(next(iter(kwargs.values())))

        if num_new_nodes == 0:
            raise ValueError("No node attribute values provided. There must be at least one attribute value")



        # Check for unexpected attributes
        unexpected_attrs = set(kwargs.keys()) - set(self.metadata["node_attr_list"])
        if unexpected_attrs:
            raise ValueError(
                f"Unexpected node attributes: {unexpected_attrs}. Expected: {list(self.metadata['node_attr_list'])}"
            )

        self.num_nodes += num_new_nodes  # Increment number of nodes
        
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
                value = self.metadata["default_attrs"][attr].repeat(num_new_nodes, 1)

            
            # If attribute has no default value, set it to zero
            else:
                attr_shape = getattr(self, attr).shape[1:]
                value = getattr(self, attr).new_zeros(attr_shape).repeat(num_new_nodes, 1)
                # warnings.warn(f"Node '{name}' attribute '{attr}' initialized to zeros (no default value provided).", UserWarning)

            if value.dim() == 0:
                value = value.unsqueeze(0)

            setattr(
                self,
                attr,
                torch.cat([getattr(self, attr), value], dim=0),
            )


    @requires_metadata
    def add_edges(self, edge_indices, **kwargs):

        """
        Adds new edges
        """
        
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
            if attr in kwargs:
                value = kwargs[attr]

                # Cast non-tensor attributes to tensor
                if not isinstance(value, torch.Tensor):
                    value = torch.tensor(value, dtype=getattr(self, attr).dtype)
                    # warnings.warn(f"Edge '({src}, {dst})' attribute '{attr}' was automatically cast to a tensor.", UserWarning)

            # If attribute is not provided, set it to default value
            elif attr in self.metadata["default_attrs"]:
                value = self.metadata["default_attrs"][attr].repeat(num_new_edges, 1)

            # If attribute has no default value, set it to zero
            else:
                attr_shape = getattr(self, attr).shape[1:]
                value = getattr(self, attr).new_zeros(attr_shape).repeat(num_new_edges, 1)
                # warnings.warn(f"Node '{name}' attribute '{attr}' initialized to zeros (no default value provided).", UserWarning)

            if value.dim() == 0:
                value = value.unsqueeze(0)

            setattr(
                self,
                attr,
                torch.cat(
                    [getattr(self, attr), value, value],
                    dim=0,
                ),
            )
       

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

    
    #???should warning be given always when nodes withh different attrs are merged? does that work with vectorized logic?
    #???should we additionally give functions that can be thrown on node_priority_attribute, edge_priority_attribute or is interface to convoluted?
    def merge(self, merge_group_id = None, node_priority_attribute: str = None, edge_priority_attribute: str = None):
        
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
                    raise ValueError(f"node_priority_attribute '{node_priority_attribute}' must be scalar (shape (n,) or (n,1)), got {tuple(priority_attribute.shape)}.")
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
                raise ValueError(f"edge_priority_attribute '{edge_priority_attribute}' must be scalar (shape (n,) or (n,1)), got {tuple(priority_attribute.shape)}.")
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


    #if attr is boolean, only true is merged
    def merge_based_on_attributes(self, attr: str, eps = 10e-4, f = None):

        if attr not in self.metadata["node_attr_list"]:
            raise ValueError(f"Unexpected node attribute '{attr}' for merging. Expected an attribute in: {list(self.metadata['node_attr_list'])}")

        value = getattr(self, attr)

        if f is not None:
            value = f(value)

        if value.dtype in (torch.float32, torch.float64):            
            value = torch.round(value / eps)

        elif value.dtype == torch.bool:
            group_id = torch.zeros_like(value)
            group_id[~value] *= -1

        elif value.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError(f"Unexpected data-type. Merge based on attributes requires a torch boolean, float or int.")


        _, group_id = torch.unique(value, dim = 0, return_inverse=True)

        self.merge(group_id)

        return

   
    #??? unfinished
    def add_radial_grid(self, res_normal: int, res_radial: int, node_attrs: dict = {}, edge_attrs: dict = {}):
    
        #save old_num_nodes to later shift edge indices 
        old_num_nodes = self.num_nodes

        u_nodes = torch.arange(res_normal).repeat_interleave(res_radial).unsqueeze(1)
        v_nodes = torch.arange(res_radial).repeat(res_normal).unsqueeze(1)

        return

    #option for diagonal edges?  
    def add_grid(self, n: int, m: int, node_attrs: dict = {}, edge_attrs: dict = {}):

        #save old_num_nodes to later shift edge indices 
        old_num_nodes = self.num_nodes

        #create default node attributes
        u_nodes = torch.arange(n).repeat_interleave(m).unsqueeze(1)
        v_nodes = torch.arange(m).repeat(n).unsqueeze(1)
        is_boundary_node = (u_nodes == 0) | (u_nodes == n-1) | (v_nodes == 0) | (v_nodes == m-1)
        is_corner_node = ((u_nodes == 0) | (u_nodes == n - 1)) & ((v_nodes == 0) | (v_nodes == m - 1))
        default_node_attrs = {"u": u_nodes, "v": v_nodes, "is_boundary": is_boundary_node, "is_corner": is_corner_node}


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
        u_src = u_nodes[edge_indices[0]]
        v_src = v_nodes[edge_indices[0]]
        u_dest = u_nodes[edge_indices[1]]
        v_dest = v_nodes[edge_indices[1]]
        u_coord = (u_src + u_dest) / 2
        v_coord = (v_src + v_dest) / 2

        u_ind = u_nodes[edge_indices[0]]
        v_ind = v_nodes[edge_indices[0]]
        is_boundary_edge = is_boundary_node[edge_indices[0]] & is_boundary_node[edge_indices[1]]
        edge_direction = torch.cat([torch.zeros(row_h.shape[0]), torch.ones(row_v.shape[0])], dim=0)

        default_edge_attrs = {"u_ind": u_ind, "v_ind": v_ind, "u_coord": u_coord, "v_coord": v_coord, "edge_direction": edge_direction,"is_boundary": is_boundary_edge}

        #resolve node attributes and add to graph
        kwargs_nodes = {}
        for attr in node_attrs:
                
            f = node_attrs[attr]

            if isinstance(f, torch.Tensor):
                
                kwargs_nodes[attr] = f

                continue

            if not callable(f):
                raise ValueError(f"Node attribute '{attr}' must be a tensor or a callable")

            f_kwargs = resolve_attrs(f, default_node_attrs, node_attrs)

            kwargs_nodes[attr] = f(**f_kwargs)

        self.add_nodes(**kwargs_nodes)

        #resolve edge attributes and add to graph
        kwargs_edges = {}
        for attr in edge_attrs:
                
            f = edge_attrs[attr]

            if isinstance(f, torch.Tensor):

                kwargs_edges[attr] = f
                continue

            if not callable(f):
                raise ValueError(f"Edge attribute '{attr}' must be a tensor or a callable")

            f_kwargs = resolve_attrs(f, default_edge_attrs, edge_attrs)

            kwargs_edges[attr] = f(**f_kwargs)
        
        self.add_edges(edge_indices + old_num_nodes, **kwargs_edges) 



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

    