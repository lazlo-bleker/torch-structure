import torch_geometric as pyg
import torch
import warnings
import inspect
import copy
import json

from torch_structure.data.view import NodeView
from torch_structure.plot import plot_data
from torch_structure.formfinding import (
    laplacian_smoothing,
    tna,
    mpcem_algorithm,
    cem_algorithm,
    seq_cem_algorithm,
    fdm,
)
from torch_structure.loss import ResidualForceLoss


class Data:
    _internal_attrs = {
        "data",
        "default_attrs",
        "node_name_to_index",
        "edge_name_to_index",
        "node_attr_list",
        "edge_attr_list",
        "graph_attr_list",
        "_num_nodes",
    }

    def __init__(
        self,
        edge_index=torch.empty((2, 0), dtype=torch.long),
        node_attrs={},
        edge_attrs={},
        graph_attrs={},
        default_attrs={},
    ):
        directed_mask = torch.empty((0, 1), dtype=torch.bool)
        reciprocal_edge = torch.empty((0, 1), dtype=torch.long)
        self.data = pyg.data.Data(
            edge_index=edge_index,
            directed_mask=directed_mask,
            reciprocal_edge=reciprocal_edge,
            **node_attrs,
            **edge_attrs,
            **graph_attrs,
        )
        self.default_attrs = default_attrs
        self.node_name_to_index = {}
        self.edge_name_to_index = {}
        self.node_attr_list = [kwarg for kwarg in node_attrs.keys()]
        self.edge_attr_list = [kwarg for kwarg in edge_attrs.keys()]
        self.graph_attr_list = [kwarg for kwarg in graph_attrs.keys()]
        self._num_nodes = edge_index.max().item() + 1 if edge_index.numel() > 0 else 0

    @classmethod
    def from_pyg_data(cls, pyg_data):
        def reconstruct_default_attrs(saved):
            return {
                k: torch.tensor(v["value"], dtype=getattr(torch, v["dtype"]))
                for k, v in saved.items()
            }
        
        # Parse metadata from JSON string
        if not hasattr(pyg_data, "metadata"):
            raise ValueError("No metadata found in the provided PyG Data object.")
        
        metadata_str = pyg_data.metadata
        metadata = json.loads(metadata_str)
        metadata["default_attrs"] = reconstruct_default_attrs(metadata["default_attrs"])

        # Create wrapper instance
        obj = cls()
        obj.data = pyg_data
        del obj.data.metadata
        for key, value in metadata.items():
            setattr(obj, key, value)

        return obj

    def __getattr__(self, name):
        """Redirect attribute getter to `self.data`"""
        if hasattr(self.data, name):
            return getattr(self.data, name)

        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )
    
    @property
    def metadata(self):
        def serialize_tensor(t):
            return {
                "value": t.tolist(),
                "dtype": str(t.dtype).replace("torch.", "")
            }

        return {
            "default_attrs": {
                k: serialize_tensor(v) for k, v in self.default_attrs.items()
            },
            "node_name_to_index": self.node_name_to_index,
            "edge_name_to_index": self.edge_name_to_index,
            "node_attr_list": self.node_attr_list,
            "edge_attr_list": self.edge_attr_list,
            "graph_attr_list": self.graph_attr_list,
            "_num_nodes": self._num_nodes,
        }
    
    def export_pyg_data(self, include_metadata=True):
        data = self.data.clone()
        if include_metadata:
            data.metadata = json.dumps(self.metadata)
        return data

    @property
    def is_support(self):
        if "is_support" in self.node_attr_list:
            return self.data.is_support

        # If 'is_support' is missing, fall back to support_condition if available
        elif "support_condition" in self.node_attr_list:
            return torch.all(self.data.support_condition, dim=1, keepdim=True)

        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute 'is_support' or 'support_condition'"
            )

    @property
    def support(self):
        return self.is_support  ### REMOVE LATER TEMP

    @property
    def length_from_coords(self):
        # derive from 'coords' if available
        if "coords" in self.node_attr_list:
            src, dst = self.edge_index
            length = torch.norm(
                self.coords[src] - self.coords[dst], dim=1, keepdim=True
            )
            return length

        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute or 'coords'"
            )

    @property
    def bbox(self):
        if "coords" not in self.node_attr_list:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute 'coords' required for bounding box calculation."
            )
        return torch.stack(
            [self.coords.min(dim=0).values, self.coords.max(dim=0).values], dim=0
        )

    @property
    def directed_edge_index(self):
        return self.edge_index[:, self.directed_mask.view(-1)]

    def __setattr__(self, name, value, attr_type=None, track_history=False):
        """Redirect attribute setter to `self.data`"""
        # Only redirect non-internal attributes
        if name in self._internal_attrs:
            object.__setattr__(self, name, value)

        else:
            # Determine attribute type (node, edge, or graph)
            if attr_type is None:
                if value.shape[0] == self.num_nodes == self.num_edges:
                    warnings.warn(
                        f"""Cannot infer attribute type for attribute '{name}' (graph has equal number of nodes and edges).
                        Defaulting to node attribute.""",
                        UserWarning,
                    )
                if value.shape[0] == self.num_nodes:
                    attr_type = "node"
                elif value.shape[0] == self.num_edges:
                    attr_type = "edge"
                else:
                    attr_type = "graph"

            # Update attribute list
            if attr_type == "node":
                if name not in self.node_attr_list:
                    self.node_attr_list.append(name)
            elif attr_type == "edge":
                if name not in self.edge_attr_list:
                    self.edge_attr_list.append(name)
            elif attr_type == "graph":
                if name not in self.graph_attr_list:
                    self.graph_attr_list.append(name)

            # Set attribute value
            if track_history:
                self._track_history(name, value)
                setattr(self.data, name, value[-1])  # Set last value as current
            else:
                setattr(self.data, name, value)

    def __repr__(self):
        return self.data.__repr__()

    @property
    def nodes(self):
        """
        Provides networkx-style access to node attributes without caching.
        """
        return NodeView(self.data, self.node_name_to_index, self.node_attr_list)

    @property
    def num_nodes(self):
        return self._num_nodes

    def set_default_attributes(self, **kwargs):
        """
        Sets default edge/node attributes.
        """
        self.default_attrs = self.default_attrs | kwargs

    def merge_nodes(self, name: str, nodes: list[str]):
        """
        Merges multiple nodes into a single node. The node inherits the set of edges connected to the
        merged nodes. Edges between merged nodes are removed.
        """
        # Check if node exists
        if name not in self.node_name_to_index:
            raise ValueError(f"Node '{name}' does not exist!")

        main_node_index = self.node_name_to_index[name]

        for node in nodes:
            if node not in self.node_name_to_index:
                raise ValueError(f"Node '{node}' does not exist!")

            node_index = self.node_name_to_index[node]

            # replace all occurence of node_index with main_node_index
            self.data.edge_index = torch.where(
                self.data.edge_index == node_index,
                main_node_index,
                self.data.edge_index,
            )

        for node in nodes:
            del_node_index = self.node_name_to_index[node]

            # remove node from all node attributes
            for attr in self.node_attr_list:
                values = getattr(self.data, attr)
                values = torch.cat(
                    [values[:del_node_index], values[del_node_index + 1 :]], dim=0
                )
                setattr(self.data, attr, values)

            # remove node from node_name_to_index
            del self.node_name_to_index[node]

            # decrement number of nodes
            self._num_nodes -= 1

            # update node_name_to_index
            for node_name, node_index in self.node_name_to_index.items():
                if node_index > del_node_index:
                    self.node_name_to_index[node_name] = node_index - 1

            # update edge_index
            self.data.edge_index = torch.where(
                self.data.edge_index > del_node_index,
                self.data.edge_index - 1,
                self.data.edge_index,
            )

    def add_node(self, name: str, **kwargs):
        """
        Adds a new node.
        """
        # Check for unexpected attributes
        unexpected_attrs = set(kwargs.keys()) - set(self.node_attr_list)
        if unexpected_attrs:
            raise ValueError(
                f"Unexpected node attributes: {unexpected_attrs}. Expected: {list(self.node_attr_list)}"
            )

        # Check if node already exists
        if name in self.node_name_to_index:
            raise ValueError(f"Node '{name}' already exists!")

        self.node_name_to_index[name] = self.num_nodes  # Add node to node_name_to_index
        self._num_nodes += 1  # Increment number of nodes

        # Add node attributes
        for attr in self.node_attr_list:
            if attr in kwargs:
                value = kwargs[attr]

                # Cast non-tensor attributes to tensor
                if not isinstance(value, torch.Tensor):
                    value = torch.tensor(value, dtype=getattr(self.data, attr).dtype)
                    # warnings.warn(f"Node '{name}' attribute '{attr}' was automatically cast to a torch tensor.", UserWarning)

            # If attribute is not provided, set it to default value
            elif attr in self.default_attrs:
                value = self.default_attrs[attr]

            # If attribute has no default value, set it to zero
            else:
                attr_shape = getattr(self.data, attr).shape[1:]
                value = getattr(self.data, attr).new_zeros(attr_shape)
                # warnings.warn(f"Node '{name}' attribute '{attr}' initialized to zeros (no default value provided).", UserWarning)

            if value.dim() == 0:
                value = value.unsqueeze(0)
            setattr(
                self.data,
                attr,
                torch.cat([getattr(self.data, attr), value.unsqueeze(0)], dim=0),
            )

    # def remove_edge(self, node_1: str, node_2: str):
    #     """
    #     Removes the edge between `node_1` to `node_2`.
    #     """
    #     if node_1 not in self.node_name_to_index or node_2 not in self.node_name_to_index:
    #         raise ValueError("One or both nodes do not exist!")

    #     src_index, dst_index = self.node_name_to_index[node_1], self.node_name_to_index[node_2]

    #     # Find and remove edge
    #     src, dst = self.edge_index
    #     mask = ~((src == src_index) & (dst == dst_index))  # would be nice to replace this with a lookup

    #     self.data.edge_index = self.edge_index[:, mask]

    #     # Remove associated edge attributes
    #     for attr in self.edge_attr_list:
    #         values = getattr(self.data, attr)
    #         values = values[mask]
    #         setattr(self.data, attr, values)

    def add_edge(self, src: str, dst: str, name=None, **kwargs):
        """
        Adds a new edge from `src` to `dst`.
        """
        # Check for unexpected attributes
        unexpected_attrs = set(kwargs.keys()) - set(self.edge_attr_list)
        if unexpected_attrs:
            raise ValueError(
                f"Unexpected edge attributes: {unexpected_attrs}. Expected: {list(self.edge_attr_list)}"
            )

        # Add nodes if they do not exist
        if src not in self.node_name_to_index:
            self.add_node(src)
        if dst not in self.node_name_to_index:
            self.add_node(dst)

        # Add edge to edge_name_to_index
        main_name = f"{src}-{dst}" if name is None else name
        self.edge_name_to_index[main_name] = self.num_edges
        reciprocal_name = f"{dst}-{src}" if name is None else f"{name}_reciprocal"
        self.edge_name_to_index[reciprocal_name] = self.num_edges + 1

        # Update directed mask and reciprocal edge
        self.data.directed_mask = torch.cat(
            [self.directed_mask, torch.tensor([[True], [False]])], dim=0
        )
        self.data.reciprocal_edge = torch.cat(
            [
                self.reciprocal_edge,
                torch.tensor([[self.num_edges + 1], [self.num_edges]]),
            ],
            dim=0,
        )

        # Add edge to edge_index
        src_index, dst_index = (
            self.node_name_to_index[src],
            self.node_name_to_index[dst],
        )
        new_edge = torch.tensor([[src_index, dst_index], [dst_index, src_index]])
        self.data.edge_index = torch.cat([self.edge_index, new_edge], dim=1)

        # Add edge attributes
        for attr in self.edge_attr_list:
            if attr in kwargs:
                value = kwargs[attr]

                # Cast non-tensor attributes to tensor
                if not isinstance(value, torch.Tensor):
                    value = torch.tensor(value, dtype=getattr(self.data, attr).dtype)
                    # warnings.warn(f"Edge '({src}, {dst})' attribute '{attr}' was automatically cast to a tensor.", UserWarning)

            # If attribute is not provided, set it to default value
            elif attr in self.default_attrs:
                value = self.default_attrs[attr]

            # If attribute has no default value, set it to zero
            else:
                attr_shape = getattr(self.data, attr).shape[1:]
                value = getattr(self.data, attr).new_zeros(attr_shape)
                # warnings.warn(f"Edge '({src}, {dst})' attribute '{attr}' initialized to zeros (no default value provided).", UserWarning)

            if value.dim() == 0:
                value = value.unsqueeze(0)

            setattr(
                self.data,
                attr,
                torch.cat(
                    [getattr(self.data, attr), value.unsqueeze(0), value.unsqueeze(0)],
                    dim=0,
                ),
            )

    def succesors(self, node_name):
        # Todo: implement networkx-style successors
        raise NotImplementedError

    neighbors = succesors

    def predecessors(self, node_name):
        # Todo: implement networkx-style predecessors
        raise NotImplementedError

    def verify_equilibrium(self, tolerance=1e-7, verbose=False, **kwargs):
        if "coords" not in kwargs:
            kwargs["coords"] = self.coords
        if "load" not in kwargs:
            kwargs["load"] = self.load
        if "is_support" not in kwargs:
            if "is_support" in self.node_attr_list:
                kwargs["is_support"] = self.is_support
            elif "support_condition" in self.node_attr_list:
                kwargs["is_support"] = torch.all(
                    self.support_condition, dim=1, keepdim=True
                )
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

    def xy_laplacian_smoothing(self, verbose=False, **kwargs):
        if "coords" not in kwargs:
            kwargs["coords"] = self.coords[:, :2]
        if "is_fixed" not in kwargs:
            kwargs["is_fixed"] = self.is_fixed
        if "edge_index" not in kwargs:
            kwargs["edge_index"] = self.edge_index

        coords, _, _ = laplacian_smoothing(verbose=verbose, **kwargs)
        self.data.coords[:, :2] = coords

    def mpcem(self, inplace=False, **kwargs):
        # Create semi-directed graph
        edge_mask = ~(self.is_trail_edge.view(-1) & ~self.directed_mask.view(-1))

        kwargs = self._prepare_kwargs(mpcem_algorithm, edge_mask=edge_mask, **kwargs)

        coords, semi_directed_force, reaction_force = mpcem_algorithm(**kwargs)
        track_history = kwargs.get("track_history", False)
        force = self.edge_attr_to_undirected(semi_directed_force, edge_mask, batched=track_history)

        return_data = self if inplace else self.copy()

        return_data.__setattr__("coords", coords, attr_type="node", track_history=track_history)
        return_data.__setattr__("reaction_force", reaction_force, attr_type="node")
        return_data.__setattr__("force", force, attr_type="edge", track_history=track_history)

        if not inplace:
            return return_data

    def cem(self, inplace=False, **kwargs):
        # Create semi-directed graph
        edge_mask = ~(self.is_trail_edge.view(-1) & ~self.directed_mask.view(-1))

        kwargs = self._prepare_kwargs(cem_algorithm, edge_mask=edge_mask, **kwargs)

        coords, semi_directed_force, reaction_force = cem_algorithm(**kwargs)
        track_history = kwargs.get("track_history", False)
        force = self.edge_attr_to_undirected(semi_directed_force, edge_mask, batched=track_history)

        return_data = self if inplace else self.copy()

        return_data.__setattr__("coords", coords, attr_type="node", track_history=track_history)
        return_data.__setattr__("reaction_force", reaction_force, attr_type="node")
        return_data.__setattr__("force", force, attr_type="edge", track_history=track_history)

        if not inplace:
            return return_data
        
    def seqcem(self, inplace=False, **kwargs):
        # Create semi-directed graph
        edge_mask = ~(self.is_trail_edge.view(-1) & ~self.directed_mask.view(-1))

        kwargs = self._prepare_kwargs(seq_cem_algorithm, edge_mask=edge_mask, **kwargs)

        coords, semi_directed_force, reaction_force = seq_cem_algorithm(**kwargs)
        track_history = kwargs.get("track_history", False)
        force = self.edge_attr_to_undirected(semi_directed_force, edge_mask, batched=track_history)

        return_data = self if inplace else self.copy()

        return_data.__setattr__("coords", coords, attr_type="node", track_history=track_history)
        return_data.__setattr__("reaction_force", reaction_force, attr_type="node")
        return_data.__setattr__("force", force, attr_type="edge", track_history=track_history)

        if not inplace:
            return return_data

    def tna(self, inplace=False, **kwargs):
        edge_mask = self.directed_mask.view(-1)
        kwargs = self._prepare_kwargs(tna, edge_mask=edge_mask, **kwargs)

        coords, directed_force, directed_force_density = tna(**kwargs)
        force = self.edge_attr_to_undirected(directed_force, edge_mask)
        force_density = self.edge_attr_to_undirected(directed_force_density, edge_mask)

        return_data = self if inplace else self.copy()

        return_data.__setattr__("coords", coords, attr_type="node")
        return_data.__setattr__("force", force, attr_type="edge")
        return_data.__setattr__("force_density", force_density, attr_type="edge")

        if not inplace:
            return return_data

    def fdm(self, inplace=False, **kwargs):
        edge_mask = self.directed_mask.view(-1)
        kwargs = self._prepare_kwargs(fdm, edge_mask=edge_mask, **kwargs)

        coords, directed_force = fdm(**kwargs, directed=True)
        force = self.edge_attr_to_undirected(directed_force, edge_mask)

        return_data = self if inplace else self.copy()

        return_data.__setattr__("coords", coords, attr_type="node")
        return_data.__setattr__("force", force, attr_type="edge")

        if not inplace:
            return return_data

    def edge_attr_to_undirected(self, edge_attr, mask, batched=False):
        mask = mask.view(-1)

        if batched:
            value = torch.empty((edge_attr.shape[0], self.num_edges), dtype=edge_attr.dtype)

            # set defined values
            value[:, mask] = edge_attr

            # set reciprocal values
            value[:, ~mask] = value[:, self.reciprocal_edge[~mask].view(-1)]

        else:
            value = torch.empty((self.num_edges, 1), dtype=edge_attr.dtype)

            # set defined values
            value[mask] = edge_attr

            # set reciprocal values
            value[~mask] = value[self.reciprocal_edge[~mask].view(-1)]

        return value

    def _prepare_kwargs(self, func, edge_mask=None, **kwargs):
        sig = inspect.signature(func)
        pos_args = [
            param_name
            for param_name, param in sig.parameters.items()
            if param_name != "self" and param.default == inspect.Parameter.empty
        ]

        for arg in pos_args:
            if arg not in kwargs:
                value = getattr(self, arg)

                # Apply edge mask
                if edge_mask is not None:
                    if arg in self.edge_attr_list:
                        value = value[edge_mask]
                    elif arg == "edge_index":
                        value = value[:, edge_mask]

                kwargs[arg] = value

        return kwargs

    def to_networkx(self, **kwargs):
        self.data.num_nodes = self.num_nodes
        return pyg.utils.to_networkx(self.data, **kwargs)

    def copy(self):
        new_obj = type(self).__new__(type(self))
        for key, value in self.__dict__.items():
            if key == "data":
                object.__setattr__(new_obj, key, self.data.clone())
            else:
                object.__setattr__(new_obj, key, copy.deepcopy(value))
        return new_obj
    
    def _track_history(self, attr_name, value):  # Todo: requires attr exists in self.data
        history_attr_name = f"{attr_name}_history"
        current_attr = getattr(self.data, attr_name)

        # Set correct view for 1D tensors
        if current_attr.dim() == 2 and current_attr.shape[1] == 1:
            current_attr = current_attr.view(-1)

        # Add batch dimension if needed
        if value.dim() == current_attr.dim() - 1:
            value = value.unsqueeze(0)

        if hasattr(self.data, history_attr_name):
            history = getattr(self.data, history_attr_name)
            new_history = torch.cat(
                [history, value], dim=0
            )
        else:
            new_history = torch.cat([current_attr.unsqueeze(0), value], dim=0)

        setattr(self.data, history_attr_name, new_history)

    def plot(self, **kwargs):
        if "coords" not in kwargs:
            kwargs["coords"] = self.coords
        if "edge_index" not in kwargs:
            kwargs["edge_index"] = self.edge_index
        if "force" not in kwargs:
            if "force" in self.edge_attr_list:
                print("Using edge force attribute for plotting.")
                kwargs["force"] = self.force
        plot_data(**kwargs)
