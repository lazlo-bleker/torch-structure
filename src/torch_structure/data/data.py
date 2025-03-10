import torch_geometric as pyg
import torch
import warnings
import inspect

from torch_structure.data.view import NodeView
from torch_structure.plot import plot_data
from torch_structure.formfinding import laplacian_smoothing, tna, mpcem, cem
from torch_structure.loss import ResidualForceLoss


class Data:
    _internal_attrs = {
        "data",
        "default_attrs",
        "node_name_to_index",
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
        self.node_attr_list = [kwarg for kwarg in node_attrs.keys()]
        self.edge_attr_list = [kwarg for kwarg in edge_attrs.keys()]
        self.graph_attr_list = [kwarg for kwarg in graph_attrs.keys()]
        self._num_nodes = edge_index.max().item() + 1 if edge_index.numel() > 0 else 0

    def __getattr__(self, name):
        """Redirect attribute getter to `self.data`"""
        if hasattr(self.data, name):
            return getattr(self.data, name)

        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

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
    def directed_edge_index(self):
        return self.edge_index[:, self.directed_mask.view(-1)]

    def __setattr__(self, name, value, attr_type=None):
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

    def add_edge(self, src: str, dst: str, **kwargs):
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
        raise NotImplementedError

    neighbors = succesors

    def predecessors(self, node_name):
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

    def mpcem(self, **kwargs):
        # Create semi-directed graph
        edge_mask = ~(self.is_trail_edge.view(-1) & ~self.directed_mask.view(-1))

        kwargs = self._prepare_kwargs(mpcem, edge_mask=edge_mask, **kwargs)

        coords, semi_directed_force, reaction_force = mpcem(**kwargs)
        force = self.edge_attr_to_undirected(semi_directed_force, edge_mask)

        self.__setattr__("coords", coords, attr_type="node")
        self.__setattr__("reaction_force", reaction_force, attr_type="node")
        self.__setattr__("force", force, attr_type="edge")

    def cem(self, **kwargs):
        # Create semi-directed graph
        edge_mask = ~(self.is_trail_edge.view(-1) & ~self.directed_mask.view(-1))

        kwargs = self._prepare_kwargs(cem, edge_mask=edge_mask, **kwargs)

        coords, semi_directed_force, reaction_force = cem(**kwargs)
        force = self.edge_attr_to_undirected(semi_directed_force, edge_mask)

        self.__setattr__("coords", coords, attr_type="node")
        self.__setattr__("reaction_force", reaction_force, attr_type="node")
        self.__setattr__("force", force, attr_type="edge")

    def tna(self, **kwargs):
        edge_mask = self.directed_mask.view(-1)
        kwargs = self._prepare_kwargs(tna, edge_mask=edge_mask, **kwargs)

        coords, directed_force, directed_force_density = tna(**kwargs)
        force = self.edge_attr_to_undirected(directed_force, edge_mask)
        force_density = self.edge_attr_to_undirected(directed_force_density, edge_mask)

        self.__setattr__("coords", coords, attr_type="node")
        self.__setattr__("force", force, attr_type="edge")
        self.__setattr__("force_density", force_density, attr_type="edge")

    def edge_attr_to_undirected(self, edge_attr, mask):
        mask = mask.view(-1)
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
