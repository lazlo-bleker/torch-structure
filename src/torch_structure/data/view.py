class NodeView:
    """
    Dynamic view for accessing nodes with NetworkX-like syntax.
    """

    def __init__(self, data, node_attr_list):
        self.data = data
        self.node_attr_list = node_attr_list

    def __getitem__(self, node_name):
        return NodeAttributeView(self.data, self.index(node_name), self.node_attr_list)

    def __iter__(self):
        return iter(self.keys())

    def keys(self):
        """Return the distinct names of all named nodes, in node order."""
        from torch_structure.data.data import decode

        return list(dict.fromkeys(decode(name) for name in self.data.name.view(-1).tolist() if name != 0))

    def index(self, node_name):
        """Return the index of the node named ``node_name``."""
        try:
            return self.data._single_node_index(node_name)
        except ValueError as e:
            raise KeyError(str(e)) from None

    def items(self):
        """Return an iterator of ``(node_name, NodeAttributeView)`` pairs."""
        return ((name, self[name]) for name in self.keys())

    def __len__(self):
        return len(self.keys())

    def __repr__(self):
        return f"{dict(self.items())}"


class NodeAttributeView:
    """
    Dynamic view for accessing node attributes with NetworkX-like syntax.
    """

    def __init__(self, data, node_index, node_attr_list):
        self.data = data
        self.node_index = node_index
        self.node_attr_list = node_attr_list

    def __getitem__(self, attr):
        """Retrieves the attribute value for the node."""
        if attr not in self.node_attr_list:
            raise KeyError(f"Attribute '{attr}' does not exist.")
        return getattr(self.data, attr)[self.node_index]

    def __setitem__(self, attr, value):
        """Sets the attribute value for the node if the attribute exists."""
        if attr not in self.node_attr_list:
            raise KeyError(f"Attribute '{attr}' does not exist.")

        existing_attr = getattr(self.data, attr)
        existing_attr[self.node_index] = value
