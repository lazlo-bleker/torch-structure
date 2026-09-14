class NodeView:
    """
    Dynamic view for accessing nodes with NetworkX-like syntax.
    """

    def __init__(self, data, node_name_to_index, node_attr_list):
        self.data = data
        self.node_name_to_index = node_name_to_index
        self.node_attr_list = node_attr_list

    def __getitem__(self, node_name):
        if node_name not in self.node_name_to_index:
            raise KeyError(f"Node '{node_name}' does not exist.")

        node_index = self.node_name_to_index[node_name]
        return NodeAttributeView(self.data, node_index, self.node_attr_list)

    def __iter__(self):
        return iter(self.node_name_to_index)

    def keys(self):
        """Return the node names."""
        return self.node_name_to_index.keys()

    def items(self):
        """Return an iterator of ``(node_name, NodeAttributeView)`` pairs."""
        return ((name, self[name]) for name in self.node_name_to_index)

    def __len__(self):
        return len(self.node_name_to_index)

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
