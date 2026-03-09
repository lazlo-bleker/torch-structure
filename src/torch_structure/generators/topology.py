import torch

from torch_structure.data import StructData


def build_trail(
    data: StructData,
    name: str,
    n_nodes: int,
    origin_coords: torch.Tensor,
    load: torch.Tensor = None,
    force_sign: float = -1.0,
    length: float = None,
    origin_node_kwargs: dict = None,
    node_kwargs=None,
    edge_kwargs: dict = None,
    last_edge_kwargs: dict = None,
):
    """
    Adds a CEM trail to data: one origin node followed by n_nodes subsequent nodes,
    connected by trail edges. The last node automatically gets support_condition=[True,True,True]
    and load=zeros.

    """

    if load is None:
        load = torch.zeros(3, dtype=torch.float)
    if origin_node_kwargs is None:
        origin_node_kwargs = {}
    if node_kwargs is None:
        node_kwargs = {}
    if edge_kwargs is None:
        edge_kwargs = {}

    def resolve_node_kwargs(j):
        return node_kwargs(j) if callable(node_kwargs) else node_kwargs

    base_edge_kwargs = {}
    if length is not None:
        base_edge_kwargs["length"] = length
    base_edge_kwargs.update(edge_kwargs)

    # Origin node
    data.add_node(
        f"{name}_node_0",
        coords=origin_coords,
        is_origin_node=torch.tensor(True),
        sequence=torch.tensor(0, dtype=torch.long),
        load=load,
        **origin_node_kwargs,
    )

    # Subsequent nodes & edges
    for j in range(1, n_nodes + 1):
        is_last = j == n_nodes

        data.add_node(
            f"{name}_node_{j}",
            is_origin_node=torch.tensor(False),
            sequence=torch.tensor(j, dtype=torch.long),
            load=torch.zeros(3, dtype=torch.float) if is_last else load,
            support_condition=(
                torch.tensor([True, True, True])
                if is_last
                else torch.tensor([False, False, False])
            ),
            **resolve_node_kwargs(j),
        )

        this_edge_kwargs = base_edge_kwargs if not is_last or last_edge_kwargs is None \
            else {**base_edge_kwargs, **last_edge_kwargs}

        edge_call_kwargs = {"is_trail_edge": torch.tensor(True), **this_edge_kwargs}
        if force_sign is not None:
            edge_call_kwargs["force_sign"] = torch.tensor(force_sign)

        data.add_edge(
            f"{name}_node_{j - 1}",
            f"{name}_node_{j}",
            **edge_call_kwargs,
        )
