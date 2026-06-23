from torch_scatter import scatter_mean


def laplacian_smoothness(pos, edge_index, batch=None, num_nodes=None, mask=None):
    """Mean squared distance from each node to its neighbours' centroid.

    The uniform graph Laplacian smoothness energy: for each node, the squared
    distance between its position and the mean position of its connected
    neighbours.

    Args:
        pos:        [N, D] node coordinates
        edge_index: [2, E] adjacency (expects both directions present)
        batch:      [N] graph assignment for each node; if given, returns [B]
                    per-graph means instead of a single scalar
        num_nodes:  total number of nodes; inferred from pos if omitted
        mask:       [N] bool; if given, only these nodes contribute to the
                    energy. Fixed/anchor nodes still act as neighbours in the
                    centroids — they're only excluded from the reduction.
    """
    if num_nodes is None:
        num_nodes = pos.size(0)
    src, dst = edge_index[0], edge_index[1]

    centroid = scatter_mean(pos[src], dst, dim=0, dim_size=num_nodes)
    offset = ((pos - centroid) ** 2).sum(dim=-1)

    if batch is None:
        return offset[mask].mean() if mask is not None else offset.mean()

    num_graphs = int(batch.max()) + 1
    if mask is not None:
        offset, batch = offset[mask], batch[mask]
    return scatter_mean(offset, batch, dim=0, dim_size=num_graphs)