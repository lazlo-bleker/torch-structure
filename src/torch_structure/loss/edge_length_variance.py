from torch_scatter import scatter_mean


def edge_length_variance(edge_length, edge_index, batch=None, num_nodes=None, mask=None):
    """Mean per-node variance of incident edge lengths.

    Batch-safe: PyG offsets node indices when batching, so the per-node
    aggregation stays local to each graph automatically.

    Args:
        edge_length: [E] or [E, 1] pre-computed edge lengths
        edge_index:  [2, E] adjacency (expects both directions present)
        batch:       [N] graph assignment for each node; if given, returns [B]
                     per-graph means instead of a single scalar
        num_nodes:   total number of nodes; inferred from edge_index if omitted
        mask:        [N] bool; if given, only these nodes contribute to the
                     loss. Excluded nodes still act as neighbours in the
                     per-node means — they are only excluded from the reduction.
    """
    edge_length = edge_length.view(-1)
    idx = edge_index[1]

    mean = scatter_mean(edge_length, idx, dim=0, dim_size=num_nodes)
    dev2 = (edge_length - mean[idx]) ** 2
    var = scatter_mean(dev2, idx, dim=0, dim_size=num_nodes)

    if batch is None:
        return var[mask].mean() if mask is not None else var.mean()

    num_graphs = int(batch.max()) + 1
    if mask is not None:
        var, batch = var[mask], batch[mask]
    return scatter_mean(var, batch, dim=0, dim_size=num_graphs)
