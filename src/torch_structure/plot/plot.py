import matplotlib.pyplot as plt
import numpy as np

from torch_structure.message_passing import ResidualForce

def plot_data(coords, edge_index, path=None, force=None, load=None, lw_scale=1.0, lw_constant=False,
              show_residual_forces=False, force_scale=1.0, show_load=False, show=False, show_axes=False, ax=None, title=None):
    """
    Creates and saves a plot of a structure stored in PyG (PyTorch Geometric) format.

    Args:
        coords (torch.Tensor): A tensor of shape (num_nodes, 3) containing the 3D coordinates of each node.
        edge_index (torch.Tensor): A tensor of shape (2, num_edges) containing the indices of the nodes that form each edge.
        path (str): The file path where the plot image will be saved.
        force (torch.Tensor, optional): A tensor of shape (num_edges, 1) containing the axial force for each edge. Default is None.
        load (torch.Tensor, optional): A tensor of shape (num_nodes, 3) containing the external load for each node. Required if `show_load` or
            `show_residual_forces` is True. Default is None.
        lw_scale (float, optional): A scaling factor for the edge line width. Default is 1.0.
        lw_constant (bool, optional): If True, all edges will have the same line width. If False, the line width will be proportional to the force
            magnitudes. Default is False.
        show_residual_forces (bool, optional): If True, the residual force for each node will be calculated and plotted. Requires `load` to be
            provided. Default is False.
        force_scale (float, optional): A scaling factor for the force vectors when plotting. Default is 1.0.
        show_load (bool, optional): If True, external load vectors will be plotted. Requires `load` to be provided. Default is False.
    """
    force = force.view(-1) if force is not None else force

    colors = {'red': '#E40714',
              'blue': '#0578BF',
              'green': '#007F00'}

    coords_np = coords.detach().cpu().numpy()
    x = coords_np[:, 0]
    y = coords_np[:, 1]
    z = coords_np[:, 2]
    num_nodes = coords.size(dim=0)
    num_edges = edge_index.size(dim=1)

    if force is not None:
        force_np = force.detach().cpu().numpy()
        edge_color = np.where(force_np > 0, colors['red'], colors['blue'])
        lw = np.abs(force_np) if not lw_constant else np.ones(num_edges) * lw_scale
    else:
        edge_color = ['gray'] * num_edges
        lw = np.ones(num_edges) * lw_scale

    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    for i, (src, dst) in enumerate(edge_index.t().cpu().numpy()):
        ax.plot([x[src], x[dst]], [y[src], y[dst]], [z[src], z[dst]], color=edge_color[i], lw=lw[i])
    
    if show_load:
        if load is None:
            raise ValueError('For showing residual forces load is a required input.')
        load_np = load.detach().cpu().numpy()
        for node in range(num_nodes):
            d_x = force_scale*load_np[node, 0]
            d_y = force_scale*load_np[node, 1]
            d_z = force_scale*load_np[node, 2]
            ax.plot([x[node], x[node] + d_x], [y[node], y[node] + d_y], [z[node], z[node] + d_z],
                    color=colors['green'], label='External Load')

    if show_residual_forces:
        if load is None:
            raise ValueError('For showing residual forces load is a required input.')
        calculate_residual_force = ResidualForce()
        residual_forces = calculate_residual_force(coords, force, edge_index, load).detach().cpu().numpy()
        for node in range(num_nodes):
            d_x = force_scale*residual_forces[node, 0]
            d_y = force_scale*residual_forces[node, 1]
            d_z = force_scale*residual_forces[node, 2]
            ax.plot([x[node], x[node] + d_x], [y[node], y[node] + d_y], [z[node], z[node] + d_z],
                    color='purple', label='Residual Force')
            
    if not show_axes:
        ax.set_axis_off()

    plt.axis('equal')
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys())
    if title is not None:
        ax.set_title(title)
    if path is not None:
        plt.savefig((f'{path}.png'), bbox_inches='tight')
    if show:
        plt.show()
