import torch
from torch_structure.message_passing import ResidualForce


def mpcem(
    coords,
    load,
    is_support,
    is_origin_node,
    edge_index,
    is_trail_edge,
    length,
    force_sign,
    force,
    max_iter=100,
    tolerance=1e-8,
    verbose=False,
):
    """Message Passing-based Combinatorial Equilibrium Modelling"""
    is_support = is_support.view(-1)
    is_origin_node = is_origin_node.view(-1)
    is_trail_edge = is_trail_edge.view(-1)
    length = length.view(-1)
    force_sign = force_sign.view(-1)
    force = force.view(-1)

    residual_force_update = ResidualForce()
    trail_src, trail_dst = edge_index[:, is_trail_edge]

    converged = False
    for i in range(max_iter):
        prev_coords = coords.clone()

        # only consider edges with a coordinate (estimate) for both nodes
        valid_nodes = ~torch.isnan(coords).any(dim=1)
        valid_edges = valid_nodes[edge_index[0]] & valid_nodes[edge_index[1]]

        # Calculate outgoing trail force
        residual_force = residual_force_update(
            coords, force[valid_edges], edge_index[:, valid_edges], load
        )
        # if i > 0 and torch.norm(residual_force[trail_src] - trail_force, dim=1).max() < tolerance:
        #     converged = True
        #     break
        trail_force = residual_force[trail_src]

        # Update force of trail edges
        trail_force_mag = torch.norm(trail_force, dim=1)
        force[is_trail_edge] = force_sign[is_trail_edge] * trail_force_mag

        # Update coordinates
        clamped_trail_force_mag = trail_force_mag.clamp(min=1e-8).unsqueeze(1)
        unit_trail_force = trail_force / clamped_trail_force_mag
        coords_update = (
            unit_trail_force
            * length[is_trail_edge].unsqueeze(1)
            * -force_sign[is_trail_edge].unsqueeze(1)
        )
        coords[trail_dst] = coords[trail_src] + coords_update

        if torch.norm(coords - prev_coords) < tolerance:
            converged = True
            break

    # Calculate reaction force
    reaction_force = torch.full((coords.shape[0], 3), float("nan"))
    reaction_force[is_support] = -residual_force[is_support]

    if verbose:
        print(f"MP-CEM finished in {i} iterations. Converged: {converged}.")

    return coords, force.unsqueeze(1), reaction_force


def cem(
    coords,
    load,
    is_support,
    is_origin_node,
    edge_index,
    is_trail_edge,
    length,
    force_sign,
    force,
    sequence,
    max_iter=100,
    tolerance=1e-8,
    verbose=False,
):
    """Combinatorial Equilibrium Modelling"""
    is_support = is_support.view(-1)
    is_origin_node = is_origin_node.view(-1)
    is_trail_edge = is_trail_edge.view(-1)
    length = length.view(-1)
    force_sign = force_sign.view(-1)
    force = force.view(-1)
    sequence = sequence.view(-1)

    residual_force_update = ResidualForce()
    trail_src, trail_dst = edge_index[:, is_trail_edge]

    converged = False
    max_k = sequence.max().item()
    for i in range(max_iter):
        prev_coords = coords.clone()
        for k in range(max_k):
            # only consider edges with a coordinate (estimate) for both nodes and pointing to a node in the current sequence
            valid_nodes = ~torch.isnan(coords).any(dim=1)
            k_mask = sequence == k
            valid_edges = valid_nodes[edge_index[0]] & k_mask[edge_index[1]]

            # Calculate outgoing trail force
            residual_force = residual_force_update(
                coords, force[valid_edges], edge_index[:, valid_edges], load
            )
            # if i > 0 and torch.norm(residual_force[trail_src] - trail_force, dim=1).max() < tolerance:
            #     converged = True
            #     break
            trail_force = residual_force[k_mask]

            # Update force of trail edges
            current_trail_edge = k_mask[edge_index[0]] & is_trail_edge
            trail_force_mag = torch.norm(trail_force, dim=1)
            force[current_trail_edge] = force_sign[current_trail_edge] * trail_force_mag

            # Update coordinates
            current_trail_src, current_trail_dst = edge_index[:, current_trail_edge]
            clamped_trail_force_mag = trail_force_mag.clamp(min=1e-8).unsqueeze(1)
            unit_trail_force = trail_force / clamped_trail_force_mag
            coords_update = (
                unit_trail_force
                * length[current_trail_edge].unsqueeze(1)
                * -force_sign[current_trail_edge].unsqueeze(1)
            )
            coords[current_trail_dst] = coords[current_trail_src] + coords_update

        if torch.norm(coords - prev_coords) < tolerance:
            converged = True
            break

    # Calculate reaction force
    reaction_force = torch.full((coords.shape[0], 3), float("nan"))
    reaction_force[is_support] = -trail_force

    if verbose:
        print(f"CEM finished in {i * max_k} iterations. Converged: {converged}.")

    return coords, force.unsqueeze(1), reaction_force
