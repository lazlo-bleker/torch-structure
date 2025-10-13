import torch
from torch_structure.message_passing import ResidualForce


def mpcem_algorithm(
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
    tolerance=1e-5,
    damping_factor=0.5,
    verbose=False,
    track_history=False,
):
    """Message Passing-based Combinatorial Equilibrium Modelling"""

    coords = torch.clone(coords)
    is_support = is_support.view(-1)
    is_origin_node = is_origin_node.view(-1)
    is_trail_edge = is_trail_edge.view(-1)
    length = length.view(-1)
    force_sign = force_sign.view(-1)
    force = force.view(-1)

    residual_force_update = ResidualForce()
    trail_src, trail_dst = edge_index[:, is_trail_edge]

    converged = False
    n_steps = 0
    for i in range(max_iter):
        prev_coords = coords.clone()

        # only consider edges with a coordinate (estimate) for both nodes
        valid_nodes = ~torch.isnan(coords).any(dim=1)
        valid_edges = (
            valid_nodes[edge_index[0]]
            & valid_nodes[edge_index[1]]
            & ~is_support[edge_index[1]]
        )

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
        new_coords = coords[trail_src] + coords_update
        if coords[trail_dst].isnan().any():
            coords[trail_dst] = new_coords
        else:
            coords[trail_dst] = coords[trail_dst] + (1 - damping_factor) * (
                new_coords - coords[trail_dst]
            )

        if track_history:
            if i == 0:
                coords_history = coords.unsqueeze(0).clone()
                force_history = force.unsqueeze(0).clone()
            else:
                coords_history = torch.cat((coords_history, coords.unsqueeze(0)), dim=0)
                force_history = torch.cat((force_history, force.unsqueeze(0)), dim=0)

        n_steps += 1

        # print(f"MPCEM Iteration {i}, delta coords = {torch.norm(coords - prev_coords)}")
        if torch.norm(coords - prev_coords) < tolerance:
            converged = True
            break

    # Calculate reaction force
    reaction_force = torch.full((coords.shape[0], 3), float("nan"), dtype=torch.float64).to(coords.device)
    reaction_force[is_support] = -residual_force[is_support]

    if verbose:
        print(f"MP-CEM finished in {i + 1}, {n_steps} steps. Converged: {converged}.")

    if track_history:
        return coords_history, force_history, reaction_force
    else:
        return coords, force.unsqueeze(1), reaction_force


def cem_algorithm(
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
    tolerance=1e-5,
    damping_factor=0.0,
    enhanced_first_iteration=False,
    verbose=False,
    track_history=False,
):
    """Combinatorial Equilibrium Modelling"""

    coords = torch.clone(coords)
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
    n_steps = 0
    for i in range(max_iter):
        prev_coords = coords.clone()
        for k in range(max_k):
            if n_steps >= max_iter:
                print(f"Maximum iterations {max_iter} reached without convergence.")
                break
            # only consider edges with a coordinate (estimate) for both nodes and pointing to a node in the current sequence
            valid_nodes = ~torch.isnan(coords).any(dim=1)
            k_mask = sequence == k
            if not enhanced_first_iteration and i == 0:
                indirect_edges = ~is_trail_edge & ~k_mask[edge_index[0]]
                valid_edges = (
                    valid_nodes[edge_index[0]]
                    & k_mask[edge_index[1]]
                    & ~indirect_edges
                    & ~is_support[edge_index[1]]
                )
            else:
                valid_edges = (
                    valid_nodes[edge_index[0]]
                    & k_mask[edge_index[1]]
                    & ~is_support[edge_index[1]]
                )

            # Calculate outgoing trail force
            # print("CEM valid edges: ", torch.where(valid_edges)[0])
            residual_force = residual_force_update(
                coords, force[valid_edges], edge_index[:, valid_edges], load
            )
            # if i > 0 and torch.norm(residual_force[trail_src] - trail_force, dim=1).max() < tolerance:
            #     converged = True
            #     break
            trail_force = residual_force[k_mask & ~is_support]

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

            new_coords = coords[current_trail_src] + coords_update
            if coords[current_trail_dst].isnan().any():
                coords[current_trail_dst] = new_coords
            else:
                coords[current_trail_dst] = coords[current_trail_dst] + (
                    1 - damping_factor
                ) * (new_coords - coords[current_trail_dst])

            if track_history:
                if n_steps == 0:
                    coords_history = coords.unsqueeze(0).clone()
                    force_history = force.unsqueeze(0).clone()
                else:
                    coords_history = torch.cat(
                        (coords_history, coords.unsqueeze(0)), dim=0
                    )
                    force_history = torch.cat(
                        (force_history, force.unsqueeze(0)), dim=0
                    )

            n_steps += 1

        # print(f'CEM iteration {i}, delta coords: {torch.norm(coords - prev_coords)}')
        if torch.norm(coords - prev_coords) < tolerance:
            converged = True
            break

    # Calculate reaction force
    k_mask = sequence == max_k
    valid_edges = (
        valid_nodes[edge_index[0]]
        & k_mask[edge_index[1]]
    )

    # Calculate outgoing trail force
    # print("CEM valid edges: ", torch.where(valid_edges)[0])
    residual_force = residual_force_update(
            coords, force[valid_edges], edge_index[:, valid_edges], load
        )
    dtype = residual_force.dtype
    reaction_force = torch.full((coords.shape[0], 3), float("nan"), dtype=dtype).to(coords.device)
    reaction_force[is_support] = -residual_force[is_support]

    if verbose:
        print(
            f"CEM finished in {(i + 1) * max_k}, {n_steps} iterations. Converged: {converged}."
        )

    if track_history:
        return coords_history, force_history, residual_force
    else:
        return coords, force.unsqueeze(1), residual_force


def seq_cem_algorithm(
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
    tolerance=1e-5,
    damping_factor=0.0,
    enhanced_first_iteration=False,
    verbose=False,
    plot=False,
    ax=None,
    ax_list=None,
    first_frame_id=None,
    last_frame_id=None,
    track_history=False,
):
    """Combinatorial Equilibrium Modelling"""

    coords = torch.clone(coords)
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
    n_steps = 0
    for i in range(max_iter):
        prev_coords = coords.clone()
        for k in range(max_k):
            # only consider edges with a coordinate (estimate) for both nodes and pointing to a node in the current sequence
            valid_nodes = ~torch.isnan(coords).any(dim=1)
            k_mask = sequence == k
            coords_cache = coords.clone()

            ## True indices of k_mask
            k_node_indices = torch.where(k_mask & ~is_support)[0]
            for j, node_idx in enumerate(k_node_indices):
                if n_steps >= max_iter:
                    print(f"Maximum iterations {max_iter} reached without convergence.")
                    break
                seq_mask = torch.zeros_like(k_mask, dtype=torch.bool)
                seq_mask[node_idx] = True

                if not enhanced_first_iteration and i == 0:
                    indirect_edges = ~is_trail_edge & ~k_mask[edge_index[0]]
                    valid_edges = (
                        valid_nodes[edge_index[0]]
                        & seq_mask[edge_index[1]]
                        & ~indirect_edges
                        & ~is_support[edge_index[1]]
                    )
                else:
                    valid_edges = (
                        valid_nodes[edge_index[0]]
                        & seq_mask[edge_index[1]]
                        & ~is_support[edge_index[1]]
                    )

                # Calculate outgoing trail force
                # print("CEM (sequential) valid edges: ", torch.where(valid_edges)[0])
                residual_force = residual_force_update(
                    coords_cache, force[valid_edges], edge_index[:, valid_edges], load
                )
                # if i > 0 and torch.norm(residual_force[trail_src] - trail_force, dim=1).max() < tolerance:
                #     converged = True
                #     break
                trail_force = residual_force[seq_mask & ~is_support]

                # Update force of trail edges
                current_trail_edge = seq_mask[edge_index[0]] & is_trail_edge
                trail_force_mag = torch.norm(trail_force, dim=1)
                force[current_trail_edge] = (
                    force_sign[current_trail_edge] * trail_force_mag
                )

                # Update coordinates
                current_trail_src, current_trail_dst = edge_index[:, current_trail_edge]
                clamped_trail_force_mag = trail_force_mag.clamp(min=1e-8).unsqueeze(1)
                unit_trail_force = trail_force / clamped_trail_force_mag
                coords_update = (
                    unit_trail_force
                    * length[current_trail_edge].unsqueeze(1)
                    * -force_sign[current_trail_edge].unsqueeze(1)
                )

                new_coords = coords[current_trail_src] + coords_update
                if coords[current_trail_dst].isnan().any():
                    coords[current_trail_dst] = new_coords
                else:
                    coords[current_trail_dst] = coords[current_trail_dst] + (
                        1 - damping_factor
                    ) * (new_coords - coords[current_trail_dst])

                if track_history:
                    if n_steps == 0:
                        coords_history = coords.unsqueeze(0).clone()
                        force_history = force.unsqueeze(0).clone()
                    else:
                        coords_history = torch.cat(
                            (coords_history, coords.unsqueeze(0)), dim=0
                        )
                        force_history = torch.cat(
                            (force_history, force.unsqueeze(0)), dim=0
                        )

                n_steps += 1

        # print(f'CEM (sequential) iteration {i}, delta coords: {torch.norm(coords - prev_coords)}')
        if torch.norm(coords - prev_coords) < tolerance:
            converged = True
            break

    # Calculate reaction force
    reaction_force = torch.full((coords.shape[0], 3), float("nan")).to(coords.device)
    reaction_force[is_support] = -residual_force[is_support]

    if verbose:
        print(
            f"CEM (sequential) finished in {(i + 1) * max_k * len(k_node_indices)}, {n_steps} iterations. Converged: {converged}."
        )

    if track_history:
        return coords_history, force_history, reaction_force
    else:
        return coords, force.unsqueeze(1), reaction_force
