import torch
from torch_scatter import scatter
from torch_structure.message_passing import ResidualForce
from torch_structure.geometry import line_plane_intersect, point_normal_to_plane, graph_edge_lengths, line_direction

ALLOWED_UPDATE_KEYS = {
    "coords", 
    "load", 
    "length", 
    "force_sign", 
    "force", 
    "constraint_plane",
}

def mpcem_algorithm(
    coords,
    load,
    is_support,
    is_origin_node,
    cem_edge_index,
    is_trail_edge,
    length,
    force_sign,
    force,
    constraint_plane=None,
    max_iter=100,
    tolerance=1e-5,
    damping_factor=0.5,
    verbose=False,
    track_history=False,
    callback=None,
):
    """Message Passing-based Combinatorial Equilibrium Modelling"""

    # Enforce expected input shapes
    is_support = is_support.view(-1)
    is_origin_node = is_origin_node.view(-1)
    is_trail_edge = is_trail_edge.view(-1)
    length = length.view(-1)
    force_sign = force_sign.view(-1)
    force = force.view(-1)

    # Assemble state and clone inputs that might be modified
    state = {
        "coords": coords.clone(),
        "load": load.clone(),
        "length": length.clone(),
        "force_sign": force_sign.clone(),
        "force": force.clone(),
        "constraint_plane": constraint_plane.clone() if constraint_plane is not None else None,
    }

    residual_force_update = ResidualForce()
    trail_src, trail_dst = cem_edge_index[:, is_trail_edge]

    if state["constraint_plane"] is not None:
        if state["constraint_plane"].size(-1) == 6:
            state["constraint_plane"] = point_normal_to_plane(state["constraint_plane"])
        constraint_plane_mask = ~torch.isnan(state["constraint_plane"]).any(dim=1)
        trail_edge_constraint_plane_mask = constraint_plane_mask[trail_dst]
    else:
        trail_edge_constraint_plane_mask = torch.zeros_like(
            state["length"][is_trail_edge], dtype=torch.bool
        )
    edge_constraint_plane_mask = torch.zeros_like(state["length"], dtype=torch.bool)
    edge_constraint_plane_mask[is_trail_edge] = trail_edge_constraint_plane_mask

    if not torch.all(
        trail_edge_constraint_plane_mask | ~torch.isnan(state["length"][is_trail_edge])
    ):
        raise ValueError(
            "All trail edges must have a specified length or associated constraint plane."
        )

    converged = False
    n_steps = 0
    for i in range(max_iter):
        # Allow user to modify state via callback
        if callback is not None:
            updates = callback(dict(state))
            if updates:
                illegal = set(updates) - ALLOWED_UPDATE_KEYS
                if illegal:
                    raise KeyError(
                        f"Callback returned updates to non-whitelisted keys: {illegal}. "
                        f"Allowed keys are {sorted(ALLOWED_UPDATE_KEYS)}"
                    )
                for key in updates:
                    state[key] = updates[key]

        prev_coords = state["coords"].clone()

        # Only consider edges with a coordinate (estimate) for both nodes
        valid_nodes = ~torch.isnan(state["coords"]).any(dim=1)
        valid_edges = (
            valid_nodes[cem_edge_index[0]]
            & valid_nodes[cem_edge_index[1]]
            & ~is_support[cem_edge_index[1]]
        )

        # Calculate outgoing trail force
        residual_force = residual_force_update(
            state["coords"], state["force"][valid_edges], cem_edge_index[:, valid_edges], state["load"]
        )
        # if i > 0 and torch.norm(residual_force[trail_src] - trail_force, dim=1).max() < tolerance:
        #     converged = True
        #     break
        trail_force = residual_force[trail_src]
        trail_force_mag = torch.norm(trail_force, dim=1)

        # Update coordinates
        clamped_trail_force_mag = trail_force_mag.clamp(min=1e-8).unsqueeze(1)
        unit_trail_force = trail_force / clamped_trail_force_mag

        # Coordinates defined by trail lengths
        length_coords_update = (
            unit_trail_force[~trail_edge_constraint_plane_mask]
            * state["length"][is_trail_edge][~trail_edge_constraint_plane_mask].unsqueeze(1)
            * -state["force_sign"][is_trail_edge][~trail_edge_constraint_plane_mask].unsqueeze(1)
        )
        new_coords_length = (
            state["coords"][trail_src][~trail_edge_constraint_plane_mask] + length_coords_update
        )

        # Coordinates defined by constraint planes
        if state["constraint_plane"] is not None:
            new_coords_plane, t = line_plane_intersect(
                plane=state["constraint_plane"][constraint_plane_mask],
                point=state["coords"][trail_src][trail_edge_constraint_plane_mask],
                vector=unit_trail_force[trail_edge_constraint_plane_mask],
                return_t=True,
            )
            state["force_sign"][edge_constraint_plane_mask] = -torch.sign(t)
        else:
            new_coords_plane = state["coords"][trail_dst][trail_edge_constraint_plane_mask]

        new_coords = torch.empty_like(state["coords"][trail_dst])
        new_coords[~trail_edge_constraint_plane_mask] = new_coords_length
        new_coords[trail_edge_constraint_plane_mask] = new_coords_plane

        # Update force of trail edges
        state["force"][is_trail_edge] = state["force_sign"][is_trail_edge] * trail_force_mag

        # Apply damping to new coordinates
        if state["coords"][trail_dst].isnan().any():
            state["coords"][trail_dst] = new_coords
        else:
            state["coords"][trail_dst] = state["coords"][trail_dst] + (1 - damping_factor) * (
                new_coords - state["coords"][trail_dst]
            )

        if track_history:
            if i == 0:
                coords_history = state["coords"].unsqueeze(0).clone()
                force_history = state["force"].unsqueeze(0).clone()
            else:
                coords_history = torch.cat((coords_history, state["coords"].unsqueeze(0)), dim=0)
                force_history = torch.cat((force_history, state["force"].unsqueeze(0)), dim=0)

        n_steps += 1

        # print(f"MPCEM Iteration {i}, delta coords = {torch.norm(state["coords"] - prev_coords)}")
        if torch.norm(state["coords"] - prev_coords) < tolerance:
            converged = True
            break

    # Calculate reaction force
    reaction_force = torch.full((state["coords"].shape[0], 3), float("nan")).to(state["coords"].device)
    reaction_force[is_support] = -residual_force[is_support]

    if verbose:
        print(f"MP-CEM finished in {n_steps} steps. Converged: {converged}.")
        # i + 1

    if track_history:
        return coords_history, force_history, reaction_force
    else:
        return state["coords"], state["force"].unsqueeze(1), reaction_force, state["load"]

def selfweight_cb(state, edge_index, edge_cem_to_undir, load_factor):
    # Only consider edges with a coordinate (estimate) for both nodes
    valid_nodes = ~torch.isnan(state["coords"]).any(dim=1)
    valid_edges = (valid_nodes[edge_index[0]] & valid_nodes[edge_index[1]])

    # Calculate edge loads due to self-weight
    length = graph_edge_lengths(state["coords"], edge_index[:, valid_edges]).view(-1)
    force = state["force"][edge_cem_to_undir][valid_edges]
    valid_edge_load = -load_factor * 0.5 * length * force.abs()  # load_factor = density / yield_strength
    edge_load = torch.zeros_like(state["force"][edge_cem_to_undir])
    edge_load[valid_edges] = valid_edge_load

    # Aggregate to node loads
    node_load = torch.zeros_like(state["load"])
    node_load[:, 2] = scatter(edge_load, edge_index[1], dim=0, dim_size=state["coords"].shape[0], reduce="sum")

    state_updates = {
        "load": node_load
    }
    return state_updates

def constrained_deck_cb(state, cem_edge_index, is_deck_trail_edge, is_mod_v, is_mod_h, sequence, deck_slope):  # TODO: Simplify
    # Enforce expected input shapes
    is_deck_trail_edge = is_deck_trail_edge.view(-1)
    is_mod_v = is_mod_v.view(-1)
    is_mod_h = is_mod_h.view(-1)
    sequence = sequence.view(-1)
    deck_slope = deck_slope.view(-1)

    # Only consider edges with a coordinate (estimate) for both nodes
    valid_nodes = ~torch.isnan(state["coords"]).any(dim=1)
    valid_edges = (valid_nodes[cem_edge_index[0]] & valid_nodes[cem_edge_index[1]])
    valid_to_deck_edges = (valid_edges & torch.isin(cem_edge_index[1, :], cem_edge_index[0, is_deck_trail_edge]))
    valid_from_deck_edges = (valid_edges & torch.isin(cem_edge_index[0, :], cem_edge_index[0, is_deck_trail_edge]))
    valid_to_deck_is_mod_v = valid_to_deck_edges & is_mod_v
    valid_from_deck_is_mod_v = valid_from_deck_edges & is_mod_v
    positive_direction = line_direction(state["coords"][cem_edge_index[0]], state["coords"][cem_edge_index[1]])[:, 1] > 0
    valid_is_mod_h_pos = valid_edges & is_mod_h & positive_direction
    valid_is_mod_h_neg = valid_edges & is_mod_h & ~positive_direction

    # Calculate residual force
    residual_force_mp = ResidualForce()
    residual_force = residual_force_mp(state["coords"], state["force"][valid_to_deck_edges],
                                       cem_edge_index[:, valid_to_deck_edges], state["load"])

    # Update forces of vertical modification edges
    mod_v_src, mod_v_dst = cem_edge_index[:, valid_to_deck_is_mod_v]
    mod_v_direction = line_direction(state["coords"][mod_v_src], state["coords"][mod_v_dst])
    x_res, _, z_res = residual_force[mod_v_dst].unbind(dim=1)
    state["force"][valid_to_deck_is_mod_v] += (z_res + deck_slope[mod_v_dst] / x_res) / mod_v_direction[:, 2]

    mod_v_dst, mod_v_src = cem_edge_index[:, valid_from_deck_is_mod_v]
    mod_v_direction = line_direction(state["coords"][mod_v_src], state["coords"][mod_v_dst])
    x_res, _, z_res = residual_force[mod_v_dst].unbind(dim=1)
    state["force"][valid_from_deck_is_mod_v] += (z_res + deck_slope[mod_v_dst] / x_res) / mod_v_direction[:, 2]

    # Update forces of horizontal modification edges
    residual_force = residual_force_mp(state["coords"], state["force"][valid_to_deck_edges],
                                       cem_edge_index[:, valid_to_deck_edges], state["load"])
    mod_h_src, mod_h_dst = cem_edge_index[:, valid_is_mod_h_pos]
    x_src, y_src, _ = residual_force[mod_h_src].unbind(dim=1)
    x_dst, y_dst, _ = residual_force[mod_h_dst].unbind(dim=1)
    state["force"][valid_is_mod_h_pos] += -(y_src * x_dst - x_src * y_dst) / (x_src + x_dst)

    mod_h_src, mod_h_dst = cem_edge_index[:, valid_is_mod_h_neg]
    x_src, y_src, _ = residual_force[mod_h_src].unbind(dim=1)
    x_dst, y_dst, _ = residual_force[mod_h_dst].unbind(dim=1)
    state["force"][valid_is_mod_h_neg] += (y_src * x_dst - x_src * y_dst) / (x_src + x_dst)

    state_updates = {
        "force": state["force"]
    }
    return state_updates

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
                print(f"Maximum steps {max_iter} reached without convergence.")
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
    reaction_force = torch.full((coords.shape[0], 3), float("nan")).to(coords.device)
    reaction_force[is_support] = -residual_force[is_support]

    if verbose:
        print(f"CEM finished in {n_steps} steps. Converged: {converged}.")
        # (i + 1) * max_k

    if track_history:
        return coords_history, force_history, reaction_force
    else:
        return coords, force.unsqueeze(1), reaction_force


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
                    print(f"Maximum steps {max_iter} reached without convergence.")
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
        print(f"CEM (sequential) finished in {n_steps} steps. Converged: {converged}.")
        # (i + 1) * max_k * len(k_node_indices)

    if track_history:
        return coords_history, force_history, reaction_force
    else:
        return coords, force.unsqueeze(1), reaction_force
