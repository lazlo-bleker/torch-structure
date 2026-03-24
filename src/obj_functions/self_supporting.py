from torch_structure.data import StructData
import torch
from config import TORCH_FLOAT

def self_supporting_loss(graph_solved: "StructData"):
    graph_solved.plot()

    grid_indices = graph_solved.grid_index
    n_trails = int(max(grid_indices[:, 0]) + 1)
    n_rings = int(max(grid_indices[:, 1]) + 1)

    all_layers = list(range(n_rings))
    state_losses  = torch.zeros(n_rings-1, dtype=TORCH_FLOAT)
    # Note: Last layer is fully supported --> Equilibrium is assured
    for j in range(1,n_rings):
        active_layers = all_layers[n_rings-j-1:-1]
        state_losses[j-1] = _solve_state(graph_solved, active_layers, n_trails, n_rings)
        graph_solved.plot(
            force = graph_solved.force_temp,
            load = graph_solved.aux_force_temp
            )
    
    return torch.sum(state_losses)

def _solve_state(graph_solved: "StructData", active_layers:list, n_trails : int, n_rings : int):
    state_loss = 0.0
    graph_solved.force_temp = torch.zeros_like(graph_solved.force)
    graph_solved.aux_force_temp = torch.zeros_like(graph_solved.load)

    for j in active_layers:
        # Solve least squares projection system at the layers
        A_sys = torch.zeros([n_trails * 3, n_trails * 2], dtype=TORCH_FLOAT)
        b_sys = torch.zeros(n_trails * 3, dtype=TORCH_FLOAT)
        for i in range(n_trails):
            # Get incoming trail force
            incoming_trail_index = graph_solved.incoming_trail[i,j]
            incoming_trail_direction = _get_edge_direction(graph_solved, incoming_trail_index)
            incoming_trail_force = incoming_trail_direction * graph_solved.force_temp[incoming_trail_index]

            # Get nodal loads
            node_index = graph_solved.node_grid[i,j]
            nodal_load = graph_solved.load[node_index]

            # Get resultant 
            res = incoming_trail_force + nodal_load

            # Get direction of the outgoing trail
            outgoing_trail_index = graph_solved.outgoing_trail[i,j]
            outgoing_trail_direction = _get_edge_direction(graph_solved, outgoing_trail_index)
            
            # Get direction of the adjacent deviation elements
            deviation_a_index = graph_solved.deviation_a[i,j]
            deviation_a_direction = _get_edge_direction(graph_solved, deviation_a_index)
            deviation_b_index = graph_solved.deviation_b[i,j]
            deviation_b_direction = -_get_edge_direction(graph_solved, deviation_b_index)

            # Add conditions for linear system to solve for the deviation forces
            sys_indices = [i,i+n_trails, i+2*n_trails]
            b_sys[sys_indices] = res
            A_sys[sys_indices, i] = deviation_a_direction
            A_sys[sys_indices, (i+1) % n_trails] = deviation_b_direction
            A_sys[sys_indices, n_trails + i] = outgoing_trail_direction


        beta = torch.linalg.solve(A_sys.T @ A_sys, A_sys.T @ b_sys)
        d_forces = beta[:n_trails]
        t_forces = beta[n_trails:]
        node_aux = -(A_sys @ beta - b_sys).reshape(3,-1).T

        # Assign trail forces to solve next layer (outgoing of layer i is the incoming of layer i+1)
        # TODO: Exception of last layer, not needed
        trail_indices = graph_solved.outgoing_trail[:,j]
        graph_solved.force_temp[trail_indices] = t_forces.unsqueeze(1)
        graph_solved.force_temp[trail_indices-1] = t_forces.unsqueeze(1)

        # Optional for log, assign deviation forces
        deviation_indices = graph_solved.deviation_a[:,j]
        graph_solved.force_temp[deviation_indices] = d_forces.unsqueeze(1)
        graph_solved.force_temp[deviation_indices-1] = d_forces.unsqueeze(1)

        # Optional for log, assign node auxiliary forces
        node_indices = graph_solved.node_grid[:,j]
        graph_solved.aux_force_temp[node_indices] = node_aux


        # Add loss contribution
        layer_loss = torch.linalg.norm(node_aux)
        state_loss += layer_loss
    
    return state_loss

def _get_edge_direction(graph_solved: "StructData", edge_index):
        if edge_index == -1:
             return torch.zeros(3, dtype=TORCH_FLOAT)
        tail, head = graph_solved.coords[graph_solved.edge_index[:,edge_index]]
        direction = head - tail
        direction /= torch.linalg.norm(direction)
        return direction