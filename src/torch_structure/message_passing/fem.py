import torch
from torch_geometric.nn import MessagePassing

class FEM(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')

    def forward(self, x, load, u, support, effective_stiffness, edge_index, length, A, E,
                direction, transformation_matrix):
        residual_force = load + self.propagate(edge_index, x=x, u=u, length=length, A=A, E=E,
                                               direction=direction, transformation_matrix=transformation_matrix)
        effective_stiffness = effective_stiffness.view(-1, 1) + 1e-9
        u_new = u + residual_force / effective_stiffness
        u_new[support] = 0
        return u_new, residual_force
    
    def message(self, u_i, u_j, length, A, E, direction, transformation_matrix):
        length = length.view(-1)
        A = A.view(-1)
        E = E.view(-1)
        local_u = torch.matmul(transformation_matrix, torch.cat([u_i, u_j], dim=1).unsqueeze(2)).squeeze(2)
        elongation = local_u[:, 0] - local_u[:, 1]
        strain = elongation / length
        stress = E * strain
        force = stress * A
        force_vector = -1 * direction * force.view(-1, 1)
        return force_vector
    
class StiffnessAggregator(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')

    def forward(self, edge_index, stiffness):
        return self.propagate(edge_index, stiffness=stiffness)

    def message(self, stiffness):
        return stiffness