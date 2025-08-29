from torch_structure.data.data import Data
import torch_geometric as pyg
import torch
from torch_scatter import scatter_add, scatter_mean

class HeteroData:
    def __init__(self, data: Data):
        self.data = data

        self.hetero_graph = pyg.data.HeteroData()

        self.face_node_table = torch.empty((0, 3), dtype=torch.int)


    def add_face(self, node_indices):

        coords = torch.empty((1, 3))  

        if 'coords' in self.hetero_graph['face_node']:
            self.hetero_graph['face_node'].coords = torch.cat([self.hetero_graph['face_node'].coords, coords], dim=0)

        else:
            self.hetero_graph['face_node'].coords = coords

        face_index = self.hetero_graph['face_node'].coords.size(0)-1

        rows = []

        for i in range(len(node_indices)):
            rows.append(torch.tensor([
                node_indices[i],
                node_indices[(i + 1) % len(node_indices)],
                face_index
            ], dtype=torch.long))

        new_rows = torch.stack(rows)  

        self.face_node_table = torch.cat([self.face_node_table, new_rows], dim=0)


    def calculate_normals(
        self,

    ):
        nodes1_indices = self.face_node_table[:, 0]
        nodes2_indices = self.face_node_table[:, 1]
        face_node_indices = self.face_node_table[:, 2]

        nodes = self.hetero_graph['node'].coords
        

        self.hetero_graph['face_node'].coords = scatter_mean(nodes[nodes1_indices], face_node_indices, dim = 0)

        face_nodes = self.hetero_graph['face_node'].coords

        vec1 = nodes[nodes2_indices] - nodes[nodes1_indices]
        vec2 = face_nodes[face_node_indices] - nodes[nodes1_indices]

        normals = torch.cross(vec1, vec2) 
        
        return normals
        

    @staticmethod
    def area_trapezoid(A, B, C, D):
        
        AB = B - A
        AC = C - A
        area = 0.5 * torch.norm(torch.cross(AB, AC))

        AD = D - A

        return area + 0.5 * torch.norm(torch.cross(AD, AC))
    


        
