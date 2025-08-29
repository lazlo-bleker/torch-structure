import torch
import math

from torch_structure.data import Data

import torch_geometric as pyg
import numpy as np
from torch_scatter import scatter_add, scatter_mean

from torch_structure.data.heterodata import HeteroData

class PneuStructure:
    def __init__(
        self,
        path,
        x_res: float,
        y_res: float
    ):
        
        self.face_node_table = torch.empty((0, 3), dtype=torch.int)
        
        self.hetero_graph = self.generate_graph(path, x_res, y_res)


    def generate_graph(
        self,
        path,
        x_res: float,
        y_res: float
    ):
        

        node_attrs = {
            "pattern_coords": torch.empty((0, 2), dtype=torch.float),
            "is_support": torch.empty((0, 1), dtype=torch.bool),
            "is_boundary": torch.empty((0, 1), dtype=torch.bool),
            "z_coord": torch.empty((0, 1), dtype=torch.float)
        }
        edge_attrs = {
        }
        default_attrs = {
            "z_coord": torch.tensor(0.0),
        }
        graph = Data(
            node_attrs=node_attrs, edge_attrs=edge_attrs, default_attrs=default_attrs
        )
        
        hetero_data = HeteroData(data = graph)        

        grid, boundary = self.grid_From_Curve_Points(path, x_res, y_res)

        k = 0

        for i in range(len(grid)):  
            for j in range(len(grid[0])):   

                if grid[i][j] is not None:
                    k += 1
                    hetero_data.data.add_node(
                    str(i) + "-" + str(j),
                    pattern_coords = torch.tensor(grid[i][j], dtype = torch.float32),
                    is_support = boundary[i][j],
                    is_boundary = boundary[i][j]
                )
                    

        for i in range(len(grid)):  
            for j in range(len(grid[0])):     
                

                if j+1 < len(grid[0]):
                    if grid[i][j] is not None and grid[i][j+1] is not None:

                        hetero_data.data.add_edge(str(i) + "-" + str(j), str(i) + "-" + str(j+1)) 

                if i+1 < len(grid):

                    if grid[i][j] is not None and grid[i+1][j] is not None:

                        hetero_data.data.add_edge(str(i) + "-" + str(j), str(i+1) + "-" + str(j))

        hetero_data.data.coords = torch.cat(
            [hetero_data.data.pattern_coords, hetero_data.data.z_coord],
            dim=1,
        )

        hetero_data.hetero_graph['node'].coords = hetero_data.data.coords

        hetero_data.hetero_graph['node', 'connected_to', 'node'].edge_index = hetero_data.data.edge_index


        for i in range(len(grid)-1):  
            for j in range(len(grid[0])-1):   
                
                indices = []
                
                if grid[i][j] is not None:
                    indices.append(self.find_Index(grid, i, j))

                if grid[i+1][j] is not None:
                    indices.append(self.find_Index(grid, i+1, j))

                if grid[i+1][j+1] is not None:
                    indices.append(self.find_Index(grid, i+1, j+1))

                if grid[i][j+1] is not None:
                    indices.append(self.find_Index(grid, i, j+1))

                
                if len(indices) > 2:

                    hetero_data.add_face(indices)
                
        return hetero_data
    

    def find_Index(self, grid, m, n):
        
        ind = 0
        for i in range(len(grid)):  
            for j in range(len(grid[0])):   
                
                if i == m and j == n:

                    return ind
                
                if grid[i][j] is not None:

                    ind += 1


        return -1

    def add_face(self, graph, node_indices):

        coords = torch.empty((1, 3))  

        if 'coords' in graph['face_node']:
            graph['face_node'].coords = torch.cat([graph['face_node'].coords, coords], dim=0)
        else:
            graph['face_node'].coords = coords

        face_index = graph['face_node'].coords.size(0)-1

        rows = []

        for i in range(len(node_indices)):
            rows.append(torch.tensor([
                node_indices[i],
                node_indices[(i + 1) % len(node_indices)],
                face_index
            ], dtype=torch.long))

        # Stack all new rows
        new_rows = torch.stack(rows)  # shape: [n, 3]

        # Append to face_node_table
        self.face_node_table = torch.cat([self.face_node_table, new_rows], dim=0)

        
    def calculate_loads(
        self,
        pressure,
        heterograph

    ):

        #these normals are not scaled to unit length because they encode the area information as well!
        normals = heterograph.calculate_normals()
        
        return normals * pressure
        

    @staticmethod
    def area_trapezoid(A, B, C, D):
        
        AB = B - A
        AC = C - A
        area = 0.5 * torch.norm(torch.cross(AB, AC))

        AD = D - A

        return area + 0.5 * torch.norm(torch.cross(AD, AC))
    

    @staticmethod
    def grid_From_Curve_Points(path, x_res, y_res):
        vertices = path.vertices
        x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
        y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
        x_min = x_min - 2 * x_res
        x_max = x_max + 2 * x_res
        y_min = y_min - 2 * y_res
        y_max = y_max + 2 * y_res
        n1 = int(math.ceil((x_max - x_min) / x_res)) 
        n2 = int(math.ceil((y_max - y_min) / y_res)) 

        grid = [[None for _ in range(n2)] for _ in range(n1)]
        boundary = [[False for _ in range(n2)] for _ in range(n1)]
        eps = .26
        for i in range(n1):  
            for j in range(n2):  
                x = x_min + i * x_res
                y = y_min + j * y_res
                pt = np.array([[x, y]])

                if path.contains_points(pt)[0]:
                    distances = np.linalg.norm(vertices - pt[0], axis=1)
                    min_dist = np.min(distances)
                    
                    if min_dist < eps:
               
                        grid[i][j] = pt[0]

                    else:
                        grid[i][j] = pt[0]
                    
                    grid[i][j] = pt[0]
        for j in range(n1):  
            for i in range(n2):  
                
                if grid[i][j] is None:

                    boundary[i][j] = True
                
                if i < n1-1:
                    if grid[i+1][j] is None:

                        boundary[i][j] = True

                if i > 0:
                    if grid[i-1][j] is None:

                        boundary[i][j] = True

                if j > 0:
                    if grid[i][j-1] is None:

                        boundary[i][j] = True

                if j < n2-1:
                    if grid[i][j+1] is None:

                        boundary[i][j] = True


                if i > 0 and j > 0:
                    if grid[i-1][j-1] is None:

                        boundary[i][j] = True

                if i > 0 and j < n2-1:
                    if grid[i-1][j+1] is None:

                        boundary[i][j] = True

                if i < n1-1 and j < n2-1:
                    if grid[i+1][j+1] is None:

                        boundary[i][j] = True

                if i < n1-1 and j > 0:
                    if grid[i+1][j-1] is None:

                        boundary[i][j] = True


            
        return grid, boundary


        


