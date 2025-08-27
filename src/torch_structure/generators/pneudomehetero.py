import torch
import math

from torch_structure.data import Data

import torch_geometric as pyg

from torch_geometric.typing import torch_scatter

from torch_scatter import scatter_add, scatter_mean

class PneuDomeHetero:
    def __init__(
        self,
        num_parallel_lines: int,
        num_meridians: int,
        radius: float
    ):
        
        self.face_node_table = torch.empty((0, 3), dtype=torch.int)
        
        self.graph, self.hetero_graph = self.generate_graph(
            num_parallel_lines,
            num_meridians,
            radius
        )

    def generate_graph(
        self,
        num_parallel_lines,
        num_meridians,
        radius
    ):
        

        node_attrs = {
            "pattern_coords": torch.empty((0, 2), dtype=torch.float),
            "is_support": torch.empty((0, 1), dtype=torch.bool),
            "is_boundary": torch.empty((0, 1), dtype=torch.bool),
            "z_coord": torch.empty((0, 1), dtype=torch.float)
        }
        edge_attrs = {
            "is_meridian": torch.empty((0, 1), dtype=torch.bool),
        }
        default_attrs = {
            "z_coord": torch.tensor(0.0),
        }
        graph = Data(
            node_attrs=node_attrs, edge_attrs=edge_attrs, default_attrs=default_attrs
        )
        
        hetero_graph = pyg.data.HeteroData()


        #rest of nodes
        for i in range(num_parallel_lines):
            for j in range(num_meridians):
                r = (num_parallel_lines - i) / num_parallel_lines * radius
                angle = 2 * math.pi * j / num_meridians
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                point = torch.tensor([x,y])
                supp = (i == 0)
                bound = (i == 0)

                graph.add_node(
                    str(i) + str(j),
                    pattern_coords = point,
                    is_support = supp,
                    is_boundary = bound
                )

        
        #add centroid
        graph.add_node(
                    "centroid",
                    pattern_coords = torch.tensor([0,0]),
                    is_support = False,
                    is_boundary = False
                )

        #edges to centroid                
        for j in range(num_meridians):
            
            graph.add_edge(str(num_parallel_lines-1) + str(j), "centroid", is_meridian = torch.tensor(True))

        #rest of edges
        for i in range(num_parallel_lines):
            for j in range(num_meridians):

                if i > 0:

                    graph.add_edge(str(i) + str(j), str(i) + str((j+1) % num_meridians), is_meridian = torch.tensor(False))
                
                if i < num_parallel_lines - 1:
                    graph.add_edge(str(i) + str(j), str(i+1) + str(j), is_meridian = torch.tensor(True))
                
        


        graph.coords = torch.cat(
            [graph.pattern_coords, graph.z_coord],
            dim=1,
        )

        #hetero_graph['face_node'].x = torch.empty((3, num_faces)) 

        hetero_graph['node'].coords = graph.coords

        hetero_graph['node', 'connected_to', 'node'].edge_index = graph.edge_index

        index = 0

        for i in range(num_parallel_lines-1):
            for j in range(num_meridians):
                

                self.add_face(hetero_graph, [index, i * num_meridians + (j + 1) % num_meridians, (i + 1) * num_meridians + (j + 1) % num_meridians, index + num_meridians])

                index += 1


        for j in range(num_meridians):


            self.add_face(hetero_graph, [index, (num_parallel_lines - 2) * num_meridians + (j + 1) % num_meridians, num_parallel_lines * num_meridians])

            index += 1


        return graph, hetero_graph
    


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
        pressure

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
        
        return (scatter_add(normals, nodes1_indices, dim = 0) + scatter_add(normals, nodes2_indices, dim = 0)) / 4 * pressure
        

    @staticmethod
    def area_trapezoid(A, B, C, D):
        
        AB = B - A
        AC = C - A
        area = 0.5 * torch.norm(torch.cross(AB, AC))

        AD = D - A

        return area + 0.5 * torch.norm(torch.cross(AD, AC))
    




    def calculate_loads_old(
        self, 
        graph,
        num_parallel_lines,
        num_meridians,
        pressure 
    ):
        

        load = torch.zeros((graph.num_nodes, 3), dtype=torch.float)

        #we assume pressure to be evenly distributed from surface to node

        #scrappy for now for trial purposes

        #nodes with four trapezoids around them
        k = 0

        for j in range(num_meridians):
                
            curr_load = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float)

            coords_center = graph.nodes[str(0) + str(j)]["pattern_coords"]
            z_cord_center = graph.nodes[str(0) + str(j)]["z_coord"]
            pt_center = torch.tensor([coords_center[0], coords_center[1],  z_cord_center], dtype=torch.float)

            coords_left_mid = graph.nodes[str(0) + str((j-1)%num_meridians)]["pattern_coords"]
            z_cord_left_mid = graph.nodes[str(0) + str((j-1)%num_meridians)]["z_coord"]
            pt_left_mid = torch.tensor([coords_left_mid[0], coords_left_mid[1],  z_cord_left_mid], dtype=torch.float)

            coords_right_mid = graph.nodes[str(0) + str((j+1)%num_meridians)]["pattern_coords"]
            z_cord_right_mid = graph.nodes[str(0) + str((j+1)%num_meridians)]["z_coord"]
            pt_right_mid = torch.tensor([coords_right_mid[0], coords_right_mid[1],  z_cord_right_mid], dtype=torch.float)

            coords_left_up = graph.nodes[str(1) + str((j-1)%num_meridians)]["pattern_coords"]
            z_cord_left_up = graph.nodes[str(1) + str((j-1)%num_meridians)]["z_coord"]
            pt_left_up = torch.tensor([coords_left_up[0], coords_left_up[1],  z_cord_left_up], dtype=torch.float)
    
            coords_right_up = graph.nodes[str(1) + str((j+1)%num_meridians)]["pattern_coords"]
            z_cord_right_up = graph.nodes[str(1) + str((j+1)%num_meridians)]["z_coord"]
            pt_right_up = torch.tensor([coords_right_up[0], coords_right_up[1],  z_cord_right_up], dtype=torch.float)

            coords_mid_up = graph.nodes[str(1) + str(j)]["pattern_coords"]
            z_cord_mid_up = graph.nodes[str(1) + str(j)]["z_coord"]
            pt_mid_up = torch.tensor([coords_mid_up[0], coords_mid_up[1],  z_cord_mid_up], dtype=torch.float)

            normal_left_up = torch.cross(pt_left_up - pt_left_mid, -pt_center + pt_left_mid)
            normal_left_up /= torch.norm(normal_left_up)
            
            area_left_up = self.area_trapezoid(pt_left_up, pt_mid_up, pt_center, pt_left_mid)
            curr_load += normal_left_up * area_left_up * pressure / 4


            normal_right_up = torch.cross(pt_mid_up - pt_center, -pt_right_mid + pt_center)
            normal_right_up /= torch.norm(normal_right_up)

            area_right_up = self.area_trapezoid(pt_mid_up, pt_right_up, pt_right_mid, pt_center)
            curr_load += normal_right_up * area_right_up * pressure / 4 
                
            load[k] = curr_load

            k += 1


        for i in range(1, num_parallel_lines-1):
            for j in range(num_meridians):

                curr_load = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float)

                coords_center = graph.nodes[str(i) + str(j)]["pattern_coords"]
                z_cord_center = graph.nodes[str(i) + str(j)]["z_coord"]
                pt_center = torch.tensor([coords_center[0], coords_center[1],  z_cord_center], dtype=torch.float)

                coords_left_mid = graph.nodes[str(i) + str((j-1)%num_meridians)]["pattern_coords"]
                z_cord_left_mid = graph.nodes[str(i) + str((j-1)%num_meridians)]["z_coord"]
                pt_left_mid = torch.tensor([coords_left_mid[0], coords_left_mid[1],  z_cord_left_mid], dtype=torch.float)

                coords_right_mid = graph.nodes[str(i) + str((j+1)%num_meridians)]["pattern_coords"]
                z_cord_right_mid = graph.nodes[str(i) + str((j+1)%num_meridians)]["z_coord"]
                pt_right_mid = torch.tensor([coords_right_mid[0], coords_right_mid[1],  z_cord_right_mid], dtype=torch.float)

                coords_left_up = graph.nodes[str(i+1) + str((j-1)%num_meridians)]["pattern_coords"]
                z_cord_left_up = graph.nodes[str(i+1) + str((j-1)%num_meridians)]["z_coord"]
                pt_left_up = torch.tensor([coords_left_up[0], coords_left_up[1],  z_cord_left_up], dtype=torch.float)
       
                coords_right_up = graph.nodes[str(i+1) + str((j+1)%num_meridians)]["pattern_coords"]
                z_cord_right_up = graph.nodes[str(i+1) + str((j+1)%num_meridians)]["z_coord"]
                pt_right_up = torch.tensor([coords_right_up[0], coords_right_up[1],  z_cord_right_up], dtype=torch.float)
          
                coords_left_down = graph.nodes[str(i-1) + str((j-1)%num_meridians)]["pattern_coords"]
                z_cord_left_down = graph.nodes[str(i-1) + str((j-1)%num_meridians)]["z_coord"]
                pt_left_down = torch.tensor([coords_left_down[0], coords_left_down[1],  z_cord_left_down], dtype=torch.float)
       
                coords_right_down = graph.nodes[str(i-1) + str((j+1)%num_meridians)]["pattern_coords"]
                z_cord_right_down = graph.nodes[str(i-1) + str((j+1)%num_meridians)]["z_coord"]
                pt_right_down = torch.tensor([coords_right_down[0], coords_right_down[1],  z_cord_right_down], dtype=torch.float)
     
                coords_mid_down = graph.nodes[str(i-1) + str(j)]["pattern_coords"]
                z_cord_mid_down = graph.nodes[str(i-1) + str(j)]["z_coord"]
                pt_mid_down = torch.tensor([coords_mid_down[0], coords_mid_down[1],  z_cord_mid_down], dtype=torch.float)
   
                coords_mid_up = graph.nodes[str(i+1) + str(j)]["pattern_coords"]
                z_cord_mid_up = graph.nodes[str(i+1) + str(j)]["z_coord"]
                pt_mid_up = torch.tensor([coords_mid_up[0], coords_mid_up[1],  z_cord_mid_up], dtype=torch.float)
   
                normal_left_down = torch.cross(pt_left_mid - pt_left_down, -pt_mid_down + pt_left_down)
                normal_left_down /= torch.norm(normal_left_down)

                area_left_down = self.area_trapezoid(pt_left_down, pt_left_mid, pt_center, pt_mid_down)
                curr_load += normal_left_down * area_left_down * pressure / 4

                normal_right_down = torch.cross(pt_center - pt_mid_down, -pt_right_down + pt_mid_down)
                normal_right_down /= torch.norm(normal_right_down)

                area_right_down = self.area_trapezoid(pt_center, pt_right_mid, pt_right_down, pt_mid_down)
                curr_load += normal_right_down * area_right_down * pressure / 4


                normal_left_up = torch.cross(pt_left_up - pt_left_mid, -pt_center + pt_left_mid)
                normal_left_up /= torch.norm(normal_left_up)

                area_left_up = self.area_trapezoid(pt_left_up, pt_mid_up, pt_center, pt_left_mid)
                curr_load += normal_left_up * area_left_up * pressure / 4


                normal_right_up = torch.cross(pt_mid_up - pt_center, -pt_right_mid + pt_center)
                normal_right_up /= torch.norm(normal_right_up)

                area_right_up = self.area_trapezoid(pt_mid_up, pt_right_up, pt_right_mid, pt_center)
                curr_load += normal_right_up * area_right_up * pressure / 4 
                
                load[k] = curr_load

                k += 1



        center_load  = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float)

        #calc loads for top center triangles 
        i = num_parallel_lines-1
        for j in range(num_meridians):
                
            curr_load = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float)

            coords_center = graph.nodes[str(i) + str(j)]["pattern_coords"]
            z_cord_center = graph.nodes[str(i) + str(j)]["z_coord"]
            pt_center = torch.tensor([coords_center[0], coords_center[1],  z_cord_center], dtype=torch.float)

            coords_mid_up = graph.nodes["centroid"]["pattern_coords"]
            z_cord_mid_up = graph.nodes["centroid"]["z_coord"]
            pt_mid_up = torch.tensor([coords_mid_up[0], coords_mid_up[1],  z_cord_mid_up], dtype=torch.float)

            coords_left_mid = graph.nodes[str(i) + str((j-1)%num_meridians)]["pattern_coords"]
            z_cord_left_mid = graph.nodes[str(i) + str((j-1)%num_meridians)]["z_coord"]
            pt_left_mid = torch.tensor([coords_left_mid[0], coords_left_mid[1],  z_cord_left_mid], dtype=torch.float)

            coords_right_mid = graph.nodes[str(i) + str((j+1)%num_meridians)]["pattern_coords"]
            z_cord_right_mid = graph.nodes[str(i) + str((j+1)%num_meridians)]["z_coord"]
            pt_right_mid = torch.tensor([coords_right_mid[0], coords_right_mid[1],  z_cord_right_mid], dtype=torch.float)

            coords_left_down = graph.nodes[str(i-1) + str((j-1)%num_meridians)]["pattern_coords"]
            z_cord_left_down = graph.nodes[str(i-1) + str((j-1)%num_meridians)]["z_coord"]
            pt_left_down = torch.tensor([coords_left_down[0], coords_left_down[1],  z_cord_left_down], dtype=torch.float)
    
            coords_right_down = graph.nodes[str(i-1) + str((j+1)%num_meridians)]["pattern_coords"]
            z_cord_right_down = graph.nodes[str(i-1) + str((j+1)%num_meridians)]["z_coord"]
            pt_right_down = torch.tensor([coords_right_down[0], coords_right_down[1],  z_cord_right_down], dtype=torch.float)
    
            coords_mid_down = graph.nodes[str(i-1) + str(j)]["pattern_coords"]
            z_cord_mid_down = graph.nodes[str(i-1) + str(j)]["z_coord"]
            pt_mid_down = torch.tensor([coords_mid_down[0], coords_mid_down[1],  z_cord_mid_down], dtype=torch.float)

            normal_left_up = torch.cross(pt_mid_up - pt_left_mid, -pt_center + pt_left_mid)
            normal_left_up /= .5 #area of triangle is half of cross product of sides
            
            curr_load += normal_left_up * pressure / 3
            center_load += normal_left_up * pressure / 3

            normal_right_up = torch.cross(pt_mid_up - pt_center, -pt_right_mid + pt_center)
            normal_right_up /= .5

            curr_load += normal_right_up * pressure / 3
                
            normal_left_down = torch.cross(pt_left_mid - pt_left_down, -pt_mid_down + pt_left_down)
            normal_left_down /= torch.norm(normal_left_down)

            area_left_down = self.area_trapezoid(pt_left_down, pt_left_mid, pt_center, pt_mid_down)
            curr_load += normal_left_down * area_left_down * pressure / 4

            normal_right_down = torch.cross(pt_center - pt_mid_down, -pt_right_down + pt_mid_down)
            normal_right_down /= torch.norm(normal_right_down)
            
            area_right_down = self.area_trapezoid(pt_center, pt_right_mid, pt_right_down, pt_mid_down)
            curr_load += normal_right_down * area_right_down * pressure / 4

            load[k] = curr_load

            k += 1
        
        load[k] = center_load

        return load
    
    


        


