import torch
import math
import numpy as np

from torch_structure.data import Data


class PneuDome:
    def __init__(
        self,
        num_parallel_lines: int,
        num_meridians: int,
        radius: float,
    ):

        self.graph = self.generate_graph(
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
            "is_boundary_edge": torch.empty((0, 1), dtype=torch.bool),
        }
        default_attrs = {
            "z_coord": torch.tensor(0.0),
        }
        graph = Data(
            node_attrs=node_attrs, edge_attrs=edge_attrs, default_attrs=default_attrs
        )

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
            
            graph.add_edge(str(num_parallel_lines) + str(j), "centroid", torch.tensor(False))


        #rest of edges
        for i in range(num_parallel_lines):
            for j in range(num_meridians):

                if i == 0:
                    is_boundary_edge = torch.tensor(True)
                else:
                    is_boundary_edge = torch.tensor(False)

                graph.add_edge(str(i) + str(j), str(i) + str((j+1) % num_meridians), is_boundary_edge)

                if i < num_parallel_lines:
                    graph.add_edge(str(i) + str(j), str(i+1) + str(j), torch.tensor(False))

        graph.coords = torch.cat(
            [graph.pattern_coords, graph.z_coord],
            dim=1,
        )

        return graph
    

    def calculate_loads(
        self, 
        graph,
        num_parallel_lines,
        num_meridians,
        pressure 
    ):
        

        load = torch.zeros((graph.num_nodes, 3), dtype=torch.float)

        #we assume pressure to be evenly distributed from surface to node

        #scrappy for now for trial purposes
        #dont forget triangles on up


        #nodes with four trazepoids around them
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
            center_load *= normal_left_up * pressure / 3

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
    
    
    @staticmethod
    def area_trapezoid(A, B, C, D):
        
        AB = B - A
        AC = C - A
        area = 0.5 * torch.norm(torch.cross(AB, AC))

        AD = D - A

        return area + 0.5 * torch.norm(torch.cross(AD, AC))


        


