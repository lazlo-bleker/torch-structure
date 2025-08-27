import numpy as np
import torch
import math

from torch_structure.data import Data

import torch_geometric as pyg

from torch_geometric.typing import torch_scatter

from torch_scatter import scatter_add, scatter_mean

class PneuTube:
    def __init__(
        self,
        polyline,
        startNormal,
        endNormal,
        startBinormal,
        endBinormal,
        radius,
        radialRes,
        verticalDamping,
        rotation,
        translation,
        scaling
    ):
        
        self.face_node_table = torch.empty((0, 3), dtype=torch.int)
        
        self.graph, self.hetero_graph, self.refNodes = self.generate_graph(
            polyline,
            startNormal,
            endNormal,
            startBinormal,
            endBinormal,
            radius,
            radialRes,
            verticalDamping,
            rotation,
            translation,
            scaling
        )

    def generate_graph(
        self,
        polyline,
        startNormal,
        endNormal,
        startBinormal,
        endBinormal,
        radius,
        radialRes,
        verticalDamping,
        rotation,
        translation,
        scaling
    ):
        refNodes = np.empty((0, 3))

        endNormal /= np.linalg.norm(endNormal)
        startNormal /= np.linalg.norm(startNormal)
        startBinormal /= np.linalg.norm(startBinormal)
        endBinormal /= np.linalg.norm(endBinormal)
        node_attrs = {
            "pattern_coords": torch.empty((0, 2), dtype=torch.float),
            "is_support": torch.empty((0, 1), dtype=torch.bool),
            "z_coord": torch.empty((0, 1), dtype=torch.float),
            "is_top_node": torch.empty((0, 1), dtype=torch.bool),
        }
        edge_attrs = {
            "is_radial": torch.empty((0, 1), dtype=torch.bool),
            "is_longitudinal": torch.empty((0, 1), dtype=torch.bool),
            "is_diagonal": torch.empty((0, 1), dtype=torch.bool),
            "is_top_edge": torch.empty((0, 1), dtype=torch.bool),
            "edges_fields": torch.empty((0, 1), dtype=torch.bool)
        }
        default_attrs = {
            "z_coord": torch.tensor(0.0),
        }
        graph = Data(
            node_attrs=node_attrs, edge_attrs=edge_attrs, default_attrs=default_attrs
        )
        
        hetero_graph = pyg.data.HeteroData()
        
        
        arcradius = np.linalg.norm((polyline[0]) - np.array(polyline[len(polyline)-1]))/2
        center = np.array([0, arcradius, 0])

        support1 = []
        support2 = []
        support3 = []
        support4 = []

        for k in range(len(rotation)):

            for i in range(1, len(polyline)+1):

                alpha = np.pi / (len(polyline)+1) * i
                
                arcpt = center + np.array([0, -arcradius * np.cos(alpha), verticalDamping * arcradius * np.sin(alpha)])
                arcpt[2] *= scaling[k]
                arcpt[1] *= scaling[k]

                #arcpt[0] *= scaling[k]
                e_1 = np.array([0, -np.cos(alpha),  verticalDamping * np.sin(alpha)])
                e_1 /= np.linalg.norm(e_1)
                vec = np.array([0, verticalDamping * np.sin(alpha),  np.cos(alpha)])
                vec /= np.linalg.norm(vec)
                e_2 = np.cross(e_1, vec) 
                is_supp = (i == 1) or (i == len(polyline))
                for j in range(radialRes):
                    pt = arcpt + radius * np.cos(2 * np.pi * j / radialRes) * e_1 + radius * np.sin(2 * np.pi * j / radialRes) * e_2
                    
                    pt = np.dot(rotation[k], pt)
                    pt = translation[k] + pt
                    graph.add_node(
                        str(k*len(polyline) * radialRes) + "-" + str(i-1) + "-" + str(j),
                        pattern_coords = [pt[0], pt[1]],
                        is_support = is_supp,
                        z_coord = pt[2],
                        is_top_node = (j == 0)
                    )
                    if(j == radialRes-1) and k == 0:
                        support1.append(pt)
                    if(j == 1) and k == 1:
                        support2.append(pt)
                    if(j == radialRes-1) and k == 1:
                        support3.append(pt)
                    if(j == 1) and k == 2:
                        support4.append(pt)
                    refNodes = np.vstack([refNodes, pt])


        '''
        e_1 = np.cross(startNormal, startBinormal)
        e_2 = np.cross(startNormal, e_1)    

        #add support nodes at beginning of polyline
        for j in range(radialRes):
            pt = radius * np.cos(2 * np.pi * j / radialRes) * e_1 + radius * np.sin(2 * np.pi * j / radialRes) * e_2 + polyline[0]

            graph.add_node(
                str(0) + "-" + str(j),
                pattern_coords = [pt[0], pt[1]],
                is_support = True,
                z_coord = pt[2],
                is_top_node = (j == 0)
            )
        #add inbetween nodes
        for i in range(1, len(polyline)-1):

            for j in range(radialRes):

                pt = radius * np.cos(2 * np.pi * j / radialRes) * e_1 + radius * np.sin(2 * np.pi * j / radialRes) * e_2 + polyline[i]
                graph.add_node(
                    str(i) + "-" + str(j),
                    pattern_coords = [pt[0], pt[1]],
                    is_support = False,
                    z_coord = pt[2],
                    is_top_node = (j == 0)
                )
    
        e_1 = np.cross(endNormal, endBinormal)
        e_2 = np.cross(endNormal, e_1)    

        #add support nodes at end of polyline
        for j in range(radialRes):
            
            pt = radius * np.cos(2 * np.pi * j / radialRes) * e_1 - radius * np.sin(2 * np.pi * j / radialRes) * e_2 + polyline[-1]

            graph.add_node(
                str(len(polyline)-1) + "-" + str(j),
                pattern_coords = [pt[0], pt[1]],
                is_support = True,
                z_coord = pt[2],
                is_top_node = (j == 0)
            )
        
        '''
        #add edges
        '''for j in range(radialRes):
            
            graph.add_edge(str(0) + "-" + str(j), str(1) + "-" + str(j), is_radial = False, is_diagonal = False, is_longitudinal = True, is_top_edge = (j==0))

            graph.add_edge(str(0) + "-" + str(j), str(1) + "-" + str((j-1) % radialRes), is_radial = False, is_diagonal = True, is_longitudinal = False, is_top_edge = False)
            
            graph.add_edge(str(0) + "-" + str(j), str(1) + "-" + str((j+1) % radialRes), is_radial = False, is_diagonal = True, is_longitudinal = False, is_top_edge = False)

        '''


        for k in range(len(rotation)):

            for i in range(0, len(polyline)):

                for j in range(radialRes):
                    
                    graph.add_edge(str(k*len(polyline) * radialRes) + "-" + str(i) + "-" + str(j), str(k*len(polyline) * radialRes) + "-" + str(i) + "-" + str((j+1) % radialRes), is_radial = True, is_longitudinal = False, is_diagonal = False, is_top_edge = False)

                    if i < len(polyline) - 1:

                        graph.add_edge(str(k*len(polyline) * radialRes) + "-" + str(i) + "-" + str(j), str(k*len(polyline) * radialRes) + "-" + str(i+1) + "-" + str(j), is_radial = False, is_diagonal = False, is_longitudinal = True, is_top_edge = (j==0))

                        #graph.add_edge(str(i) + "-" + str(j), str(i+1) + "-" + str((j+1) % radialRes), is_radial = False, is_diagonal = True, is_longitudinal = False, is_top_edge = False)

                        #graph.add_edge(str(i) + "-" + str(j), str(i+1) + "-" + str((j-1) % radialRes), is_radial = False, is_diagonal = True, is_longitudinal = False, is_top_edge = False)


        #hetero_graph['face_node'].x = torch.empty((3, num_faces)) 

        # add faces
        for k in range(len(rotation)):
            n = k * len(polyline) * radialRes

            for j in range(radialRes):
                
                self.add_face(hetero_graph, [n + j, n + (j+1) % radialRes, n + radialRes + (j+1) % radialRes, n + radialRes + j])
                

            for i in range(1, len(polyline)-1):

                for j in range(radialRes):
                    
                    self.add_face(hetero_graph, [n + i * radialRes + j, n + i * radialRes + (j+1) % radialRes, n + (i+1) * radialRes + (j+1) % radialRes,  n + (i+1) * radialRes + j])

        
        #res of inbetween grids
        n = len(polyline) - 1
        m = 5



        base = len(rotation) * len(polyline) * radialRes
        
        for i in range(n+1):
            for j in range(m+1):

                pt = j/(m) * support1[i] + (1-(j/m)) * support2[i]
                graph.add_node(
                        "left" + "-" + str(i) + "-" + str(j),
                        pattern_coords = [pt[0], pt[1]],
                        is_support = (j == 0  or j == m),
                        z_coord = pt[2],
                        is_top_node = (j == 0)
                    )
        
        for i in range(n+1):
            for j in range(m+1):
                
                if i < n:
                                        
                    graph.add_edge("left" + "-" + str(i) + "-" + str(j), "left" + "-" + str(i+1) + "-" + str(j), edges_fields = False, is_radial = False, is_longitudinal = False, is_diagonal = False, is_top_edge = False)

                if j < m:
                    
                    graph.add_edge("left" + "-" + str(i) + "-" + str(j), "left" + "-" + str(i) + "-" + str(j+1), edges_fields = (i == n or i == 0), is_radial = False, is_longitudinal = False, is_diagonal = False, is_top_edge = False)


        for i in range(n):
            for j in range(m):
             
                self.add_face(hetero_graph, [base + i * (m+1) + j, base + i * (m+1) + j + 1, base + (i+1) * (m+1) + j + 1, base + (i+1) * (m+1) + j])

        


        base = base + (n+1) * (m+1)

        for i in range(n+1):
            for j in range(m+1):

                pt = j/(m) * support4[i] + (1-(j/m)) * support3[i]
                graph.add_node(
                        "right" + "-" + str(i) + "-" + str(j),
                        pattern_coords = [pt[0], pt[1]],
                        is_support = (j == 0  or j == m),
                        z_coord = pt[2],
                        is_top_node = (j == 0)
                    )
        
        for i in range(n+1):
            for j in range(m+1):
                
                if i < n:
                                        
                    graph.add_edge("right" + "-" + str(i) + "-" + str(j), "right" + "-" + str(i+1) + "-" + str(j), edges_fields = False, is_radial = False, is_longitudinal = False, is_diagonal = False, is_top_edge = False)

                if j < m:
                    
                    graph.add_edge("right" + "-" + str(i) + "-" + str(j), "right" + "-" + str(i) + "-" + str(j+1), edges_fields = (i == n or i == 0), is_radial = False, is_longitudinal = False, is_diagonal = False, is_top_edge = False)


        for i in range(n):
            for j in range(m):
             
                self.add_face(hetero_graph, [base + (i+1) * (m+1) + j, base + (i+1) * (m+1) + j + 1, base + i * (m+1) + j + 1, base + i * (m+1) + j])


        graph.coords = torch.cat(
            [graph.pattern_coords, graph.z_coord],
            dim=1,
        )
        
        hetero_graph['node'].coords = graph.coords

        hetero_graph['node', 'connected_to', 'node'].edge_index = graph.edge_index
        
        return graph, hetero_graph, refNodes


    def addTube(self, radial_res, longitudinal_res, newcoords):

        count = 0
        for i in range(1, longitudinal_res+1):

            for j in range(radial_res):
            
                self.graph.add_node2(
                    str(radial_res * longitudinal_res) + "-" + str(i-1) + "-" + str(j),
                    pattern_coords = newcoords[count][:2].tolist(),
                    z_coord = float(newcoords[count][2]),
                    is_top_node = (j == 0)
                )
                count += 1
                


        for i in range(0, longitudinal_res):

            for j in range(radial_res):

                self.graph.add_edge2(str(radial_res * longitudinal_res) + "-" + str(i) + "-" + str(j), str(radial_res * longitudinal_res) + "-" + str(i) + "-" + str((j+1) % radial_res), is_radial = True, is_longitudinal = False, is_diagonal = False, is_top_edge = False)

                if i < longitudinal_res - 1:

                    self.graph.add_edge2(str(radial_res * longitudinal_res) + "-" + str(i) + "-" + str(j), str(radial_res * longitudinal_res) + "-" + str(i+1) + "-" + str(j), is_radial = False, is_diagonal = False, is_longitudinal = True, is_top_edge = (j==0))


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
        
        load = -(scatter_add(normals, nodes1_indices, dim = 0) + scatter_add(normals, nodes2_indices, dim = 0)) / 4
        
        #if pressure.dim == 0:
        load[:135] *= pressure
        load[135:] *= 5
        #else:
        #load[:int(len(load)/3)] *= pressure[0]
        #load[int(len(load)/3):2 * int(len(load)/3)] *= pressure[1]
        #load[int(len(load)/3):] *= pressure[2]
        return load
        

    @staticmethod
    def area_trapezoid(A, B, C, D):
        
        AB = B - A
        AC = C - A
        area = 0.5 * torch.norm(torch.cross(AB, AC))

        AD = D - A

        return area + 0.5 * torch.norm(torch.cross(AD, AC))
    




        


