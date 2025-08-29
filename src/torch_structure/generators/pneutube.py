import numpy as np
import torch

from torch_structure.data import Data

import torch_geometric as pyg

from torch_geometric.typing import torch_scatter

from torch_scatter import scatter_add

from torch_structure.data.heterodata import HeteroData

class PneuTube:
    def __init__(
        self,
        polyline,
        radius,
        radialRes,
        verticalDamping,
        rotation,
        translation,
        scaling,
        hybrid = False
    ):
                
        self.hetero_graph, self.refNodes = self.generate_graph(
            polyline,
            radius,
            radialRes,
            verticalDamping,
            rotation,
            translation,
            scaling,
            hybrid
        )

    def generate_graph(
        self,
        polyline,
        radius,
        radialRes,
        verticalDamping,
        rotation,
        translation,
        scaling,
        hybrid
    ):
        refNodes = np.empty((0, 3))

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
        
        hetero_data = HeteroData(data = graph)        
        
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
                    hetero_data.data.add_node(
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


        for k in range(len(rotation)):

            for i in range(0, len(polyline)):

                for j in range(radialRes):
                    
                    hetero_data.data.add_edge(str(k*len(polyline) * radialRes) + "-" + str(i) + "-" + str(j), str(k*len(polyline) * radialRes) + "-" + str(i) + "-" + str((j+1) % radialRes), is_radial = True, is_longitudinal = False, is_diagonal = False, is_top_edge = False)

                    if i < len(polyline) - 1:

                        hetero_data.data.add_edge(str(k*len(polyline) * radialRes) + "-" + str(i) + "-" + str(j), str(k*len(polyline) * radialRes) + "-" + str(i+1) + "-" + str(j), is_radial = False, is_diagonal = False, is_longitudinal = True, is_top_edge = (j==0))

                      


        # add faces
        for k in range(len(rotation)):
            n = k * len(polyline) * radialRes

            for j in range(radialRes):
                
                hetero_data.add_face([n + j, n + (j+1) % radialRes, n + radialRes + (j+1) % radialRes, n + radialRes + j])
                

            for i in range(1, len(polyline)-1):

                for j in range(radialRes):
                    
                    hetero_data.add_face([n + i * radialRes + j, n + i * radialRes + (j+1) % radialRes, n + (i+1) * radialRes + (j+1) % radialRes,  n + (i+1) * radialRes + j])


        if hybrid:
            #res of inbetween grids
            n = len(polyline) - 1
            m = 5



            base = len(rotation) * len(polyline) * radialRes
            
            for i in range(n+1):
                for j in range(m+1):

                    pt = j/(m) * support1[i] + (1-(j/m)) * support2[i]
                    hetero_data.data.add_node(
                            "left" + "-" + str(i) + "-" + str(j),
                            pattern_coords = [pt[0], pt[1]],
                            is_support = (j == 0  or j == m),
                            z_coord = pt[2],
                            is_top_node = (j == 0)
                        )
            
            for i in range(n+1):
                for j in range(m+1):
                    
                    if i < n:
                                            
                        hetero_data.data.add_edge("left" + "-" + str(i) + "-" + str(j), "left" + "-" + str(i+1) + "-" + str(j), edges_fields = False, is_radial = False, is_longitudinal = False, is_diagonal = False, is_top_edge = False)

                    if j < m:
                        
                        hetero_data.data.add_edge("left" + "-" + str(i) + "-" + str(j), "left" + "-" + str(i) + "-" + str(j+1), edges_fields = (i == n or i == 0), is_radial = False, is_longitudinal = False, is_diagonal = False, is_top_edge = False)


            for i in range(n):
                for j in range(m):
                
                    hetero_data.add_face([base + i * (m+1) + j, base + i * (m+1) + j + 1, base + (i+1) * (m+1) + j + 1, base + (i+1) * (m+1) + j])

            


            base = base + (n+1) * (m+1)

            for i in range(n+1):
                for j in range(m+1):

                    pt = j/(m) * support4[i] + (1-(j/m)) * support3[i]
                    hetero_data.data.add_node(
                            "right" + "-" + str(i) + "-" + str(j),
                            pattern_coords = [pt[0], pt[1]],
                            is_support = (j == 0  or j == m),
                            z_coord = pt[2],
                            is_top_node = (j == 0)
                        )
            
            for i in range(n+1):
                for j in range(m+1):
                    
                    if i < n:
                                            
                        hetero_data.data.add_edge("right" + "-" + str(i) + "-" + str(j), "right" + "-" + str(i+1) + "-" + str(j), edges_fields = False, is_radial = False, is_longitudinal = False, is_diagonal = False, is_top_edge = False)

                    if j < m:
                        
                        hetero_data.data.add_edge("right" + "-" + str(i) + "-" + str(j), "right" + "-" + str(i) + "-" + str(j+1), edges_fields = (i == n or i == 0), is_radial = False, is_longitudinal = False, is_diagonal = False, is_top_edge = False)


            for i in range(n):
                for j in range(m):
                
                    hetero_data.add_face([base + (i+1) * (m+1) + j, base + (i+1) * (m+1) + j + 1, base + i * (m+1) + j + 1, base + i * (m+1) + j])


        hetero_data.data.coords = torch.cat(
            [hetero_data.data.pattern_coords, hetero_data.data.z_coord],
            dim=1,
        )
        
        hetero_data.hetero_graph['node'].coords = hetero_data.data.coords

        hetero_data.hetero_graph['node', 'connected_to', 'node'].edge_index = hetero_data.data.edge_index
        
        return hetero_data, refNodes
        
    def calculate_loads(
        self,
        pressure,
        heterograph,
        hybrid = False  

    ):
        
        nodes1_indices = heterograph.face_node_table[:, 0]
        nodes2_indices = heterograph.face_node_table[:, 1]
   
        
        normals = heterograph.calculate_normals()
        load = -(scatter_add(normals, nodes1_indices, dim = 0) + scatter_add(normals, nodes2_indices, dim = 0)) / 4
        
        if hybrid:
            load[:135] *= pressure
            load[135:] *= 5
        else:
            load *= pressure
        
        return load
        

        


