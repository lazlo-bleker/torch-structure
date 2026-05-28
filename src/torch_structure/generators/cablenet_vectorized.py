import torch
import numpy as np
import math
from torch_structure.data import StructData
from torch_structure.generators.base_generator import BaseGenerator


class CableNetGeneratorVectorized(BaseGenerator):
    def __init__(self, **overrides):
        super().__init__(**overrides)
        self.node_attrs = {
            "coords": torch.empty((0, 3), dtype=torch.float),
            "merge_group_id" : torch.empty((0,1), dtype=torch.int), 
            "load": torch.empty((0, 3), dtype=torch.float),
            "is_support": torch.empty((0, 1), dtype=torch.bool),
        }

        self.edge_attrs = {
            "force_density": torch.empty((0), dtype=torch.float),
        }



    #???which inputs do we want specifically? uniform height or more customizable? same for x,y of support?
    def generate_merge_with_index(
        self,
        n,
        grid_res,
        support_height,
        force_density,
        force_density_at_boundary
    ):
        
        data = StructData(
            node_attrs=self.node_attrs,
            edge_attrs=self.edge_attrs
        )


        #higher force densities at boundary
        def force_densities(u_ind, v_ind):

            res =  torch.ones_like(u_ind, dtype=torch.float) * force_density

            res[(u_ind == grid_res-1) | (v_ind ==grid_res-1)] *= force_density_at_boundary / force_density
            
            return res
        
        
        edge_attrs = {
            "force_density": force_densities,
   
        }


        def load(v):
            
            return torch.hstack([torch.zeros_like(v), torch.zeros_like(v), -torch.ones_like(v)])            


        for i in range(n):

            def coords(u,v):

                angle = i/n * 2 * math.pi

                x_coord = u * math.cos(angle) - v * math.sin(angle) 
                y_coord = u * math.sin(angle) + v * math.cos(angle) 
                                
                z_coord = torch.zeros_like(u)
                z_coord[(u == grid_res-1) & (v == grid_res-1)] += support_height[i]

                return torch.hstack([x_coord, y_coord, z_coord])            


            def merge_group_id(u,v):

                res = - torch.ones_like(u)

                res[v == 0] = (torch.arange(grid_res) + grid_res * i) % (grid_res * n)                
                res[u == 0] = (torch.arange(grid_res) + grid_res * ((i + 1) % n)) % (grid_res * n)

                #centroid
                res[(u == 0) & (v == 0)] = grid_res * n

                return res
            

            def support(u,v):
                
                res = torch.zeros_like(u, dtype=torch.bool)

                res[(u == grid_res-1) & (v == grid_res-1)] = torch.ones(1, dtype=torch.bool)

                return res
            

            node_attrs = {
                "coords": coords,
                "merge_group_id": merge_group_id,
                "load": load,
                "is_support": support
            }
            
            data.add_grid(grid_res, grid_res, node_attrs=node_attrs, edge_attrs=edge_attrs)

        data.merge()
        data = data.fdm()

        return data
    

    def generate(
        self,
        n,
        grid_res,
        support_height,
        force_density,
        force_density_at_boundary
    ):
        
        data = StructData(
            node_attrs=self.node_attrs,
            edge_attrs=self.edge_attrs
        )


        #higher force densities at boundary
        def force_densities(u_ind, v_ind):

            res =  torch.ones_like(u_ind, dtype=torch.float) * force_density

            res[(u_ind == grid_res-1) | (v_ind ==grid_res-1)] *= force_density_at_boundary / force_density
            
            return res
        
        
        edge_attrs = {
            "force_density": force_densities,
   
        }


        def load(v):
            
            return torch.hstack([torch.zeros_like(v), torch.zeros_like(v), -torch.ones_like(v)])            


        for i in range(n):

            def coords(u, v):

                theta_i = (i / n) * 2 * math.pi
                theta_j = ((i + 1) / n) * 2 * math.pi

                a0 = math.cos(theta_i)
                a1 = math.sin(theta_i)

                b0 = math.cos(theta_j)
                b1 = math.sin(theta_j)

                x_coord = u * a0 + v * b0
                y_coord = u * a1 + v * b1

                z_coord = torch.zeros_like(u)
                is_corner = (u == grid_res - 1) & (v == grid_res - 1)
                z_coord = z_coord.masked_fill(is_corner, support_height[i])

                return torch.hstack([x_coord, y_coord, z_coord])
            

            def support(u,v):
                
                res = torch.zeros_like(u, dtype=torch.bool)

                res[(u == grid_res-1) & (v == grid_res-1)] = torch.ones(1, dtype=torch.bool)

                return res
            

            node_attrs = {
                "coords": coords,
                "load": load,
                "is_support": support
            }
            
            data.add_grid(grid_res, grid_res, node_attrs=node_attrs, edge_attrs=edge_attrs)


        data.merge_based_on_attributes("coords")
        data = data.fdm()

        return data
    
    
    def sample_input(

        self,
        n = None,
        grid_res = None,
        support_height = None,
        force_density = None,
        force_density_at_boundary = None

    ):

        if n == None:
            n = 4
        if grid_res == None:
            grid_res = np.random.randint(4,6)
        if support_height == None:
            support_height = torch.randint(0, 8, (n,), dtype=torch.long)
        if force_density == None:
            force_density = 20.0
        if force_density_at_boundary == None:
            force_density_at_boundary = 50.0

        input = {

            "n": n,
            "grid_res": grid_res,
            "support_height": support_height,
            "force_density": force_density,
            "force_density_at_boundary": force_density_at_boundary

        }

        return input