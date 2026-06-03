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

            fd =  torch.ones_like(u_ind, dtype=torch.float) * force_density

            cablenet_boundary = (u_ind == grid_res-1) | (v_ind == grid_res-1)
            fd[cablenet_boundary] *= force_density_at_boundary / force_density
            
            return fd
        
        
        edge_attrs = {
            "force_density": force_densities,
        }
 

         # create a tensor of shape (1,3) for a constant load that is later automatically applied to all nodes.
        load = torch.tensor([[0, 0, -1]])   


        # add one support of the cablenet for each grid
        def support(u,v):
            
            support = torch.zeros_like(u, dtype=torch.bool)

            cablenet_support = (u == grid_res-1) & (v == grid_res-1)
            support[cablenet_support] = torch.ones(1, dtype=torch.bool)

            return support
            

        #create n grids that are merged together to form a cablenet
        for i in range(n):


            # interpret u and v indices of grid as coordinates and rotate them such that the supports of all grids form a regular n-gon (One corner of each grid is taken as a support of the cablenet).
            def coords(u,v):
                
                angle = i/n * 2 * math.pi
                                
                x_coord = u * math.cos(angle) - v * math.sin(angle) 
                y_coord = u * math.sin(angle) + v * math.cos(angle) 
                                
                z_coord = torch.zeros_like(u)

                cablenet_support = (u == grid_res-1) & (v == grid_res-1)
                z_coord[cablenet_support] += support_height[i]

                return torch.hstack([x_coord, y_coord, z_coord])            


            #assign identical merge ids to grid nodes that should be merged together.
            def merge_group_id(u,v):

                id = - torch.ones_like(u)

                #assign merge ids for two neighbouring sides of the grid. The [u == 0] side of grid i will be merged with the [v == 0] side of grid (i + 1) % n
                id[v == 0] = (torch.arange(grid_res) + grid_res * i) % (grid_res * n)                
                id[u == 0] = (torch.arange(grid_res) + grid_res * ((i + 1) % n)) % (grid_res * n)

                #assign the same merge id for all corners that will form the centroid of the cable net (The opposite corner of the support corner of each grid will be the centroid).
                centroid = (u == 0) & (v == 0)
                id[centroid] = grid_res * n

                return id

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

            fd =  torch.ones_like(u_ind, dtype=torch.float) * force_density

            cablenet_boundary = (u_ind == grid_res-1) | (v_ind == grid_res-1)
            fd[cablenet_boundary] *= force_density_at_boundary / force_density  

            return fd
        
        
        edge_attrs = {
            "force_density": force_densities,
   
        }

        # create a tensor of shape (1,3) for a constant load that is later automatically applied to all nodes.
        load = torch.tensor([[0, 0, -1]])            


        # add one support of the cablenet for each grid
        def support(u,v):
            
            support = torch.zeros_like(u, dtype=torch.bool)

            cablenet_support = (u == grid_res-1) & (v == grid_res-1)
            support[cablenet_support] = torch.ones(1, dtype=torch.bool)

            return support
        

        for i in range(n):

            # Merge will be based on geometrically identical nodes. Thus interpret u and v indices of grid as coordinates and transform them to a wedge s.t. sides of neighbouring grids align.
            def coords(u, v):
                
                # create angles that are i/n-th and (i+1)%n)/n-th fraction of 360°.
                theta_i = (i / n) * 2 * math.pi
                theta_j = ((i + 1) / n) * 2 * math.pi

                # create direction vectors (a0, a1) and (b0, b1) on the unit cycle
                a0 = math.cos(theta_i)
                a1 = math.sin(theta_i)

                b0 = math.cos(theta_j)
                b1 = math.sin(theta_j)
                

                # create a wedge-like grid through expressing u,v in terms of the new basis (a0, a1) and (b0, b1)
                x_coord = u * a0 + v * b0
                y_coord = u * a1 + v * b1

                
                z_coord = torch.zeros_like(u)

                # set support height
                corner = (u == grid_res - 1) & (v == grid_res - 1)
                z_coord[corner] = support_height[i]

                return torch.hstack([x_coord, y_coord, z_coord])
            

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