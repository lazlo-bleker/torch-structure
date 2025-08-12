import torch
import math
import numpy as np

from torch_structure.data import StructData
from torch_structure.generators.base_generator import BaseGenerator


class UVGridGenerator(BaseGenerator):
    def __init__(self, **overrides):
        super().__init__(**overrides)
        self.max_attempts = 1
        self.node_attrs = {
            "coords": torch.empty((0, 3), dtype=torch.float),
            "uv_coords": torch.empty((0, 2), dtype=torch.float),
            "load": torch.empty((0, 3), dtype=torch.float),
            "support_condition": torch.empty((0, 3), dtype=torch.long),
            "is_origin_node": torch.empty((0, 1), dtype=torch.bool),
            "sequence": torch.empty((0, 1), dtype=torch.long),
        }
        self.edge_attrs = {
            "force": torch.empty((0, 1), dtype=torch.float),
            "length": torch.empty((0, 1), dtype=torch.float),
            "is_trail_edge": torch.empty((0, 1), dtype=torch.bool),
            "force_sign": torch.empty((0, 1), dtype=torch.float),
            "active_dof" : torch.empty((0, 1), dtype=torch.bool),
        }
        self.default_attrs = {
            "force": torch.tensor([torch.nan]),
            "length": torch.tensor([torch.nan]),
            "coords": torch.full((3,), torch.nan),
            "load": torch.zeros(3, dtype=torch.float),
            "support_condition": torch.zeros(3, dtype=torch.bool),
            "is_origin_node": torch.tensor(0, dtype=torch.bool),
            "active_dof" : torch.tensor(1, dtype=torch.bool),
        }

    def sample_input(
        self,
        nu : int = 10,
        nv : int = 10,
    ):
    
        default_length : float = 1.0 / (nv-1),
        default_magnitude : float = -.0,

        input = {
            "nu": nu,
            "nv": nv,
            "default_length": default_length,
            "default_magnitude": default_magnitude,
        }

        return input

    def generate(
        self,
        nu, 
        nv, 
        default_length,
        default_magnitude,
    ) -> StructData:
        graph = StructData(
            node_attrs=self.node_attrs,
            edge_attrs=self.edge_attrs,
            default_attrs=self.default_attrs,
        )

        # Grid variables
        nw_point = torch.tensor([0.0, 0.0, 1.0])
        ne_point = torch.tensor([1.0, 0.0, 1.0])
        default_load = torch.tensor([0.0, 0.0, -1.0])
        
        # Generate origin nodes
        _v = 0
        for _u in range(nu):
            omega_u = _u/(nu-1)
            xyz_coords = nw_point * (1-omega_u) + ne_point * omega_u
            graph.add_node(
                f"u_{_u}_v_{_v}",
                coords = xyz_coords,
                load = default_load,
                is_origin_node = torch.tensor(True), 
                sequence = torch.tensor(_v),
                uv_coords = torch.tensor([_u, _v])
            )

        # Generate inner nodes
        for _v in range(1, nv-1):
            for _u in range(nu):
                graph.add_node(
                    f"u_{_u}_v_{_v}",
                    load = default_load,
                    support_condition = torch.tensor([False, False, False]), 
                    is_origin_node = torch.tensor(False), 
                    sequence = torch.tensor(_v),
                    uv_coords = torch.tensor([_u, _v])
                )

        # Generate support nodes
        _v = nv-1
        for _u in range(nu):
            graph.add_node(
                f"u_{_u}_v_{_v}",
                load = default_load,
                support_condition = torch.tensor([True, True, True]), 
                is_origin_node = torch.tensor(False), 
                sequence = torch.tensor(_v),
                uv_coords = torch.tensor([_u, _v])
            )

        # Generate deviation edges
        for _v in range(nv):
            graph.add_edge(
                f"u_{0}_v_{_v}",
                f"u_{1}_v_{_v}",
                is_trail_edge=torch.tensor(False),
                force=default_magnitude,
                # active_dof=torch.tensor(False)
            )
            graph.add_edge(
                f"u_{nu-2}_v_{_v}",
                f"u_{nu-1}_v_{_v}",
                is_trail_edge=torch.tensor(False),
                force=default_magnitude,
                # active_dof=torch.tensor(False)
            )
            for _u in range(1,nu-2):
                # Generate deviation edge (u,v) --> (u+1,v)
                graph.add_edge(
                    f"u_{_u}_v_{_v}",
                    f"u_{_u+1}_v_{_v}",
                    is_trail_edge=torch.tensor(False),
                    force=default_magnitude,
                )
        
        # Generate trail edges
        for _v in range(nv-1):
            for _u in range(nu):
                # Generate trail edge (u,v) --> (u,v+1)
                graph.add_edge(
                    f"u_{_u}_v_{_v}",
                    f"u_{_u}_v_{_v+1}",
                    is_trail_edge=torch.tensor(True),
                    length=default_length,
                    force_sign=torch.tensor(-1.0),
                )

        return graph
