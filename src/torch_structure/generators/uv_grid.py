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
            "coords": torch.empty((0, 3), dtype=torch.float64),
            "uv_coords": torch.empty((0, 2), dtype=torch.float64),
            "load": torch.empty((0, 3), dtype=torch.float64),
            "is_support": torch.empty((0, 1), dtype=torch.bool),
            "is_origin_node": torch.empty((0, 1), dtype=torch.bool),
            "sequence": torch.empty((0, 1), dtype=torch.long),
            "active_ndof" : torch.empty((0, 1), dtype=torch.bool),
            "loss" : torch.empty((0, 1), dtype=torch.float64),
            "res": torch.empty((0, 3), dtype=torch.float64),
        }
        self.edge_attrs = {
            "force": torch.empty((0, 1), dtype=torch.float64),
            "length": torch.empty((0, 1), dtype=torch.float64),
            "is_trail_edge": torch.empty((0, 1), dtype=torch.bool),
            "force_sign": torch.empty((0, 1), dtype=torch.float64),
            "active_edof" : torch.empty((0, 1), dtype=torch.bool),
        }
        self.default_attrs = {
            "force": torch.tensor([torch.nan]),
            "length": torch.tensor([torch.nan]),
            "coords": torch.full((3,), torch.nan),
            "load": torch.zeros(3, dtype=torch.float64),
            "res": torch.zeros(3, dtype=torch.float64),
            "is_support": torch.tensor(0, dtype=torch.bool),
            "is_origin_node": torch.tensor(0, dtype=torch.bool),
            "active_edof" : torch.tensor(1, dtype=torch.bool),
            "active_ndof" : torch.tensor(0, dtype=torch.bool),
            "loss" : torch.tensor(0., dtype=torch.float64),
        }

    def sample_input(
        self,
        origin_nodes : torch.tensor = torch.tensor([]),
        nu : int = 10,
        nv : int = 10,
        default_length : float = 1.0,
        default_magnitude : float = -.0,
        default_load : torch.tensor = torch.tensor([0.0, 0.0, -1.0])
    ):

        input = {
            "origin_nodes" : origin_nodes,
            "nu": nu,
            "nv": nv,
            "default_length": default_length,
            "default_magnitude": default_magnitude,
            "default_load" : default_load
        }

        return input

    def generate(
        self,
        origin_nodes,
        nu, 
        nv, 
        default_length,
        default_magnitude,
        default_load
    ) -> StructData:
        graph = StructData(
            node_attrs=self.node_attrs,
            edge_attrs=self.edge_attrs,
            default_attrs=self.default_attrs,
        )
        # Generate origin nodes
        _v = 0
        for _u in range(nu):
            xyz_coords = origin_nodes[_u]
            graph.add_node(
                f"u_{_u}_v_{_v}",
                coords = xyz_coords,
                load = default_load,
                is_origin_node = torch.tensor(True), 
                is_support = torch.tensor(False), 
                sequence = torch.tensor(_v),
                uv_coords = torch.tensor([_u, _v]),
            )

        # Generate inner nodes
        for _v in range(1, nv-1):
            for _u in range(nu):
                graph.add_node(
                    f"u_{_u}_v_{_v}",
                    # No self-weight, only load on top
                    load = .0 * default_load,
                    is_support = torch.tensor(False), 
                    is_origin_node = torch.tensor(False), 
                    sequence = torch.tensor(_v),
                    uv_coords = torch.tensor([_u, _v])
                )

        # Generate support nodes
        _v = nv-1
        for _u in range(nu):
            graph.add_node(
                f"u_{_u}_v_{_v}",
                is_support = torch.tensor(True), 
                is_origin_node = torch.tensor(False), 
                sequence = torch.tensor(_v),
                uv_coords = torch.tensor([_u, _v])
            )

        # Generate top deviation edges (zero force and not dof)
        _v = 0
        for _u in range(nu-1):
                graph.add_edge(
                f"u_{_u}_v_{_v}",
                f"u_{_u+1}_v_{_v}",
                is_trail_edge=torch.tensor(False),
                # Init deviation edge to default of zero and keep like that
                force = torch.tensor(0.0),
                active_edof = torch.tensor(False)
            )
        # Generate deviation edges
        for _v in range(1,nv-1):
            for _u in range(nu-1):
                # Generate deviation edge (u,v) --> (u+1,v)
                graph.add_edge(
                    f"u_{_u}_v_{_v}",
                    f"u_{_u+1}_v_{_v}",
                    is_trail_edge=torch.tensor(False),
                    force=default_magnitude,
                )
        # Generate bottom deviation edges
        _v = nv-1
        for _u in range(nu-1):
            # Generate deviation edge (u,v) --> (u+1,v)
            graph.add_edge(
                f"u_{_u}_v_{_v}",
                f"u_{_u+1}_v_{_v}",
                is_trail_edge=torch.tensor(False),
                force = torch.tensor(0.0),
                active_edof = torch.tensor(False)
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
