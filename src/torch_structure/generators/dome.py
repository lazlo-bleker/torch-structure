import torch
import math

from torch_structure.data import Data


class Dome:
    def __init__(
        self,
        n_trails,
        n_rings,
        trail_length,
        center_deviation_force,
        opening=False,
    ):
        # Topological parameters
        self.n_trails = n_trails
        self.n_rings = n_rings
        self.opening = opening

        # Metric parameters
        self.trail_length = trail_length
        self.center_deviation_force = center_deviation_force

        if self.n_trails % 2 != 0:
            raise ValueError("Number of trails must be even.")

        self.generate_graph()
        self.graph = self.graph.mpcem()
        if not self.opening:
            self.fix_graph()

    def generate_graph(self):
        # Initialize data object
        node_attrs = {
            "coords": torch.empty((0, 3), dtype=torch.float),
            "load": torch.empty((0, 3), dtype=torch.float),
            "support_condition": torch.empty((0, 3), dtype=torch.long),
            "is_origin_node": torch.empty((0, 1), dtype=torch.bool),
            "sequence": torch.empty((0, 1), dtype=torch.long),
        }
        edge_attrs = {
            "force": torch.empty((0, 1), dtype=torch.float),
            "length": torch.empty((0, 1), dtype=torch.float),
            "is_trail_edge": torch.empty((0, 1), dtype=torch.bool),
            "force_sign": torch.empty((0, 1), dtype=torch.float),
        }
        default_attrs = {
            "force": torch.tensor([torch.nan]),
            "length": torch.tensor([torch.nan]),
            "coords": torch.full((3,), torch.nan),
            "load": torch.zeros(3, dtype=torch.float),
            "support_condition": torch.zeros(3, dtype=torch.bool),
            "is_origin_node": torch.tensor(0, dtype=torch.bool),
        }
        self.graph = Data(
            node_attrs=node_attrs, edge_attrs=edge_attrs, default_attrs=default_attrs
        )

        # Create topology diagram
        centroid = torch.tensor([0.0, 0.0, 0.0])
        angles = torch.linspace(0, 2 * math.pi, self.n_trails + 1)[:-1]

        # Generate trails
        for i in range(self.n_trails):
            angle = angles[i]
            if self.opening:
                origin_diameter = self.trail_length
                x = torch.cos(angle) * 0.5 * origin_diameter
                y = torch.sin(angle) * 0.5 * origin_diameter
                origin_coords = centroid + torch.tensor([x, y, 0.0])
                origin_load = torch.tensor([0.0, 0.0, -1.0])
            else:
                origin_coords = centroid
                x_load = torch.cos(angle) * self.center_deviation_force
                y_load = torch.sin(angle) * self.center_deviation_force
                origin_load = torch.tensor([x_load, y_load, -1.0 / self.n_trails])
            self.generate_trail(
                origin_coords=origin_coords, origin_load=origin_load, id=i
            )

        # Add ring deviations
        for i in range(1, self.n_rings):
            force_sign = torch.randint(2, (1,)) * 2 - 1
            force_magnitude = torch.rand(1) * 2 + 1
            force = force_sign * force_magnitude
            for j in range(self.n_trails):
                self.graph.add_edge(
                    f"trail_{j}_node_{i}",
                    f"trail_{(j + 1) % self.n_trails}_node_{i}",
                    is_trail_edge=torch.tensor(False),
                    force=force,
                )

        if self.opening:
            ring_force = self.center_deviation_force * origin_diameter * 50
            for i in range(self.n_trails):
                self.graph.add_edge(
                    f"trail_{i}_node_0",
                    f"trail_{(i + 1) % self.n_trails}_node_0",
                    is_trail_edge=torch.tensor(False),
                    force=ring_force,
                )

    def generate_trail(self, origin_coords, origin_load, id):
        self.graph.add_node(
            f"trail_{id}_node_0",
            coords=origin_coords,
            is_origin_node=torch.tensor(True),
            sequence=torch.tensor(0),
            load=origin_load,
        )
        for i in range(1, self.n_rings + 1):
            support_condition = (
                torch.tensor([True, True, True])
                if i == self.n_rings
                else torch.tensor([False, False, False])
            )
            self.graph.add_node(
                f"trail_{id}_node_{i}",
                is_origin_node=torch.tensor(False),
                sequence=torch.tensor(i),
                load=torch.tensor([0.0, 0.0, -1.0]),
                support_condition=support_condition,
            )
            self.graph.add_edge(
                f"trail_{id}_node_{i - 1}",
                f"trail_{id}_node_{i}",
                is_trail_edge=torch.tensor(True),
                length=self.trail_length,
                force_sign=torch.tensor(-1.0),
            )

    def fix_graph(self):
        self.graph.add_node(
            "centroid",
            coords=torch.tensor([0.0, 0.0, 0.0]),
            load=torch.tensor([0.0, 0.0, -1.0]),
        )
        merge_nodes = [f"trail_{i}_node_0" for i in range(self.n_trails)]
        self.graph.merge_nodes("centroid", merge_nodes)
