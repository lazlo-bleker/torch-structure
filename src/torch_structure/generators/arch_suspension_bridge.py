import torch
import math
import numpy as np
from torch_scatter import scatter
from functools import partial

from torch_structure.data import StructData
from torch_structure.generators.base_generator import BaseGenerator
from torch_structure.formfinding.cem import constrained_deck_cb
from torch_structure.geometry.utils import line_direction


class ArchSuspensionBridgeGenerator(BaseGenerator):
    def __init__(self, **overrides):
        super().__init__(**overrides)
        self.max_attempts = 1000

        self.node_attrs = {
            "coords": torch.empty((0, 3), dtype=torch.float),
            "load": torch.empty((0, 3), dtype=torch.float),
            "support_condition": torch.empty((0, 3), dtype=torch.long),
            "is_origin_node": torch.empty((0, 1), dtype=torch.bool),
            "constraint_plane": torch.empty((0, 6), dtype=torch.float),
            "sequence": torch.empty((0, 1), dtype=torch.long),
            "deck_slope": torch.empty((0, 1), dtype=torch.float),
            "is_deck_node": torch.empty((0, 1), dtype=torch.bool),
            "is_left_side_node": torch.empty((0, 1), dtype=torch.bool),
            "is_right_side_node": torch.empty((0, 1), dtype=torch.bool),
            "is_center_node": torch.empty((0, 1), dtype=torch.bool),
        }
        self.edge_attrs = {
            "force": torch.empty((0, 1), dtype=torch.float),
            "length": torch.empty((0, 1), dtype=torch.float),
            "is_trail_edge": torch.empty((0, 1), dtype=torch.bool),
            "force_sign": torch.empty((0, 1), dtype=torch.float),
            "is_mod_v": torch.empty((0, 1), dtype=torch.bool),
            "is_mod_h": torch.empty((0, 1), dtype=torch.bool),
            "is_deck_trail_edge": torch.empty((0, 1), dtype=torch.bool),
            "is_center_deviation_edge": torch.empty((0, 1), dtype=torch.bool),
        }
        self.default_attrs = {
            "force": torch.tensor([torch.nan]),
            "length": torch.tensor([torch.nan]),
            "coords": torch.full((3,), torch.nan),
            "load": torch.zeros(3, dtype=torch.float),
            "support_condition": torch.zeros(3, dtype=torch.bool),
            "is_origin_node": torch.tensor(0, dtype=torch.bool),
            "constraint_plane": torch.full((6,), torch.nan),
            "is_mod_v": torch.tensor(0, dtype=torch.bool),
            "is_mod_h": torch.tensor(0, dtype=torch.bool),
            "is_deck_trail_edge": torch.tensor(0, dtype=torch.bool),
            "is_center_deviation_edge": torch.tensor(0, dtype=torch.bool),
            "is_deck_node": torch.tensor(0, dtype=torch.bool),
            "is_left_side_node": torch.tensor(0, dtype=torch.bool),
            "is_right_side_node": torch.tensor(0, dtype=torch.bool),
            "is_center_node": torch.tensor(0, dtype=torch.bool),
        }

    def validate_input(
        self,
        span,
        n_cables,
        n_bays,
        midspan_height,
        deck_width,
        inter_cable_distance,
        twist,
        connected_cables,
        inclined_cables,
        deck_force,
        cable_force,
        inter_cable_force,
        deck_rise,
    ):
        if n_cables not in [1,  2]:
            raise ValueError("Number of cables must be either 1 or 2.")

    def sample_input(
        self,
        span=None,
        n_cables=None,
        n_bays=None,
        midspan_height=None,
        deck_width=None,
        inter_cable_distance=None,
        twist=None,
        connected_cables=None,
        inclined_cables=None,
        deck_force=None,
        cable_force=None,
        inter_cable_force=None,
        deck_rise=None,
    ):
        if cable_force is None:
            cable_force_sign = np.random.choice([-1, 1])
            cable_force = cable_force_sign * np.random.uniform(3.0, 15.0)
        if span is None:
            span = np.random.uniform(40.0, 100.0)
        if n_cables is None:
            n_cables = np.random.choice([1, 2], p=[0.5, 0.5])
        if n_bays is None:
            n_bays = np.random.randint(math.ceil(0.5 * span / 7), math.floor(0.5 * span / 3) + 1)
        if midspan_height is None:
            midspan_height = span * np.sign(cable_force) * np.random.uniform(-0.2, 0.05)
        if deck_width is None:
            deck_width = np.random.uniform(2.0, 5.0)
        if twist is None:
            twisted = np.random.choice([False, True], p=[0.7, 0.3])
            twist = twisted * np.random.uniform(5.0, 20.0)
        if connected_cables is None:
            connected_cables = np.random.choice([False, True], p=[0.3, 0.7]) if not twisted else 1
        if inclined_cables is None:
            if n_cables == 1:
                inclined_cables = False
            else:
                inclined_cables = np.random.choice([False, True], p=[0.3, 0.7]) if not twisted and not connected_cables else 1
        if inter_cable_distance is None:
            inter_cable_distance = np.random.uniform(3.0, 20.0) if inclined_cables else deck_width
        if deck_force is None:
            deck_force = np.random.uniform(-3.0, -1.5)
        if inter_cable_force is None:
            inter_cable_force = np.random.uniform(-1.0, 1.0)
        if deck_rise is None:
            deck_rise = np.random.choice([0.0, np.random.uniform(2.0, 12.0)])

        return {
            "span": span,
            "n_cables": n_cables,
            "n_bays": n_bays,
            "midspan_height": midspan_height,
            "deck_width": deck_width,
            "inter_cable_distance": inter_cable_distance,
            "twist": twist,
            "connected_cables": connected_cables,
            "inclined_cables": inclined_cables,
            "deck_force": deck_force,
            "cable_force": cable_force,
            "inter_cable_force": inter_cable_force,
            "deck_rise": deck_rise,
        }

    def generate(
        self,
        span,
        n_cables,
        n_bays,
        midspan_height,
        deck_width,
        inter_cable_distance,
        twist,
        connected_cables,
        inclined_cables,
        deck_force,
        cable_force,
        inter_cable_force,
        deck_rise,
    ):
        # Initialize data object
        data = StructData(
            node_attrs=self.node_attrs,
            edge_attrs=self.edge_attrs,
            default_attrs=self.default_attrs,
        )

        # Global parameters
        bay_size = 0.5 * span / (n_bays + 0.5)
        line_load = 0.5
        load_mag = line_load * span / (2 * (2 * n_bays + 2))
        load = torch.tensor([0.0, 0.0, -load_mag], dtype=torch.float)
        twist_rad = math.radians(twist)
        twist_offset = 0.5 * bay_size * math.tan(twist_rad)

        # Origin nodes
        deck_origin_coords = torch.tensor([
            [0.5*bay_size, 0.5*deck_width, 0.0],
            [0.5*bay_size, -0.5*deck_width, 0.0],
            [-0.5*bay_size, 0.5*deck_width, 0.0],
            [-0.5*bay_size, -0.5*deck_width, 0.0],
        ], dtype=torch.float)
        profile_side = ["right", "left", "right", "left"]

        if n_cables == 1:
            cable_origin_coords = torch.tensor([
                [0.5*bay_size, twist_offset, midspan_height],
                [-0.5*bay_size, -twist_offset, midspan_height],
            ], dtype=torch.float)
            profile_side += ["center", "center"]
        elif n_cables == 2:
            cable_origin_coords = torch.tensor([
                [0.5*bay_size, 0.5*inter_cable_distance + twist_offset, midspan_height],
                [0.5*bay_size, -0.5*inter_cable_distance + twist_offset, midspan_height],
                [-0.5*bay_size, 0.5*inter_cable_distance - twist_offset, midspan_height],
                [-0.5*bay_size, -0.5*inter_cable_distance - twist_offset, midspan_height],
            ], dtype=torch.float)
            profile_side += ["right", "left", "right", "left"]
    
        origin_coords = torch.cat([deck_origin_coords, cable_origin_coords], dim=0)

        for i, origin_coord in enumerate(origin_coords):
            side = 1 if origin_coord[0] > 0 else -1
            is_deck = i < 4
            data.add_node(
                f"trail_{i}_node_0",
                coords=origin_coord,
                is_origin_node=torch.tensor(True),
                load=load if is_deck else torch.tensor([0.0, 0.0, 0.0]),
                sequence=torch.tensor([0], dtype=torch.long),
                deck_slope=torch.tensor(side*8*deck_rise/span * 1 / (n_bays + 0.5)),
                is_deck_node=torch.tensor(is_deck),
                is_left_side_node=torch.tensor(profile_side[i] == "left"),
                is_right_side_node=torch.tensor(profile_side[i] == "right"),
                is_center_node=torch.tensor(profile_side[i] == "center"),
            )
            for j in range(n_bays):
                data.add_node(
                    f"trail_{i}_node_{j+1}",
                    load=load if is_deck else torch.tensor([0.0, 0.0, 0.0]),
                    support_condition=torch.tensor([True, True, True]) if j == n_bays - 1 else torch.tensor([False, False, False]),
                    constraint_plane=torch.tensor([side*(j + 1.5)*bay_size, 0.0, 0.0, 1.0, 0.0, 0.0]),
                    sequence=torch.tensor([j + 1], dtype=torch.long),
                    deck_slope=torch.tensor(side*8*deck_rise/span * (j + 2) / (n_bays + 0.5)),
                    is_deck_node=torch.tensor(is_deck),
                    is_left_side_node=torch.tensor(profile_side[i] == "left"),
                    is_right_side_node=torch.tensor(profile_side[i] == "right"),
                    is_center_node=torch.tensor(profile_side[i] == "center"),
                )
                # Trail edges
                data.add_edge(
                    f"trail_{i}_node_{j}",
                    f"trail_{i}_node_{j+1}",
                    is_trail_edge=torch.tensor(True),
                    is_deck_trail_edge=torch.tensor(is_deck),
                )

        # Deviation edges
        center_deck_pairs = [(0, 2), (1, 3)]
        inter_deck_pairs = [(0, 1), (2, 3)]
        if n_cables == 1:
            center_cable_pairs = [(4, 5)]
            inter_cable_pairs = []
            deck_cable_pairs = [(0, 4), (1, 4), (2, 5), (3, 5)]
        elif n_cables == 2:
            center_cable_pairs = [(4, 6), (5, 7)]
            inter_cable_pairs = [(4, 5), (6, 7)]
            deck_cable_pairs = [(0, 4), (1, 5), (2, 6), (3, 7)]

        # Center deck edges
        for pair in center_deck_pairs:
            data.add_edge(
                f"trail_{pair[0]}_node_0",
                f"trail_{pair[1]}_node_0",
                is_trail_edge=torch.tensor(False),
                force=deck_force,
                is_center_deviation_edge=torch.tensor(True),
            )

        # Center cable edges
        for pair in center_cable_pairs:
            data.add_edge(
                f"trail_{pair[0]}_node_0",
                f"trail_{pair[1]}_node_0",
                is_trail_edge=torch.tensor(False),
                force=cable_force,
                is_center_deviation_edge=torch.tensor(True),
            )

        temp_force = 0.0
        for i in range(n_bays):
            # Inter-deck edges
            for pair in inter_deck_pairs:
                data.add_edge(
                    f"trail_{pair[0]}_node_{i}",
                    f"trail_{pair[1]}_node_{i}",
                    is_trail_edge=torch.tensor(False),
                    is_mod_h=torch.tensor(True),
                    force=temp_force,
                )

            # Deck-cable edges
            for pair in deck_cable_pairs:
                data.add_edge(
                    f"trail_{pair[0]}_node_{i}",
                    f"trail_{pair[1]}_node_{i}",
                    is_trail_edge=torch.tensor(False),
                    is_mod_v=torch.tensor(True),
                    force=temp_force,
                )

            # Inter-cable edges
            if connected_cables:
                for pair in inter_cable_pairs:
                    data.add_edge(
                        f"trail_{pair[0]}_node_{i}",
                        f"trail_{pair[1]}_node_{i}",
                        is_trail_edge=torch.tensor(False),
                        force=inter_cable_force,
                    )

        # Formfinding
        callback = partial(constrained_deck_cb,
                           cem_edge_index=data.cem_edge_index,
                           is_mod_v=data.is_mod_v[data.cem_edge_mask],
                           is_mod_h=data.is_mod_h[data.cem_edge_mask],
                           is_deck_trail_edge=data.is_deck_trail_edge[data.cem_edge_mask],
                           sequence=data.sequence,
                           deck_slope=data.deck_slope)
        data = data.mpcem(callback=callback, verbose=False)

        # bbox filter
        _, y_extent, z_extent = (data.bbox[1] - data.bbox[0]).unbind(dim=0)
        y_factor = 0.8
        z_factor = 0.5
        if y_extent > span * y_factor:
            # print("bbox (wide)")
            raise ValueError(f"Bridge geometry is too wide ({y_extent.item()} > {span * y_factor}).")
        if z_extent > span * z_factor:
            # print("bbox (tall)")
            raise ValueError(f"Bridge geometry is too tall ({z_extent.item()} > {span * z_factor}).")
        
        # # smoothness filter
        max_angle = 45  # degrees

        edge_mask = (data.is_trail_edge.view(-1) | data.is_center_deviation_edge.view(-1))
        src, dst = data.edge_index[:, edge_mask]
        unit_vector = line_direction(data.coords[src], data.coords[dst])
        vector_sum = scatter(unit_vector, src, dim=0, reduce="sum", dim_size=data.num_nodes)
        degree = scatter(torch.ones_like(src), src, dim=0, reduce="sum", dim_size=data.num_nodes)
        degree_2_mask = degree == 2

        R2 = vector_sum[degree_2_mask].pow(2).sum(dim=1)
        cos_theta = R2 * 0.5 - 1
        cos_theta = cos_theta.clamp(-1.0, 1.0)
        cos_min = torch.cos(torch.deg2rad(torch.tensor(180 - max_angle, dtype=torch.float)))
        is_invalid = cos_theta > cos_min
        if is_invalid.any():
            # print("Invalid nodes:", is_invalid.sum())
            # print("vector sums:", vector_sum[degree_2_mask][is_invalid])
            # print("smoothness")
            raise ValueError(f"Trails have too sharp angles (>{max_angle}°).")

        # # force density filter
        data.force_density = data.force / data.length_from_coords
        abs_force_density = data.force_density.abs()
        # if abs_force_density[~edge_mask].max() > abs_force_density[edge_mask].min():
        #     # print("force density")
        #     raise ValueError("Some secondary edges have higher force density magnitude than deck/main cable edges.")
        
        # # force density magnitude filter
        mag_threshold = 30
        if abs_force_density.max() > mag_threshold:
            raise ValueError(f"Some edges have too high force density magnitude (>{mag_threshold}).")
        return data
