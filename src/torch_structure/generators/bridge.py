# ToDo: Refactor to inherit from BaseGenerator
# Todo: Update outdated data structure

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import math
import torch_geometric
from torch_geometric.data import Data
import torch

from torch_structure.generators import Structure


class Bridge(Structure):
    """Legacy dict-based generator for arch, suspension, and truss bridges built from mirrored half-spans.

    Unlike the CEM-based generators, force densities are prescribed
    directly per element type rather than solved via form-finding.
    """

    param_dtypes = {
        "span": float,
        "n_cables": int,
        "n_bays": int,
        "midspan_height": float,
        "deck_width": float,
        "twist": float,
        "cable_distance": float,
        "deck_rise": float,
        "deck_force": float,
        "cable_force": float,
        "brace_force": float,
        "inter_cable_force": float,
    }

    def __init__(self, **kwargs):
        """
        Args:
            span (float): Approximate span of the bridge.
            n_cables (int): Number of cables (or arches). Should be equal to 1 or 2.
            n_bays (int): Number of bays on each half of the bridge span.
            midspan_height (float): Vertical distance between the cables (or arches) and the deck at midspan.
            deck_width (float): Horizontal width of the deck.
            twist (float): Twist angle (in degrees) of the cables(s) at midspan.
            cable_distance (float): Horizontal distance between the cables when n_cables = 2.
            deck_rise (float): Rise of the deck as a fraction of the span.
            deck_force (float): Force in the longitudinal deck elements at midspan.
            cable_force (float): Force in the longitudinal cable elements at midspan.
            brace_force (float): Force in the diagonal brace elements.
            inter_cable_force (float): Force in the elements connecting the cables when n_cable = 2.
        """

        # Check validity of input arguments
        for param, value in kwargs.items():
            if param in self.param_dtypes and not np.issubdtype(
                type(value), self.param_dtypes[param]
            ):
                raise TypeError(
                    f"Expected type {self.param_dtypes[param]} for parameter '{param}', got {type(value)}"
                )
            elif param not in self.param_dtypes:
                raise ValueError(f"Unexpected parameter: '{param}'")
        self.kwargs = kwargs

        # Sample random (missing) input parameters
        span = self.kwargs["span"] if "span" in self.kwargs else None
        self.params = self.sample_input(span=span)
        self._set_dict_attr(self.params)

        # Derived attributes
        self.bay_size = 0.5 * self.span / self.n_bays
        line_load = 0.5
        point_load = line_load * self.span / (2 * (2 * self.n_bays + 1))
        self.load = np.array((0.0, 0.0, -point_load))

        # Initialize bridge
        self.nodes = {}
        self.edges = {}
        self.init_procedure()

    def init_procedure(self):
        """Procedure run at object initialization"""
        self.formfind()
        self.mirror_structure()
        self.title_color = "black" if self._verify_equilibrium(verbose=False) else "red"

    @staticmethod
    def sample_input(span=None):
        """Sample random input parameters"""
        cable_force_sign = np.random.choice([-1, 1])
        braced = np.random.choice([True, False], p=[0.3, 0.7])
        twisted = np.random.choice([True, False], p=[0.3, 0.7])
        span = np.random.uniform(40, 100) if span is None else span
        params = {
            "span": span,
            "n_cables": np.random.choice([1, 2]),
            "n_bays": np.random.randint(
                math.ceil(0.5 * span / 7), math.floor(0.5 * span / 3) + 1
            ),  # max, min bay_size = 7, 3
            "midspan_height": (
                span * cable_force_sign * np.random.uniform(-0.3, -0.1)
                if braced
                else span * cable_force_sign * np.random.uniform(-0.2, 0.05)
            ),
            "deck_width": np.random.uniform(2.0, 5.0),
            "twist": np.random.uniform(5.0, 20.0) if twisted and not braced else 0.0,
            "cable_distance": np.random.uniform(3.0, 20.0),
            "deck_rise": 0.0
            if braced
            else np.random.choice([0.0, np.random.uniform(0.02, 0.1)], p=[0.7, 0.3]),
            "deck_force": np.random.uniform(-1.5, -3.0),
            "cable_force": cable_force_sign * np.random.uniform(3.0, 15.0),
            "brace_force": 1.0 if braced else 0.0,
            "inter_cable_force": 0.0
            if braced
            else np.random.choice([0, np.random.uniform(-1.0, 1.0)]),
        }
        return params

    def formfind(self, threshold=5e-4, update_straight_trail_force=True):
        """Main form-finding algorithm"""
        # Initialize first deck nodes
        self._set_node(
            0,
            np.array([0.0, 0.5 * self.deck_width, 0.0]),
            0.5 * self.load,
            external_force=np.array([-self.deck_force, 0.0, 0.0]),
            sem_type="deck",
            origin=True,
        )
        self._set_node(
            1,
            np.array([0.0, -0.5 * self.deck_width, 0.0]),
            0.5 * self.load,
            external_force=np.array([-self.deck_force, 0.0, 0.0]),
            sem_type="deck",
            origin=True,
        )

        # Initialize first cable nodes
        cable_force_vector = np.array([-self.cable_force, 0.0, 0.0])
        twist_radians = np.radians(self.twist)
        R_z = np.array(
            [
                [np.cos(twist_radians), -np.sin(twist_radians), 0],
                [np.sin(twist_radians), np.cos(twist_radians), 0],
                [0, 0, 1],
            ]
        )
        cable_force_vector = R_z.dot(cable_force_vector)

        if self.n_cables == 1:
            self._set_node(
                2,
                np.array([0.0, 0.0, self.midspan_height]),
                np.zeros(3),
                external_force=np.copy(cable_force_vector),
                sem_type="cable",
                origin=True,
            )
        else:
            self._set_node(
                2,
                np.array([0.0, 0.5 * self.cable_distance, self.midspan_height]),
                np.zeros(3),
                external_force=np.copy(cable_force_vector),
                sem_type="cable",
                origin=True,
            )
            self._set_node(
                3,
                np.array([0.0, -0.5 * self.cable_distance, self.midspan_height]),
                np.zeros(3),
                external_force=np.copy(cable_force_vector),
                sem_type="cable",
                origin=True,
            )

        for i in range(self.n_bays):
            # Set node indices
            prev_left_deck_node_idx = i * (2 + self.n_cables)
            prev_right_deck_node_idx = prev_left_deck_node_idx + 1
            prev_left_cable_node_idx = prev_right_deck_node_idx + 1
            next_left_deck_node_idx = prev_left_cable_node_idx + self.n_cables
            next_right_deck_node_idx = next_left_deck_node_idx + 1
            next_left_cable_node_idx = next_right_deck_node_idx + 1
            if self.n_cables == 2:
                prev_right_cable_node_idx = prev_left_cable_node_idx + 1
                next_right_cable_node_idx = next_left_cable_node_idx + 1

            # Determine vertical node offset
            z_offset = (2 * i + 1) * self.deck_rise * self.span / self.n_bays**2

            # Determine hangar forces
            left_hangar_force = self._hangar_force(
                prev_left_deck_node_idx, prev_left_cable_node_idx, z_offset=z_offset
            )
            self._set_edge(
                (prev_left_deck_node_idx, prev_left_cable_node_idx),
                left_hangar_force,
                "deviation",
                "hangar",
            )
            cable_node_idx = (
                prev_left_cable_node_idx
                if self.n_cables == 1
                else prev_right_cable_node_idx
            )
            right_hangar_force = self._hangar_force(
                prev_right_deck_node_idx, cable_node_idx, z_offset=z_offset
            )
            self._set_edge(
                (prev_right_deck_node_idx, cable_node_idx),
                right_hangar_force,
                "deviation",
                "hangar",
            )

            # Set inter-deck edge
            inter_deck_force = self._inter_deck_force(
                prev_left_deck_node_idx, prev_right_deck_node_idx
            )
            self._set_edge(
                (prev_left_deck_node_idx, prev_right_deck_node_idx),
                inter_deck_force,
                "deviation",
                "inter-deck",
            )

            # Determine next deck nodes and edges
            next_left_deck_coords, left_deck_force = self._next_node(
                prev_left_deck_node_idx, self.bay_size
            )
            self._set_node(
                next_left_deck_node_idx,
                next_left_deck_coords,
                self.load,
                sem_type="deck",
            )
            self._set_edge(
                (prev_left_deck_node_idx, next_left_deck_node_idx),
                left_deck_force,
                "trail",
                "deck",
            )
            next_right_deck_coords, right_deck_force = self._next_node(
                prev_right_deck_node_idx, self.bay_size
            )
            self._set_node(
                next_right_deck_node_idx,
                next_right_deck_coords,
                self.load,
                sem_type="deck",
            )
            self._set_edge(
                (prev_right_deck_node_idx, next_right_deck_node_idx),
                right_deck_force,
                "trail",
                "deck",
            )

            # Determine brace forces
            if self.brace_force != 0.0:
                brace_factor = self.brace_force / (-self.n_cables + 3)
                left_brace_force = self._brace_force(
                    prev_left_cable_node_idx,
                    next_left_deck_node_idx,
                    factor=brace_factor,
                )
                right_brace_force = self._brace_force(
                    cable_node_idx, next_right_deck_node_idx, factor=brace_factor
                )
                self._set_edge(
                    (prev_left_cable_node_idx, next_left_deck_node_idx),
                    left_brace_force,
                    "deviation",
                    "brace",
                )
                self._set_edge(
                    (cable_node_idx, next_right_deck_node_idx),
                    right_brace_force,
                    "deviation",
                    "brace",
                )

            # Set inter-cable edges
            if self.n_cables == 2 and self.inter_cable_force != 0.0:
                self._set_edge(
                    (prev_left_cable_node_idx, prev_right_cable_node_idx),
                    self.inter_cable_force,
                    "deviation",
                    "inter-cable",
                )

            # Determine next cable node and edge
            next_left_cable_coords, cable_force = self._next_node(
                prev_left_cable_node_idx, self.bay_size
            )
            self._set_node(
                next_left_cable_node_idx,
                next_left_cable_coords,
                np.zeros(3),
                sem_type="cable",
            )
            self._set_edge(
                (prev_left_cable_node_idx, next_left_cable_node_idx),
                cable_force,
                "trail",
                "cable",
            )
            if self.n_cables == 2:
                next_right_cable_coords, cable_force = self._next_node(
                    prev_right_cable_node_idx, self.bay_size
                )
                self._set_node(
                    next_right_cable_node_idx,
                    next_right_cable_coords,
                    np.zeros(3),
                    sem_type="cable",
                )
                self._set_edge(
                    (prev_right_cable_node_idx, next_right_cable_node_idx),
                    cable_force,
                    "trail",
                    "cable",
                )

        # Add reaction force as external force
        support_nodes = [
            next_left_deck_node_idx,
            next_right_deck_node_idx,
            next_left_cable_node_idx,
        ]
        if self.n_cables == 2:
            support_nodes.append(next_right_cable_node_idx)
        for node_idx in support_nodes:
            resultant = self._resultant(node_idx)
            self.nodes[node_idx]["external_force"] = -resultant
            self.nodes[node_idx]["support"] = True

        # Update straight trail forces for horizontal reaction forces to be zero
        if update_straight_trail_force:
            # Update deck forces and corresponding external forces
            deck_straight = (
                1
                - self._force_direction(
                    prev_left_deck_node_idx, next_left_deck_node_idx
                )[0]
                < threshold
            )
            if (
                deck_straight
                and abs(self.nodes[next_left_deck_node_idx]["external_force"][0])
                > threshold
            ):
                deck_force_update = (
                    -self.nodes[next_left_deck_node_idx]["external_force"][0] - 1e-4
                )
                self.params["deck_force"] += deck_force_update
                self.nodes[next_left_deck_node_idx]["external_force"][0] += (
                    deck_force_update
                )
                self.nodes[0]["external_force"][0] -= deck_force_update
                self.nodes[next_right_deck_node_idx]["external_force"][0] += (
                    deck_force_update
                )
                self.nodes[1]["external_force"][0] -= deck_force_update
                for edge, edge_info in self.edges.items():
                    if edge_info["semantic_type"] == "deck":
                        self.edges[edge]["force"] += deck_force_update

            # Update cable forces and corresponding external forces
            cable_straight = (
                1
                - self._force_direction(
                    prev_left_cable_node_idx, next_left_cable_node_idx
                )[0]
                < threshold
            )
            if (
                cable_straight
                and abs(self.nodes[next_left_cable_node_idx]["external_force"][0])
                > threshold
            ):
                cable_force_update = (
                    -self.nodes[next_left_cable_node_idx]["external_force"][0] - 1e-5
                )
                self.params["cable_force"] += cable_force_update
                self.nodes[next_left_cable_node_idx]["external_force"][0] += (
                    cable_force_update
                )
                self.nodes[2]["external_force"][0] -= cable_force_update
                if self.n_cables == 2:
                    self.nodes[next_right_cable_node_idx]["external_force"][0] += (
                        cable_force_update
                    )
                    self.nodes[3]["external_force"][0] -= cable_force_update
                for edge, edge_info in self.edges.items():
                    if edge_info["semantic_type"] == "cable":
                        self.edges[edge]["force"] += cable_force_update

            self._set_dict_attr(self.params)

    def _hangar_force(self, deck_node_idx, cable_node_idx, z_offset=0):
        """Determine hangar force that ensures a next deck node with specified vertical offset"""
        direction = self._force_direction(deck_node_idx, cable_node_idx)
        resultant = self._resultant(deck_node_idx)
        force = -(resultant[2] + z_offset / self.bay_size * resultant[0]) / direction[2]
        return force

    def _inter_deck_force(self, left_deck_node_idx, right_deck_node_idx):
        """Determine inter-deck force that ensures a constant-width deck"""
        left_resultant = self._resultant(left_deck_node_idx)
        right_resultant = self._resultant(right_deck_node_idx)
        x_1, y_1, _ = left_resultant
        x_2, y_2, _ = right_resultant
        force = (x_2 * y_1 - x_1 * y_2) / (x_1 + x_2)
        return force

    def _brace_force(self, cable_node_idx, deck_node_idx, factor):
        """Determine brace force that ensures a flat cable"""
        direction = self._force_direction(cable_node_idx, deck_node_idx)
        resultant = self._resultant(cable_node_idx, ignore_sem_type="cable")
        force = -resultant[2] / direction[2]
        return force * factor

    def _next_node(self, prev_node_idx, bay_size):
        """Determine the next node coordinates and corresponding connecting edge force"""
        coords = self.nodes[prev_node_idx]["coordinates"]
        force_vector = self._resultant(prev_node_idx)
        sign = -np.sign(force_vector[0])
        force = sign * np.linalg.norm(force_vector)
        factor = bay_size / abs(force_vector[0])
        new_coords = -sign * factor * force_vector + coords
        return new_coords, force

    def mirror_structure(self):
        """Mirror structure across the yz plane"""
        mirrored_nodes = {}
        nodes_to_skip = (
            {0: 1, 1: 0, 2: 2} if self.n_cables == 1 else {0: 1, 1: 0, 2: 3, 3: 2}
        )
        new_idx_offset = max(self.nodes.keys()) + 1 - len(nodes_to_skip)

        # Mirror nodes and adjust load for nodes not mirrored
        for node_idx, node_info in self.nodes.items():
            if node_idx in nodes_to_skip:
                if "external_force" in node_info:
                    del node_info["external_force"]
                node_info["load"] = np.array(node_info["load"]) * 2
                continue
            if "external_force" in node_info:
                mirrored_external_force = np.array(node_info["external_force"])
                mirrored_external_force[0] = -mirrored_external_force[0]
                mirrored_external_force[1] = -mirrored_external_force[1]
            mirrored_coords = np.array(node_info["coordinates"])
            mirrored_coords[0] = -mirrored_coords[0]
            mirrored_coords[1] = -mirrored_coords[1]
            new_idx = node_idx + new_idx_offset
            mirrored_nodes[new_idx] = {**node_info, "coordinates": mirrored_coords}
            if "external_force" in node_info:
                mirrored_nodes[new_idx]["external_force"] = mirrored_external_force
        self.nodes.update(mirrored_nodes)

        # Mirror edges and adjust force for edges not mirrored
        mirrored_edges = {}
        for (start_idx, end_idx), edge_info in self.edges.items():
            mirrored_start_idx = (
                nodes_to_skip[start_idx]
                if start_idx in nodes_to_skip
                else start_idx + new_idx_offset
            )
            mirrored_end_idx = (
                nodes_to_skip[end_idx]
                if end_idx in nodes_to_skip
                else end_idx + new_idx_offset
            )
            if not (start_idx in nodes_to_skip and end_idx in nodes_to_skip):
                mirrored_edges[(mirrored_start_idx, mirrored_end_idx)] = edge_info
            else:
                edge_info["force"] *= 2
        self.edges.update(mirrored_edges)

    def create_feature_dict(self):
        """Derive a dict of descriptive features (typology, dimensions, load path, etc.) from this bridge.

        Returns:
            dict: the sampled parameters plus derived features such as
            ``typology``, ``curved_deck``, ``alignment``, ``height``,
            ``load_path``, and member length statistics.
        """
        self.array_output()
        coords = self.array_dict["coordinates_node"]
        # support_node_indices = set(self.array_dict['support_node_indices'][0])
        # origin_node_indices = set(self.array_dict['origin_node_indices'][0])
        # cable_node_indices = set(self.array_dict['cable_node_indices'][0])
        # arch_mid_height = coords[list(cable_node_indices.intersection(origin_node_indices))[0]][2]
        # arch_base_height = coords[list(cable_node_indices.intersection(support_node_indices))[0]][2]
        # arch_rise = arch_mid_height - arch_base_height
        bbox_length, bbox_width, bbox_height = coords.max(axis=0) - coords.min(axis=0)
        feature_dict = self.params.copy()

        # Populate feature dictionary
        if self.brace_force > 1e-4:
            feature_dict["typology"] = "truss bridge"
            feature_dict["number_of_cables"] = 0
            feature_dict["number_of_arches"] = 0
        elif self.cable_force < 0:
            feature_dict["typology"] = "arch bridge"
            feature_dict["number_of_arches"] = self.n_cables
            feature_dict["number_of_cables"] = 0
        elif self.cable_force > 0:
            feature_dict["number_of_cables"] = self.n_cables
            feature_dict["number_of_arches"] = 0
            if self.midspan_height > 0:
                feature_dict["typology"] = "suspension bridge"
        if "typology" not in feature_dict:
            feature_dict["typology"] = "hybrid"

        if self.deck_rise > 1e-4:
            feature_dict["curved_deck"] = True
        else:
            feature_dict["curved_deck"] = False
        if self.twist > 1e-4:
            feature_dict["alignment"] = "twisted"
        else:
            feature_dict["alignment"] = "straight"

        feature_dict["height"] = bbox_height
        feature_dict["load_path"] = self.load_path()
        feature_dict["average_member_length"] = self.average_member_length()
        feature_dict["minimum_member_length"] = self.minimum_member_length()
        feature_dict["maximum_member_length"] = self.maximum_member_length()
        # feature_dict['passed_filter'] = self.passed_filter()

        return feature_dict

    def create_request_dict(self):
        """Return [create_feature_dict][torch_structure.generators.bridge.Bridge.create_feature_dict]'s output, stripped of the underlying sampling parameters."""
        request_dict = self.create_feature_dict()
        # del request_dict['passed_filter']
        del request_dict["n_cables"]
        del request_dict["twist"]
        del request_dict["brace_force"]
        del request_dict["inter_cable_force"]
        del request_dict["cable_force"]
        del request_dict["deck_force"]
        del request_dict["midspan_height"]
        del request_dict["deck_rise"]
        del request_dict["cable_distance"]
        return request_dict

    # def passed_filter(self):
    #     if bridge_filter.maximum_load_path(self, factor=1.0) and bridge_filter.within_bbox(self) and bridge_filter.deck_smoothness(self) and bridge_filter.maximum_force(self) and self._verify_equilibrium():
    #         return True
    #     else:
    #         return False

    def array_output(self):
        """Create edge and node attribute numpy array outputs"""
        self.array_dict = {}

        # Save edge attributes
        self.array_dict["edge_index"] = np.array(list(self.edges.keys()))
        self.array_dict["trail_edge"] = np.array(
            [edge["cem_type"] == "trail" for edge in self.edges.values()]
        )
        self.array_dict["deviation_edge"] = np.array(
            [edge["cem_type"] == "deviation" for edge in self.edges.values()]
        )
        self.array_dict["deck_edge"] = np.array(
            [edge["semantic_type"] == "deck" for edge in self.edges.values()]
        )
        self.array_dict["cable_edge"] = np.array(
            [edge["semantic_type"] == "cable" for edge in self.edges.values()]
        )
        self.array_dict["hangar_edge"] = np.array(
            [edge["semantic_type"] == "hangar" for edge in self.edges.values()]
        )
        self.array_dict["force_edge"] = np.array(
            [edge["force"] for edge in self.edges.values()]
        )
        self.array_dict["length_edge"] = np.array(
            [edge["length"] for edge in self.edges.values()]
        )

        # Save node attributes
        self.array_dict["coordinates_node"] = np.array(
            [node["coordinates"] for node in self.nodes.values()]
        )
        self.array_dict["load_node"] = np.array(
            [node["load"] for node in self.nodes.values()]
        )
        self.array_dict["support_node"] = np.array(
            [node["support"] for node in self.nodes.values()]
        )
        self.array_dict["support_node_indices"] = np.where(
            self.array_dict["support_node"]
        )
        self.array_dict["origin_node"] = np.array(
            [node["origin"] for node in self.nodes.values()]
        )
        self.array_dict["origin_node_indices"] = np.where(
            self.array_dict["origin_node"]
        )
        self.array_dict["deck_node"] = np.array(
            [node["semantic_type"] == "deck" for node in self.nodes.values()]
        )
        self.array_dict["deck_node_indices"] = np.where(self.array_dict["deck_node"])
        self.array_dict["cable_node"] = np.array(
            [node["semantic_type"] == "cable" for node in self.nodes.values()]
        )
        self.array_dict["cable_node_indices"] = np.where(self.array_dict["cable_node"])
        return self.array_dict

    def pyg_data(self):
        """Convert this bridge to a `torch_geometric.data.Data` graph.

        Returns:
            tuple[torch_geometric.data.Data, dict]: the graph, with node
            features ``[x, y, z, x_load, y_load, z_load, support]`` and
            edge feature ``force_density``, and a metadata dict describing
            each feature column.
        """
        self.array_output()
        meta_data = {}

        # Node features
        coordinates_node = torch.tensor(
            self.array_dict["coordinates_node"], dtype=torch.float64
        )
        load_node = torch.tensor(self.array_dict["load_node"], dtype=torch.float64)
        support_node = torch.tensor(
            self.array_dict["support_node"], dtype=torch.float64
        ).unsqueeze(1)
        meta_data["node_features"] = {
            0: {"description": "x_coordinate", "type": "numerical"},
            1: {"description": "y_xoordinate", "type": "numerical"},
            2: {"description": "z_xoordinate", "type": "numerical"},
            3: {"description": "x_load", "type": "numerical"},
            4: {"description": "y_load", "type": "numerical"},
            5: {"description": "z_load", "type": "numerical"},
            6: {
                "description": "support condition (0 = free, 1 = pinned)",
                "type": "categorical",
            },
        }
        node_features = torch.cat([coordinates_node, load_node, support_node], dim=1)

        # Edge features
        force_edge = torch.tensor(
            self.array_dict["force_edge"], dtype=torch.float64
        ).unsqueeze(1)
        length_edge = torch.tensor(
            self.array_dict["length_edge"], dtype=torch.float64
        ).unsqueeze(1)
        force_density_edge = force_edge / length_edge
        meta_data["edge_features"] = {
            # 0: {'description': 'force', 'type': 'numerical'},
            0: {"description": "force_density", "type": "numerical"}
        }
        edge_features = torch.cat([force_density_edge], dim=1)

        # Edge index
        edge_index = (
            torch.tensor(self.array_dict["edge_index"], dtype=torch.long)
            .t()
            .contiguous()
        )

        undirected_edge_index, undirected_edge_features = (
            torch_geometric.utils.to_undirected(edge_index, edge_attr=edge_features)
        )

        # Create PyTorch Geometric data object
        pyg_data = Data(
            x=node_features,
            edge_index=undirected_edge_index,
            edge_attr=undirected_edge_features,
        )

        return pyg_data, meta_data

    def plot(self, ax=None, plot_threshold=1e-4, plot_deck=True, path=None):
        """Plot bridge structure in 3D"""
        show = False
        if ax is None:
            fig = plt.figure(figsize=(20, 20))
            ax = fig.add_subplot(111, projection="3d")
            show = True
        for edge in self.edges:
            force = self.edges[edge]["force"]
            if abs(force) < plot_threshold:
                continue
            color = self.red_color if force >= 0 else self.blue_color
            x_coords = (
                self.nodes[edge[0]]["coordinates"][0],
                self.nodes[edge[1]]["coordinates"][0],
            )
            y_coords = (
                self.nodes[edge[0]]["coordinates"][1],
                self.nodes[edge[1]]["coordinates"][1],
            )
            z_coords = (
                self.nodes[edge[0]]["coordinates"][2],
                self.nodes[edge[1]]["coordinates"][2],
            )
            ax.plot(x_coords, y_coords, z_coords, color=color)

        if plot_deck:
            self.array_output()
            coords = self.array_dict["coordinates_node"][self.array_dict["deck_node"]]
            coords = np.concatenate(
                (
                    coords[0 : (self.n_bays + 1) * 2 : 2],
                    coords[(self.n_bays + 1) * 2 - 1 : 0 : -2],
                    coords[(self.n_bays + 2) * 2 :: 2],
                    coords[-1 : (self.n_bays + 1) * 2 + 1 : -2],
                )
            )
            ax.add_collection3d(Poly3DCollection([coords], color="grey", alpha=0.4))

        ax.set_proj_type("ortho")
        ax.axis("equal")
        ax.set_axis_off()
        load_path = self.load_path()
        ax.set_title(
            f"Span = {self.span:.1f} | LP = {load_path:.1f}", color=self.title_color
        )
        if path:
            plt.savefig(f"{path}.png", bbox_inches="tight")
        elif show:
            plt.show()
