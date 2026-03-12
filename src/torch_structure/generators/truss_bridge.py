import torch
import math
import numpy as np

from torch_structure.data import StructData
from torch_structure.generators.base_generator import BaseGenerator, InvalidSampleError
from torch_structure.formfinding import create_branch_node_matrix


class TrussBridgeGenerator(BaseGenerator):
    def __init__(self, analysis=True, **overrides):
        super().__init__(**overrides)
        self.max_attempts = 100
        self.analysis = analysis

        self.node_attrs = {
            "coords": torch.empty((0, 3), dtype=torch.float),
            "load": torch.empty((0, 3), dtype=torch.float),
            "support_condition": torch.empty((0, 3), dtype=torch.bool),
            "sequence": torch.empty((0, 1), dtype=torch.long),
            "is_deck_node": torch.empty((0, 1), dtype=torch.bool),
            "is_left_side_node": torch.empty((0, 1), dtype=torch.bool),
            "is_right_side_node": torch.empty((0, 1), dtype=torch.bool),
            "is_center_node": torch.empty((0, 1), dtype=torch.bool),
        }
        self.default_attrs = {
            "coords": torch.full((3,), torch.nan),
            "load": torch.zeros(3, dtype=torch.float),
            "support_condition": torch.zeros(3, dtype=torch.bool),
            "is_deck_node": torch.tensor(0, dtype=torch.bool),
            "is_left_side_node": torch.tensor(0, dtype=torch.bool),
            "is_right_side_node": torch.tensor(0, dtype=torch.bool),
            "is_center_node": torch.tensor(0, dtype=torch.bool),
        }
        self.cat_dict = {
            0: {"generator": {0: "truss", 1: "arch_susp"}},
            # if truss
            1: {"truss_type": {0: "pratt", 1: "howe", 2: "parker"}},
            2: {"triangle": {0: False, 1: True}},
            # if not triangle
            3: {"deck_truss": {0: False, 1: True}},
            4: {"inclined": {0: False, 1: True}},
            # endif not triangle
            5: {"n_bays_truss": {i: i for i in range(30)}},
            # endif truss
            # if arch_susp
            6: {
                "n_cables": {
                    0: 1,
                    1: 2,
                }
            },
            7: {"n_bays_arch_susp": {i: i for i in range(30)}},
            # if 2 cables
            8: {"connected_cables": {0: False, 1: True}},
        }

    def validate_input(
        self,
        span,
        deck_truss,
        triangle,
        n_bays,
        truss_height,
        deck_width,
        inclined_truss,
        truss_inclination,
        truss_type,
    ):
        if truss_type not in ["pratt", "howe", "parker"]:
            raise ValueError("Truss type must be either 'pratt', 'howe', or 'parker'.")

    def sample_input(
        self,
        span=None,
        deck_truss=None,
        triangle=None,
        n_bays=None,
        truss_height=None,
        deck_width=None,
        inclined_truss=None,
        truss_inclination=None,
        truss_type=None,
    ):
        if span is None:
            span = np.random.uniform(40.0, 80.0)
        if deck_truss is None:
            deck_truss = np.random.choice([True, False], p=[0.5, 0.5])
        if triangle is None:
            if deck_truss:
                triangle = np.random.choice([True, False], p=[0.3, 0.7])
            else:
                triangle = False
        if deck_width is None:
            deck_width = np.random.uniform(2.0, 5.0)
        if truss_height is None:
            truss_height = np.random.uniform(0.1, 0.2) * span * (deck_width / 5)
        if n_bays is None:
            n_bays = (
                np.random.randint(
                    int(span / truss_height + 0.5) - 1,
                    int(span / truss_height + 0.5) + 5,
                )
                // 2
            )
        if inclined_truss is None:
            if triangle:
                inclined_truss = False
            else:
                inclined_truss = np.random.choice([True, False], p=[0.7, 0.3])
        if truss_inclination is None:
            truss_inclination = np.random.uniform(-5.0, 30.0) if inclined_truss else 0.0
        if truss_type is None:
            truss_type = np.random.choice(
                ["pratt", "howe", "parker"], p=[1 / 3, 1 / 3, 1 / 3]
            )

        return {
            "span": span,
            "deck_truss": deck_truss,
            "triangle": triangle,
            "n_bays": n_bays,
            "truss_height": truss_height,
            "deck_width": deck_width,
            "inclined_truss": inclined_truss,
            "truss_inclination": truss_inclination,
            "truss_type": truss_type,
        }

    def generate(
        self,
        span,
        deck_truss,
        triangle,
        n_bays,
        truss_height,
        deck_width,
        inclined_truss,
        truss_inclination,
        truss_type,
    ):
        # Initialize data object
        data = StructData(
            node_attrs=self.node_attrs,
            default_attrs=self.default_attrs,
        )

        # Global parameters
        bay_size = 0.5 * span / n_bays
        line_load = 0.5
        load_mag = line_load * span / (2 * (2 * n_bays + 1))
        load = torch.tensor([0.0, 0.0, -load_mag], dtype=torch.float)
        if deck_truss:
            chord_z = -truss_height
        else:
            chord_z = truss_height
        if inclined_truss:
            chord_y_offset = truss_height * math.tan(math.radians(truss_inclination))
        else:
            chord_y_offset = 0.0
        # center nodes
        deck_center_coords = torch.tensor(
            [
                [0.0, 0.5 * deck_width, 0.0],
                [0.0, -0.5 * deck_width, 0.0],
            ],
            dtype=torch.float,
        )
        profile_side = ["right", "left"]

        if triangle:
            chord_center_coords = torch.tensor(
                [
                    [0.0, 0.0, chord_z],
                ],
                dtype=torch.float,
            )
            profile_side.append("center")
        else:
            chord_center_coords = torch.tensor(
                [
                    [0.0, 0.5 * deck_width + chord_y_offset, chord_z],
                    [0.0, -0.5 * deck_width - chord_y_offset, chord_z],
                ],
                dtype=torch.float,
            )
            profile_side.extend(["right", "left"])

        center_coords = torch.cat([deck_center_coords, chord_center_coords], dim=0)

        # chords
        for i, center_coord in enumerate(center_coords):
            is_deck = i < 2
            data.add_node(
                f"trail_{i}_node_0",
                coords=center_coord,
                load=load if is_deck else torch.tensor([0.0, 0.0, 0.0]),
                sequence=torch.tensor([0], dtype=torch.long),
                is_deck_node=torch.tensor(is_deck),
                is_left_side_node=torch.tensor(profile_side[i] == "left"),
                is_right_side_node=torch.tensor(profile_side[i] == "right"),
                is_center_node=torch.tensor(profile_side[i] == "center"),
            )
            for side in [-1, 1]:
                for j in range(n_bays):
                    if not is_deck and j == n_bays - 1:
                        if truss_type == "parker":
                            continue
                        elif truss_type == "pratt" and deck_truss:
                            continue
                        elif truss_type == "howe" and not deck_truss:
                            continue
                    coord = center_coord + torch.tensor(
                        [side * (j + 1) * bay_size, 0.0, 0.0]
                    )
                    if truss_type == "parker" and not is_deck:
                        z_shift = math.copysign(
                            3 * truss_height / span**2 * coord[0] ** 2, chord_z
                        )
                        coord[2] -= z_shift
                        if inclined_truss and not triangle:
                            factor = 1 if profile_side[i] == "right" else -1
                            coord[1] -= (
                                factor
                                * z_shift
                                * math.tan(math.radians(truss_inclination))
                            )

                    if j == n_bays - 1 and is_deck:
                        y_fixed = profile_side[i] == "left" and side == 1
                        y_fixed = True
                        support_condition = (
                            torch.tensor([True, y_fixed, True])
                            if side == -1
                            else torch.tensor([False, y_fixed, True])
                        )
                    else:
                        support_condition = torch.tensor([False, False, False])
                    data.add_node(
                        f"trail_{i}_node_{j + 1}_side_{side}",
                        load=load if is_deck else torch.tensor([0.0, 0.0, 0.0]),
                        support_condition=support_condition,
                        coords=coord,
                        sequence=torch.tensor([j + 1], dtype=torch.long),
                        is_deck_node=torch.tensor(is_deck),
                        is_left_side_node=torch.tensor(profile_side[i] == "left"),
                        is_right_side_node=torch.tensor(profile_side[i] == "right"),
                        is_center_node=torch.tensor(profile_side[i] == "center"),
                    )
                    # Trail edges
                    data.add_edge(
                        f"trail_{i}_node_{j}_side_{side}"
                        if j > 0
                        else f"trail_{i}_node_0",
                        f"trail_{i}_node_{j + 1}_side_{side}",
                    )

        # diagonals and verticals
        if triangle:
            chord_pairs = [(0, 2), (1, 2)]
        else:
            chord_pairs = [(0, 2), (1, 3)]

        for pair in chord_pairs:
            for side in [-1, 1]:
                for j in range(n_bays):
                    index_0 = j
                    index_1 = j + 1
                    if deck_truss:
                        index_0, index_1 = index_1, index_0
                    if truss_type in ["pratt", "parker"]:
                        if (
                            truss_type == "parker"
                            and not deck_truss
                            and j == n_bays - 1
                        ):
                            data.add_edge(
                                f"trail_{pair[0]}_node_{index_1}_side_{side}"
                                if index_1 > 0
                                else f"trail_{pair[0]}_node_0",
                                f"trail_{pair[1]}_node_{index_0}_side_{side}"
                                if index_0 > 0
                                else f"trail_{pair[1]}_node_0",
                            )
                        else:
                            data.add_edge(
                                f"trail_{pair[0]}_node_{index_0}_side_{side}"
                                if index_0 > 0
                                else f"trail_{pair[0]}_node_0",
                                f"trail_{pair[1]}_node_{index_1}_side_{side}"
                                if index_1 > 0
                                else f"trail_{pair[1]}_node_0",
                            )
                    if truss_type == "howe":
                        data.add_edge(
                            f"trail_{pair[0]}_node_{index_1}_side_{side}"
                            if index_1 > 0
                            else f"trail_{pair[0]}_node_0",
                            f"trail_{pair[1]}_node_{index_0}_side_{side}"
                            if index_0 > 0
                            else f"trail_{pair[1]}_node_0",
                        )
                    if j == n_bays - 1:
                        if truss_type == "parker":
                            continue
                        elif truss_type == "pratt" and deck_truss:
                            continue
                        elif truss_type == "howe" and not deck_truss:
                            continue
                    data.add_edge(
                        f"trail_{pair[0]}_node_{j + 1}_side_{side}",
                        f"trail_{pair[1]}_node_{j + 1}_side_{side}",
                    )
            data.add_edge(
                f"trail_{pair[0]}_node_0",
                f"trail_{pair[1]}_node_0",
            )

        if inclined_truss or triangle:
            for side in [-1, 1]:
                for i in range(n_bays - 1):
                    # Inter-deck edges
                    data.add_edge(
                        f"trail_0_node_{i + 1}_side_{side}",
                        f"trail_1_node_{i + 1}_side_{side}",
                    )
                data.add_edge(
                    "trail_0_node_0",
                    "trail_1_node_0",
                )

        # Analysis
        if self.analysis:
            support = data.support_condition
            load = data.load

            coords = torch.clone(data.coords)  # check if this is necessary
            C = create_branch_node_matrix(data.directed_edge_index)

            CxT = C[:, ~support[:, 0]].T  # (n_free_x, E)
            CyT = C[:, ~support[:, 1]].T  # (n_free_y, E)
            CzT = C[:, ~support[:, 2]].T  # (n_free_z, E)

            u = torch.mv(C, coords[:, 0])
            v = torch.mv(C, coords[:, 1])
            w = torch.mv(C, coords[:, 2])
            U = torch.diag(u)
            V = torch.diag(v)
            W = torch.diag(w)

            A = torch.vstack((CxT @ U, CyT @ V, CzT @ W))

            b = torch.hstack(
                (
                    load[~support[:, 0], 0],
                    load[~support[:, 1], 1],
                    load[~support[:, 2], 2],
                )
            )

            force_density = torch.linalg.lstsq(A, b).solution

            data.force_density = data.edge_attr_to_undirected(
                force_density.view(-1, 1), mask=data.directed_mask
            )
            data.force = data.force_density * data.length_from_coords
            if not data.verify_equilibrium():
                raise InvalidSampleError("Equilibrium not found.")

            prev_coords = torch.clone(data.coords)
            data = data.fdm()
            if (data.coords - prev_coords).abs().max() > 1e-4:
                # print("Inconsistent geometry detected.")
                # data.plot()
                # data.plot(show=True, force=None, coords=prev_coords)
                raise InvalidSampleError("Inconsistent geometry.")

        # Text labels
        typology = "deck_truss" if deck_truss else "through_truss"
        text_label_dict = {
            "typology": typology,
            "truss_type": str(truss_type),
            "triangle": bool(triangle),
            "n_bays": 2 * n_bays,
            "inclined": bool(inclined_truss),
            "inclination": truss_inclination,
            "span": span,
            "deck_width": deck_width,
        }

        data.topology_params = torch.tensor(
            [
                0,
                {"pratt": 0, "howe": 1, "parker": 2}[truss_type],
                int(triangle),
                int(deck_truss) if not triangle else -100,
                int(inclined_truss) if not triangle else -100,
                n_bays,
                -100,
                -100,
                -100,
            ],
            dtype=torch.long,
        ).view(1, -1)

        return data, text_label_dict
