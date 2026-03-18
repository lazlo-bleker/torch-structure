"""
Example: Optimize Cable Forces in a Cable-Stayed Bridge using CEM and scipy

This example demonstrates how to use torch_structure to optimize the force distribution
in a cable-stayed bridge so that the bridge deck is as flat as possible.
"""

import torch
import matplotlib.pyplot as plt
import torch_structure as ts
from torch_structure.data import StructData
from optimizer import (
    Optimizer,
    DesignVariableConfig,
    SolverConfig,
    ObjectiveConfig,
)


def main():
    # 1. Generate structure
    bridge_generator = ts.generators.CableStayedBridge(
        n_towers=5,  # Number of vertical towers
        n_cables=5,  # Number of cables on each side of a tower
        deck_trail_length=1.0,  # Spacing between deck nodes
        tower_trail_length=0.3,  # Spacing between tower nodes
        center_deviation_force=torch.tensor(
            -10.0
        ),  # Axial force halfway along the deck
        cable_deviation_force=torch.tensor(1.0),  # Tension in cables
        deck_load=torch.tensor([0.0, 0.0, -1.0]),  # Downward gravity load on the deck
        tower_height=5.0,  # Height of each tower
        tower_offset=1.0,  # Horizontal offset from the deck to the tower peaks
        back_stay_offset=5.0,  # Horizontal offset from the tower peaks to the back stay
        back_stay_force=torch.tensor(10.0),  # Force in back stays
    )
    data = bridge_generator()

    # 2. Define optimization variables
    # Exclude deviation edge in the center
    edge_to_exclude = "deck_trail_0_node_0-deck_trail_1_node_0"
    edge_to_exclude = data.metadata["edge_name_to_index"][edge_to_exclude]
    data.active_edof[edge_to_exclude] = False
    dv_config_list = [
        DesignVariableConfig(
            name="deviation_forces",
            attr_name="force",
            mask_keyword="deviation_elements",
            is_dual_edge=True,
        )
    ]

    # 3. Define objective function
    deck_node_indices = [
        i
        for name, i in data.metadata["node_name_to_index"].items()
        if "deck_trail" in name
    ]
    deck_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    deck_mask[deck_node_indices] = True

    def deck_flatness(graph_solved: StructData, deck_mask):
        # Compute mean square Z deviation (measure of flatness)
        deck_z = graph_solved.coords[deck_mask][:, 2]
        return torch.mean(deck_z**2)

    obj_func_config_list = [
        ObjectiveConfig(
            name="deck_flatness", obj_function=lambda x: deck_flatness(x, deck_mask)
        )
    ]

    # 4. Define constraint (not needed)

    # 5. Initialize & run optimizer
    solver_config = SolverConfig(
        solver_name="cem", solver_kwargs={"max_iter": 10 * data.num_nodes}
    )

    optimizer = Optimizer(
        graph=data,
        solver_config=solver_config,
        dv_config_list=dv_config_list,
        obj_func_config_list=obj_func_config_list,
    )
    optimizer.run(100)

    # 6. Visualize result
    data.plot(
        title="Optimized Trail",
        legend=False,
        show_supports=True,
        show_load=True,
        force_scale=0.1,
    )
    plt.show()
    print("Finish")


if __name__ == "__main__":
    main()
