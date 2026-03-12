"""
Example: Optimize a single trail sequence using CEM and scipy

This is a minimal example of optimization with torch_structure
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
    ConstraintConfig,
)


def main():
    # 1. Generate structure
    n_nodes = 10
    trail_generator = ts.generators.SingleTrailGenerator(
        n_nodes=n_nodes,
        trail_element_length=0.1,
        nodal_load=[0.0, 0.0, -0.1],
        origin_node_load=[1.0, 0.0, 0.0],
    )
    trail = trail_generator()

    # 2. Define optimization variables
    eps = 1e-3
    dv_config_list = [
        DesignVariableConfig(
            name="trail_lengths",
            attr_name="length",
            mask_keyword="trail_elements",
            is_dual_edge=True,
            lower_bound=eps,
        )
    ]

    # 3. Define objective function
    def obj_func(graph_solved: StructData, target_mask, target_coords):
        # Compare current coords with target coords
        diff = graph_solved.coords[target_mask] - target_coords
        return torch.sum(diff * diff)

    def reg_func(graph_solved: StructData):
        lengths = graph_solved.length_from_coords
        mean_length = torch.mean(lengths)
        diff = lengths - mean_length
        return torch.sum(diff * diff)

    target_coords = torch.tensor([[1.0, 0.0, -1.0]])
    target_mask = trail.is_support.squeeze()
    assert trail.coords[target_mask].shape == target_coords.shape, (
        "Target and mask do not match their shape"
    )

    obj_func_config_list = [
        ObjectiveConfig(name="length_similarity", obj_function=reg_func, weight=1e1),
    ]

    # 4. Define constraint
    constr_config_list = [
        ConstraintConfig(
            name="support_coords",
            constr_function=lambda g: obj_func(g, target_mask, target_coords),
            lower_bound=0.0,
            upper_bound=1e-1,
        )
    ]

    # 5. Initialize & run optimizer
    solver_config = SolverConfig(
        solver_name="cem", solver_kwargs={"max_iter": 10 * n_nodes}
    )

    optimizer = Optimizer(
        graph=trail,
        solver_config=solver_config,
        dv_config_list=dv_config_list,
        obj_func_config_list=obj_func_config_list,
        constr_config_list=constr_config_list,
    )
    optimizer.run(100)

    # 6. Visualize result
    trail.plot(
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
