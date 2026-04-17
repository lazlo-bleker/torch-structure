import torch
import torch_structure as ts
from torch_structure.data import StructData
from optimizer import (
    Optimizer,
    DesignVariableConfig,
    SolverConfig,
    ObjectiveConfig,
    ConstraintConfig,
)
from obj_functions.orthogonal import orthogonal_func, orthogonal_cache
from obj_functions.self_supporting import supporting_loss_func, supporting_loss_cache
from utils import deviation_force_function, trail_length_function, origin_node_function


def main():
    # 1. Generate structure
    n_trails = 13
    n_rings = 8
    n_nodes = n_trails * n_rings
    data_generator = ts.generators.DomeAssemblyGenerator(
        n_trails=n_trails,
        n_rings=n_rings,
        deviation_force_function=deviation_force_function,
        trail_length_function=trail_length_function,
        origin_node_function=origin_node_function,
    )
    data = data_generator()

    # 2. Define optimization variables
    eps = 1e-1
    dv_config_list = [
        DesignVariableConfig(
            name="trail_lengths",
            attr_name="length",
            mask_keyword="trail_elements",
            is_dual_edge=True,
            lower_bound=eps,
        ),
        DesignVariableConfig(
            name="deviation_forces",
            attr_name="force",
            mask_keyword="deviation_elements",
            is_dual_edge=True,
        ),
    ]

    # 3. Define objective function
    obj_func_config_list = [
        ObjectiveConfig(
            name="length_similarity",
            obj_function=supporting_loss_func,
            weight=1e0,
            kwargs=supporting_loss_cache(data),
        ),
    ]

    # 4. Define constraint
    def fixed_support_func(graph_solved: StructData, target_mask, target_coords):
        # Compare current coords with target coords
        diff = graph_solved.coords[target_mask] - target_coords
        return torch.sum(diff * diff)

    data.cem(inplace=True)
    target_mask = data.is_support.squeeze()
    target_coords = data.coords[target_mask].detach().clone()
    constr_config_list = [
        ConstraintConfig(
            name="support_coords",
            constr_function=lambda g: fixed_support_func(g, target_mask, target_coords),
            lower_bound=0.0,
            upper_bound=1e-3,
        )
    ]

    # 5. Initialize & run optimizer
    solver_config = SolverConfig(
        solver_name="cem", solver_kwargs={"max_iter": 10 * n_nodes}
    )

    optimizer = Optimizer(
        graph=data,
        solver_config=solver_config,
        dv_config_list=dv_config_list,
        obj_func_config_list=obj_func_config_list,
        constr_config_list=constr_config_list,
    )

    optimizer.run(50)

    print("Finish")


if __name__ == "__main__":
    main()
