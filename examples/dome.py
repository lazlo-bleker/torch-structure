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
from obj_functions.self_supporting import cache_supporting_loss, supporting_loss, plot_supporting_loss

def load_path_func(graph_solved: StructData):
    lengths = graph_solved.length_from_coords
    forces = graph_solved.force
    return torch.sum(torch.abs(lengths * forces))


def orthogonal_func(graph_solved: "StructData", quad_indices):
    coords = graph_solved.coords
    x_qni = coords[quad_indices]
    length_q0 = torch.linalg.norm(x_qni[:, 0] - x_qni[:, 2], dim=1)
    length_q1 = torch.linalg.norm(x_qni[:, 1] - x_qni[:, 3], dim=1)
    diff = length_q0 - length_q1
    return torch.sum(diff * diff)


def cache_quad(graph_solved: StructData):
    kwargs = {}
    nodes_uv = graph_solved.uv_coords
    nu = int(max(nodes_uv[:, 0]) + 1)
    nv = int(max(nodes_uv[:, 1]) + 1)
    n_elements = (nu) * (nv - 1)
    quad_indices = -torch.ones([n_elements, 4], dtype=torch.int)
    for v in range(nv - 1):
        for u in range(nu):
            k = u * (nv - 1) + v
            quad_indices[k, 0] = int(((u) % nu) * (nv) + (v))
            quad_indices[k, 1] = int(((u) % nu) * (nv) + (v + 1))
            quad_indices[k, 2] = int(((u + 1) % nu) * (nv) + (v + 1))
            quad_indices[k, 3] = int(((u + 1) % nu) * (nv) + (v))
    kwargs["quad_indices"] = quad_indices
    return kwargs


def deviation_force_function(n_trails: int, n_rings: int) -> torch.Tensor:
    u = torch.linspace(-torch.pi, torch.pi, n_trails)
    v = torch.linspace(-torch.pi, torch.pi, n_rings)
    forces_u = 1 + 0.015 * torch.cos(2 * u)
    forces_v = torch.cos(v)
    return 0.4 * torch.outer(forces_u, forces_v)


def trail_length_function(n_trails: int, n_rings: int) -> torch.Tensor:
    u = torch.linspace(-torch.pi, torch.pi, n_trails)
    v = torch.linspace(-torch.pi, torch.pi, n_rings)
    lentghts_u = 1 + 0.05 * torch.cos(2 * u)
    lentghts_v = torch.ones_like(v)
    return 3.0 / (n_rings - 1) * torch.outer(lentghts_u, lentghts_v)


def main():
    # 1. Generate structure
    n_trails = 12
    n_rings = 20
    n_nodes = n_trails * n_rings
    data_generator = ts.generators.DomeUVGenerator(
        n_trails=n_trails,
        n_rings=n_rings,
        deviation_force_function=deviation_force_function,
        trail_length_function=trail_length_function,
    )
    data = data_generator()

    # DEV: plot self-supporting loss
    data.cem(inplace=True)
    kwargs = cache_supporting_loss(data)
    loss, step_losses = supporting_loss(data, **kwargs)
    import matplotlib.pyplot as plt
    plt.plot(step_losses)
    plt.title("Aux. Force Loss Throughout Assembly")
    plt.xlabel("Assembly Step")
    plt.ylabel("Aux. Force Norm")
    plt.savefig("./img/plot.png")
    plt.close()
    plot_supporting_loss(data, step_losses, **kwargs)

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
    # kwargs = cache_quad(data)
    obj_func_config_list = [
        # ObjectiveConfig(
        #     name="length_similarity",
        #     obj_function=orthogonal_func,
        #     weight=1e0,
        #     kwargs=kwargs
        # ),
    ]

    # 4. Define constraint
    def fixed_support_func(graph_solved: StructData, target_mask, target_coords):
        # Compare current coords with target coords
        diff = graph_solved.coords[target_mask] - target_coords
        return torch.sum(diff * diff)

    target_mask = data.is_support.squeeze()
    data.cem(inplace=True)
    target_coords = data.coords[target_mask].detach().clone()

    t = torch.linspace(-torch.pi, torch.pi, n_trails + 1)[:-1]
    target_coords[:, 0] *= 2.0
    target_coords[:, 1] *= 2.0
    target_coords[:, 2] += 0.25 * torch.cos(2 * t)

    constr_config_list = [
        ConstraintConfig(
            name="support_coords",
            constr_function=lambda g: fixed_support_func(g, target_mask, target_coords),
            lower_bound=0.0,
            upper_bound=1e-3,
        )
    ]

    # # Pre-Visualize result
    # data.plot(
    #     title="Optimized Trail",
    #     legend=False,
    #     show_supports=True,
    #     show_load=True,
    #     force_scale=0.1,
    # )
    # plt.show()

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
    optimizer.run(500)

    # # 6. Visualize result
    # data.plot(
    #     title="Optimized Trail",
    #     legend=False,
    #     show_supports=True,
    #     show_load=True,
    #     force_scale=0.1,
    # )
    # plt.show()

    print("Finish")


if __name__ == "__main__":
    main()
