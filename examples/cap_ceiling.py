from torch_structure.generators import CapCeilingAssemblyGenerator
import numpy as np

from pathlib import Path
import json, os
from post.core import post_process, PostProcessConfig
from obj_functions.self_supporting import graph_post_process as graph_post_process
from post.core import post_process, PostProcessConfig
from post.custom import export_assembly_states_img


def func_surf(u, v):
    coords_hat = np.zeros([u.shape[0], u.shape[1], 3])
    coords_hat[:, :, 0] = u
    coords_hat[:, :, 1] = v
    coords_hat[:, :, 2] = -(u**2)
    return coords_hat


for i in range(10):
    generator = CapCeilingAssemblyGenerator(
        n_u=40,
        n_v=40,
        support_sides=[True, True, True, True],
        surface_function=func_surf,
        fd_init=-4.0,
        fd_boundary_init=-1.0,
        seed=1,
        force_per_area=10,
    )
    data = generator()
    data = data.fdm()
    data.plot()

    export_dir = Path(f"./results/rs_{i:03d}")
    print(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    data_log = data.to_log()
    json_path = os.path.join(export_dir, f"state_0.json")
    with open(json_path, "w") as f:
        json.dump(data_log, f, indent=2)

    config = PostProcessConfig(
        dir=export_dir,
        export_img=export_assembly_states_img,
        post_process_func=graph_post_process,
    )
    post_process(config)
