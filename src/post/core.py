from dataclasses import dataclass, field
from typing import Callable, Any
from pathlib import Path
import json, os
import imageio.v2 as imageio
import matplotlib.pyplot as plt

from torch_structure.data import StructData
from torch_geometric.utils import to_networkx

from utils.utils import export_graph_to_vtp


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def identity(x: Any) -> Any:
    return x


def return_none(x: Any) -> Any:
    return None


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
@dataclass
class PostProcessConfig:
    """
    Configuration for post-processing structural simulation results.
    """

    dir: str
    post_process_func: Callable[[StructData], StructData] = identity
    label: str = "export"
    files: list[str] = None
    export_vtk: bool = False
    export_img: bool = False
    export_steps: bool = False
    img_to_gif: bool = True
    edge_attr_list: list[str] = field(default_factory=lambda: ["force", "length"])
    node_attr_list: list[str] = field(default_factory=lambda: ["coords"])


# ---------------------------------------------------------------------
# Main post-processing routine
# ---------------------------------------------------------------------
def post_process_export_opt(config: PostProcessConfig) -> None:
    # --------------------------------------------------------------
    # Validate base directory
    # --------------------------------------------------------------
    base_dir = Path(config.dir)
    print(f"Post-processing files in dir: {base_dir}\n")

    if not base_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {base_dir}")
    if not base_dir.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {base_dir}")

    # --------------------------------------------------------------
    # Find and sort JSON files
    # --------------------------------------------------------------
    if config.files is None:
        json_files = sorted(base_dir.glob("*.json"))
        if not json_files:
            raise RuntimeError(f"No JSON files found in {base_dir}")
    else:
        json_files = []
        for filename in config.files:
            json_files.append(base_dir / filename)

    # --------------------------------------------------------------
    # Prepare export directories
    # --------------------------------------------------------------
    export_dir = base_dir / config.label

    if config.export_vtk:
        (export_dir / "paraview").mkdir(parents=True, exist_ok=True)

    if config.export_img:
        (export_dir / "img").mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------
    # Process each JSON snapshot
    # --------------------------------------------------------------
    progress = len(json_files)
    for i, json_path in enumerate(json_files):
        # Print to terminal
        iteration_str = f"iter_{i:05d}"
        bar_length = 30
        filled = int(bar_length * (i + 1) / progress)
        bar = "#" * filled + "-" * (bar_length - filled)
        print(f"[{bar}] {i + 1}/{progress}", end="\r", flush=True)

        # Load log
        with open(json_path, "r") as f:
            loaded_log = json.load(f)

        # Parse to StructData
        data = StructData.from_log(loaded_log)

        # User-defined post-processing hook
        post_data = config.post_process_func(data)

        # ----------------------------------------------------------
        # VTK export (ParaView)
        # ----------------------------------------------------------
        if config.export_vtk:
            suffix = "aggregate"
            graph = to_networkx(
                post_data,
                node_attrs=["coords", "aux_force_total"],
                edge_attrs=[],
                to_undirected=True,
            )
            export_graph_to_vtp(
                graph, export_dir / "paraview" / f"shot_{i:05d}_{suffix}.vtp"
            )

            # Find smarter way
            if config.export_steps:
                n_steps = post_data.aux_force_steps.shape[1]
                for step in range(n_steps):
                    state_str = f"state_{step:05d}"
                    (export_dir / "paraview" / iteration_str).mkdir(
                        parents=True, exist_ok=True
                    )
                    post_data.aux_force = post_data.aux_force_steps[:, step]
                    post_data.internal_force = post_data.internal_force_steps[:, step]
                    graph = to_networkx(
                        post_data,
                        node_attrs=["coords", "aux_force"],
                        edge_attrs=["internal_force"],
                        to_undirected=True,
                    )
                    export_graph_to_vtp(
                        graph,
                        export_dir / "paraview" / iteration_str / (state_str + ".vtp"),
                    )

        # ----------------------------------------------------------
        # Image export (placeholder)
        # ----------------------------------------------------------
        if config.export_img:
            # Find smarter way
            if config.export_steps:
                n_steps = post_data.aux_force_steps.shape[1]
                for step in range(n_steps):
                    state_str = f"state_{step:05d}"
                    (export_dir / "img" / iteration_str).mkdir(
                        parents=True, exist_ok=True
                    )
                    post_data.aux_force = post_data.aux_force_steps[:, step]
                    post_data.internal_force = post_data.internal_force_steps[:, step]
                    # Implement image rendering
                    filepath = export_dir / "img" / iteration_str / (state_str + ".png")
                    post_data.plot(
                        path=filepath,
                        force=post_data.internal_force,
                        load=post_data.aux_force,
                        show_load=True,
                        force_scale=5e1,
                    )
                    plt.close()

                # --------------------------------------------------------------
                # GIF creation
                # --------------------------------------------------------------
                if config.export_img and config.img_to_gif:
                    # image-to-gif conversion
                    save_as_gif(export_dir / "img" / iteration_str)
                    pass


def save_as_gif(dir):
    # Folder containing your images
    output_gif = dir / f"animation.gif"

    # Collect all image files (sorted)
    images = []
    for filename in sorted(os.listdir(dir)):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(dir, filename)
            images.append(imageio.imread(image_path))

    # Save as GIF
    if len(images) > 0:
        imageio.mimsave(
            output_gif, images, duration=2.0
        )  # duration = time per frame in seconds


from obj_functions.self_supporting import graph_post_process

if __name__ == "__main__":
    config = PostProcessConfig(
        dir="./results/run_ext",
        label="opt",
        # files = ["state_0000.json"],
        export_img=False,
        export_vtk=True,
        img_to_gif=False,
        export_steps=False,
        post_process_func=graph_post_process,
    )
    post_process_export_opt(config)
    config = PostProcessConfig(
        dir="./results/run_ext",
        label="initial",
        files = ["state_0000.json"],
        export_img=False,
        export_vtk=True,
        img_to_gif=False,
        export_steps=True,
        post_process_func=graph_post_process,
    )
    post_process_export_opt(config)
    config = PostProcessConfig(
        dir="./results/run_ext",
        label="final",
        files = ["state_0300.json"],
        export_img=False,
        export_vtk=True,
        img_to_gif=False,
        export_steps=True,
        post_process_func=graph_post_process,
    )
    post_process_export_opt(config)
