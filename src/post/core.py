from dataclasses import dataclass, field
from typing import Callable
from pathlib import Path
import json

from torch_structure.data import StructData
from .utils import identity, return_none


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
    export_vtp: Callable[[StructData], StructData] = None
    export_img: Callable[[StructData], StructData] = None
    verbose: bool = False


# ---------------------------------------------------------------------
# Main post-processing routine
# ---------------------------------------------------------------------
def post_process(config: PostProcessConfig) -> None:
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

    if config.export_vtp:
        (export_dir / "paraview").mkdir(parents=True, exist_ok=True)

    if config.export_img:
        (export_dir / "img").mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------
    # Process each JSON snapshot
    # --------------------------------------------------------------
    progress = len(json_files)
    cache_dict_vtp = {}
    cache_dict_img = {}
    for i, json_path in enumerate(json_files):
        # Print to terminal
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
        if config.export_vtp is not None:
            config.export_vtp(post_data, export_dir, f"shot_{i:05d}", cache_dict_vtp)

        # ----------------------------------------------------------
        # Image export
        # ----------------------------------------------------------
        if config.export_img is not None:
            config.export_img(post_data, export_dir, f"shot_{i:05d}", cache_dict_img)

    if config.verbose:
        print("Finished post processing")
        print(cache_dict_vtp)
        print(cache_dict_img)
