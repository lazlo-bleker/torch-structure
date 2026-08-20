from dataclasses import dataclass

@dataclass
class LoggerConfig:
    """
    Data container to initialize a Logger object
    """
    enable_tensorboard: bool = True
    enable_matplotlib : bool = True
    export_dir: str = "./runs"
    name: str = "opt_log"

from collections import defaultdict
from pathlib import Path
import shutil
import re

import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch_structure.data import StructData

class Logger:
    def __init__(self, logger_config : LoggerConfig):
        self.history = defaultdict(list)
        self.steps = []

        self.enable_tensorboard = logger_config.enable_tensorboard
        self.enable_matplotlib = logger_config.enable_matplotlib

        self.path = Path(logger_config.export_dir) / logger_config.name
        shutil.rmtree(self.path, ignore_errors=True)
        self.path.mkdir(parents=True, exist_ok=True)

        self.writer = (
            SummaryWriter(log_dir=str(self.path))
            if self.enable_tensorboard
            else None
        )

    def log(self, log_dict, step: int):
        """
        Parameters
        ----------
        log_dict : dict[str, Tensor | float]
            Dictionary of losses / metrics.
        step : int
            Training iteration.
        """
        self.steps.append(step)

        for key, value in log_dict.items():
            value = float(value.detach())

            self.history[key].append(value)

            if self.writer is not None:
                self.writer.add_scalar(
                    key,
                    value,
                    global_step=step,
                )

    def save_plots(self):
        """
        Save all logged metrics into a single figure.
        """
        if not self.enable_matplotlib:
            return

        fig, ax = plt.subplots(figsize=(8, 5))

        for key, values in self.history.items():
            ax.plot(
                self.steps[: len(values)],
                values,
                label=key,
            )

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Value")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(self.path / "losses.png", dpi=300)
        plt.close(fig)

    def save_individual_plots(self):
        """
        Save one plot per metric.
        """
        if not self.enable_matplotlib:
            return

        for key, values in self.history.items():
            fig, ax = plt.subplots(figsize=(8, 5))

            ax.plot(self.steps[: len(values)], values, marker="o", color="darkorange", markersize=2)

            ax.set_title(key)
            ax.set_xlabel("Iteration")
            ax.set_ylabel(key)
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)

            fig.tight_layout()
            fig.savefig(self.path / f"{key}.png", dpi=300)
            plt.close(fig)

    def close(self):
        self.save_individual_plots()

        if self.writer is not None:
            self.writer.close()

    def plot_state(self, struct_data: StructData, title=""):
        safe_title = re.sub(r'[<>:"/\\|?*]', "_", title)
        path = self.path / f"_{safe_title}.png"
        struct_data.plot(show_load=False, force_scale=1e0, title=title, path=path)