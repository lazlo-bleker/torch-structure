from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
import shutil
import json
import re
import matplotlib.pyplot as plt
from scipy.optimize import minimize, OptimizeResult
from ..data import save as save_struct_data, StructData
from .design_variables import DesignVariableHandler
from .objectives import ObjectiveHandler
from .constraints import ConstraintHandler


@dataclass
class LoggerConfig:
    """
    Data container to initialize a Logger object
    """

    enable_tensorboard: bool = True
    enable_matplotlib: bool = True
    verbose: bool = True
    flush: bool = True
    export_dir: str = "./results"
    name: str = "opt_log"
    save_cycle : int = 10                   # Save the state of the optimized object every save_cycle iterations
    plot_cycle : int = 10                   # Create a plot of the optimized object every plot_cycle iterations
    log_cycle : int  =  5                   # Log the optimization data every log_cycle iterations


class Logger:
    def __init__(
        self,
        logger_config: LoggerConfig,
        dv_handler: DesignVariableHandler,
        obj_handler: ObjectiveHandler,
        constr_handler: ConstraintHandler,
    ):
        self.history = defaultdict(list)
        self.iter = 0
        self.logged_iters = []

        self.enable_tensorboard = logger_config.enable_tensorboard
        self.enable_matplotlib = logger_config.enable_matplotlib
        self.verbose = logger_config.verbose
        self.flush = logger_config.flush
        self.save_cycle = logger_config.save_cycle
        self.plot_cycle = logger_config.plot_cycle
        self.log_cycle = logger_config.log_cycle

        self.path = Path(logger_config.export_dir) / logger_config.name
        shutil.rmtree(self.path, ignore_errors=True)
        self.path.mkdir(parents=True, exist_ok=True)

        self.dv_handler = dv_handler
        self.obj_handler = obj_handler
        self.constr_handler = constr_handler

        if self.enable_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as e:
                raise RuntimeError(
                    "TensorBoard logging is enabled (enable_tensorboard=True), "
                    "but TensorBoard is not installed. Install it with:\n"
                    "pip install tensorboard"
                ) from e

            self.writer = SummaryWriter(log_dir=str(self.path))
        else:
            self.writer = None

    def __call__(self, intermediate_result: OptimizeResult):
        i = self.iter
        will_checkpoint = i % self.save_cycle == 0
        will_log = i % self.log_cycle == 0
        will_plot = i % self.plot_cycle == 0

        if will_checkpoint or will_plot:
            sturc_state = self.dv_handler.solve_graph(self.dv_handler.x)

        log_dict = {
            # "Design Variables" : self.dv_hander.log,
            "Objectives": self.obj_handler.log,
            "Constraints": self.constr_handler.log,
        }
        
        if will_checkpoint:
            save_struct_data(
                [sturc_state],
                root=self.path / "checkpoints" / f"iter_{i:05d}",
            )

        if will_plot:
            plot_state(
                self.path,
                sturc_state,
                title=f"Structure Iteration {i:05d}",
            )

        if will_log:
            log_dict_flat = flatten_dict(log_dict)
            self.log(log_dict_flat, i)
            self.logged_iters.append(i)

        if self.verbose:
            msg = format_status(i, log_dict)
            print("\r" + msg, end="", flush=self.flush)

        self.iter += 1
        

    def log(self, log_dict, n_iter: int):
        """
        Parameters
        ----------
        log_dict : dict[str, Tensor | float]
            Dictionary of losses / metrics.
        step : int
            Training iteration.
        """
        for key, value in log_dict.items():
            value = value.item()

            self.history[key].append(value)

            if self.writer is not None:
                self.writer.add_scalar(
                    key,
                    value,
                    global_step=n_iter,
                )

    def save_individual_plots(self):
        """
        Save one plot per metric.
        """
        if not self.enable_matplotlib:
            return

        for key, values in self.history.items():
            fig, ax = plt.subplots(figsize=(8, 5))

            ax.plot(
                self.logged_iters,
                values,
                marker="o",
                color="darkorange",
                markersize=2,
            )

            ax.set_title(key)
            ax.set_xlabel("Iteration")
            ax.set_ylabel(key)
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)

            fig.tight_layout()
            safe_key = key.replace("\\", "_").replace("/", "_")
            fig.savefig(self.path / f"{safe_key}.png", dpi=300)
            plt.close(fig)

    def close(self):
        # Close instance of writer
        if self.writer is not None:
            self.writer.close()

        # Create plots for logged data
        self.save_individual_plots()

        # Save raw logged data
        filepath = self.path / "log.json"
        log = {
            "iters": self.logged_iters,
            "entries": self.history,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2, ensure_ascii=False)


def flatten_dict(d, prefix=""):
    items = {}

    for k, v in d.items():
        key = f"{prefix}/{k}" if prefix else str(k)

        if isinstance(v, dict):
            items.update(flatten_dict(v, key))
        else:
            items[key] = v

    return items


def format_status(n_iter, log_dict):
    log_txt = format_log_dict(log_dict, indent=1)
    return f"Iter {n_iter:4d} \n {log_txt}\n"


def format_log_dict(d, indent=0):
    lines = []

    for key, value in d.items():
        prefix = "    " * indent

        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.extend(format_log_dict(value, indent + 1).splitlines())
        else:
            lines.append(f"{prefix}{key}: {value}")

    return "\n".join(lines)

def plot_state(path, struct_data: StructData, title=""):
    safe_title = re.sub(r'[<>:"/\\|?*]', "", title)
    path = path / f"{safe_title}"
    struct_data.plot(title=title, path=path)
    plt.close()