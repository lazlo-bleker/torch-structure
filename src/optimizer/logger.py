import os
import shutil
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from typing import TYPE_CHECKING
from torch_geometric.utils import to_networkx

from .utils import export_graph_to_vtp

if TYPE_CHECKING:
    from .core import Optimizer

from config import optimizer_export_paraview, optimizer_export_tensorboard


class Logger:
    def __init__(self, optimizer, log_interval=10, flush=False, export_dir=None):
        self.optimizer = optimizer
        self.log_interval = log_interval
        self.iteration = 0
        self.flush = flush

        # Determine directory to export data
        if export_dir is not None:
            self.export_dir = export_dir
        else:
            timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
            self.export_dir = f"{os.getcwd()}/results/{timestamp}"
        # Check that base_dir exists
        os.makedirs(self.export_dir, exist_ok=True)

        if optimizer_export_paraview:
            pv_dir = os.path.join(self.export_dir, "paraview")
            if os.path.exists(pv_dir):
                shutil.rmtree(pv_dir)
            os.makedirs(pv_dir, exist_ok=True)

        if optimizer_export_tensorboard:
            self.writer = SummaryWriter(log_dir=self.export_dir)

    def __call__(self, _):
        self.iteration += 1

        if self.iteration % self.log_interval != 0:
            return None

        obj_log = self.optimizer.objective_function_handler.log
        constr_log = self.optimizer.constraint_function_handler.log

        msg = self.format_status(obj_log, constr_log)
        print("\r" + msg, end="", flush=self.flush)

        if optimizer_export_tensorboard:
            metrics = {}

            # Scalars from objective_function_handler
            for obj_name, obj in self.optimizer.objective_function_handler.log.items():
                for obj_attr, obj_value in obj.items():
                    name = f"{obj_name}_{obj_attr}"
                    metrics[name] = float(obj_value)

            # Scalars from constraint_function_handler
            for (
                constr_name,
                constr,
            ) in self.optimizer.constraint_function_handler.log.items():
                for constr_attr, constr_value in constr.items():
                    name = f"{constr_name}_{constr_attr}"
                    metrics[name] = float(constr_value)

            for metric_key, metric_value in metrics.items():
                self.writer.add_scalar(
                    metric_key, metric_value, global_step=self.iteration
                )

        # ParaView export
        if optimizer_export_paraview:
            # Export cem result as graph in vtp
            solved_graph = self.optimizer.graph
            filepath = f"{self.export_dir}/paraview/shot_{self.iteration:05d}.vtp"
            solved_graph.length = solved_graph.length_from_coords
            graph = to_networkx(
                solved_graph,
                to_undirected=True,
                node_attrs=["coords"],
                edge_attrs=["force", "length"],
            )
            export_graph_to_vtp(graph, filepath, True)

    def format_status(self, obj_log, constr_log):
        obj_txt = format_log_dict(obj_log)
        constr_txt = format_log_dict(constr_log)
        return (
            f"Iter {self.iteration:4d} \t| Obj: {obj_txt} \t| Constr: {constr_txt} \n"
        )


def format_log_dict(log_dicts, precision=3):
    parts = []
    for key_0, log_dict in log_dicts.items():
        for key_1, value in log_dict.items():
            key = f"{key_0}.{key_1}"
            parts.append(f"{key}={format_value(value, precision)}")
    return ", ".join(parts)


def format_value(value, precision=3):
    if isinstance(value, float):
        return f"{value:.{precision}e}"
    if hasattr(value, "item") and getattr(value, "ndim", None) == 0:
        return f"{value.item():.{precision}e}"
    return str(value)
