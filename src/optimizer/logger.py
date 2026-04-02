import os, json, shutil
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from config import optimizer_export_tensorboard


class Logger:
    def __init__(self, optimizer, log_interval=10, flush=False, export_dir=None):
        self.optimizer = optimizer
        self.log_interval = log_interval
        self.iteration = -1
        self.flush = flush

        # Determine directory to export data
        if export_dir is not None:
            self.export_dir = export_dir
        else:
            timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
            self.export_dir = f"{os.getcwd()}/results/{timestamp}"
        # Check that base_dir exists
        clean_dir([self.export_dir])

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

        data_log = self.optimizer.graph.to_log()
        json_path = os.path.join(self.export_dir, f"state_{self.iteration:04d}.json")
        with open(json_path, "w") as f:
            json.dump(data_log, f, indent=2)

        if optimizer_export_tensorboard:
            metrics = {}

            # Scalars from objective_function_handler
            for (
                obj_name,
                obj_value,
            ) in self.optimizer.objective_function_handler.log.items():
                metrics[obj_name] = float(obj_value)

            # Scalars from constraint_function_handler
            for (
                constr_name,
                constr_value,
            ) in self.optimizer.constraint_function_handler.log.items():
                metrics[constr_name] = float(constr_value)

            # TODO: support histograms and statistics, currently only scalars are logged
            for metric_key, metric_value in metrics.items():
                self.writer.add_scalar(
                    metric_key, metric_value, global_step=self.iteration
                )

    def format_status(self, obj_log, constr_log):
        obj_txt = format_log_dict(obj_log)
        constr_txt = format_log_dict(constr_log)
        return (
            f"Iter {self.iteration:4d} \t| Obj: {obj_txt} \t| Constr: {constr_txt} \n"
        )


def format_log_dict(log_dicts, precision=3):
    parts = []
    for key, value in log_dicts.items():
        parts.append(f"{key}={format_value(value, precision)}")
    return ", ".join(parts)


def format_value(value, precision=3):
    if isinstance(value, float):
        return f"{value:.{precision}e}"
    if hasattr(value, "item") and getattr(value, "ndim", None) == 0:
        return f"{value.item():.{precision}e}"
    return str(value)


def clean_dir(dir_list):
    pv_dir = os.path.join(*dir_list)
    if os.path.exists(pv_dir):
        shutil.rmtree(pv_dir)
    os.makedirs(pv_dir, exist_ok=True)
