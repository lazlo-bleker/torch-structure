import os
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Optimizer


class Logger:
    def __init__(self, optimizer, log_interval=2, flush=False):
        self.optimizer = optimizer
        self.log_interval = log_interval
        self.iteration = 0
        self.flush = flush

    def __call__(self, _):
        self.iteration += 1

        if self.iteration % self.log_interval != 0:
            return None

        obj_log = self.optimizer.objective_function_handler.log
        constr_log = self.optimizer.constraint_function_handler.log

        msg = self.format_status(obj_log, constr_log)
        print("\r" + msg, end="", flush=self.flush)

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
