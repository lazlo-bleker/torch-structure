import os
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Optimizer


class Logger:
    def __init__(self, optimizer: "Optimizer", log_interval=10, export_dir=None):
        """
        Initialize a logging object with full access to the optimizer
        """
        self.iteration = 0
        self.optimizer = optimizer
        self.log_interval = log_interval

        # Determine directory to export data
        if export_dir is not None:
            self.base_dir = export_dir
        else:
            timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
            self.base_dir = f"{os.getcwd()}/results/{timestamp}"
        # Check that base_dir exists
        os.makedirs(self.base_dir, exist_ok=True)

    def __call__(self, _):
        """
        When called, logs data and prints the optimizer status to terminal.
        """
        # Advance iteration counter
        self.iteration += 1
        # Skip logger very log_interval
        if self.iteration % self.log_interval == 0:
            return None

        # Total loss printout
        out = f"Iteration {self.iteration:3d}"
        for (
            loss_name,
            loss_value,
        ) in self.optimizer.objective_function_handler.loss_dict.items():
            out += f"\t| {loss_name} : {float(loss_value):.6f}"
        print(out)
