
import os
import shutil
import wandb
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import imageio
import meshio
import numpy as np
from datetime import datetime

from .utils import export_graph_to_vtp
from .config import (
    export_paraview,
    export_tensorboard,
    export_wandb
)

class Logger:
    def __init__(self, optimizer, shot_interval = 1, export_dir = None):
        """
        Same interface as before, but logs to Weights & Biases instead of TensorBoard.
        Keeps your directory layout and plotting/ParaView exports.
        """
        self.iteration = 0
        self.optimizer = optimizer
        self.shot_interval = shot_interval

        # Determine directory to export data
        if export_dir is not None:
            self.base_dir = export_dir
        else:
            timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
            self.base_dir = f"{os.getcwd()}/results/{timestamp}"
        # Check that base_dir exists
        os.makedirs(self.base_dir, exist_ok=True)

        if export_paraview:
            pv_dir = os.path.join(self.base_dir, "paraview")
            if os.path.exists(pv_dir):
                shutil.rmtree(pv_dir)
            os.makedirs(pv_dir, exist_ok=True)

        if export_tensorboard:
            self.writer = SummaryWriter(log_dir=self.base_dir)
        
        if export_wandb:
            self.run = wandb.init(project=os.getenv("WANDB_PROJECT", "torch-structure"))

    def __call__(self, _):
        """
        When called, logs data to W&B and prints the optimizer status to terminal.
        Mirrors your TensorBoard scalar logging, but via wandb.log(..., step=...).
        """
        if self.iteration % self.shot_interval == 0:
            self.iteration += 1
            return None
        
        if export_tensorboard or export_wandb:
            metrics = {}

            # Scalars from objective_function_handler
            for loss_name, loss_value in self.optimizer.objective_function_handler.loss_dict.items():
                # Ensure plain floats for JSON serialization
                metrics[loss_name] = float(loss_value)

            # Scalars from constraint_function_handler
            for constr_name, constr_object in self.optimizer.constraint_function_handler.constraint_objects.items():
                log_dict = constr_object.get_log_dict()
                for name, value in log_dict.items():
                    metrics[f"{constr_name}/{name}"] = float(value)

            if export_wandb:
                # Single consolidated log call per iteration
                wandb.log(metrics, step=self.iteration)
            if export_tensorboard:
                for metric_key, metric_value in metrics.items():
                    self.writer.add_scalar(metric_key, metric_value, global_step=self.iteration)


        # ParaView export
        if export_paraview:
            # Export vtp with Gauss points (specific to application)
            log_dict = self.optimizer.objective_function_handler.log()
            sub_dict_name = "Loss/Orthogonal"
            if sub_dict_name in log_dict.keys():
                filepath = f"{self.base_dir}/paraview/shot_{self.iteration:05d}.vtk"
                export_gauss_to_vtp(log_dict[sub_dict_name], self.iteration, filepath)
            
            # Export cem result as graph in vtp
            solved_graph = self.optimizer.graph
            filepath = f"{self.base_dir}/paraview/shot_{self.iteration:05d}.vtp"
            solved_graph.length = solved_graph.length_from_coords
            graph = solved_graph.to_networkx(
                node_attrs=["coords", "loss", "res"],
                edge_attrs=["force", "length"]
            )
            export_graph_to_vtp(graph, filepath, True)

        # Total loss printout
        total_loss = self.optimizer.objective_function_handler.loss_dict["Loss/Total"]
        print(f"Iteration {self.iteration:3d} | Loss: {float(total_loss.detach()):.6f}")

        # Advance iteration counter
        self.iteration += 1

def export_gauss_to_vtp(debug_dict,shot, filepath):
    """
    Export Gauss-point data to .vtp (VTK PolyData).

    _gauss_points: (Q, G, 3)
    _design_jacobian: (Q, G, 2, 3)
    _target_jacobian: (Q, G, 2, 3)
    """
    # flatten quads × gps -> (M, ...)
    pts = debug_dict["points"].reshape(-1,3).detach().numpy()
    J = debug_dict["J"].reshape(-1,2,3).detach().numpy()
    _J = debug_dict["_J"].reshape(-1,2,3).detach().numpy()
    loss = debug_dict["loss"].reshape(-1).detach().numpy()

    # split into column vectors, since ParaView treats 3-tuples as vectors
    point_data = {
        "Jd_col0": J[:,0,:],   # (M,3)
        "Jd_col1": J[:,1,:],   # (M,3)
        "Jt_col0": _J[:,0,:],   # (M,3)
        "Jt_col1": _J[:,1,:],   # (M,3)
        "loss" : loss
    }

    # connectivity: treat each Gauss point as a vertex
    npoints = pts.shape[0]
    cells = [("vertex", np.arange(npoints).reshape(-1,1))]

    # write as VTK PolyData (.vtp)
    mesh = meshio.Mesh(points=pts, cells=cells, point_data=point_data)
    mesh.write(filepath, file_format="vtk")