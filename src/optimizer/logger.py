import os,shutil
import numpy as np
import meshio
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from .utils import export_graph_to_vtp
from .config import (
    export_plt, 
    export_paraview,
    export_tensorboard
)
same_dir = True
class Logger():
    def __init__(self, optimizer):
        """
        Creates an object that logs optimization data to tensorboard, plots snapshots of the structure and prints the total loss to the terminal
        """
        self.iteration = 0
        # Link to parent optimizer to access its attributes
        self.optimizer = optimizer
        # Initialize tensorboard writer
        self.writer = SummaryWriter()
        if same_dir :
            self.base_dir = f"{os.getcwd()}/result"
        else:
            self.base_dir = self.writer.log_dir
        
        if export_plt:
            # Find folder to save plot it
            folder_path = f"{self.base_dir}/img"
            os.makedirs(folder_path, exist_ok=True)
        if export_paraview:
            folder_path = os.path.join(self.base_dir, "paraview")

            # If the folder exists, delete it completely
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)

            # Recreate the empty folder
            os.makedirs(folder_path, exist_ok=True)
        # Option to add additional scatter plots
        self.additional_scatter_plots = []
    
    def __call__(self, _):
        """
        When called, logs data to tensorboard and prints the optimizer status to terminal
        """
        if export_tensorboard:
            # Collect scalar values in objective_function_handler
            # NOTE: Partial losses are directly stores in an attribute
            for loss_name, loss_value in self.optimizer.objective_function_handler.loss_dict.items():
                self.writer.add_scalar(loss_name, loss_value, self.iteration)
            
            # Collect scalar values in constraint_function_handler
            for constr_name, constr_object in self.optimizer.constraint_function_handler.constraint_objects.items():
                # Get log data from constraint
                constr_log_dict = constr_object.get_log_dict()
                for name, value in constr_log_dict.items():
                    full_name = f"{constr_name}/{name}"
                    self.writer.add_scalar(full_name, value, self.iteration)

        # Get value of total loss
        total_loss = self.optimizer.objective_function_handler.loss_dict["Loss/Total"]
        # Print status to terminal
        print(
            f"Iteration {self.iteration:3d} | Loss: {total_loss:.6f}"
        )

        log_dict = self.optimizer.objective_function_handler.log()
        sub_dict_name = "Loss/Orthogonal"
        if sub_dict_name in log_dict.keys():
            log_dict_export = log_dict[sub_dict_name]
            export_gauss_to_vtp(log_dict_export, self.iteration)

        solved_graph = self.optimizer.graph
        filepath = f"{self.base_dir}/paraview/shot_{self.iteration:05d}.vtp"
        solved_graph.length = solved_graph.length_from_coords
        graph = solved_graph.to_networkx(node_attrs=["coords","loss","res"], edge_attrs=["force","length"])
        export_graph_to_vtp(graph, filepath, True)
        
        # Locally keep iteration count (assumes that the logger is called every iteration)
        self.iteration += 1

    def plot(self, solved_graph, shot):
        """
        Handles the plot of the instance of StrucData
        """
        # log_dict = self.optimizer.objective_function_handler.log()
        # log_dict_export = log_dict["Loss/Orthogonal"]
        # export_gauss_to_vtp(log_dict_export, shot)

        if export_plt:
            # Plot the optimized structure    
            data_plot = solved_graph.plot(title="Optimized Structure", legend=False, force_scale = 15.0)
            # Add additional (optional) scatters
            for additional_scatter_plot in self.additional_scatter_plots:
                data_plot.scatter(**additional_scatter_plot)
            # Save plot and close afterwards
            plt.savefig(f"{self.base_dir}/img/shot_{shot:05d}.png", dpi=200)
            plt.close()

        # if export_paraview: 
        #     if same_dir:
        #         filepath = f"{self.base_dir}/paraview/shot_{shot:05d}.vtp"
        #     else:
        #         filepath = f"{self.base_dir}/paraview/shot_{shot:05d}.vtp"
            
        #     solved_graph.length = solved_graph.length_from_coords
        #     graph = solved_graph.to_networkx(node_attrs=["coords","loss","res"], edge_attrs=["force","length"])
        #     export_graph_to_vtp(graph, filepath, True)
    
    def generate_gif(self):
        """
        Collects all the snapshot saved plots and creates a GIF out of them
        """
        if export_plt:
            # Folder containing your images
            folder_path = f"{self.base_dir}/img"
            output_gif = f"{self.base_dir}/shots.gif"

            # Collect all image files (sorted)
            images = []
            for filename in sorted(os.listdir(folder_path)):
                if filename.endswith((".png", ".jpg", ".jpeg")):
                    image_path = os.path.join(folder_path, filename)
                    images.append(imageio.imread(image_path))

            # Save as GIF
            imageio.mimsave(output_gif, images, duration=10.0)

def export_gauss_to_vtp(debug_dict,shot):
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
    path = f"./result/paraview/rect_{shot:05d}.vtk"
    mesh.write(path, file_format="vtk")