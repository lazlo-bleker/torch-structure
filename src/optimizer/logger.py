import os,shutil
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
        # Locally keep iteration count (assumes that the logger is called every iteration)
        self.iteration += 1

    def plot(self, solved_graph, shot):
        """
        Handles the plot of the instance of StrucData
        """
        if export_plt:
            # Plot the optimized structure    
            data_plot = solved_graph.plot(title="Optimized Structure", legend=False, force_scale = 15.0)
            # Add additional (optional) scatters
            for additional_scatter_plot in self.additional_scatter_plots:
                data_plot.scatter(**additional_scatter_plot)
            # Save plot and close afterwards
            plt.savefig(f"{self.base_dir}/img/shot_{shot:05d}.png", dpi=200)
            plt.close()

        if export_paraview: 
            if same_dir:
                filepath = f"{self.base_dir}/paraview/shot_{shot}.vtp"
            else:
                filepath = f"{self.base_dir}/paraview/shot_{shot}.vtp"
            
            solved_graph.length = solved_graph.length_from_coords
            graph = solved_graph.to_networkx(node_attrs=["coords"], edge_attrs=["force","length"])
            export_graph_to_vtp(graph, filepath, True)
    
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
