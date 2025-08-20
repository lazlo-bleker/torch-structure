import os
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

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
        # Option to add additional scatter plots
        self.additional_scatter_plots = []
    
    def __call__(self, _):
        """
        When called, logs data to tensorboard and prints the optimizer status to terminal
        """
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
        # Plot the optimized structure    
        data_plot = solved_graph.plot(title="Optimized Structure", legend=False, force_scale = 6.0)
        # Add additional (optional) scatters
        for additional_scatter_plot in self.additional_scatter_plots:
            data_plot.scatter(**additional_scatter_plot)
        # Find folder to save plot it
        folder_path = f"{self.writer.log_dir}/shots"
        os.makedirs(folder_path, exist_ok=True)
        # Save plot and close afterwards
        plt.savefig(f"{folder_path}/{shot:05d}.png", dpi=400)
        plt.close()

    def generate_gif(self):
        """
        Collects all the snapshot saved plots and creates a GIF out of them
        """
        # Folder containing your images
        folder_path = f"{self.writer.log_dir}/shots"
        output_gif = f"{self.writer.log_dir}/shots.gif"

        # Collect all image files (sorted)
        images = []
        for filename in sorted(os.listdir(folder_path)):
            if filename.endswith((".png", ".jpg", ".jpeg")):
                image_path = os.path.join(folder_path, filename)
                images.append(imageio.imread(image_path))

        # Save as GIF
        imageio.mimsave(output_gif, images, duration=10.0)
