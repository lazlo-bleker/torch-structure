from torch_structure.plot import plot_data

class PlotMixin:
    def plot(self, **kwargs):
        if "coords" not in kwargs:
            kwargs["coords"] = self.coords
        if "edge_index" not in kwargs:
            kwargs["edge_index"] = self.edge_index
        if "force" not in kwargs and hasattr(self, "force"):
            kwargs["force"] = self.force

        plot_data(**kwargs)