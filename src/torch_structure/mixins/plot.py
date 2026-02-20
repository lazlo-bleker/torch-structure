from torch_structure.plot import plot_data
from torch_structure.mixins.utils import OverrideResolveMixin

class PlotMixin(OverrideResolveMixin):
    def plot(self, **kwargs):
        if "coords" not in kwargs:
            kwargs["coords"] = self.coords
        if "edge_index" not in kwargs:
            kwargs["edge_index"] = self.edge_index
        if "force" not in kwargs and hasattr(self, "force"):
            kwargs["force"] = self.force

        plot_data(**kwargs)