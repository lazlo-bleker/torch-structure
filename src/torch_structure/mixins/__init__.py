from .cem import CEMMixin
from .fdm import FDMMixin
from .laplacian_smooth import LaplacianSmoothingMixin
from .plot import PlotMixin
from .tna import TNAMixin

class TSMixin(CEMMixin, FDMMixin, LaplacianSmoothingMixin, PlotMixin, TNAMixin):
    pass

__all__ = ["TSMixin", "FDMMixin", "LaplacianSmoothingMixin", "PlotMixin", "TNAMixin"]