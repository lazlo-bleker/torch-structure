from .cem import CEMMixin
from .fdm import FDMMixin
from .laplace import LaplacianMixin
from .plot import PlotMixin
from .tna import TNAMixin
from .utils import OverrideResolveMixin

class TSMixin(CEMMixin, FDMMixin, LaplacianMixin, PlotMixin, TNAMixin):
    pass

__all__ = ["TSMixin", "CEMMixin", "FDMMixin", "LaplacianMixin", "PlotMixin", "TNAMixin", "OverrideResolveMixin"]