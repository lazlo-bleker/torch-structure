from .cem import CEMMixin
from .fdm import FDMMixin
from .laplace import LaplacianMixin
from .plot import PlotMixin
from .tna import TNAMixin
from .utils import OverrideResolveMixin

class TSMixin(CEMMixin, FDMMixin, LaplacianMixin, PlotMixin, TNAMixin):
    """Combines all form-finding, smoothing, and plotting mixins onto a single data class."""

    pass

__all__ = ["TSMixin", "CEMMixin", "FDMMixin", "LaplacianMixin", "PlotMixin", "TNAMixin", "OverrideResolveMixin"]