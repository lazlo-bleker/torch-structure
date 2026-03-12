from .cem import CEMMixin
from .fdm import FDMMixin
from .laplacian_smooth import LaplacianSmoothingMixin
from .plot import PlotMixin
from .tna import TNAMixin
from .utils import OverrideResolveMixin


class TSMixin(CEMMixin, FDMMixin, LaplacianSmoothingMixin, PlotMixin, TNAMixin):
    pass


__all__ = [
    "TSMixin",
    "CEMMixin",
    "FDMMixin",
    "LaplacianSmoothingMixin",
    "PlotMixin",
    "TNAMixin",
    "OverrideResolveMixin",
]
method_names = [
    m for m in dir(TSMixin) if callable(getattr(TSMixin, m)) and not m.startswith("__")
]
