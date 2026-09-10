from .cem import CEMMixin
from .fdm import FDMMixin
from .laplacian_smooth import LaplacianSmoothingMixin
from .tna import TNAMixin
from .utils import OverrideResolveMixin

class TSMixin(CEMMixin, FDMMixin, LaplacianSmoothingMixin, TNAMixin):
    pass

__all__ = ["TSMixin", "CEMMixin", "FDMMixin", "LaplacianSmoothingMixin", "TNAMixin", "OverrideResolveMixin"]
