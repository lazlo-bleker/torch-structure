from abc import ABC, abstractmethod
from torch_structure.data import StructData

class BaseGenerator(ABC):
    r"""An abstract base class for writing structure generators."""

    def __init__(self, **overrides):
        self.overrides = overrides
        self.max_attempts = 1

    def __call__(self):
        return self._build()
    
    def _build(self) -> StructData:
        if self.max_attempts == 1:
            self.input = self.sample_input(**self.overrides)
            self.validate_input(**self.input)
            return self.generate(**self.input)
        else:
            for _ in range(self.max_attempts):
                self.input = self.sample_input(**self.overrides)
                self.validate_input(**self.input)
                try:
                    return self.generate(**self.input)
                except Exception:
                    continue

        raise RuntimeError(f"Failed to generate structure after {self.max_attempts} attempts.")

    @abstractmethod
    def sample_input(self, **overrides) -> dict:
        """Samples missing input parameters."""
        pass

    def validate_input(self, **input) -> None:
        """Raises errors if inputs are invalid."""
        pass

    @abstractmethod
    def generate(self, **input) -> StructData:
        """Generates a StructData object."""
        pass
