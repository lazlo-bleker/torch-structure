from abc import ABC, abstractmethod
from torch_structure.data import StructData

class BaseGenerator(ABC):
    r"""An abstract base class for writing structure generators."""

    def __init__(self, **overrides):
        self.overrides = overrides

    def __call__(self):
        self.input = self.sample_input(**self.overrides)
        self.validate_input(**self.input)
        data = self.generate(**self.input)
        return data

    @abstractmethod
    def sample_input(self, **overrides) -> dict:
        """Samples missing input parameters."""
        pass

    @abstractmethod
    def validate_input(self, **input) -> None:
        """Raises errors if inputs are invalid."""
        pass

    @abstractmethod
    def generate(self, **input) -> StructData:
        """Generates a StructData object."""
        pass
