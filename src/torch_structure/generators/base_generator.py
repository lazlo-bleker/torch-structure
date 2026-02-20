from abc import ABC, abstractmethod
from torch_structure.data import StructData
import traceback

class InvalidSampleError(Exception):
    """Sample violated generator constraints."""
    pass

class BaseGenerator(ABC):
    r"""An abstract base class for writing structure generators."""

    def __init__(self, **overrides):
        self.overrides = overrides

        # Optional attempt logic for handling invalid samples
        self.max_attempts = 1
        self.attempt_count = 0
        self.success_count = 0
        self.verbose = False

    def __call__(self):
        return self._build()

    def _build(self) -> StructData:
        if self.max_attempts == 1:
            self.input = self.sample_input(**self.overrides)
            self.validate_input(**self.input)
            return self.generate(**self.input)
        else:
            for attempt in range(self.max_attempts):
                self.attempt_count += 1
                self.input = self.sample_input(**self.overrides)
                self.validate_input(**self.input)
                try:
                    result = self.generate(**self.input)
                    self.success_count += 1
                    return result
                except InvalidSampleError as e:
                    error_message = f"Attempt {attempt+1}/{self.max_attempts} failed: {str(e)}"
                    if self.verbose:
                        print(error_message)
                    continue

            raise RuntimeError(
                f"Failed to generate structure after {self.max_attempts} attempts. {error_message}"
            )

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
