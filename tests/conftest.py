import pytest
from torch_structure.generators import DomeGenerator, DomeAssemblyGenerator
import torch
import random


@pytest.fixture
def dome_data():
    torch.manual_seed(42)
    random.seed(42)

    generator = DomeGenerator()
    dome = generator()
    return dome


@pytest.fixture
def dome_data_alt():
    n_trails = 8
    n_rings = 12
    # NOTE: Using new fixture because dome_data results in some nan when using the cem/mpcem method
    data_generator = DomeAssemblyGenerator(
        n_trails=n_trails,
        n_rings=n_rings,
    )
    return data_generator()
