import pytest
from torch_structure.generators import (
    DomeGenerator,
    DomeAssemblyGenerator,
    CableNetGenerator,
)
import torch
import random


@pytest.fixture
def cablenet_data(request):
    generator = CableNetGenerator(seed=request.param)
    return generator()


@pytest.fixture
def dome_data(request):
    generator = DomeGenerator(seed=request.param)
    return generator()


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
