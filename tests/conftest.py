import pytest
from torch_structure.generators.dome import DomeGenerator
import torch
import random


@pytest.fixture
def dome_data():
    torch.manual_seed(42)
    random.seed(42)

    generator = DomeGenerator()
    dome = generator()
    return dome
