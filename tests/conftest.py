import pytest
from torch_structure.generators.dome import Dome
import torch
import random


def dome_graph():  # TODO: Make fully deterministic
    torch.manual_seed(42)
    random.seed(42)

    dome = Dome(n_trails=4, n_rings=3, trail_length=1.0, center_deviation_force=0.1)
    return dome.graph


@pytest.fixture
def dome_data():
    return dome_graph()
