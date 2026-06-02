from pathlib import Path

import pytest
import json
from torch_structure.data import StructData

DATA_DIR = Path(__file__).parent / "files"
PLOT_DIR = Path(__file__).parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

ALL_CASES = sorted(DATA_DIR.glob("*.json"))


@pytest.fixture(params=ALL_CASES, ids=lambda p: p.stem)
def struc_data(request):
    json_path = request.param

    with json_path.open() as f:
        json_data = json.load(f)

    data = StructData.from_log(json_data)
    # data.plot(path =  PLOT_DIR / f"{json_path.stem}.png")

    return data
