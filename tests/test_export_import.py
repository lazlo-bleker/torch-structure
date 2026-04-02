import json
import torch


def test_structdata_export_import_equivalence(tmp_path, dome_data_alt):
    """
    Test that:
      StructData -> to_log -> JSON -> from_log -> StructData
    produces numerically equivalent results.

    This is the strongest possible test
    for correctness of the export/import pipeline.
    """

    data = dome_data_alt

    # --------------------------------------------------------------
    # Export
    # --------------------------------------------------------------
    data_log = data.to_log()

    json_path = tmp_path / "structdata_test.json"
    with open(json_path, "w") as f:
        json.dump(data_log, f, indent=2)

    # --------------------------------------------------------------
    # Import
    # --------------------------------------------------------------
    with open(json_path, "r") as f:
        loaded_log = json.load(f)

    new_data = data.from_log(loaded_log)

    # --------------------------------------------------------------
    # Run CEM on both structures
    # --------------------------------------------------------------
    data.cem(inplace=True)
    new_data.cem(inplace=True)

    # --------------------------------------------------------------
    # Test numeric equivalence
    # --------------------------------------------------------------
    # 1) coords
    assert torch.allclose(data.coords, new_data.coords, atol=1e-12, rtol=1e-12), (
        "Coordinates differ after export/import round-trip."
    )

    # 2) forces
    assert torch.allclose(data.force, new_data.force, atol=1e-12, rtol=1e-12), (
        "Forces differ after export/import round-trip."
    )
