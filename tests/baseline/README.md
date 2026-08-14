# Plotter baseline

Reference PNGs captured from the old function-based plotter (`plot/plot.py`,
`mixins/plot.py`) on `main`, before the class-based rewrite. Used to visually
confirm each later phase reproduces the same pictures. Not compared by any
committed test.

## How they were made

```
python tests/baseline/capture_baseline.py
```

The script runs every `.py` file in `examples/` with the `Agg` backend, and
saves every matplotlib figure the script produces to
`<script_stem>_<index>.png` at `dpi=100`. Figure size is whatever
`plot_data`/`plot_data_xz` set internally (`figsize=(10, 8)`), which is
constant across all current examples.

## Environment

- Python 3.12.13 (Anaconda, `tstruct` env)
- Windows-11-10.0.26100-SP0
- matplotlib 3.11.1
- torch 2.11.0+cu128
- torch-geometric 2.8.0.post1
- numpy 2.5.1
- pydantic 2.13.4

## Contents

One PNG per figure produced by each `examples/` script (11 total, across 9
scripts — `cem_simple.py` produces three figures, one per CEM variant).
