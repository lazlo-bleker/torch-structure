"""Throwaway script: run every script in examples/ and save each figure it
produces as a PNG under tests/baseline/, at a fixed dpi.

This captures the output of the old function-based plotter (plot.py,
mixins/plot.py) before it is replaced, per Step 0 of the plotter rewrite. Not
part of the pytest suite. Re-run by hand with:

    python tests/baseline/capture_baseline.py
"""

import runpy
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from pathlib import Path  # noqa: E402

DPI = 100

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples"
BASELINE_DIR = REPO_ROOT / "tests" / "baseline"


def main():
    scripts = sorted(EXAMPLES_DIR.glob("*.py"))
    for script in scripts:
        before = set(plt.get_fignums())
        runpy.run_path(str(script), run_name="__main__")
        after = [n for n in plt.get_fignums() if n not in before]

        for i, fignum in enumerate(after):
            fig = plt.figure(fignum)
            out_path = BASELINE_DIR / f"{script.stem}_{i:03d}.png"
            fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
            print(f"wrote {out_path}")

        plt.close("all")


if __name__ == "__main__":
    main()
