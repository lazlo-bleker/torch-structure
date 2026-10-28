"""Renders the figures of docs/plotting.md into docs/assets.

Run from anywhere: python docs_scripts/gen_plotting_figs.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch_structure as ts

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"


def save(ax, name):
    ax.figure.savefig(ASSETS / f"plotting_{name}.png", dpi=110, bbox_inches="tight")
    plt.close("all")


# -----------------------------------------
# A form-found cable net, plotted many ways
# -----------------------------------------
# the seed covers torch, numpy and random, which the generator all samples from
data = ts.generators.CableNetGenerator(
    n=5, boundary_density=5, diagonals=False, centroid_support=False, pattern="standard",
    unsupported_boundaries=True, curve_boundaries=True, rectangle=False, square=False, circle=False,
    support_height=0.6, seed=0,
)()

save(data.plot(), "default")
save(data.plot(edge_color="force"), "colormap")

data.material_id = data.is_boundary_edge.long()
save(data.plot(edge_color="material_id"), "categories")

data.height = data.coords[:, 2:]
save(data.plot(view="xy", edge_color="black", edge_width=0.8, node_color="height"), "plan_nodes")

flat = data.coords.clone()
flat[:, 2] = 0.0
save(data.plot(ghost=flat, ghost_label="Flat", edge_color="black", edge_width=0.8), "ghost")

save(data.plot(view="xy", edge_color="grey", edge_width=0.8, node_labels="index"), "labels")
save(data.plot(loads=True, vector_scale=1.5, edge_width=0.8), "loads")
save(data.plot(reactions=True, vector_scale=0.0025, edge_width=0.8), "reactions")
