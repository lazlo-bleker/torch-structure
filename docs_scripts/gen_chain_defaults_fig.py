import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

DARK = os.environ.get("TS_DARK") == "1"

n = 6

u_node = torch.arange(n).float()
x_node = u_node / (n - 1)
x_edge = (x_node[:-1] + x_node[1:]) / 2

node_x = x_node.numpy()

ORANGE      = "#F05F42" if DARK else "#E7461E"
DGREY       = "#9A9A9A" if DARK else "#555555"
GREY        = "#888888" if DARK else "#AAAAAA"
SWATCH_GREY = GREY      if DARK else "#999999"
TEXT        = "#E8E8E8" if DARK else "black"
NODE_FILL   = "#1E2129" if DARK else "white"
BG          = "none"    if DARK else "white"

if DARK:
    plt.rcParams["text.color"] = TEXT

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), facecolor=BG)

for ax in (ax1, ax2):
    ax.set_xlim(-0.05, 1.05)
    ax.axis("off")

ax1.set_ylim(-0.18, 0.22)
ax2.set_ylim(-0.28, 0.22)

# ── Panel 1: Node defaults ──
ax1.set_title("Node defaults", fontsize=16, loc="left", pad=4)
ax1.plot(node_x, np.zeros(n), color=DGREY, lw=1.5, zorder=1)

for i in range(n):
    ax1.scatter(node_x[i], 0, color=NODE_FILL, s=80, marker="o", zorder=3,
                edgecolors=ORANGE, linewidths=1.2)
    ax1.text(node_x[i], 0.03, f"{i}",
             ha="center", va="bottom", fontsize=12.5, color=TEXT)
    ax1.text(node_x[i], -0.03, f"{x_node[i]:.2f}",
             ha="center", va="top", fontsize=12.5, color=DGREY)

# ── Panel 2: Edge defaults ──
ax2.set_title("Edge defaults", fontsize=16, loc="left", pad=4)
ax2.plot(node_x, np.zeros(n), color=ORANGE, lw=1.5, zorder=1)
ax2.scatter(node_x, np.zeros(n), color=NODE_FILL, s=80, zorder=3,
            edgecolors=DGREY, linewidths=1.2)

E_FRAC = 0.075   # offset for edge labels, as a fraction of edge length --
                 # same ratio used for the polar/tri/grid defaults figures
edge_length = 1 / (n - 1)
e = E_FRAC * edge_length

for i in range(n - 1):
    mid = x_edge[i].item()
    ax2.text(mid, e, f"{i}",
             ha="center", va="bottom", fontsize=12.5, color=TEXT)
    ax2.text(mid, -e, f"{x_edge[i]:.2f}",
             ha="center", va="top", fontsize=12.5, color=DGREY)

# axis arrow bottom-left of edge defaults panel
ax2.annotate("", xy=(0.0447, -0.22), xytext=(0.0, -0.22),
             arrowprops=dict(arrowstyle="->", color=TEXT, lw=1.2))
ax2.text(0.055, -0.22, "u_ind", ha="left", va="center",
         fontsize=12.5, color=TEXT)

leg = [
    mpatches.Patch(color=TEXT,        label="u_ind"),
    mpatches.Patch(color=SWATCH_GREY, label="x_unit_coord"),
]
legend_kwargs = dict(handles=leg, loc="upper right", fontsize=12.5, frameon=True,
                      edgecolor=(GREY if DARK else "none"), framealpha=(0.0 if DARK else 1.0),
                      bbox_to_anchor=(1.024, -0.16), bbox_transform=ax2.transData)
if DARK:
    legend_kwargs["facecolor"] = "none"
legend = ax2.legend(**legend_kwargs)


plt.tight_layout()
OUT_PATH = "docs/assets/add_chain_defaults_dark.png" if DARK else "docs/assets/add_chain_defaults.png"
plt.savefig(OUT_PATH, dpi=150, facecolor=BG)
print(f"Saved {OUT_PATH}")
