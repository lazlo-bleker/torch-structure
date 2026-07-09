import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

DARK = os.environ.get("TS_DARK") == "1"

n = 3
FONT  = 12.5
FONT_TITLE = 16
GREY   = "#888888" if DARK else "#AAAAAA"
ORANGE = "#F05F42" if DARK else "#E7461E"
DGREY  = "#9A9A9A" if DARK else "#555555"
TEXT      = "#E8E8E8" if DARK else "black"
NODE_FILL = "#1E2129" if DARK else "white"
BG        = "none"    if DARK else "white"

if DARK:
    plt.rcParams["text.color"] = TEXT

xu = lambda u: u / (n - 1)
yv = lambda v: v / (n - 1)

nodes   = [(u, v) for u in range(n)     for v in range(n - u)]
h_edges = [(u, v) for u in range(n)     for v in range(n - 1 - u)]   # dir=0: (u,v)→(u,v+1)
v_edges = [(u, v) for u in range(n - 1) for v in range(n - 1 - u)]   # dir=1: (u,v)→(u+1,v)
d_edges = [(u, v) for u in range(n - 1) for v in range(n - 1 - u)]   # dir=2: (u,v+1)→(u+1,v)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8), facecolor=BG)

for ax in (ax1, ax2):
    ax.set_xlim(-0.18, 1.18)
    ax.set_ylim(-0.55, 1.18)
    ax.set_aspect("equal")
    ax.axis("off")

ax1.set_anchor("W")
ax2.set_anchor("E")

D = 0.03    # diagonal offset for node labels
E_FRAC = 0.075   # perpendicular offset for edge labels, as a fraction of edge
                 # length -- same ratio used for the polar defaults figure
                 # (there E=0.15 on a SCALE=2 edge length of 2.0)

# ── Panel 1: Node defaults ──
_node_title_text = "Node defaults"   # positioned after canvas.draw()

for u, v in h_edges:
    ax1.plot([xu(u), xu(u)], [yv(v), yv(v + 1)], color=DGREY, lw=1.5)
for u, v in v_edges:
    ax1.plot([xu(u), xu(u + 1)], [yv(v), yv(v)], color=DGREY, lw=1.5)
for u, v in d_edges:
    ax1.plot([xu(u), xu(u + 1)], [yv(v + 1), yv(v)], color=DGREY, lw=1.5)

for u, v in nodes:
    ax1.scatter(xu(u), yv(v), color=NODE_FILL, s=80, zorder=3,
                edgecolors=ORANGE, linewidths=1.2)
    ax1.text(xu(u) + D, yv(v) + D, f"({u}, {v})",
             ha="left", va="bottom", fontsize=FONT, color=TEXT)
    ax1.text(xu(u) - D, yv(v) - D, f"({xu(u):.2f}, {yv(v):.2f})",
             ha="right", va="top", fontsize=FONT, color=DGREY)

# ── Panel 2: Edge defaults ──
ax2.set_title("Edge defaults", fontsize=FONT_TITLE, loc="left", pad=4)

for u, v in nodes:
    ax2.scatter(xu(u), yv(v), color=NODE_FILL, s=80, zorder=3,
                edgecolors=DGREY, linewidths=1.2)

# Single, uniform rule for every edge (all 3 directions): the label pair
# sits at the edge's own midpoint, offset along the edge's normal, and
# rotated to read along the edge's own direction. Walking from source to
# destination, the index label goes to the left, the coord label to the
# right.
def plot_edge_with_normal_labels(ax, x1, y1, x2, y2, label_i, label_c, linestyle):
    ax.plot([x1, x2], [y1, y2], color=ORANGE, lw=1.5, linestyle=linestyle, zorder=2)
    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
    dx, dy = x2 - x1, y2 - y1
    length = np.hypot(dx, dy)
    ux, uy = dx / length, dy / length
    right_n = (uy, -ux)
    left_n = (-uy, ux)

    angle = np.degrees(np.arctan2(uy, ux))
    if angle > 90:
        angle -= 180
    elif angle <= -90:
        angle += 180

    e = E_FRAC * length
    ax.text(mid_x + e * left_n[0], mid_y + e * left_n[1], label_i,
             ha="center", va="center", rotation=angle, rotation_mode="anchor",
             fontsize=FONT, color=TEXT)
    ax.text(mid_x + e * right_n[0], mid_y + e * right_n[1], label_c,
             ha="center", va="center", rotation=angle, rotation_mode="anchor",
             fontsize=FONT, color=DGREY)

# direction 0: (u,v)→(u,v+1) — goes +y in the plot
for u, v in h_edges:
    label_c = f"({xu(u):.2f}, {(yv(v) + yv(v+1))/2:.2f})"
    plot_edge_with_normal_labels(ax2, xu(u), yv(v), xu(u), yv(v + 1),
                                  f"({u}, {v})", label_c, "-")

# direction 1: (u,v)→(u+1,v) — goes +x in the plot
for u, v in v_edges:
    label_c = f"({(xu(u)+xu(u+1))/2:.2f}, {yv(v):.2f})"
    plot_edge_with_normal_labels(ax2, xu(u), yv(v), xu(u + 1), yv(v),
                                  f"({u}, {v})", label_c, "--")

# direction 2: (u,v+1)→(u+1,v) — goes +x, −y in the plot
for u, v in d_edges:
    mid_x = (xu(u) + xu(u + 1)) / 2
    mid_y = (yv(v + 1) + yv(v)) / 2
    label_c = f"({mid_x:.2f}, {mid_y:.2f})"
    plot_edge_with_normal_labels(ax2, xu(u), yv(v + 1), xu(u + 1), yv(v),
                                  f"({u}, {v + 1})", label_c, ":")

leg = [
    mpatches.Patch(color=TEXT, label="(u_ind, v_ind)"),
    mpatches.Patch(color=GREY,    label="(x_unit_coord, y_unit_coord)"),
    Line2D([0], [0], color=ORANGE, lw=1.5, linestyle="-",  label="edge_direction=0  (vertical)"),
    Line2D([0], [0], color=ORANGE, lw=1.5, linestyle="--", label="edge_direction=1  (horizontal)"),
    Line2D([0], [0], color=ORANGE, lw=1.5, linestyle=":",  label="edge_direction=2  (diagonal)"),
]
legend_kwargs = dict(handles=leg, loc="upper right", fontsize=FONT, frameon=True,
                      edgecolor=(GREY if DARK else "none"), framealpha=(0.0 if DARK else 1.0),
                      bbox_to_anchor=(1.2018, -0.22), bbox_transform=ax2.transData)
if DARK:
    legend_kwargs["facecolor"] = "none"
legend = ax2.legend(**legend_kwargs)

fig.subplots_adjust(left=0.04, right=0.96, top=0.93, bottom=0.07, wspace=0.08)

# Align "Node defaults" title and arrows with add_chain / add_grid.
CHAIN_AXES_LEFT_FRAC = 0.010714
CHAIN_ARROW_FRAC     = 0.055195
fig.canvas.draw()

_fig_px       = fig.get_figwidth() * fig.dpi
_target_px    = CHAIN_AXES_LEFT_FRAC * _fig_px
_ax1_left_px  = ax1.transAxes.transform((0, 0))[0]
_ax1_right_px = ax1.transAxes.transform((1, 0))[0]
_ax1_width_px = _ax1_right_px - _ax1_left_px
_title_x_axes = (_target_px - _ax1_left_px) / _ax1_width_px
_title_y_axes = 1.0 + 4 / (ax1.get_position().height * fig.get_figheight() * 72)
ax1.text(_title_x_axes, _title_y_axes, _node_title_text,
         fontsize=FONT_TITLE, va="bottom", transform=ax1.transAxes, clip_on=False)

target_px = CHAIN_ARROW_FRAC * _fig_px
ox = ax1.transData.inverted().transform((target_px, 0))[0]
oy = -0.45

# Triangular grid: u_ind increases rightward (+x), v_ind increases upward (+y).
ax1.annotate("", xy=(ox + 0.14, oy), xytext=(ox, oy),
             arrowprops=dict(arrowstyle="->", color=TEXT, lw=1.2))
ax1.text(ox + 0.15, oy, "u_ind", ha="left", va="center", fontsize=FONT, color=TEXT)
ax1.annotate("", xy=(ox, oy + 0.15), xytext=(ox, oy),
             arrowprops=dict(arrowstyle="->", color=TEXT, lw=1.2))
ax1.text(ox + 0.01, oy + 0.15, "v_ind", ha="left", va="center", fontsize=FONT, color=TEXT)

OUT_PATH = "docs/assets/add_tri_defaults_dark.png" if DARK else "docs/assets/add_tri_defaults.png"
plt.savefig(OUT_PATH, dpi=150, facecolor=BG)
print(f"Saved {OUT_PATH}")
