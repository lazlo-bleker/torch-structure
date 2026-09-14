import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

n_rings   = 1
n_sectors = 5

GREY   = "#AAAAAA"
ORANGE = "#E7461E"
DGREY  = "#555555"

SCALE = 2.0   # draw at 2× unit coords so labels have breathing room

def ang(a):
    return a / n_sectors * 2 * np.pi

def node_unit(r, a):
    if r == 0:
        return 0.0, 0.0
    θ = ang(a)
    return r / n_rings * np.cos(θ), r / n_rings * np.sin(θ)

def node_plot(r, a):
    x, y = node_unit(r, a)
    return x * SCALE, y * SCALE

def ha_va(dx, dy):
    if abs(dx) >= abs(dy):
        return ("left" if dx >= 0 else "right"), "center"
    return "center", ("bottom" if dy > 0 else "top")

nodes_ra = [(0, 0)] + [(r, a) for a in range(n_sectors) for r in range(1, n_rings + 1)]

radial_edges = (
    [(0, 0, 1, a) for a in range(n_sectors)] +
    [(r, a, r + 1, a) for a in range(n_sectors) for r in range(1, n_rings)]
)
angular_edges = [
    (r, a, r, (a + 1) % n_sectors)
    for r in range(1, n_rings + 1)
    for a in range(n_sectors)
]

D = 0.20   # radial offset for node index labels
E = 0.20   # offset for edge labels

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8), facecolor="white")

for ax in (ax1, ax2):
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.7, 2.5)
    ax.set_aspect("equal")
    ax.axis("off")

ax1.set_anchor("W")
ax2.set_anchor("E")

# ── Panel 1: Node defaults ──
ax1.set_title("Node defaults", fontsize=16, loc="left", pad=4)

for sr, sa, dr, da in radial_edges:
    ax1.plot([node_plot(sr, sa)[0], node_plot(dr, da)[0]],
             [node_plot(sr, sa)[1], node_plot(dr, da)[1]], color=DGREY, lw=1.5)
for sr, sa, dr, da in angular_edges:
    ax1.plot([node_plot(sr, sa)[0], node_plot(dr, da)[0]],
             [node_plot(sr, sa)[1], node_plot(dr, da)[1]], color=DGREY, lw=1.5)

for r, a in nodes_ra:
    xp, yp = node_plot(r, a)
    xu, yu = node_unit(r, a)
    ax1.scatter(xp, yp, color="white", s=100, zorder=3,
                edgecolors=ORANGE, linewidths=1.5)
    # Center node: index top-left, coord bottom-right (nudged left and down)
    if r == 0:
        ax1.text(xp + D * 0.6, yp + D * 0.4, f"({r}, {a})",
                 ha="left", va="bottom", fontsize=12.5, color="black")
        ax1.text(xp + D * 0.6, yp - D * 0.4, f"({xu:.2f}, {yu:.2f})",
                 ha="left", va="top", fontsize=12.5, color=DGREY)
        continue
    # Bottom node (1,4): index top-left, coord bottom-right
    if r == 1 and a == 4:
        ax1.text(xp - D * 0.9, yp + D * 1.1, f"({r}, {a})",
                 ha="right", va="bottom", fontsize=12.5, color="black")
        ax1.text(xp + D * 0.5, yp - D * 0.3, f"({xu:.2f}, {yu:.2f})",
                 ha="left", va="top", fontsize=12.5, color=DGREY)
        continue
    # Mid-right node (1,0): index top-left, coord bottom-right
    if r == 1 and a == 0:
        ax1.text(xp - D, yp + D * 0.3, f"({r}, {a})",
                 ha="right", va="bottom", fontsize=12.5, color="black")
        ax1.text(xp + D * 0.4, yp - D * 0.3, f"({xu:.2f}, {yu:.2f})",
                 ha="left", va="top", fontsize=12.5, color=DGREY)
        continue
    # Top node (1,1): index top-left, coord bottom-right
    if r == 1 and a == 1:
        ax1.text(xp - D, yp + D * 0.3, f"({r}, {a})",
                 ha="right", va="bottom", fontsize=12.5, color="black")
        ax1.text(xp + D, yp - D * 0.3, f"({xu:.2f}, {yu:.2f})",
                 ha="left", va="top", fontsize=12.5, color=DGREY)
        continue
    # Index: radially outward; center node: straight up (90° clears all 5 radial edges)
    if r == 0:
        odx, ody = 0.0, D
    else:
        θ = ang(a)
        odx, ody = D * np.cos(θ), D * np.sin(θ)
    ha_i, va_i = ha_va(odx, ody)
    ax1.text(xp + odx, yp + ody, f"({r}, {a})",
             ha=ha_i, va=va_i, fontsize=12.5, color="black")
    # Coord: radially inward (opposite side from index); ring1 at radius 2 so no center overlap
    ha_c, va_c = ha_va(-odx, -ody)
    coord_y_shift = 0.08 if (r == 1 and a == 2) else (-0.08 if (r == 1 and a == 3) else 0.0)
    coord_x_shift = 0.10 if (r == 1 and a in (2, 3)) else 0.0
    ax1.text(xp - odx + coord_x_shift, yp - ody + coord_y_shift, f"({xu:.2f}, {yu:.2f})",
             ha=ha_c, va=va_c, fontsize=12.5, color=DGREY)

# r_ind: diagonal arrow (45°, top-right); a_ind: counter-clockwise
# sixth-circle arc from the r_ind direction up to 105°, matching the actual
# CCW convention used by add_polar_grid (x=r·cos θ, y=r·sin θ, θ increasing).
fig.canvas.draw()
ox = ax1.get_xlim()[0] + 0.15   # just inside left edge of ax1
oy = -2.3

L = 0.40       # arc radius — scaled to match physical indicator size of tri figure
EXT = 0.10     # extra length added to each end of the r_ind arrow
Dr = np.array([np.cos(np.radians(45)), np.sin(np.radians(45))])
arr_start = np.array([ox, oy]) - EXT * Dr
arr_tip   = np.array([ox, oy]) + (L + EXT * 0.4) * Dr

# r_ind: diagonal arrow at 45°, extended beyond the arc in both directions
ax1.annotate("", xy=arr_tip, xytext=arr_start,
             arrowprops=dict(arrowstyle="->", color="black", lw=1.2))
ax1.text(arr_start[0] - 0.02, arr_start[1] + 0.01, "r_ind", ha="right", va="bottom", fontsize=12.5, color="black")

# a_ind: CCW sixth-circle from 45° (r_ind direction) to 105°
arc = mpatches.Arc((ox, oy), 2 * L, 2 * L, angle=0, theta1=45, theta2=105,
                   color="black", lw=1.2)
ax1.add_patch(arc)
# arrowhead at end of arc (angle 105°), pointing CCW
end_x = ox + L * np.cos(np.radians(105))
end_y = oy + L * np.sin(np.radians(105))
prev_x = ox + L * np.cos(np.radians(104))
prev_y = oy + L * np.sin(np.radians(104))
ax1.annotate("", xy=(end_x, end_y), xytext=(prev_x, prev_y),
             arrowprops=dict(arrowstyle="->", color="black", lw=1.2))
ax1.text(end_x - 0.05, end_y + 0.03, "a_ind", ha="right", va="bottom", fontsize=12.5, color="black")

# ── Panel 2: Edge defaults ──
ax2.set_title("Edge defaults", fontsize=16, loc="left", pad=4)

for r, a in nodes_ra:
    ax2.scatter(*node_plot(r, a), color="white", s=100, zorder=3,
                edgecolors=DGREY, linewidths=1.5)

# direction 0 — radial edges: draw all; label only ring_r→ring_r+1.
# Each label is placed at an explicit (x, y) position in the same plot
# coordinates as everything else in this axes (ax2.set_xlim(-2.5, 2.5) etc.)
# -- edit these directly to nudge a label anywhere.
# Per-edge (keyed by da, the ring1 sector the edge points to):
#   (index_x, index_y, coord_x, coord_y)
_radial_label_pos = {
    0: (1, 0.08, 1, -0.08),
    1: (0.08, 0.4, 0.25, 0.54),
    2: (-1.2, 0.69, -0.60, 0.84),
    3: (-0.40, -0.5, -1.05, -0.4),
    4: (0.41, -1.07, 0.25, -0.95)
}
for sr, sa, dr, da in radial_edges:
    x1p, y1p = node_plot(sr, sa)
    x2p, y2p = node_plot(dr, da)
    ax2.plot([x1p, x2p], [y1p, y2p], color=ORANGE, lw=1.5, linestyle="-", zorder=2)
    x1u, y1u = node_unit(sr, sa)
    x2u, y2u = node_unit(dr, da)
    mxu, myu = (x1u + x2u) / 2, (y1u + y2u) / 2
    θ = ang(da)
    tx, ty = -np.sin(θ), np.cos(θ)           # tangential (CCW perp to radial), for text alignment only
    ix_p, iy_p, cx_p, cy_p = _radial_label_pos[da]
    # a_ind: for edges sourced at the center (sr == 0), the source node has
    # no angular index of its own, so the sector the edge points into (da)
    # is used instead -- matching add_polar_grid's default a_ind.
    label_a = da if sr == 0 else sa
    ha_i, va_i = ha_va(tx, ty)
    ax2.text(ix_p, iy_p, f"({sr}, {label_a})",
             ha=ha_i, va=va_i, fontsize=12.5, color="black")
    ha_c, va_c = ha_va(-tx, -ty)
    ax2.text(cx_p, cy_p, f"({mxu:.2f}, {myu:.2f})",
             ha=ha_c, va=va_c, fontsize=12.5, color=DGREY)

# direction 1 — angular edges: above/below placement to stay clear of center
for sr, sa, dr, da in angular_edges:
    x1p, y1p = node_plot(sr, sa)
    x2p, y2p = node_plot(dr, da)
    x1u, y1u = node_unit(sr, sa)
    x2u, y2u = node_unit(dr, da)
    ax2.plot([x1p, x2p], [y1p, y2p], color=ORANGE, lw=1.5, linestyle="--", zorder=2)
    mxp, myp = (x1p + x2p) / 2, (y1p + y2p) / 2
    mxu, myu = (x1u + x2u) / 2, (y1u + y2u) / 2
    # per-edge nudges: (sa): (ind_x_shift, crd_x_shift, ind_y_shift, crd_y_shift)
    _nudge = {0: (-0.25, 0.6, -0.3, 0.1), 1: (0.0, 0.0, -0.1, 0.0), 
              4: (-0.19, +0.5, -0.1, 0.18), 2: (-0.23, +0.49, 0.00, 0.38), 
              3: (0.35, -0.35, -0.2, 0.1)}
    ix, cx, iy, cy = _nudge.get(sa, (0.0, 0.0, 0.0, 0.0))
    ax2.text(mxp + ix, myp + E + iy, f"({sr}, {sa})",
             ha="center", va="bottom", fontsize=12.5, color="black")
    ax2.text(mxp + cx, myp - E + cy, f"({mxu:.2f}, {myu:.2f})",
             ha="center", va="top", fontsize=12.5, color=DGREY)

leg = [
    mpatches.Patch(color="black", label="(r_ind, a_ind)"),
    mpatches.Patch(color=GREY,    label="(x_unit_coord, y_unit_coord)"),
    Line2D([0], [0], color=ORANGE, lw=1.5, linestyle="-",  label="edge_direction=0  (radial)"),
    Line2D([0], [0], color=ORANGE, lw=1.5, linestyle="--", label="edge_direction=1  (angular)"),
]
ax2.legend(handles=leg, loc="upper right", fontsize=12.5, frameon=True,
           edgecolor=GREY, framealpha=1.0,
           bbox_to_anchor=(2.5, -2.1), bbox_transform=ax2.transData)

fig.subplots_adjust(left=0.04, right=0.96, top=0.93, bottom=0.07, wspace=0.08)

plt.savefig("docs/assets/add_polar_defaults.png", dpi=150, facecolor="white")
print("Saved docs/assets/add_polar_defaults.png")
