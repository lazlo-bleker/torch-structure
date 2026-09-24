

import math

import matplotlib.pyplot as plt
import torch

from torch_structure.data.data import StructData


def ring_point(radius, angle_deg, z):
    angle = math.radians(angle_deg)
    return torch.tensor([[radius * math.cos(angle), radius * math.sin(angle), z]])


def assert_symmetric(data, symmetry):

    group_id = data.metadata["name_to_symmetry"][symmetry]
    matrices = data.symmetry_matrices[data.symmetry_matrix_id == group_id]
    orbit_id = data.orbit_id.view(-1)
    position = data.orbit_position.view(-1)
    members = torch.where(data.symmetry_id.view(-1) == group_id)[0]

    # nodes
    seeds = members - position[members]
    m = matrices[position[members]]
    expected = (m[:, :3, :3] @ data.coords[seeds].unsqueeze(-1)).squeeze(-1) + m[:, :3, 3]
    assert torch.allclose(data.coords[members], expected, atol=1e-5), "nodes are not symmetric"

    # edges: matrices[g] @ matrices[p] == matrices[table[g, p]], so symmetry g moves the node
    # at orbit position p to orbit position table[g, p] of the same orbit
    products = matrices.unsqueeze(1) @ matrices.unsqueeze(0)
    distance = (products.unsqueeze(2) - matrices).abs().amax(dim=(-2, -1))
    assert torch.all(distance.amin(dim=2) < 1e-5), "matrices do not form a group"
    table = distance.argmin(dim=2)

    node_at = {(int(orbit_id[i]), int(position[i])): int(i) for i in members}
    force = {tuple(sorted(e)): f for e, f in zip(data.edge_index.t().tolist(), data.force.view(-1).tolist())}
    member_set = set(members.tolist())

    for (u, v), f in force.items():
        if u not in member_set or v not in member_set:
            continue
        for g in range(matrices.shape[0]):
            gu = node_at[(int(orbit_id[u]), int(table[g, position[u]]))]
            gv = node_at[(int(orbit_id[v]), int(table[g, position[v]]))]
            image = tuple(sorted((gu, gv)))
            assert image in force, f"edge {(u, v)} has no symmetric image {image}"
            assert math.isclose(force[image], f), f"edge {image} has force {force[image]}, expected {f}"


# ------------------------------
# 1. Create an empty structure
# ------------------------------
node_attrs = {
    "coords": torch.empty((0, 3), dtype=torch.float),
    "load": torch.empty((0, 3), dtype=torch.float),
    "is_support": torch.empty((0, 1), dtype=torch.bool),
}

edge_attrs = {
    "force": torch.empty((0, 1), dtype=torch.float),
}

data = StructData(node_attrs=node_attrs, edge_attrs=edge_attrs)

# -------------------------------------
# 2. Define and register the symmetry
# -------------------------------------
rot6 = data.create_rotational_symmetry(6)  # 6 copies about the z-axis
mirror = data.create_mirror_symmetry(normal=torch.tensor([0.0, 1.0, 0.0]))  # mirror plane y = 0
d6 = data.combine_symmetry(mirror, rot6)  # mirror nested inside the rotation: 12 copies

data.add_symmetry({"d6": d6}, copy_attrs=["load", "is_support", "force"])
data.view_symmetries()

# A seed at 15° becomes 12 nodes. Orbit position p is the seed, mirrored if p is odd,
# then rotated by 60° * (p // 2):
#   p:       0     1     2     3     4     5   ...
#   angle:  15   -15    75    45   135   105   ...

# ---------------------------------
# 3. Add nodes: one seed per ring
# ---------------------------------
# add_nodes stores every orbit contiguously and ordered by orbit position:
# nodes 0-11 are the inner ring, nodes 12-23 the outer ring.
data.add_nodes(
    names=["inner", "outer"],
    symmetry="d6",
    coords=torch.cat([ring_point(2.0, 15.0, 2.0), ring_point(6.0, 15.0, 0.0)]),
    load=torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, 0.0]]),
    is_support=torch.tensor([[False], [True]]),
)
INNER, OUTER = 0, 1  # orbit ids, in the order the seeds were added

# The mast top lies on the rotation axis, so it is added without a symmetry.
data.add_nodes(names=["mast"], coords=torch.tensor([[0.0, 0.0, 4.0]]), is_support=torch.tensor([[True]]))
MAST = 24

# ------------------------------------------
# 4. Add edges: one seed edge per family
# ------------------------------------------
# Rings: connect position 0 (15°) to its neighbours at position 1 (-15°) and position 3 (45°).
data.add_edges_by_orbit(
    src_orbit_ids=[INNER, INNER, OUTER, OUTER],
    dest_orbit_ids=[INNER, INNER, OUTER, OUTER],
    src_orbit_positions=[0, 0, 0, 0],
    dest_orbit_positions=[1, 3, 1, 3],
    force=torch.tensor([[-5.0], [-5.0], [-8.0], [-8.0]]),
)

# Radial cables (inner 15° -> outer 15°) and cross cables (inner 15° -> outer -15°).
data.add_edges(torch.tensor([[0, 0], [12, 13]]), force=torch.tensor([[3.0], [1.0]]))

# Mast cables: the mast top is outside the symmetry, so these edges are added exactly as given.
data.add_edges(
    torch.stack([torch.full((12,), MAST), torch.arange(12)]),
    force=torch.full((12, 1), 2.0),
)

print(f"built: {data.num_nodes} nodes, {data.edge_index.shape[1] // 2} edges")
assert data.num_nodes == 25
assert data.edge_index.shape[1] // 2 == 5 * 12  # 2 rings, radial, cross and mast cables
assert_symmetric(data, "d6")

# -----------------------------------
# 5. Change it with the setters
# -----------------------------------
# set_node_attr: widen the outer ring by moving the node at orbit position 3 (45°).
# The rest of the ring follows, and the selected node ends up exactly at the given point.
mask = torch.zeros(data.num_nodes, dtype=torch.bool)
mask[12 + 3] = True
data.set_node_attr("coords", mask, ring_point(7.0, 45.0, 0.0))

# set_node_attr_by_name: increase the load on the inner ring. "inner" names the whole orbit,
# which is fine for a copy attribute.
data.set_node_attr_by_name("load", ["inner"], torch.tensor([[0.0, 0.0, -1.5]]))

# set_node_attr_by_orbit: raise the inner ring, addressed via orbit position 7 (165°).
data.set_node_attr_by_orbit("coords", [INNER], [7], ring_point(2.5, 165.0, 2.5))

# set_edge_attr: select one radial cable, by its reciprocal row (outer -> inner), to update all 12.
radial = (data.edge_index[0] == 12 + 6) & (data.edge_index[1] == 6)
data.set_edge_attr("force", radial, torch.tensor([[4.0]]))

# set_edge_attr_by_orbit: update only the inner ring edges between sectors (15° - 45°).
data.set_edge_attr_by_orbit("force", [INNER], [INNER], [0], [3], torch.tensor([[-6.0]]))

assert_symmetric(data, "d6")

# With consider_symmetry=False only the selected node changes, e.g. an extra load on one node.
single = torch.zeros(data.num_nodes, dtype=torch.bool)
single[0] = True
data.set_node_attr("load", single, torch.tensor([[0.0, 0.0, -3.0]]), consider_symmetry=False)

# Edges with an endpoint outside the symmetry are never propagated, so one mast cable
# can be changed on its own even with consider_symmetry=True.
mast_cable = (data.edge_index[0] == MAST) & (data.edge_index[1] == 5)
data.set_edge_attr("force", mast_cable, torch.tensor([[5.0]]))

# --------------------
# 6. Check the result
# --------------------
radius = data.coords[:, :2].norm(dim=1)
assert torch.allclose(radius[12:24], torch.tensor(7.0))
assert torch.allclose(data.coords[12 + 3], ring_point(7.0, 45.0, 0.0))
assert torch.allclose(radius[:12], torch.tensor(2.5)) and torch.allclose(data.coords[:12, 2], torch.tensor(2.5))
assert data.load[0, 2] == -3.0 and torch.all(data.load[1:12, 2] == -1.5)

expected_edges = {  # force: number of edges with that force
    -8.0: 12,  # outer ring
    -6.0: 6,   # inner ring, between sectors
    -5.0: 6,   # inner ring, across the mirror planes
    1.0: 12,   # cross cables
    4.0: 12,   # radial cables
    2.0: 11,   # mast cables
    5.0: 1,    # the single changed mast cable
}
for value, num_edges in expected_edges.items():
    assert (data.force.view(-1) == value).sum() == 2 * num_edges  # both directions are stored

assert_symmetric(data, "d6")
print("all checks passed")

# ---------------
# 7. Plot
# ---------------
data.plot(show_supports=True, show_load=True, title="D6-symmetric cable roof")

plt.show()
