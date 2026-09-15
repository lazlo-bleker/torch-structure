import torch
from torch_structure.data.data import StructData
import matplotlib.pyplot as plt

n_square = 4
n_pentagon = 5
n_triangle = 3

square_radius = 1.0
pentagon_radius = 3.0
triangle_radius = 10.0

node_attrs = {"coords": torch.empty((0, 3), dtype=torch.float)}
edge_attrs = {"force": torch.empty((0, 1), dtype=torch.float)}
data = StructData(node_attrs=node_attrs, edge_attrs=edge_attrs)

square = data.create_rotational_symmetry(n_square, origin=torch.tensor([pentagon_radius, 0.0, 0.0]))
pentagon = data.create_rotational_symmetry(n_pentagon, origin=torch.tensor([triangle_radius, 0.0, 0.0]))
triangle = data.create_rotational_symmetry(n_triangle, origin=torch.tensor([0.0, 0.0, 0.0]))

square_in_pentagon = data.combine_symmetry(square, pentagon)
triangle_of_pentagons_of_squares = data.combine_symmetry(square_in_pentagon, triangle)

data.add_symmetry({"triangle_of_pentagons_of_squares": triangle_of_pentagons_of_squares}, transform_attrs=["coords"])

data.add_nodes(symmetry="triangle_of_pentagons_of_squares", coords=torch.stack([
    torch.tensor([square_radius + pentagon_radius, 0.0, 0.0]), 
    torch.tensor([square_radius + pentagon_radius, 0.0, 1.0])
]))

#on creation symmetric vals should be copied over (also names) is this actually happening right now?

data.add_edges_by_orbit(
    src_orbit_ids=[0,0],
    dest_orbit_ids=[0,1],
    src_orbit_position=[0,0],
    dest_orbit_position=[1,0],
    force=torch.tensor([[1.0],[1.0]]),
)

data.plot()
plt.show()
