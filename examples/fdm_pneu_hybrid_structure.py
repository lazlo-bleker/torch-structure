

import torch
import torch_structure as ts
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np

radial_res = 5
longitudinal_res = 8
length = 10
tuberadius = .8
verticalDamping = .8

def createStraightPolyline(segments: int, length: int):

    seg_length = length / segments
    polyline = []

    for i in range(segments + 1):
        polyline.append([0, i * seg_length, 0])
    return polyline



def rotation_matrix_y(theta_degrees):
    theta = np.radians(theta_degrees) 
    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

polyline = createStraightPolyline(longitudinal_res, length)

rotation = [rotation_matrix_y(-55), rotation_matrix_y(0), rotation_matrix_y(55)]
translation = [[-1.2,0,0], [0,-1.1,.7], [1.2,0,0]]
scaling = [1, 1.3, 1]

input_params = {
    'polyline': polyline,                     
    'radius': tuberadius,
    'radialRes': radial_res,
    'verticalDamping': verticalDamping,
    'rotation' : rotation,
    'translation' : translation,
    'scaling' : scaling,
    'hybrid': True
}




pneu_tube = ts.generators.PneuTube(**input_params)

hetero_data = pneu_tube.hetero_graph  

data = hetero_data.data


refNodes = torch.from_numpy(pneu_tube.refNodes)


force_densities = torch.full((data.num_edges,), 30.0, dtype=torch.float64)
force_densities[data.edges_fields.view(-1)] = 700.0

pressure = torch.tensor([90, 90, 90])


eps = 1e-5
max_its = 500
ftol = 1e-12
gtol = 1e-15
data.force_density = force_densities

norm_steps = []

C = ts.formfinding.utils.create_branch_node_matrix(data.edge_index[:, data.directed_mask.view(-1)])




def iterative_fdm(data, pressure, hetero_data, eps, max_its, useRealLoad = False):

    for i in range(max_its):
        if useRealLoad:
            hetero_data.hetero_graph['node'].coords = data.coords
        else:
            
            coords = data.coords.clone()
            coords[:len(refNodes)] = refNodes.float()
            hetero_data.hetero_graph['node'].coords = coords
        load = pneu_tube.calculate_loads(pressure, hetero_data, True)

        data.load = load

        old_coords = data.coords

        data.fdm(inplace=True, C=C)

        data.pattern_coords = data.coords[:, :2]
        data.z_coord = data.coords[:, 2]
        
        step = old_coords - data.coords

        norm_step = torch.norm(step)
        norm_steps.append(norm_step)
        if i == max_its - 1:
            print("maxits")
        if norm_step < eps:
            break

    return data


def make_callback():
    def callback(x):
        print(f"Iteration {callback.iteration:3d} | Val: {arc.best_loss:.6f}")
        callback.iteration += 1

        global count
        count += 1
    callback.iteration = 0
    return callback


f_val = []

count = 0
@ts.utils.scipy_jacobian  
def arc(x, data, hetero_data, eps, max_its, useRealLoad = False):

    global count

    data.force_density = x[:-3]   
    pressure = x[-3:]   
    
    data = iterative_fdm(data, pressure, hetero_data, eps, max_its, useRealLoad)
    
    oldcoords = data.coords

    data.coords = refNodes


    data.coords = oldcoords

    lengthsRadial = data.length_from_coords[data.is_radial.view(-1)]


    evenlengthsradial = torch.sum((lengthsRadial - torch.mean(lengthsRadial))**2)
    arc =  torch.norm(refNodes-data.coords[:len(refNodes)])**2

   
    return evenlengthsradial + arc/70

lowerbound = 10
upperbound = None


n_vars = force_densities.shape[0]

bounds = [(lowerbound, upperbound) for _ in range(n_vars)]

bounds.append((10, None))
bounds.append((10, None))
bounds.append((10, None))

result = minimize(
    fun=arc,
    args=(data, hetero_data, eps, max_its, False),
    x0=np.append(force_densities.detach().numpy(), pressure),
    method="SLSQP",
    jac=True,
    bounds=bounds,
    callback=make_callback(),
    options={
        'disp': True,
        'ftol': ftol,
        'gtol': gtol,
        'maxiter': 2000,
    }
)

res = torch.tensor(result.x, dtype=torch.float32)
optimized_forces = res[:-3]
pressure = res[-3:]

data.force_density = optimized_forces
data = iterative_fdm(data, pressure, hetero_data, eps, 1000, False)

data.plot(title="Pneu Tube", legend=False, show_load=True, load=data.load, force_scale = .005, lw_scale = .03)
plt.show()
data.plot(lw_scale = .03)
plt.show()





