"""
Example: Form-find a Cable Net with the Force Density Method (FDM)
"""

import torch
import torch_structure as ts
import matplotlib.pyplot as plt
import torch_geometric as pyg
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np




radial_res = 5
longitudinal_res = 5
length = 10
tuberadius = .5
verticalDamping = 1

fd1 = torch.tensor([ 50.0000,  50.0000,   2.0674,  50.0000,  50.0000,  50.0000,   6.2530,
         50.0000,  50.0000,  50.0000,   7.5458,  50.0000,  50.0000,  50.0000,
         10.0682,  50.0000,  50.0000,  50.0000,   6.3591,  50.0000, 188.0469,
         50.0000,   4.2280,  50.0000, 179.3576,  50.0000,   7.1369,  50.0000,
        172.2500,  50.0000,   6.5755,  50.0000, 179.2852,  50.0000,   7.1610,
         50.0000, 187.8705,  50.0000,   6.5233,  50.0000, 187.5598,  50.0000,
          3.2458,  50.0000, 177.1587,  50.0000,   7.5889,  50.0000, 172.4621,
         50.0000,   6.8982,  50.0000, 177.0580,  50.0000,   4.8001,  50.0000,
        186.9694,  50.0000,   9.0197,  50.0000, 185.8143,  50.0000,   5.7365,
         50.0000, 180.2046,  50.0000,   3.3847,  50.0000, 173.0951,  50.0000,
         11.1269,  50.0000, 178.0423,  50.0000,   7.8258,  50.0000, 186.6521,
         50.0000,   4.1065,  50.0000, 186.1736,  50.0000,   4.5073,  50.0000,
        184.6012,  50.0000,   2.7028,  50.0000, 173.6736,  50.0000,  12.0673,
         50.0000, 183.0013,  50.0000,  10.6971,  50.0000, 185.3916,  50.0000,
          2.7305,  50.0000,  50.0000,  50.0000,  50.0000,  50.0000,  50.0000,
         50.0000,  50.0000,  50.0000,  50.0000,  50.0000], dtype=torch.float64)


print(fd1)


fd2 = torch.tensor([5.0000e+01, 5.0000e+01, 1.0000e-05, 5.0000e+01, 5.0000e+01, 5.0000e+01,
        1.0032e+01, 5.0000e+01, 5.0000e+01, 5.0000e+01, 3.1329e+00, 5.0000e+01,
        5.0000e+01, 5.0000e+01, 4.7835e+00, 5.0000e+01, 5.0000e+01, 5.0000e+01,
        9.8992e+00, 5.0000e+01, 1.6886e+02, 5.0000e+01, 1.0349e+00, 5.0000e+01,
        1.5742e+02, 5.0000e+01, 4.5151e+00, 5.0000e+01, 1.5119e+02, 5.0000e+01,
        7.1302e+00, 5.0000e+01, 1.5395e+02, 5.0000e+01, 5.6146e+00, 5.0000e+01,
        1.6889e+02, 5.0000e+01, 1.0068e+01, 5.0000e+01, 1.6376e+02, 5.0000e+01,
        6.9674e+00, 5.0000e+01, 1.6152e+02, 5.0000e+01, 1.6713e-01, 5.0000e+01,
        1.5320e+02, 5.0000e+01, 9.5978e+00, 5.0000e+01, 1.6040e+02, 5.0000e+01,
        1.0350e+01, 5.0000e+01, 1.6685e+02, 5.0000e+01, 1.9025e+00, 5.0000e+01,
        1.6703e+02, 5.0000e+01, 1.0033e+00, 5.0000e+01, 1.5990e+02, 5.0000e+01,
        9.8759e+00, 5.0000e+01, 1.5237e+02, 5.0000e+01, 4.3266e+00, 5.0000e+01,
        1.6171e+02, 5.0000e+01, 7.9110e+00, 5.0000e+01, 1.6391e+02, 5.0000e+01,
        5.2269e+00, 5.0000e+01, 1.6863e+02, 5.0000e+01, 1.0000e-05, 5.0000e+01,
        1.5544e+02, 5.0000e+01, 6.4353e+00, 5.0000e+01, 1.4855e+02, 5.0000e+01,
        5.5640e+00, 5.0000e+01, 1.5616e+02, 5.0000e+01, 4.3135e+00, 5.0000e+01,
        1.7125e+02, 5.0000e+01, 1.1800e+01, 5.0000e+01, 5.0000e+01, 5.0000e+01,
        5.0000e+01, 5.0000e+01, 5.0000e+01, 5.0000e+01, 5.0000e+01, 5.0000e+01,
        5.0000e+01, 5.0000e+01], dtype=torch.float64)

def createStraightPolyline(segments: int, length: int):

    seg_length = length / segments
    polyline = []

    for i in range(segments + 1):
        polyline.append([0, i * seg_length,0])

    return polyline


def rotation_matrix_y(theta_degrees):
    theta = np.radians(theta_degrees) 
    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])



fdtwotubes = torch.cat([fd2, fd1], dim=0) 

polyline = createStraightPolyline(longitudinal_res, length)


rotation = [rotation_matrix_y(-45)]
translation = [[10,0,0]]


input_params = {
    'polyline': polyline,                     
    'startNormal': [0,0,1],  
    'endNormal': [0,0,1],  
    'startBinormal': [1,0,0],
    'endBinormal': [-1,0,0],             
    'radius': tuberadius,
    'radialRes': radial_res,
    'verticalDamping': verticalDamping,
    'rotation' : rotation,
    'translation' : translation    
}

pneu_tube = ts.generators.PneuTube(**input_params)

hetero_data = pneu_tube.hetero_graph  

data = pneu_tube.graph


refNodes = torch.from_numpy(pneu_tube.refNodes)

force_densities = torch.full((data.num_edges,), 50.0, dtype=torch.float64)
#force_densities = fdtwotubes
#force_densities = fd1

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
            hetero_data['node'].coords = data.coords
        else:
            hetero_data['node'].coords = refNodes.float()


        load = pneu_tube.calculate_loads(pressure)

        data.load = load

        old_coords = data.coords

        data.fdm(inplace=True, C=C)

        data.pattern_coords = data.coords[:, :2]
        data.z_coord = data.coords[:, 2]
        
        step = old_coords - data.coords

        norm_step = torch.norm(step)
        norm_steps.append(norm_step)
        if i == max_its -1:
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

    data.force_density = x[:-1]   
    pressure = x[-1]   
    
    data = iterative_fdm(data, pressure, hetero_data, eps, max_its, useRealLoad)
    
    oldcoords = data.coords

    data.coords = refNodes


    data.coords = oldcoords

    lengthsRadial = data.length_from_coords[data.is_radial.view(-1)]


    evenlengthsradial = torch.sum((lengthsRadial - torch.mean(lengthsRadial))**2)
    arc =  torch.norm(refNodes-data.coords)**2

    if useRealLoad:

        return evenlengthsradial + arc/70
    
    else:
        hetero_data['node'].coords = refNodes.float()
        load1 = pneu_tube.calculate_loads(pressure)
        hetero_data['node'].coords = data.coords
        load2 = pneu_tube.calculate_loads(pressure)

        #return (torch.norm(load1 - load2)**2)/100 + evenlengthsradial + arc/70
        return evenlengthsradial + arc/40

lowerbound = 0.00001
upperbound = None


n_vars = force_densities.shape[0]

bounds = [(lowerbound, upperbound) for _ in range(n_vars)]

pressure = 300

bounds.append((10, 500))

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
optimized_forces = res[:-1]
pressure = res[-1]

data.force_density = optimized_forces
data = iterative_fdm(data, pressure, hetero_data, eps, 1000, False)

newcoords = data.coords.numpy() + np.array([10, 0, 0])
print(optimized_forces)
#pneu_tube.addTube(radial_res, longitudinal_res, newcoords)

data.plot(title="Pneu Tube", legend=False, show_load=True, load=data.load, force_scale = .01, lw_scale = .1)

plt.show()





