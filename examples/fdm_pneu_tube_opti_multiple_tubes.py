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


fd1 = torch.tensor([ 50.0000,  50.0000,  71.1764,  50.0000,  50.0000,  50.0000,   1.0000,
         50.0000,  50.0000,  50.0000,   1.3556,  50.0000,  50.0000,  50.0000,
          1.3557,  50.0000,  50.0000,  50.0000,   1.0000,  50.0000, 135.3599,
         50.0000,  35.2393,  50.0000, 121.9061,  50.0000,  18.0402,  50.0000,
        121.3795,  50.0000,   2.5284,  50.0000, 121.9062,  50.0000,   2.5283,
         50.0000, 135.3599,  50.0000,  18.0401,  50.0000, 140.2158,  50.0000,
         22.7265,  50.0000, 134.0166,  50.0000,  14.7081,  50.0000, 130.4610,
         50.0000,  12.2956,  50.0000, 134.0163,  50.0000,  12.2948,  50.0000,
        140.2157,  50.0000,  14.7082,  50.0000, 151.1134,  50.0000,  18.0992,
         50.0000, 144.4076,  50.0000,  11.9503,  50.0000, 138.7677,  50.0000,
         16.6873,  50.0000, 144.4075,  50.0000,  16.6870,  50.0000, 151.1135,
         50.0000,  11.9503,  50.0000, 154.8129,  50.0000,  17.8944,  50.0000,
        147.2489,  50.0000,  11.9109,  50.0000, 142.1288,  50.0000,  16.9609,
         50.0000, 147.2490,  50.0000,  16.9606,  50.0000, 154.8129,  50.0000,
         11.9110,  50.0000, 151.1072,  50.0000,  22.3779,  50.0000, 144.0353,
         50.0000,  14.5583,  50.0000, 138.8616,  50.0000,  12.5709,  50.0000,
        144.0353,  50.0000,  12.5702,  50.0000, 151.1069,  50.0000,  14.5587,
         50.0000, 140.2917,  50.0000,  34.2792,  50.0000, 133.3455,  50.0000,
         19.0732,  50.0000, 130.6514,  50.0000,   2.2689,  50.0000, 133.3457,
         50.0000,   2.2685,  50.0000, 140.2910,  50.0000,  19.0735,  50.0000,
        135.0490,  50.0000,  66.6527,  50.0000, 121.9659,  50.0000,   3.2872,
         50.0000, 121.5477,  50.0000,   1.0000,  50.0000, 121.9662,  50.0000,
          1.0000,  50.0000, 135.0484,  50.0000,   3.2869,  50.0000,  50.0000,
         50.0000,  50.0000,  50.0000,  50.0000,  50.0000,  50.0000,  50.0000,
         50.0000,  50.0000], dtype=torch.float64)

pressure1 = 94.1900

fd2 = torch.tensor([ 50.0000,  50.0000,  45.6195,  50.0000,  50.0000,  50.0000,   5.3701,
         50.0000,  50.0000,  50.0000,   5.4185,  50.0000,  50.0000,  50.0000,
          5.4184,  50.0000,  50.0000,  50.0000,   5.3700,  50.0000, 189.0337,
         50.0000,  34.3235,  50.0000, 189.5775,  50.0000,   6.7898,  50.0000,
        188.0607,  50.0000,  11.2145,  50.0000, 189.5776,  50.0000,  11.2127,
         50.0000, 189.0338,  50.0000,   6.7902,  50.0000, 210.4343,  50.0000,
          7.9712,  50.0000, 204.8218,  50.0000,  12.5039,  50.0000, 196.6325,
         50.0000,  18.4333,  50.0000, 204.8219,  50.0000,  18.4343,  50.0000,
        210.4344,  50.0000,  12.5036,  50.0000, 225.4232,  50.0000,  14.7818,
         50.0000, 221.3008,  50.0000,   1.2969,  50.0000, 207.8374,  50.0000,
         25.6459,  50.0000, 221.3006,  50.0000,  25.6426,  50.0000, 225.4232,
         50.0000,   1.2970,  50.0000, 222.0545,  50.0000,  13.0302,  50.0000,
        230.2661,  50.0000,   2.9272,  50.0000, 215.1903,  50.0000,  24.6209,
         50.0000, 230.2675,  50.0000,  24.6503,  50.0000, 222.0553,  50.0000,
          2.9271,  50.0000, 225.5298,  50.0000,   8.1074,  50.0000, 221.4843,
         50.0000,  12.7101,  50.0000, 207.8557,  50.0000,  18.2093,  50.0000,
        221.4857,  50.0000,  18.2090,  50.0000, 225.5290,  50.0000,  12.7110,
         50.0000, 210.4298,  50.0000,  33.9373,  50.0000, 204.8073,  50.0000,
          6.6455,  50.0000, 196.6174,  50.0000,  11.4255,  50.0000, 204.8073,
         50.0000,  11.4261,  50.0000, 210.4308,  50.0000,   6.6460,  50.0000,
        189.2704,  50.0000,  47.4015,  50.0000, 189.8798,  50.0000,   3.7948,
         50.0000, 188.1328,  50.0000,   6.2673,  50.0000, 189.8797,  50.0000,
          6.2672,  50.0000, 189.2702,  50.0000,   3.7948,  50.0000,  50.0000,
         50.0000,  50.0000,  50.0000,  50.0000,  50.0000,  50.0000,  50.0000,
         50.0000,  50.0000], dtype=torch.float64)


pressure2 = 107.0232

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
    'startNormal': [0,0,1],  
    'endNormal': [0,0,1],  
    'startBinormal': [1,0,0],
    'endBinormal': [-1,0,0],             
    'radius': tuberadius,
    'radialRes': radial_res,
    'verticalDamping': verticalDamping,
    'rotation' : rotation,
    'translation' : translation,
    'scaling' : scaling
}

pneu_tube = ts.generators.PneuTube(**input_params)

hetero_data = pneu_tube.hetero_graph  

data = pneu_tube.graph


refNodes = torch.from_numpy(pneu_tube.refNodes)
print("------------------------------------")
print(data.coords[len(refNodes):])

force_densities = torch.full((data.num_edges,), 30.0, dtype=torch.float64)
force_densities[data.edges_fields.view(-1)] = 700.0
#force_densities = fd1
#force_densities = fdtwotubes
#pressure = twopressures
#pressure = torch.tensor([200,200,200])
#pressure = 150

#force_densities = result = torch.cat((fd1, fd2, fd1), dim=0)

pressure = torch.tensor([pressure1, pressure2, pressure1])


eps = 1e-5
max_its = 500
ftol = 1e-12
gtol = 1e-15
data.force_density = force_densities

norm_steps = []
#data.plot()
#plt.show()

C = ts.formfinding.utils.create_branch_node_matrix(data.edge_index[:, data.directed_mask.view(-1)])


def iterative_fdm(data, pressure, hetero_data, eps, max_its, useRealLoad = False):

    for i in range(max_its):
        if useRealLoad:
            hetero_data['node'].coords = data.coords
        else:
            
            coords = data.coords.clone()
            coords[:len(refNodes)] = refNodes.float()
            hetero_data['node'].coords = coords
        load = pneu_tube.calculate_loads(pressure)

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

    if useRealLoad:

        return evenlengthsradial + arc/70
    
    else:
        #hetero_data['node'].coords = refNodes.float()
        #load1 = pneu_tube.calculate_loads(pressure)
        #hetero_data['node'].coords = data.coords
        #load2 = pneu_tube.calculate_loads(pressure)

        #return (torch.norm(load1 - load2)**2)/100 + evenlengthsradial + arc/70
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
#print(optimized_forces)
#print(pressure)
data.plot(title="Pneu Tube", legend=False, show_load=True, load=data.load, force_scale = .005, lw_scale = .03)
plt.show()
data.plot(lw_scale = .03)
plt.show()





