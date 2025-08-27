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


fd1 = torch.tensor([5.0000e+01, 5.0000e+01, 5.5436e+01, 5.0000e+01, 5.0000e+01, 5.0000e+01,
        9.3537e+00, 5.0000e+01, 5.0000e+01, 5.0000e+01, 1.0000e-05, 5.0000e+01,
        5.0000e+01, 5.0000e+01, 1.0000e-05, 5.0000e+01, 5.0000e+01, 5.0000e+01,
        9.3534e+00, 5.0000e+01, 1.3183e+02, 5.0000e+01, 3.1719e+01, 5.0000e+01,
        1.2224e+02, 5.0000e+01, 1.7653e+01, 5.0000e+01, 1.2148e+02, 5.0000e+01,
        4.9498e+00, 5.0000e+01, 1.2224e+02, 5.0000e+01, 4.9499e+00, 5.0000e+01,
        1.3183e+02, 5.0000e+01, 1.7654e+01, 5.0000e+01, 1.4034e+02, 5.0000e+01,
        1.9470e+01, 5.0000e+01, 1.3389e+02, 5.0000e+01, 1.3240e+01, 5.0000e+01,
        1.2957e+02, 5.0000e+01, 1.5604e+01, 5.0000e+01, 1.3389e+02, 5.0000e+01,
        1.5604e+01, 5.0000e+01, 1.4034e+02, 5.0000e+01, 1.3240e+01, 5.0000e+01,
        1.5046e+02, 5.0000e+01, 1.6577e+01, 5.0000e+01, 1.4428e+02, 5.0000e+01,
        1.1675e+01, 5.0000e+01, 1.3787e+02, 5.0000e+01, 1.7630e+01, 5.0000e+01,
        1.4428e+02, 5.0000e+01, 1.7630e+01, 5.0000e+01, 1.5046e+02, 5.0000e+01,
        1.1675e+01, 5.0000e+01, 1.5397e+02, 5.0000e+01, 1.6674e+01, 5.0000e+01,
        1.4692e+02, 5.0000e+01, 1.1674e+01, 5.0000e+01, 1.4138e+02, 5.0000e+01,
        1.7484e+01, 5.0000e+01, 1.4692e+02, 5.0000e+01, 1.7483e+01, 5.0000e+01,
        1.5397e+02, 5.0000e+01, 1.1674e+01, 5.0000e+01, 1.5049e+02, 5.0000e+01,
        1.9603e+01, 5.0000e+01, 1.4457e+02, 5.0000e+01, 1.3285e+01, 5.0000e+01,
        1.3780e+02, 5.0000e+01, 1.5537e+01, 5.0000e+01, 1.4457e+02, 5.0000e+01,
        1.5537e+01, 5.0000e+01, 1.5049e+02, 5.0000e+01, 1.3285e+01, 5.0000e+01,
        1.4042e+02, 5.0000e+01, 3.1675e+01, 5.0000e+01, 1.3435e+02, 5.0000e+01,
        1.7597e+01, 5.0000e+01, 1.2949e+02, 5.0000e+01, 4.9958e+00, 5.0000e+01,
        1.3435e+02, 5.0000e+01, 4.9958e+00, 5.0000e+01, 1.4042e+02, 5.0000e+01,
        1.7597e+01, 5.0000e+01, 1.3199e+02, 5.0000e+01, 5.5560e+01, 5.0000e+01,
        1.2229e+02, 5.0000e+01, 8.9672e+00, 5.0000e+01, 1.2170e+02, 5.0000e+01,
        3.2094e-01, 5.0000e+01, 1.2229e+02, 5.0000e+01, 3.2094e-01, 5.0000e+01,
        1.3199e+02, 5.0000e+01, 8.9674e+00, 5.0000e+01, 5.0000e+01, 5.0000e+01,
        5.0000e+01, 5.0000e+01, 5.0000e+01, 5.0000e+01, 5.0000e+01, 5.0000e+01,
        5.0000e+01, 5.0000e+01], dtype=torch.float64)


pressure1 = 93.5405

fd2 = torch.tensor([ 50.0000,  50.0000,  67.9777,  50.0000,  50.0000,  50.0000,   1.0000,
         50.0000,  50.0000,  50.0000,   3.1589,  50.0000,  50.0000,  50.0000,
          3.1589,  50.0000,  50.0000,  50.0000,   1.1016,  50.0000, 135.5380,
         50.0000,  33.1816,  50.0000, 122.6982,  50.0000,  17.7474,  50.0000,
        121.4431,  50.0000,   4.0786,  50.0000, 122.6981,  50.0000,   4.0783,
         50.0000, 135.5379,  50.0000,  17.7474,  50.0000, 140.3616,  50.0000,
         22.4516,  50.0000, 134.1313,  50.0000,  14.8838,  50.0000, 131.7919,
         50.0000,  12.2534,  50.0000, 134.1309,  50.0000,  12.2521,  50.0000,
        140.3614,  50.0000,  14.8841,  50.0000, 150.4275,  50.0000,  17.4539,
         50.0000, 143.8289,  50.0000,  12.1722,  50.0000, 139.1050,  50.0000,
         16.9602,  50.0000, 143.8287,  50.0000,  16.9598,  50.0000, 150.4274,
         50.0000,  12.1721,  50.0000, 154.3517,  50.0000,  17.6482,  50.0000,
        148.1672,  50.0000,  12.1618,  50.0000, 142.0498,  50.0000,  16.8068,
         50.0000, 148.1672,  50.0000,  16.8063,  50.0000, 154.3515,  50.0000,
         12.1618,  50.0000, 150.3535,  50.0000,  23.0594,  50.0000, 143.6996,
         50.0000,  14.9598,  50.0000, 139.2181,  50.0000,  11.8353,  50.0000,
        143.6994,  50.0000,  11.8339,  50.0000, 150.3532,  50.0000,  14.9602,
         50.0000, 140.1773,  50.0000,  35.1079,  50.0000, 133.7085,  50.0000,
         17.2148,  50.0000, 132.0210,  50.0000,   3.5242,  50.0000, 133.7083,
         50.0000,   3.5237,  50.0000, 140.1769,  50.0000,  17.2149,  50.0000,
        135.4725,  50.0000,  68.9448,  50.0000, 122.0190,  50.0000,   2.3163,
         50.0000, 121.2305,  50.0000,   1.0000,  50.0000, 122.0193,  50.0000,
          1.0000,  50.0000, 135.4720,  50.0000,   2.3163,  50.0000,  50.0000,
         50.0000,  50.0000,  50.0000,  50.0000,  50.0000,  50.0000,  50.0000,
         50.0000,  50.0000], dtype=torch.float64)

pressure2 = 94.2037

radial_res = 8
longitudinal_res = 8
length = 10
tuberadius = .8
verticalDamping = 1




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





polyline = createStraightPolyline(longitudinal_res, length)


rotation = [rotation_matrix_y(0)]
translation = [[0,0,0]]
scaling = [1.3]

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

force_densities = torch.full((data.num_edges,), 50, dtype=torch.float64)
#force_densities = fd1
#force_densities = fd2
pressure = pressure2
pressure = 150



eps = 1e-5
max_its = 500
ftol = 1e-12
gtol = 1e-15
data.force_density = force_densities

norm_steps = []

C = ts.formfinding.utils.create_branch_node_matrix(data.edge_index[:, data.directed_mask.view(-1)])

data.plot()
plt.show()

def iterative_fdm(data, pressure, hetero_data, eps, max_its, useRealLoad = False):

    for i in range(max_its):
        

        #if torch.max(data.coords) > 20:
         #   print("hallo")
         #   break

        if useRealLoad:
            print("hallo")
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


f_val = []

def make_callback():
    def callback(x):
        print(f"Iteration {callback.iteration:3d} | Val: {arc.best_loss:.6f}")
        callback.iteration += 1
        f_val.append(arc.best_loss)
        global count
        count += 1
    callback.iteration = 0
    return callback



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
        return evenlengthsradial + arc/70

lowerbound = 1
upperbound = None


n_vars = force_densities.shape[0]

bounds = [(lowerbound, upperbound) for _ in range(n_vars)]


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
optimized_forces = res[:-1]
pressure = res[-1:]

data.force_density = optimized_forces
data = iterative_fdm(data, pressure, hetero_data, eps, 1000, False)

data.plot(title="Pneu Tube", legend=False, show_load=True, load=data.load, force_scale = .005, lw_scale = .09)
plt.show()
data.plot(lw_scale = .09)
plt.show()



iterations = range(len(f_val))
plt.plot(iterations, f_val, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Norm of Optimization Step')
plt.yscale('log')
plt.grid(True)
plt.show()





