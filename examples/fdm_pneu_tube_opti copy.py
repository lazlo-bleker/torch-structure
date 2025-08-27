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




radial_res = 10
longitudinal_res = 10
length = 10
tuberadius = .5
verticalDamping = 1

def createStraightPolyline(segments: int, length: int):

    seg_length = length / segments
    polyline = []

    for i in range(segments + 1):
        #polyline.append([0,i * seg_length,min(i, segments - i)])
        polyline.append([0, i * seg_length,0])

    return polyline


polyline = createStraightPolyline(longitudinal_res, length)

input_params = {
    'polyline': polyline,                     
    'startNormal': [0,0,1],  
    'endNormal': [0,0,1],  
    'startBinormal': [1,0,0],
    'endBinormal': [-1,0,0],             
    'radius': tuberadius,
    'radialRes': radial_res,
    'verticalDamping': verticalDamping      
}

pneu_tube = ts.generators.PneuTube(**input_params)

hetero_data = pneu_tube.hetero_graph  

data = pneu_tube.graph
data.pattern_coords = pneu_tube.refNodes[:, :2]
data.z_coord = pneu_tube.refNodes[:, 2]

refNodes = torch.from_numpy(pneu_tube.refNodes)

def radialforcedensitydistribution(i, j, low=0, high=1):
    x = np.linspace(0, (j-1)/j * 2 * np.pi, j)  
    wave = (np.sin(x) + 1) / 2    
    wave = wave * (high - low) + low  
    return np.repeat(np.tile(wave, i),2)


force_densities = torch.full((data.num_edges,), 50.0, dtype=torch.float64)
#force_densities[data.is_radial.view(-1)] = 500
#force_densities[~data.is_radial.view(-1)] = torch.tensor(radialforcedensitydistribution(longitudinal_res-1, radial_res, 10, 100))

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
            #hetero_data['node'].coords = (1 - min(count, 800) / 800) * refNodes.float() + (min(count, 800) / 800) * data.coords
            hetero_data['node'].coords = refNodes.float()


        load = pneu_tube.calculate_loads(pressure)

        data.load = load

        #data.load[data.is_top_node.view(-1)] += torch.tensor([0, 0, -1])

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

@ts.utils.scipy_jacobian  
def same_lengths(force_densities, data, hetero_data, eps, max_its):
   
    data.force_density = force_densities

    data = iterative_fdm(data, hetero_data, eps, max_its)

    lengths = data.length_from_coords

    val = torch.sum((lengths - torch.mean(lengths))**2)
    f_val.append(float(val))

    return val




#data.plot()
#plt.show()

count = 0
@ts.utils.scipy_jacobian  
def arc(x, data, hetero_data, eps, max_its, useRealLoad = False):

    global count

    data.force_density = x[:-1]   
    pressure = x[-1]   
    
    '''
    if count > 800:
        data = iterative_fdm(data, pressure, hetero_data, eps, max_its, True)
    else:
        data = iterative_fdm(data, pressure, hetero_data, eps, max_its, False)
    '''
    data = iterative_fdm(data, pressure, hetero_data, eps, max_its, useRealLoad)


    
    oldcoords = data.coords

    data.coords = refNodes


    data.coords = oldcoords

    lengthsRadial = data.length_from_coords[data.is_radial.view(-1)]


    evenlengthsradial = torch.sum((lengthsRadial - torch.mean(lengthsRadial))**2)
    arc =  torch.norm(refNodes-data.coords)**2
    #return val
    
    #return evenlengthsradial + arc /70
    #return samelengths/10 + evenlengthsradial + arc/70

    #return arc/70 + evenlengthsradial

    if useRealLoad:

        print(1)
        return evenlengthsradial + arc/70
    else:
        hetero_data['node'].coords = refNodes.float()
        load1 = pneu_tube.calculate_loads(pressure)
        hetero_data['node'].coords = data.coords
        load2 = pneu_tube.calculate_loads(pressure)

        #return (torch.norm(load1 - load2)**2)/100 + evenlengthsradial + arc/70
        return evenlengthsradial + arc/70
    #return arc
    '''if torch.isnan(res):
        return torch.tensor(1e6, dtype=res.dtype, device=res.device) 
    else:
        return res
    '''
    #return val + torch.norm(refNodes-data.coords)**2 + evenlengthsradial + evenlengthslongitudinal
    #return torch.norm(refNodes-data.coords)**2

lowerbound = 0.00001
upperbound = None


n_vars = force_densities.shape[0]

bounds = [(lowerbound, upperbound) for _ in range(n_vars)]

pressure = 150

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
print(optimized_forces)
print(pressure)
data.force_density = optimized_forces
data = iterative_fdm(data, pressure, hetero_data, eps, 1000, False)
print(data.verify_equilibrium())

hetero_data['node'].coords = refNodes.float()
load1 = pneu_tube.calculate_loads(pressure)
hetero_data['node'].coords = data.coords
load2 = pneu_tube.calculate_loads(pressure)

print(torch.norm(load1 -load2))
print((torch.norm(refNodes - data.coords)))
#data.plot()
data.plot(title="Pneu Tube", legend=False, show_load=True, load=data.load, force_scale = .05, lw_scale = .05)

plt.show()





