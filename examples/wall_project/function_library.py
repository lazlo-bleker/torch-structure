import torch
from torch_structure.data import StructData

from params import nu, sw_point, se_point, mw_point, me_point, nw_point, ne_point, s

move = 0.0
k = 1. * torch.pi

def evaluate_top_coords_target():
    # Linearly interpolate southwest point and southeast point
    ts = torch.linspace(0.0, 1.0, nu).unsqueeze(dim=1)
    coords_target = nw_point * (1-ts) + ne_point * ts
    coords_target = coords_target + torch.tensor([0.0, 0.0, move]) * torch.sin(k * ts)
    return coords_target

def evaluate_center_coords_target():
    # Linearly interpolate southwest point and southeast point
    ts = torch.linspace(0.0, 1.0, nu).unsqueeze(dim=1)
    coords_target = mw_point * (1-ts) + me_point * ts
    coords_target = coords_target - torch.tensor([0.0, 0.0, 0.0]) * torch.sin(k * ts)
    return coords_target

def evaluate_bottom_coords_target():
    # Linearly interpolate southwest point and southeast point
    ts = torch.linspace(0.0, 1.0, nu).unsqueeze(dim=1)
    coords_target = sw_point * (1-ts) + se_point * ts
    coords_target = coords_target - torch.tensor([0.0, 0.0, move]) * torch.sin(k * ts)
    return coords_target

def center_coords_constr_func(graph_solved : StructData, target_coords):
    log = {}
    # Get bottom coords
    v_coords = graph_solved.uv_coords[:,1]
    v_coord = int(max(v_coords)/2)
    center_mask = (v_coords == v_coord)
    center_coords = graph_solved.coords[center_mask]
    center_coords = torch.flatten(center_coords)
    center_coords = torch.reshape(center_coords, [-1,3])
    # Compare current coords with target coords
    diff = center_coords - target_coords
    diff = diff[:,[0]]
    return diff.flatten(), log

def bottom_coords_constr_func(graph_solved : StructData, target_coords):
    log = {}
    # Get bottom coords
    v_coords = graph_solved.uv_coords[:,1]
    bottom_mask = (v_coords == max(v_coords))
    bottom_coords = graph_solved.coords[bottom_mask]
    bottom_coords = torch.flatten(bottom_coords)
    bottom_coords = torch.reshape(bottom_coords, [-1,3])
    # Compare current coords with target coords
    diff = bottom_coords - target_coords
    diff = diff[:,[0,2]]
    return diff.flatten(), log

def bottom_coords_obj_func(graph_solved : StructData, target_coords):
    log = {}
    # Get bottom coords
    v_coords = graph_solved.uv_coords[:,1]
    bottom_mask = (v_coords == max(v_coords))
    bottom_coords = graph_solved.coords[bottom_mask]
    bottom_coords = torch.flatten(bottom_coords)
    bottom_coords = torch.reshape(bottom_coords, [-1,3])
    # Compare current coords with target coords
    diff = bottom_coords - target_coords
    diff = diff[:,1]
    return torch.linalg.norm(diff) / (diff.numel() ** 0.5), log

def support_reaction_force_constr_func(graph_solved : StructData):
    log = {}
    # Get bottom coords
    graph_solved.res = graph_solved.reaction_force
    mask = graph_solved.is_support
    reac_all = graph_solved.reaction_force
    reac = reac_all[mask.squeeze()]
    reac = reac[:,[0,1]]
    return reac.flatten(), log


def load_path(graph_solved : StructData):
    log = {}
    edge_load_path = torch.abs(graph_solved.force * graph_solved.length_from_coords)
    return  torch.sum(edge_load_path) / (edge_load_path.numel()), log


def cache_laplacian(graph_solved : StructData):
    kwargs = {}
    # Pre-compute gauss points for quadrature
    gauss_points = []
    s = (1/3) ** 0.5
    w = 1.0
    gp_1d = [(w,-s),(w,s)]
    for weight_xi, coord_xi in gp_1d:
        for weight_eta, coord_eta in gp_1d:
            weight = weight_xi * weight_eta
            gauss_points.append((weight, coord_xi, coord_eta))
    kwargs["gauss_points"] = gauss_points

    # Pre compute element indices
    nodes_uv = graph_solved.uv_coords
    nu = int(max(nodes_uv[:,0])+1)
    nv = int(max(nodes_uv[:,1])+1)
    n_elements = (nu-1) * (nv-1)
    quad_indices = -torch.ones([n_elements,4], dtype=torch.int)
    for u in range(nu-1):
        for v in range(nv-1):
            k = v * (nu - 1) + u
            quad_indices[k,0] = int((v+1) * (nu) + (u))
            quad_indices[k,1] = int((v+1) * (nu) + (u+1))
            quad_indices[k,2] = int(  (v) * (nu) + (u+1))
            quad_indices[k,3] = int(  (v) * (nu) + (u))
    kwargs["quad_indices"] = quad_indices

    return kwargs

def laplacian(graph_solved: "StructData", gauss_points, quad_indices, omega):
    log = {}
    omega_a = omega
    omega_b = 1. - omega_a
    coords   = graph_solved.coords
    device = coords.device
    dtype  = coords.dtype
    loss  = torch.zeros((), dtype=dtype, device=device)
    _loss = torch.zeros(coords.shape[0], dtype=dtype, device=device)
    
    for gp_w, gp_xi, gp_eta in gauss_points:
        # Get shape function grad
        dN_dxi_nj = dN_dxi(gp_xi, gp_eta, dtype=dtype, device=device) 
        # Get node coords
        x_qni = coords[quad_indices]
        # Evaluate Jacobians
        _J_qij = torch.einsum('qni,nj->qij',x_qni, dN_dxi_nj)
        J_qij = _J_qij * (omega_a + omega_b / torch.linalg.norm(_J_qij,dim=1, keepdim=True))
        # Evaluate covariance Jacobians
        JJ_qij = torch.einsum('qki,qkj->qij',J_qij, J_qij)
        # Evaluate determinant of Jacobian
        det_J_q = JJ_qij[:,0,0] * JJ_qij[:,1,1] - JJ_qij[:,0,1] * JJ_qij[:,1,0]
        dA_q = torch.sqrt(det_J_q)
        # Evaluate singular values
        sval_qi = sval(JJ_qij)
        # Least square conformal mapping based on singular values
        lscm_q = (1/sval_qi[:,0] - 1/sval_qi[:,1]) ** 2
        # Add LSCM loss weighted by tributary area
        loss_q = gp_w * dA_q * lscm_q

        # Add losses
        loss = loss + torch.sum(loss_q)
        # Add vertex losses (for visualization)
        _loss[quad_indices[:,0]] += loss_q
        _loss[quad_indices[:,1]] += loss_q
        _loss[quad_indices[:,2]] += loss_q
        _loss[quad_indices[:,3]] += loss_q

    graph_solved.loss = _loss
    return loss, log

def N(xi,eta):
    return 0.25 * torch.tensor([
        (1-xi) * (1-eta),
        (1+xi) * (1-eta),
        (1+xi) * (1+eta),
        (1-xi) * (1+eta),
    ], dtype=torch.float64)

def dN_dxi(xi, eta, dtype, device):
    # rows: nodes (N1..N4), cols: [∂N/∂ξ, ∂N/∂η]
    return 0.25 * torch.tensor([
        [-(1-eta), -(1-xi)],
        [+(1-eta), -(1+xi)],
        [+(1+eta), +(1+xi)],
        [-(1+eta), +(1-xi)],
    ], dtype=dtype, device=device)

def sval(JJ_qij):
    # B = A @ A.T
    disc_q = JJ_qij[:,0,0] * JJ_qij[:,0,0] + JJ_qij[:,1,1] * JJ_qij[:,1,1] + 4 * JJ_qij[:,1,0] * JJ_qij[:,0,1] - 2 * JJ_qij[:,0,0] * JJ_qij[:,1,1]
    b_q = 0.5 * torch.sqrt(disc_q)
    a_q = 0.5 * (JJ_qij[:,0,0] + JJ_qij[:,1,1])
    s_0_q = torch.sqrt(a_q + b_q)
    s_1_q = torch.sqrt(a_q - b_q)
    return torch.stack([s_0_q, s_1_q], dim = 1)


def cache_rectangular(graph_solved : StructData):
    kwargs = {}
    dtype = graph_solved.coords.dtype
    # Pre-compute gauss points for quadrature
    s20 = 1 / (3 ** 0.5)
    w30 = 5/9
    w31 = 8/9
    s30 = 0.6 ** 0.5
    s31 = 0.0
    gp_1d_dict = {
        1 : [(2.0,0.0)],
        2 : [(1.0,s20),(1.0,-s20)],
        3 : [(w30,s30),(w31,s31),(w30,-s30)]
    }
    gp_1d = gp_1d_dict[3]
    gauss_points = []
    for weight_xi, coord_xi in gp_1d:
        for weight_eta, coord_eta in gp_1d:
            weight = weight_xi * weight_eta
            gauss_points.append((weight, coord_xi, coord_eta))
    kwargs["gauss_points"] = gauss_points

    # Pre compute element indices
    nodes_uv = graph_solved.uv_coords
    nu = int(max(nodes_uv[:,0])+1)
    nv = int(max(nodes_uv[:,1])+1)
    n_elements = (nu-1) * (nv-1)
    quad_indices = -torch.ones([n_elements,4], dtype=torch.int)
    for u in range(nu-1):
        for v in range(nv-1):
            k = v * (nu - 1) + u
            quad_indices[k,0] = int((v+1) * (nu) + (u))
            quad_indices[k,1] = int((v+1) * (nu) + (u+1))
            quad_indices[k,2] = int(  (v) * (nu) + (u+1))
            quad_indices[k,3] = int(  (v) * (nu) + (u))
    kwargs["quad_indices"] = quad_indices

    kwargs["normals"] = torch.ones([graph_solved.num_nodes,1], dtype=dtype) @ torch.tensor([0.0, 1.0, 0.0], dtype=dtype).unsqueeze(0)
    return kwargs

def rectangular(graph_solved: "StructData", gauss_points, quad_indices, normals):
    log = {}
    coords   = graph_solved.coords
    device = coords.device
    dtype  = coords.dtype
    loss  = torch.zeros((), dtype=dtype, device=device)
    dim_gp = len(gauss_points)
    dim_quad = quad_indices.shape[0]
    _gauss_points = torch.zeros([dim_quad, dim_gp, 3], device="cpu")
    _design_jacobian = torch.zeros([dim_quad, dim_gp, 2, 3], dtype=dtype, device=device)
    _target_jacobian = torch.zeros([dim_quad, dim_gp, 2, 3], dtype=dtype, device=device)
    _jac_loss = torch.zeros([dim_quad, dim_gp], dtype=dtype, device=device)
    
    x_qni = coords[quad_indices]
    for i_gp, (gp_w, gp_xi, gp_eta) in enumerate(gauss_points):
        # Get shape function grad
        dN_dxi_nj = dN_dxi(gp_xi, gp_eta, dtype=dtype, device=device) 
        N_n = N(gp_xi, gp_eta)
        # Get node coords
        # normals_qni = normals[quad_indices]
        # Evaluate Jacobians
        J_qij = torch.einsum('qni,nj->qij',x_qni, dN_dxi_nj)
        
        with torch.no_grad():
            # SVD of J to get singular values
            _, S_qj, _ = torch.linalg.svd(J_qij, full_matrices=False)

            # build both diagonal matrices in a batch
            B_pqij = torch.diag_embed(torch.stack([S_qj, S_qj.flip(1)]))

            # compute covariance P
            P_pqij= torch.einsum('qik,pqkj->pqij',J_qij, B_pqij)

            # batched SVD: returns H, _, Kh
            H_pqij, _, Kh_pqij = torch.linalg.svd(P_pqij, full_matrices=False)

            # rotations R for both options, and corresponding J-hats
            R_pqij= torch.einsum('pqik,pqkj->pqij',H_pqij, Kh_pqij)
            Jhat_pqij = torch.einsum('pqik,pqkj->pqij',R_pqij, B_pqij)

            # pick the one closest to J in Frobenius norm
            diffs_pqij = J_qij.unsqueeze(0) - Jhat_pqij
            norms_pq = torch.linalg.norm(diffs_pqij, dim=(-2,-1))
            id_q = norms_pq.argmin(dim=0)

            _J_qij = Jhat_pqij[id_q, torch.arange(id_q.shape[0], device=Jhat_pqij.device)]

        e = J_qij - _J_qij
        loss_q = torch.sum(e * e, dim=(1,2))
        loss = loss + gp_w * torch.sum(loss_q)
    
        # Log for debug
        _gauss_points[:,i_gp] = torch.einsum('qni,n->qi',x_qni, N_n)
        _design_jacobian[:,i_gp] = J_qij.transpose(-1,-2)
        _target_jacobian[:,i_gp] = _J_qij.transpose(-1,-2)
        _jac_loss[:,i_gp] = loss_q

    log['J'] = _design_jacobian.reshape(-1, 2, 3)
    log['_J'] = _target_jacobian.reshape(-1, 2, 3)
    log['loss'] = _jac_loss.reshape(-1)
    log['points'] = _gauss_points.reshape(-1, 3)

    return loss, log