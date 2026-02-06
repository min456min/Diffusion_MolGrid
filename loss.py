import torch

def calc_boundary_violation_loss(G, boundary, w=0.5, alpha=3.0):
    pos = G.pos.view(-1, 3)
    sig = G.sigma.view(-1)
    boundary = boundary.view(-1, 3)
    d = torch.cdist(pos, boundary)  
    d_min = d.min(dim=1).values        
    sig_cut = sig / 2.0
    viol = torch.relu(sig_cut - d_min)
    loss =  viol.pow(alpha)
    return loss.mean()

def calc_overlap_loss(G, alpha=2, w=0.4):
    pos = G.pos
    sig = G.sigma
    N = pos.size(0)
    idx = torch.triu_indices(N, N, offset=1, device=pos.device) # i < j
    dist = ((pos[idx[0]] - pos[idx[1]]).pow(2).sum(-1) + 1e-12).sqrt()
    min_dist = (sig[idx[0]] + sig[idx[1]]) / 2.0
    min_dist *= w
    penalty = torch.relu(min_dist - dist)
    loss = (penalty).pow(alpha)
    return loss.mean()

def calc_norm_loss(query, target):
    return torch.sum((query - target) ** 2)/torch.sum(query ** 2 + target ** 2)

def calc_grid_loss(GA, GB):
    # GA: target grids or AA molecules
    # GB: query grids or CG molecules
    G_temp = GB.get_grid(store=False)
    total_loss = 0 
    for idx, name in enumerate(GA.G):
        # G[name][0]: Grid_type
        # G[name][1]: Tau
        # G[name][2]: Grid_value
        g_a= GA.G[name][2]
        g_b = G_temp[name][2]
        loss = calc_norm_loss(g_a, g_b)
        print(f'{name} Los:,   grid_type:{GA.G[name][0]},   tau:{GA.G[name][1]},   norm_loss: {loss.item():.6f}')
        total_loss += loss
    return total_loss
