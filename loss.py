import torch

def calc_norm_loss(query, target):
    # Normalized loss
    loss = torch.sum((query - target) ** 2)/torch.sum(query ** 2 + target ** 2)
    return loss

def calc_grid_loss(GA, GB):
    # Grids (GA.G and GA.G) 
    # density grids with multi taus
    # total, positive, negative charge grids
    # vdw grids with multi taus 

    total_loss = 0
    for gA, gB in zip(GA.G, GB.G):
        # Calculate normalized loss
        loss = calc_norm_loss(gA, gB)
        total_loss += loss #.item()
    return total_loss

def update_G_features(G, loss, lr):
    pos  = G.pos
    chg  = G.charge      
    eps  = G.epsilon
    sig  = G.sigma

    # set grad zero
    if pos.grad is not None: pos.grad.zero_()
    if chg.grad is not None: chg.grad.zero_()
    if eps.grad is not None: eps.grad.zero_()
    if sig.grad is not None: sig.grad.zero_()

    # back propagation
    loss.backward()
    
    # update features
    with torch.no_grad():
        # param = param - lr * grad
        if pos.grad is not None:
            pos -= lr['lr_pos_grad'] * pos.grad
        if chg.grad is not None:
            chg -= lr['lr_chg_grad'] * chg.grad
        if eps.grad is not None:
            eps -= lr['lr_eps_grad'] * eps.grad
        if sig.grad is not None:
            sig -= lr['lr_sigma_grad'] * sig.grad
    
    # update grid object
    G.pos      = pos.detach().clone().requires_grad_(True)
    G.charge   = chg.detach().clone().requires_grad_(True)
    G.epsilon  = eps.detach().clone().requires_grad_(True)
    G.sigma    = sig.detach().clone().requires_grad_(True)
    G.get_grid()

    return G

def decline_lr(lr, t, decay_r=0.9, t_dec_pos=200, t_dec_attr=100):
    if t % t_dec_pos == 0:
        lr['lr_pos_grad'] = max(lr['lr_pos_grad'] * 0.8, 1e-4) 
    if t % t_dec_attr == 0:
        for k in ['lr_chg_grad', 'lr_eps_grad', 'lr_sigma_grad']:
            lr[k] = max(lr[k] * 0.8, 1e-5) 
    return lr

def run_step(GA, GB, lr, t):
    # GA: target grids or molecule
    # GB: query grids or molecule which will be updated
    # calculate loss
    loss = calc_grid_loss(GA, GB)
    GB = update_G_features(GB, loss, lr)
    lr = decline_lr(lr, t)
    return GB, lr, loss
