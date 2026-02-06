import torch
import math

def get_boundary_dots(G, n_rays = 2048, sig_scale = 0.5, margin = 2.0, max_dots = 5000):
    
    device = G.device
    dtype = G.dtype

    pos = G.pos.reshape(-1, 3).to(device=device, dtype=dtype)
    sig = G.sigma.reshape(-1).to(device=device, dtype=dtype)

    dots_list = []
    # Multi-try for each atoms
    for pos0 in pos:
        # Atom radii
        r = (sig_scale * sig).clamp_min(torch.tensor(1e-8, device=device, dtype=dtype)) 

        # Sphere
        i = torch.arange(n_rays, device=device, dtype=dtype)
        z = 1.0 - 2.0 * (i + 0.5) / float(n_rays)
        angle = math.pi * (3.0 - math.sqrt(5.0))
        theta = i * angle
        xy = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
        dirs = torch.stack([xy * torch.cos(theta), xy * torch.sin(theta), z], dim=1)
        dirs = dirs / (dirs.norm(dim=1, keepdim=True) + 1e-12)  # ensure unit 

        rel = pos - pos0[None, :]
        outer = rel.norm(dim=1).max() + r.max() + torch.tensor(float(margin), device=device, dtype=dtype)

        p0 = pos0[None, :] + outer * dirs
        d = -dirs

        # Ray-sphere intersection
        # Ray: p(t) = p0 + t*d
        # Sphere j: ||p(t) - c_j||^2 = r_j^2
        # oc = p0 - c_j; solve t^2 + 2*b*t + c = 0
        #  where b = dot(oc, d), c = dot(oc,oc) - r^2

        oc = p0[:, None, :] - pos[None, :, :]
        b = (oc * d[:, None, :]).sum(dim=2)
        c = (oc * oc).sum(dim=2) - (r[None, :] ** 2)    
        disc = b * b - c

        # Valid intersections where discriminant >= 0
        valid = disc >= 0.0
        sqrt_disc = torch.zeros_like(disc)
        sqrt_disc[valid] = torch.sqrt(torch.clamp(disc[valid], min=0.0))

        # Two solutions: t = -b ± sqrt_disc
        t1 = -b - sqrt_disc
        t2 = -b + sqrt_disc

        INF = torch.tensor(1e30, device=device, dtype=dtype)
        t1_pos = torch.where((t1 > 1e-10) & valid, t1, INF)
        t2_pos = torch.where((t2 > 1e-10) & valid, t2, INF)
        t_first = torch.minimum(t1_pos, t2_pos)

        # Pick nearest atom hit per ray                                                                  
        t_min, hit_atom = torch.min(t_first, dim=1)
        hit_mask = t_min < (INF * 0.5)

        # Compute hit points for rays that hit                                                           
        t_hit = t_min[hit_mask]
        p0_hit = p0[hit_mask]
        d_hit = d[hit_mask]
        dots = p0_hit + t_hit[:, None] * d_hit  
        dots_list.append(dots)

    # Concat all dots
    dots = torch.cat(dots_list, dim=0)

    # Farthest point sampling
    N = dots.shape[0]


    idx = torch.zeros(max_dots, dtype=torch.long, device=device)
    dist = torch.full((N,), float('inf'), device=device)
    farthest = torch.randint(0, N, (1,), device=device)
    for i in range(max_dots):
        centroid = dots[farthest].view(1, 3)
        d = torch.sum((dots - centroid) ** 2, dim=1)
        dist = torch.minimum(dist, d)
        farthest = torch.argmax(dist)
        idx[i] = farthest

    dots = dots[idx]

    return dots                                 
