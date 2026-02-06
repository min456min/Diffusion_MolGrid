import torch
import math
import torch.nn.functional as F

class MolGrid:
    def __init__(
        self,
        pos:            torch.Tensor,
        charge:         torch.Tensor,
        epsilon:        torch.Tensor | None = None,   
        sigma:          torch.Tensor | None = None,   
        grid_interval:  float = 0.3,
        grid_buffer :   float = 5.0,
        center :        list[float] | None = None, 
        grid_size :     list[float] | None = None, 
        grid_coords :   torch.Tensor | None = None,
        taus:           dict[str, list[float]] | None = None
    ):

        self.pos           = pos
        self.charge        = charge
        self.epsilon       = epsilon
        self.sigma         = sigma

        self.dtype         = pos.dtype
        self.device        = pos.device

        self.grid_interval = grid_interval # Distance Among Grids
        self.grid_buffer   = grid_buffer # 

        self.grid_size     = self.set_grid_size() if grid_size is None else grid_size
        self.center        = self.pos.mean(dim=0) if center is None else center
        self.grid_coords   = self.build_grid_coords() if grid_coords is None else grid_coords

        self.taus = taus


    def set_grid_size(self):
        """
        Calculate grid box size with buffer around atomic positions.
        """
        grid_buffer = self.grid_buffer
        pos = self.pos
        max_vals = torch.max(pos, dim=0).values
        min_vals = torch.min(pos, dim=0).values
        lengths = max_vals - min_vals
        size = [math.ceil(s) + 2.0 * grid_buffer for s in lengths.tolist()]
        return torch.tensor(size, device=self.device, dtype=self.dtype)

    def build_grid_coords(self):
        """
        Generate 3D coordinates for the FCC grid voxels.
        Nearest neighbor distance is maintained at self.grid_interval.
        """
        center = self.center
        grid_size = self.grid_size
        d  = self.grid_interval
        device = self.device
        dtype = self.dtype
        
        a1 = torch.tensor([d, 0.0], device=device, dtype=dtype)
        a2 = torch.tensor([0.5 * d, 0.5 * math.sqrt(3.0) * d], device=device, dtype=dtype)
        
        half = 0.5 * grid_size
        x_min, x_max = center[0] - half[0], center[0] + half[0]
        y_min, y_max = center[1] - half[1], center[1] + half[1]
        
        dx = d
        dy = float(a2[1].item())  # sqrt(3)/2 * d
                                                                                           
        nx = int(math.ceil(float(half[0].item()) / dx)) + 3
        ny = int(math.ceil(float(half[1].item()) / dy)) + 3
                                                                                           
        ii = torch.arange(-nx, nx + 1, device=device)
        jj = torch.arange(-ny, ny + 1, device=device)
        I, J = torch.meshgrid(ii, jj, indexing="ij")
        IJ = torch.stack([I.reshape(-1), J.reshape(-1)], dim=1).to(dtype)
        xy0 = IJ[:, 0:1] * a1[None, :] + IJ[:, 1:2] * a2[None, :]
        in_box = (
            (xy0[:, 0] + center[0] >= x_min) & (xy0[:, 0] + center[0] <= x_max) &
            (xy0[:, 1] + center[1] >= y_min) & (xy0[:, 1] + center[1] <= y_max))
        xy0 = xy0[in_box]
        
        dz = math.sqrt(2.0 / 3.0) * d
        z_min, z_max = center[2] - half[2], center[2] + half[2]
        nz = int(math.ceil(float(half[2].item()) / dz)) + 2
        kk = torch.arange(-nz, nz + 1, device=device, dtype=dtype)
        z_levels = center[2] + kk * dz
        keep_z = (z_levels >= z_min) & (z_levels <= z_max)
        z_levels = z_levels[keep_z]
        kk_int = kk[keep_z].to(torch.int64)
        
        s = (a1 + a2) / 3.0
        phase = torch.remainder(kk_int, 3)  
        shifts = torch.zeros((z_levels.numel(), 2), device=device, dtype=dtype)
        shifts[phase == 1] = s
        shifts[phase == 2] = 2.0 * s
        
        coords_list = []
        for li in range(z_levels.numel()):
            xy = xy0 + shifts[li][None, :]
            z = torch.full((xy.size(0), 1), z_levels[li], device=device, dtype=dtype)
            coords_list.append(torch.cat([xy + center[None, :2], z], dim=1))
        
        coords = torch.cat(coords_list, dim=0)
        return coords


    def center_to_origin(self):
        current_centroid = self.pos.mean(dim=0)
        
        new_pos = self.pos - current_centroid
        self.pos = new_pos.detach().clone().to(self.device).requires_grad_(True)
        self.center = torch.zeros(3, device=self.device, dtype=self.dtype)

    
    def softplus_attr(self, add=0.001):
        self.sigma = F.softplus(self.log_sigma) + 0.001
        self.epsilon = F.softplus(self.log_epsilon) + 0.001


    def get_grid(self, store=True):
        taus        = self.taus
        pos         = self.pos.view(1, 1, 1, -1, 3)
        chg         = self.charge.view(1, 1, 1, -1)
        sig         = self.sigma.view(1, 1, 1, -1)
        eps         = self.epsilon.view(1, 1, 1, -1)
        grid_coords = self.grid_coords.unsqueeze(-2)
    
        sq_r = torch.sum((grid_coords - pos)**2, dim=-1) 
        r    = torch.sqrt(sq_r)    
        grids = {}
        for gtype, taus in taus.items():
            for tau in taus:
                # --- Grid 1: Density Grid ---
                if gtype == 'density':
                    w = tau * sig
                    g_den = torch.exp(-sq_r / (2 * w**2))
                    g_den = torch.sum(g_den, dim=-1)
                    grids[f'{gtype}_{tau}'] = [gtype, tau, g_den]
    
                # --- Grid 2: Charge Potential ---
                elif gtype == 'charge':
                    pos = torch.clamp(chg, min=0.0)
                    neg = torch.clamp(chg, max=0.0) 
        
                    denom = torch.sqrt(sq_r + tau**2 + 1e-12)
                    g_chg = torch.sum(chg/denom, dim=-1)
                    g_pos = torch.sum(pos/denom, dim=-1)
                    g_neg = torch.sum(neg/denom, dim=-1)

                    grids[f'{gtype}_{tau}'] = [gtype, tau, g_chg]
                    grids[f'{gtype}_pos_{tau}'] = [gtype, tau, g_pos]
                    grids[f'{gtype}_neg_{tau}'] = [gtype, tau, g_neg]

                # --- Grid 3: Epsilon VDW Potential ---
                elif gtype == 'vdw':
                    w = tau * sig
                    # repulsion (Gaussian at r=0)
                    g_rep = (4.0 * eps) * torch.exp(-0.5 * (r / w) ** 2)

                    # attraction (Gaussian centered near LJ minimum)
                    r_min = (2.0 ** (1.0 / 6.0)) * sig
                    g_att = (4.0 * eps) * torch.exp(- (r - r_min) ** 2 / (2.0 * w ** 2))
                
                    g_rep = torch.sum(g_rep, dim=-1)
                    g_att = torch.sum(g_att, dim=-1)
                
                    grids[f'{gtype}_rep_{tau}'] = [gtype, tau, g_rep]
                    grids[f'{gtype}_att_{tau}'] = [gtype, tau, g_att]

        if store:
            self.G = grids
        else:
            return grids

if  __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    taus = {
        "density": [0.2, 0.5],
        "charge": [0.2, 1.0],
        "vdw": [0.2, 2.0]
        }


    POS  = torch.tensor([[0.0,1.0,2.0], [3.0,4.0,5.0], [6.0,7.0,8.0]], device=device)
    CHG = torch.tensor([1.0,1.0,1.0], device=device) 
    EPS = torch.tensor([2.0,2.0,2.0], device=device) 
    SIG = torch.tensor([3.0,3.0,3.0], device=device) 

    POS2  = torch.tensor([[1.0,2.0,3.0], [4.0,2.0,4.0], [3.0,2.0,1.0]], device=device)
    CHG2 = torch.tensor([0.5,0.4,0.3], device=device) 
    EPS2 = torch.tensor([1.0,1.0,1.0], device=device) 
    SIG2 = torch.tensor([2.0,2.0,2.0], device=device) 


    G_target = MolGrid(
            pos     = POS,
            charge  = CHG,
            sigma   = SIG,                      
            epsilon = EPS,
            taus    = taus
            )
    G_target.center_to_origin()

    G_query = MolGrid(
            pos     = POS2,
            charge  = CHG2,
            sigma   = SIG2,                      
            epsilon = EPS2,
            grid_coords = G_target.grid_coords,
            taus   = taus

            )
    G_query.center_to_origin()



