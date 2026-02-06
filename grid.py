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
        # Conventional unit cell side length
        a = math.sqrt(2) * self.grid_interval
        grid_size = self.grid_size
        center = self.center

        half = grid_size / 2.0
        # Start coordinates for the unit cell corners
        g_min = center - half + a/2
        g_max = center + half - a/2

        nx = int(round((grid_size[0] / a).item()))
        ny = int(round((grid_size[1] / a).item()))
        nz = int(round((grid_size[2] / a).item()))

        # Generate 1D axes for the conventional cubic lattice
        gx = torch.linspace(g_min[0], g_max[0], nx, dtype=self.dtype, device=self.device)
        gy = torch.linspace(g_min[1], g_max[1], ny, dtype=self.dtype, device=self.device)
        gz = torch.linspace(g_min[2], g_max[2], nz, dtype=self.dtype, device=self.device)
        
        # L1: Corner points (0, 0, 0)
        g1x, g1y, g1z = torch.meshgrid(gx, gy, gz, indexing='ij')
        l1 = torch.stack([g1x, g1y, g1z], dim=-1).view(-1, 3)

        # Offset for the face centers (a/2)
        h = a / 2.0

        # L2: Face centers on XY plane (1/2, 1/2, 0)
        l2 = l1.clone()
        l2[:, 0] += h
        l2[:, 1] += h

        # L3: Face centers on XZ plane (1/2, 0, 1/2)
        l3 = l1.clone()
        l3[:, 0] += h
        l3[:, 2] += h

        # L4: Face centers on YZ plane (0, 1/2, 1/2)
        l4 = l1.clone()
        l4[:, 1] += h
        l4[:, 2] += h

        # Combine all 4 interlocking sub-lattices
        fcc_coords = torch.cat([l1, l2, l3, l4], dim=0)
        
        return fcc_coords

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



