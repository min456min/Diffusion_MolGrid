import torch
import math

class MolGrid:
    def __init__(
        self,
        pos: torch.Tensor,
        charge: torch.Tensor,
        epsilon: torch.Tensor = None,   
        sigma: torch.Tensor = None, 
        center : list[float] | None = None,
        grid_interval: float = 0.3,
        grid_size : list[float] | None = None,
        k_coulomb: float = 1.0,
        grid_buffer : float = 5.0,
        taus: dict[str, list[float]] | None = None
    ):


        self.pos    = pos
        self.charge = charge
        self.epsilon = epsilon
        self.sigma = sigma
        self.dtype  = pos.dtype
        self.device = pos.device

        self.grid_buffer = grid_buffer
        self.grid_interval = float(grid_interval)
        self.k_coulomb = float(k_coulomb)

        if center is None:
            self.center = torch.zeros(3, device=self.device, dtype=self.dtype)
        else:
            self.center = torch.as_tensor(center, device=self.device, dtype=self.dtype)
    
        if grid_size is None:
            calculated_size = self.set_grid_size()
            self.grid_size = torch.tensor(calculated_size, device=self.device, dtype=self.dtype)
        elif isinstance(grid_size, torch.Tensor):
            self.grid_size = grid_size.clone().detach().to(device=self.device, dtype=self.dtype)


        if taus is None:
            self.taus = {
                "density": [0.1, 0.5, 1.0],
                "charge": [1],
                "epsilon": [0.1, 0.5, 1.0]
            }
        else:
            self.taus = taus


    def set_grid_size(self):
        grid_buffer = self.grid_buffer
        pos    = self.pos

        max_vals = torch.max(pos, dim=0).values
        min_vals = torch.min(pos, dim=0).values
    
        lengths = max_vals - min_vals
    
        size  = [math.ceil(s) + 2.0 * grid_buffer for s in lengths.tolist()]
        return size


    def center_to_origin(self):
        current_centroid = self.pos.mean(dim=0)
        
        new_pos = self.pos - current_centroid
        self.pos = new_pos.detach().clone().to(self.device).requires_grad_(True)
        self.center = torch.zeros(3, device=self.device, dtype=self.dtype)

    def build_grid_coords(self):
        grid_interval = self.grid_interval
        grid_size     = self.grid_size
        center        = self.center.to(device=self.device, dtype=self.dtype)

        half = grid_size / 2.0

        g_min = center - half + grid_interval/2
        g_max = center + half - grid_interval/2

        nx = int(round((grid_size[0] / grid_interval).item()))
        ny = int(round((grid_size[1] / grid_interval).item()))
        nz = int(round((grid_size[2] / grid_interval).item()))

        gx = torch.linspace(g_min[0], g_max[0], nx, dtype=self.dtype, device=self.device)
        gy = torch.linspace(g_min[1], g_max[1], ny, dtype=self.dtype, device=self.device)
        gz = torch.linspace(g_min[2], g_max[2], nz, dtype=self.dtype, device=self.device)
        gx, gy, gz = torch.meshgrid(gx, gy, gz, indexing='ij')
        return torch.stack([gx, gy, gz], dim=-1)

    def get_grid(self):
        pos         = self.pos
        num_atoms   = self.pos.shape[0]
        charge      = self.charge
        epsilon     = self.epsilon
        sigma       = self.sigma
        taus        = self.taus
    
        grid_coords = self.build_grid_coords()
        pos         = pos.view(1, 1, 1, -1, 3)
        charges     = charge.view(1, 1, 1, -1)
        sigmas      = sigma.view(1, 1, 1, -1)
        epsilons    = epsilon.view(1, 1, 1, -1)
        grid_coords = grid_coords.unsqueeze(-2)
    
        squared_dist = torch.sum((grid_coords - pos)**2, dim=-1) # (32, 32, 32, N)
        distance     = torch.sqrt(squared_dist + 1e-6)          # (Gx,Gy,Gz,N)
    
    
        grids = []
        for grid_type, tau_list in taus.items():
            for tau in tau_list:
                # --- Grid 1: Density Grid ---
                if grid_type == 'density':
                    gaussian = torch.exp(-squared_dist / (2 * (tau * sigmas)**2))
                    density_grid = torch.sum(gaussian, dim=-1)
                    grids.append(density_grid)
    
                # --- Grid 2: Charge Potential ---
                elif grid_type == 'charge':
                    charges_pos = torch.clamp(charges, min=0.0)
                    charges_neg = torch.clamp(charges, max=0.0) 
        
                    denom = torch.sqrt(squared_dist + tau**2 + 1e-12)
                    charge_grid = torch.sum((self.k_coulomb * charges) / denom, dim=-1)
                    charge_grid_pos = torch.sum((self.k_coulomb * charges_pos) / denom, dim=-1)
                    charge_grid_neg = torch.sum((self.k_coulomb * charges_neg) / denom, dim=-1)

                    grids += [charge_grid, charge_grid_pos, charge_grid_neg]

                # --- Grid 3: Epsilon VDW Potential ---
                elif grid_type == 'epsilon':
                    r_min = (2 ** (1/6)) * sigmas         # LJ potential minimum radius
                    w = tau * sigmas            
                    numerator = (distance - r_min)**2
                    denominator = 2.0 * (w**2)
                    vdw_values = epsilons * torch.exp(-numerator / (denominator + 1e-6))
                    vdw_grid = torch.sum(vdw_values, dim=-1) 
                    grids.append(vdw_grid)

        self.G = grids

if  __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    VOXEL_SIZE = 0.3
    GRID_SIZE  = 100

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
            )
    G_target.center_to_origin()

    G_query = MolGrid(
            pos     = POS2,
            charge  = CHG2,
            sigma   = SIG2,                      
            epsilon = EPS2,
            )
    G_query.center_to_origin()

    G_target.get_grid()
