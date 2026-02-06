import torch
from grid import MolGrid
from loss import calc_grid_loss
from loss import calc_overlap_loss
from loss import calc_boundary_violation_loss
from boundary import get_boundary_dots
def set_test_grids(device):
    taus = {
        "density": [0.2, 0.5],
        "charge": [0.2, 1.0],
        "vdw": [0.2, 2.0]}


    POS  = torch.tensor([[1.0,0.0,0.0], [0.0,1.0,0.0], [0.0,0.0,1.0]], device=device)
    CHG = torch.tensor([0.5,0.0,-0.5], device=device)
    SIG = torch.tensor([1.0,1.0,1.0], device=device) 
    EPS = torch.tensor([1.0,1.0,1.0], device=device)

    POS2  = torch.tensor([[1.0,1.0,0.0], [0.0,1.0,1.0], [1.0,0.0,1.0]], device=device, requires_grad=True)
    CHG2 = torch.tensor([0.5,-0.4,0.3], device=device, requires_grad=True) 
    SIG2 = torch.tensor([2.0,2.0,2.0], device=device, requires_grad=True) 
    EPS2 = torch.tensor([1.0,1.0,1.0], device=device, requires_grad=True) 


    G_target = MolGrid(
            pos     = POS,
            charge  = CHG,
            sigma   = SIG,                      
            epsilon = EPS,
            taus    = taus
            )

    G_query = MolGrid(
            pos       = POS2,
            charge    = CHG2,
            sigma     = SIG2,                      
            epsilon   = EPS2,
            grid_coords = G_target.grid_coords,
            taus    = taus
            )

    G_target.center_to_origin()
    G_query.center_to_origin()

    return G_target, G_query

def main():
    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # get target (answer) and query grid objects
    G_target, G_query = set_test_grids(device)
    
    # get inital grids
    G_target.get_grid()
    G_target.boundary = get_boundary_dots(G_target)

    total_grid_loss = calc_grid_loss(G_target, G_query)
    boundary_loss = calc_boundary_violation_loss(G_query, G_target.boundary)
    print(f'Boundary Loss: {boundary_loss:.6f}')
    overlap_loss = calc_overlap_loss(G_query)
    print(f'Overlap Loss: {overlap_loss:.6f}')
if  __name__ == '__main__':
    main()
