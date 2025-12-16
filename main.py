import torch
from grid import MolGrid
from loss import run_step

def set_test_grids(device):
    POS  = torch.tensor([[1.0,0.0,0.0], [0.0,1.0,0.0], [0.0,0.0,1.0]], device=device, requires_grad=True)
    CHG = torch.tensor([0.5,0.0,-0.5], device=device, requires_grad=True )
    SIG = torch.tensor([1.0,1.0,1.0], device=device, requires_grad=True) 
    EPS = torch.tensor([1.0,1.0,1.0], device=device, requires_grad=True) 

    POS2  = torch.tensor([[1.0,1.0,0.0], [0.0,1.0,1.0], [1.0,0.0,1.0]], device=device, requires_grad=True)
    CHG2 = torch.tensor([0.5,-0.4,0.3], device=device, requires_grad=True) 
    SIG2 = torch.tensor([2.0,2.0,2.0], device=device, requires_grad=True) 
    EPS2 = torch.tensor([1.0,1.0,1.0], device=device, requires_grad=True) 


    G_target = MolGrid(
            pos     = POS,
            charge  = CHG,
            sigma   = SIG,                      
            epsilon = EPS,
            )
    G_target.center_to_origin()

    G_query = MolGrid(
            pos       = POS2,
            charge    = CHG2,
            sigma     = SIG2,                      
            epsilon   = EPS2,
            grid_size = G_target.grid_size
            )

    G_target.center_to_origin()
    G_query.center_to_origin()

    return G_target, G_query

def main():
    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set learning rate
    lr = {
        "lr_pos_grad"   : 1e-1,
        "lr_sigma_grad" : 1e-2,
        "lr_chg_grad"   : 1e-2,
        "lr_eps_grad"   : 1e-2,
        }
    
    # get target (answer) and query grid objects
    G_target, G_query = set_test_grids(device)
    
    # get inital grids
    G_target.get_grid()
    G_query.get_grid()

    # set target grids no_grad
    with torch.no_grad():
        G_target.get_grid()


    # update query grid
    for t in range(1000):
        G_query, lr, loss = run_step(G_target, G_query, lr, t)
        print(f"Step {t+1} Loss: {loss.item()}")


if  __name__ == '__main__':
    main()
