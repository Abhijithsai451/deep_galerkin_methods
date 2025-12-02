import torch
import torch.nn as nn

#%% PDE Residual Loss Function in 1D and 2D

def pde_residual_loss_pinn(model, x, f_x):
    # Calculating the residual for a Schrodinger's Equation in 1D
    x.requires_grad_(True)
    u = model(x)

    # First Derivative
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]

    # Second Derivative
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]

    # PDE Residual
    residual = u_xx - f_x
    x.requires_grad_(False)
    return residual



#%% Loss Functions for Schrodinger's Equation
def loss_function_pinn(model:nn.Module,
                  x_int: torch.Tensor,
                  source_term_domain: torch.Tensor,
                  x_bc: torch.Tensor,
                  lambda_pde: float = 1.0,
                  lambda_bc: float = 100.0) -> tuple:
    """
    Computes total loss of the Neural Network which has the losses from Pde_residual_loss and boundary_condition_loss.
    """
    criterion = nn.MSELoss()

    # 1. PDE Loss
    residual_interior = pde_residual_loss_pinn(model, x_int,source_term_domain)
    L_pde = criterion(residual_interior, torch.zeros_like(residual_interior))

    # 2. Initial Condition Loss

    # 3. Boundary Condition Loss
    u_predicted_bc = model(x_bc)
    L_bc = criterion(u_predicted_bc, torch.zeros_like(u_predicted_bc))

    L_total = (lambda_pde * L_pde) + (lambda_bc * L_bc)

    return L_total, L_pde.item(), L_bc.item()



