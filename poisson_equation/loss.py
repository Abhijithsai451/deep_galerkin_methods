import torch
import torch.nn as nn

#%% PDE Residual Loss Functions in 1D and 2D
def pde_residual_loss(model, x, f_x):
    # Calculating the residual for a Poisson's Equation in 1D (Time Independent)
    x.requires_grad_(True)
    u = model(x)

    # First Derivative
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),create_graph=True, retain_graph=True)[0]

    # Second Derivative
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]

    # PDE Residual
    residual = u_xx - f_x
    x.requires_grad_(False)
    return residual

def pde_residual_loss_2d(model, x, y, f_xy):
    # Calculating the residual for a Poisson's Equation in 1D (Time Independent)
    x.requires_grad_(True)
    y.requires_grad_(True)
    u = model(torch.cat([x, y],1))

    # Calculating the first derivative w r t to x,y
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]

    # Calculating the second derivative w r t x, y
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True, retain_graph=True)[0]

    u_xx_yy = u_xx + u_yy
    residual = u_xx_yy - f_xy
    x.requires_grad_(False); y.requires_grad_(False)

    return residual

#%% Loss Functions for Poisson's Equation

def loss_function(model:nn.Module,
                  x_int: torch.Tensor,
                  source_term_domain: torch.Tensor,
                  x_bc: torch.Tensor,
                  lambda_pde: float = 1.0,
                  lambda_bc: float = 100.0) -> tuple:
    """
    Computes total loss of the Neural Network which has the losses from pde_residual_loss and boundary_condition_loss.
    """
    criterion = nn.MSELoss()

    # 1. PDE Loss
    residual_interior = pde_residual_loss(model, x_int,source_term_domain)
    L_pde = criterion(residual_interior, torch.zeros_like(residual_interior))

    # 2. Initial Condition Loss (We implement Homogenous Dirichlet Conditions)

    # 3 Boundary Condition Loss
    u_predicted_bc = model(x_bc)
    L_bc = criterion(u_predicted_bc, torch.zeros_like(u_predicted_bc))

    L_total = (lambda_pde * L_pde) + (lambda_bc * L_bc)

    return L_total, L_pde.item(), L_bc.item()

def loss_function_2d(model:nn.Module,
                     x_int: torch.Tensor,
                     y_int: torch.Tensor,
                     source_term_domain: torch.Tensor,
                     x_bc: torch.Tensor,
                     y_bc: torch.Tensor,
                     source_term_bc: torch.Tensor,
                     lambda_pde: float = 1.0,
                     lambda_bc: float = 100.0):

    criterion = nn.MSELoss()

    # 1. PDE loss
    residual_interior = pde_residual_loss_2d(model, x_int, y_int, source_term_domain)
    L_pde = criterion(residual_interior, torch.zeros_like(residual_interior))

    # 2. BC loss
    u_predicted_bc = model(torch.cat([x_bc, y_bc],1))
    L_bc = criterion(u_predicted_bc, source_term_bc)

    L_total = (lambda_pde * L_pde) + (lambda_bc * L_bc)

    return L_total, L_pde.item(), L_bc.item()



