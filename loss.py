import torch
import torch.nn as nn
Alpha = 0.5 # Thermal Diffusivity Constant


def pde_residual_loss(model, t,x, alpha, f_tx):
    # Calculates the residual for a time-dependent PDE
    t.requires_grad_(True)
    x.requires_grad_(True)
    u = model(t, x)

    # First derivatives
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]

    # Second derivative
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]

    # General PDE Residual (R = u_t - alpha * u_xx - f_tx)
    residual = u_t - alpha * u_xx - f_tx

    t.requires_grad_(False)
    x.requires_grad_(False)

    return residual

def loss_function(  model: nn.Module,
    # Interior Points
    t_int: torch.Tensor, x_int: torch.Tensor, source_term_interior: torch.Tensor,
    # Initial Condition Points
    t_ic: torch.Tensor, x_ic: torch.Tensor, target_ic: torch.Tensor,
    # Boundary Condition Points
    t_bc: torch.Tensor, x_bc: torch.Tensor, target_bc: torch.Tensor,
    # Hyperparameters/PDE constants
    alpha: float,
    lambda_pde: float = 1.0,
    lambda_ic: float = 100.0,
    lambda_bc: float = 100.0) -> tuple:

    """
        Computes total loss: L_total = lambda_pde*L_pde + lambda_ic*L_ic + lambda_bc*L_bc.
    """
    criterion = nn.MSELoss()

    # 1. PDE Loss (L_pde): Enforce residual R=0 in the domain
    R_interior = pde_residual_loss(model, t_int, x_int, alpha, source_term_interior)
    L_pde = criterion(R_interior, torch.zeros_like(R_interior))

    # 2. Initial Condition Loss (L_ic): Enforce u(x, t=0) = u_IC
    u_predicted_ic = model(t_ic, x_ic)
    L_ic = criterion(u_predicted_ic, target_ic)

    # 3. Spatial Boundary Condition Loss (L_bc): Enforce u(x_bnd, t) = u_BC
    u_predicted_bc = model(t_bc, x_bc)
    L_bc = criterion(u_predicted_bc, target_bc)

    # Total Weighted Loss
    L_total = (lambda_pde * L_pde) + (lambda_ic * L_ic) + (lambda_bc * L_bc)

    return L_total, L_pde.item(), L_ic.item(), L_bc.item()

