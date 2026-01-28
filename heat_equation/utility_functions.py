import torch

#%% All the source, boundary and initial condition functions in 1D
# Example for u(x, t) = exp(-t) * sin(pi*x)
def source_term_fn_1D(t, x, alpha):
    # u(x,t) = u_t - alpha*u_xx
    # u_t = -exp(-t)*sin(pi*x)
    # u_xx = -pi^2 * exp(-t) * sin(pi*x)
    return (-torch.exp(-t) + alpha * torch.pi**2 * torch.exp(-t)) * torch.sin(torch.pi * x)

def initial_condition_fn_1D(x):
    # u(x, 0) = sin(pi*x)
    return torch.sin(torch.pi * x)

def boundary_condition_fn_1D(t, x):
    # u(0, t) = u(1, t) = exp(-t)*sin(0 or pi) = 0
    return torch.zeros_like(t)

def analytical_solution(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    The known analytical solution for the 1D Heat Equation example: u(x, t) = exp(-t) * sin(pi*x)
    """
    # Ensure all inputs are tensors and on the correct device for multiplication
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=torch.float32, device=x.device)

    # Ensure t has the same shape as x for element-wise multiplication
    if t.dim() == 1:
        t = t.reshape(-1, 1)

    return torch.exp(-t) * torch.sin(torch.pi * x)

#%% All the source, boundary and initial condition functions in 2D

# Example for u(x, t) = exp(-t) * sin(pi*x) * sin(pi*y)
def source_term_fn_2D(t, x, y , alpha):
    # u_exact(t,x,y) = exp(-t) * sin(pi*x) * sin(pi*y)
    u_exact = torch.exp(-t) * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)
    const = (2.0 * (torch.pi**2) * alpha) - 1.0
    return u_exact * const

def initial_condition_fn_2D(x,y):
    # u(x, 0) = sin(pi*x)
    return torch.sin(torch.pi * x) * torch.sin(torch.pi * y)

def boundary_condition_fn_2D(t, x, y ):
    # u(0, t) = u(1, t) = exp(-t)*sin(0 or pi) = 0
    return torch.zeros_like(t)

def analytical_solution_2d(t: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
        The known analytical solution for the 2D Heat Equation example: u(x, y, t) = exp(-t) * sin(pi*x) * sin(pi*y)
    """
    # Ensuring all the imputs are tensors and on the correct device for multiplication
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=torch.float32, device=x.device)

    # Ensuring t has the same shape as x and y for element wise multiplication
    if t.dim() == 1:
        t = t.reshape(-1, 1)
    return torch.exp(-t) * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)

def calculate_relative_l2_error(model, t, x, analytical_fn):
    model.eval()
    with torch.no_grad():
        u_pred = model(t, x)
        u_true = analytical_fn(t, x)
        error = torch.norm(u_true - u_pred, 2) / torch.norm(u_true, 2)
    return error.item()

def calculate_relative_l2_error_2d(model, t, x, y, analytical_fn):
    model.eval()
    with torch.no_grad():
        u_pred = model(t, torch.cat([x, y], 1))
        u_true = analytical_fn(t, x, y)
        error = torch.norm(u_true - u_pred, 2) / torch.norm(u_true, 2)
    return error.item()


#%%