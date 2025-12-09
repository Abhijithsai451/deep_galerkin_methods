import torch

#%% All the source, boundary and initial condition functions in 1D

def source_term_fn_1D(x, alpha):
    """
    Calculates the source term for the steady state heat equation in 1D
    """
    f = -alpha * torch.pi**2 * torch.sin(torch.pi * x)
    return f

def boundary_condition_fn_1D(x: torch.Tensor) -> torch.Tensor:
    """
    Calculates the boundary condition for the heat's equation in 1D
    """
    return torch.zeros_like(x)

def analytical_function_1d(x: torch.Tensor)-> torch.Tensor:
    """
    Analytical Solution of the 1D heat equation with homogeneous Dirichlet Boundary conditions
    """
    return torch.sin(torch.pi * x)

#%% All the source, boundary and initial condition functions in 2D

def source_term_fn_2D(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """
    Calculates the source term for heat Equation in 2D, assuming u(x, y) = sin(pi*x) * sin(pi*y).
    """
    return -2 * alpha * torch.pi**2 * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)


def boundary_condition_fn_2D(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Calculates the boundary condition for heat Equation in 2D, assuming u(x, y) = 0 on the boundary.
    """
    return torch.zeros_like(x)


def analytical_function_2d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Analytical solution of the 2D heat equation with homogeneous Dirichlet boundary conditions.
    """
    return torch.sin(torch.pi * x) * torch.sin(torch.pi * y)
