import torch

#%% All the Source and Boundary condition functions in 1D

def source_term_fn_1D(x)-> torch.Tensor:
    """
    Calculates the source term for Poisson's Equation in 1D.
    Analytical function is 'sin(πx)'. We return the second order derivative of the analytical function
    f(x) = sin(πx). second order derivative wrt x is -π^2 sin(πx)
    """
    f = - torch.pi**2 * torch.sin(torch.pi *x)
    return f

def boundary_condition_fn_1D(x)-> torch.Tensor:
    """"
    Calculates the boundary condition for Poisson's Equation in 1D.
    Analytical function is 'sin(πx)'
    f(0) = sin(π*0) = 0
    f(1) = sin(π*1) = 0
    """
    f = torch.zeros_like(x)
    return f

def analytical_function_1d(x)-> torch.Tensor:
    """
    Calculates the Analytical Function for Poisson's Equation in 1D.
    Analytical function is 'sin(πx)'
    """
    return torch.sin(torch.pi * x)


#%% All the Source and Boundary condition functions in 1D

def source_term_fn_2d(x,y)-> torch.Tensor:
    """
    Calculates the source term for Poisson's Equation in 2D.
    Analytical function is 'sin(πx) * sin(πy)'. We return the second order derivative of the analytical function
    f(x) = sin(πx) * sin(πy). second order derivative wrt x is -2π^2 sin(πx) * sin(πy)
    """
    f = -2.0 * (torch.pi**2) * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)
    return f

def boundary_condition_fn_2d(x,y)-> torch.Tensor:
    """"
    Calculates the boundary condition for Poisson's Equation in 1D.
    Analytical function is 'sin(πx)' on the boundary [0, 1]x[0, 1]
    f(0) = sin(π*0)* sin(π*0) = 0
    f(1) = sin(π*1)* sin(π*1) = 0
    """
    return torch.zeros_like(x)

def analytical_function_2d(x,y)-> torch.Tensor:
    """
    Calculates the Analytical Function for Poisson's Equation in 1D.
    Analytical function is 'sin(πx) * sin(πy)'
    """
    f = torch.sin(torch.pi * x) * torch.sin(torch.pi * y)
    return f