import torch

#%% All the Source and Boundary Condition function in 1D

def source_term_function(x)-> torch.Tensor:
    f = torch.zeros_like(x)
    return f

def boundary_condition_function(x)-> torch.Tensor:
    f = torch.zeros_like(x)
    return f

def initial_condition_function(x)-> torch.Tensor:
    f = torch.zeros_like(x)
    return f

def analytical_function(x)-> torch.Tensor:
    f = torch.zeros_like(x)
    return f



