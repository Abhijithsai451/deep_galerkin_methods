from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
from matplotlib import cm
from matplotlib.patches import Patch

from poisson_equation.utility_functions import analytical_function_1d, analytical_function_2d

#%% Visualization function in 1D
def visualize_points_1d(domain_points: torch.Tensor,
                        boundary_points: torch.Tensor,
                        bounds: list,
                        title: str = "Visualization of the Generated Points in 1D"):
    """
    Visualizes 1D internal and boundary points in 1D
    """
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 4))

    if not bounds or len(bounds[0]) != 2:
        raise ValueError("Bounds must be in format [[x_min, x_max]].")

    x_min, x_max = bounds[0]

    # 1. Plot Domain Points
    plt.plot([x_min,x_max], [0,0],
             linestyle='-', color='gray', linewidth=3, label='Domain')

    # 2. Plot Internal Points
    if domain_points.numel() >0:
        x_int = domain_points.cpu().detach().flatten().numpy()
        y_int = torch.zeros_like(domain_points).cpu().detach().flatten().numpy()
        plt.scatter(x_int, y_int,
                    s=40, color='#337AFF', marker='o', alpha=0.6,
                    zorder=3, label='Internal Points')

    # 3. Plot Boundary Poinnts
    if boundary_points.numel() > 0:
        x_bnd = boundary_points.cpu().detach().flatten().numpy()
        y_bnd = torch.zeros_like(boundary_points).cpu().detach().flatten().numpy()
        plt.scatter(x_bnd, y_bnd,
                    s=100, color='#FF5733', marker='o',
                    edgecolors='black', linewidths=1.5, zorder=5,
                    label='Boundary Points (BCs)')

    # Set Plot Properties
    x_range = x_max - x_min
    plt.xlim(x_min - 0.1 * x_range, x_max + 0.1 * x_range)
    plt.ylim(-0.5, 0.5)
    plt.title(title, fontsize = 14)
    plt.xlabel(f"Dimension 1 (x-axis) in {bounds[0]}", fontsize = 12)
    plt.yticks([])
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)
    plt.show()

def visualize_solution_1d(model: nn.Module, domain_bound: float, n_test_points: int = 500):
    """
    Generates the test points and Plots the NN solution vs Analytical solution.
    """
    model.eval()
    device = next(model.parameters()).device

    # 1. Generate test Points
    x_test_np = np.linspace(0, domain_bound, n_test_points, dtype=torch.float32)
    x_test = torch.from_numpy(x_test_np).reshape(-1, 1).to(device)

    # 2. Calculate the Neural Network Solution
    with torch.no_grad():
        u_nn = model(x_test)
        u_nn_np = u_nn.cpu().numpy().flatten()

    # 3. Calculate Analytical solution
    u_exact = analytical_function_1d(x_test)
    u_exact_np = u_exact.cpu().numpy().flatten()

    # 4. Error estimation (Absolute Error)
    error_np = np.abs(u_nn_np - u_exact_np)

    # 5. Plotting the error

    # Create the Figure with two subplots. Solution and Error.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # Top Subplot: Solution comparision
    ax1.plot(x_test_np,u_exact_np, label='Analytical Solution', color='tab:blue', linewidth=2)
    ax1.plot(x_test_np, u_nn_np, label="DGM Solution (u_NN)", color='tab:red', linestyle='--', linewidth=2)
    ax1.set_title(f'Solution Comparison (Time Independent) ', fontsize = 14)
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.7)

    # Bottom subplot: Absolute Error
    ax2.plot(x_test_np, error_np, label='Absolute Error $|u_{NN} - u_{exact}|$', color='tab:green', linewidth=1)
    ax2.set_xlabel("Spatial Coordinate $x$", fontsize = 12)
    ax2.set_ylabel("Absolute Error", fontsize = 12)
    ax2.set_ylim(0, np.max(error_np) * 1.1)
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))

    plt.tight_layout()
    plt.show()
