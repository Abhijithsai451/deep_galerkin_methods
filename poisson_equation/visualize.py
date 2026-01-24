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

def visualize_loss(trainer, title="Training Loss History"):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(trainer.loss_history, label='Total Loss')
    plt.plot(trainer.pde_loss_history, label='PDE Loss')
    plt.plot(trainer.bc_loss_history, label='BC Loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.show()

def visualize_solution_1d(model: nn.Module, domain_bound: float, n_test_points: int = 500):
    """
    Generates the test points and Plots the NN solution vs Analytical solution.
    """
    model.eval()
    device = next(model.parameters()).device

    # 1. Generate test Points
    x_test_np = np.linspace(0, domain_bound, n_test_points, dtype= np.float32)
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

#%% Visualization Functions for 2D
def visualize_points_2d(domain_points:torch.Tensor,
                        boundary_points:torch.Tensor,
                        bounds: list,
                        title:str = "Visualization of the Generated Points in 2D"):
    """
        Visualizes 2D internal (collocation) and boundary points within a rectangular domain.
    """
    if len(bounds) != 2 or domain_points.shape[1] != 2:
        raise ValueError("This function requires 2D bounds and 2D points (N, 2).")

    sns.set_style("whitegrid")
    plt.figure(figsize=(7, 7))

    # --- Convert to CPU/NumPy/DataFrame for Plotting ---

    # FIX: Use .cpu().detach().numpy() for all tensors
    i_points_np = domain_points.cpu().detach().numpy()
    b_points_np = boundary_points.cpu().detach().numpy()

    df_int = pd.DataFrame(i_points_np, columns=['X', 'Y'])
    df_bnd = pd.DataFrame(b_points_np, columns=['X', 'Y'])

    # --- 1. Draw the Domain Boundary Box ---
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]

    plt.plot(
        [x_min, x_max, x_max, x_min, x_min],
        [y_min, y_min, y_max, y_max, y_min],
        linestyle='--',
        color='gray',
        linewidth=2.0,
        label='Domain Boundary'
    )
    # --- 2. Plot Internal Points ---
    if not df_int.empty:
        plt.scatter(
            df_int['X'], df_int['Y'],
            s=8,
            color='#337AFF',
            alpha=0.4,
            label='Internal Points'
        )

    # --- 3. Plot Boundary Points (BCs) ---
    if not df_bnd.empty:
        plt.scatter(
            df_bnd['X'], df_bnd['Y'],
            s=25,
            color='#FF5733',
            edgecolors='black',
            linewidths=0.5,
            label='Boundary Points (BCs)'
        )

    # --- Set Plot Properties ---
    x_range = x_max - x_min
    y_range = y_max - y_min
    plt.xlim(x_min - 0.05 * x_range, x_max + 0.05 * x_range)
    plt.ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)

    plt.title(title, fontsize=14)
    plt.xlabel(f'Dimension 1 ({bounds[0]})', fontsize=12)
    plt.ylabel(f'Dimension 2 ({bounds[1]})', fontsize=12)
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


def visualize_2d(model: nn.Module, bounds: List[List[float]], n_grid: int = 100):
    """
    Plots the DGM solution as a contour map in the x-y plane.
    """
    model.eval()
    device = next(model.parameters()).device
    # 1. create a 2d Meshgrid
    x_min  , x_max = bounds[0]
    y_min, y_max =  bounds[1]

    x_np = np.linspace(x_min, x_max, n_grid)
    y_np = np.linspace(y_min, y_max, n_grid)
    X, Y = np.meshgrid(x_np, y_np)

    # Flatten grid and convert to tensors for model input
    x_test = torch.from_numpy(X.flatten()).float().reshape(-1, 1).to(device)
    y_test = torch.from_numpy(Y.flatten()).float().reshape(-1, 1).to(device)
    spatial_coords = torch.cat([x_test, y_test], 1)

    # 2. Calculate Solution
    with torch.no_grad():
        u_nn = model(spatial_coords)
        u_nn_np = u_nn.cpu().numpy().reshape(n_grid, n_grid)

        # Analytical Solution
        u_exact = analytical_function_2d(x_test, y_test)
        u_exact_np = u_exact.cpu().numpy().reshape(n_grid, n_grid)

    # 3. Calculate absolute Error
    error_np = np.abs(u_nn_np - u_exact_np)

    # 4 Plotting
    fig, axes = plt.subplots(1,2,figsize=(14, 6))

    # -- Determine Global Colorbar limits for Comparision
    vmax_solution = max(u_nn_np.max(), u_exact_np.max())
    vmin_solution = min(u_nn_np.min(), u_exact_np.min())
    levels_solution = np.linspace(vmin_solution, vmax_solution, 10)

    ax1 = axes[0]
    contour1 = ax1.contourf(X, Y, u_nn_np, levels=levels_solution, cmap='viridis')
    fig.colorbar(contour1, ax=ax1, label='$u_{NN}(x, y, t)$')
    ax1.set_xlabel('$x$', fontsize=12)
    ax1.set_ylabel('$y$', fontsize=12)
    ax1.set_title(f'DGM Solution for Poisson\'s Equation', fontsize=14)
    ax1.set_aspect('equal', adjustable='box')

    ax2 = axes[1]

    eror_vmax = error_np.max()
    contour2 = ax2.contourf(X, Y, error_np, levels=50, cmap='Reds', vmax=eror_vmax, extend='max')
    cbar = fig.colorbar(contour2, ax=ax2, format = '%.1e',label='Absolute Error $|u_{NN} - u_{exact}|$')
    ax2.set_xlabel('$x$', fontsize=12)
    ax2.set_ylabel('$y$', fontsize=12)
    ax2.set_title(f'Absolute Error Poisson\'s Equation', fontsize=14)
    ax2.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()

def visualize_solution_2d(model: nn.Module, bounds: List[List[float]], n_grid: int = 100):
    """
    Plots the DGM solution as a contour map in the x-y plane.
    """
    model.eval()
    device = next(model.parameters()).device
    # 1. create a 2d Meshgrid
    x_min  , x_max = bounds[0]
    y_min, y_max =  bounds[1]

    x_np = np.linspace(x_min, x_max, n_grid)
    y_np = np.linspace(y_min, y_max, n_grid)
    X, Y = np.meshgrid(x_np, y_np)

    # Flatten grid and convert to tensors for model input
    x_test = torch.from_numpy(X.flatten()).float().reshape(-1, 1).to(device)
    y_test = torch.from_numpy(Y.flatten()).float().reshape(-1, 1).to(device)
    spatial_coords = torch.cat([x_test, y_test], 1)

    # 2. Calculate Solution
    with torch.no_grad():
        u_nn = model(spatial_coords)
        u_nn_np = u_nn.cpu().numpy().reshape(n_grid, n_grid)

        # Analytical Solution
        u_exact = analytical_function_2d(x_test, y_test)
        u_exact_np = u_exact.cpu().numpy().reshape(n_grid, n_grid)

    # Determine limits for Z axis
    v_max = max(u_nn_np.max(), u_exact_np.max()) * 1.05
    v_min = min(u_nn_np.min(), u_exact_np.min()) * 1.05

    # 3. Plotting the graphs
    fig = plt.figure(figsize=(20,16))
    ax1 = plt.subplot2grid((2,2),(0,0),colspan=2, projection='3d')
    ax2 = plt.subplot2grid((2, 2), (1, 0),  projection='3d')
    ax3 = plt.subplot2grid((2, 2), (1, 1),projection='3d')

    # Plot 1: Analytica vs DGM Solution
    surf_dgm_top = ax1.plot_surface(X, Y, u_nn_np, cmap='viridis',edgecolor='none', alpha=0.9)
    surf_exact_top = ax1.plot_wireframe(X, Y, u_exact_np, color= 'blue', linewidth=1.0, rstride=5, cstride=5)
    # Legend for combined plot
    ax1.legend(
        handles=[
            Patch(color=cm.viridis(0.7), alpha=0.8, label='DGM Solution'),
            Patch(color='blue', alpha=0.8, label='Analytical Solution')
        ],
        loc='upper right'
    )
    ax1.set_title(f'Combined DGM Solution vs Analytical Solution', fontsize=16)
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$y$')
    ax1.set_zlabel('$u(x, y, t)$')
    ax1.set_zlim(v_min, v_max)
    ax1.view_init(elev=30, azim=45)
    # Add one colorbar for the DGM solution on the combined plot
    cbar_ax1 = fig.add_axes([ax1.get_position().x1 + 0.01, ax1.get_position().y0, 0.02, ax1.get_position().height])
    fig.colorbar(surf_dgm_top, cax=cbar_ax1, label='$u_{NN}(x, y, t)$')

    # Plot 2 (bottom left) - Analytical solution
    surf2 = ax2.plot_surface(X, Y, u_exact_np, cmap='plasma',edgecolor='none', alpha=0.9)
    ax2.set_title('Bottom-Left: Analytical Solution (Reference)', fontsize=14)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('u')
    ax2.set_zlim(v_min, v_max)
    ax2.view_init(elev=30, azim=45)

    # Plot 3 (Bottom Right) - DGM Solution
    surf3 = ax3.plot_surface(X, Y, u_nn_np, cmap='viridis', edgecolor='none', alpha=0.9)
    ax3.set_title('Bottom-Right: DGM Solution (Reference)', fontsize=14)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('u')
    ax3.set_zlim(v_min, v_max)
    ax3.view_init(elev=30, azim=45)

    fig.suptitle(f'DGM Solution vs Analytical Solution ', fontsize=20, y = 0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
