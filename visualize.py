import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
def visualize_points_1d(domain_points:torch.Tensor,
                        boundary_points:torch.Tensor,
                        bounds: list,
                        title:str = "Visualization of the Generated Points in 1D"):
    """
    Visualizes 1D internal and boundary points in 1D.
    """
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 4))

    if not bounds or len(bounds[0]) != 2:
        raise ValueError("Bounds must be in format [[x_min, x_max]].")

    x_min, x_max = bounds[0]

    # --- 1. Draw the Domain  ---
    plt.plot([x_min, x_max], [0, 0],
             linestyle='-', color='gray', linewidth=3, label='Domain')

    # --- 2. Plot Internal Points ---
    if domain_points.numel() > 0:
        x_int = domain_points.cpu().detach().flatten().numpy()
        y_int = torch.zeros_like(domain_points).cpu().detach().flatten().numpy()
        plt.scatter(x_int, y_int,
                    s=40, color='#337AFF', marker='o', alpha=0.6,
                    zorder=3, label='Internal Points')

    # --- 3. Plot Boundary Points ---
    if boundary_points.numel() > 0:
        x_bnd = boundary_points.cpu().detach().flatten().numpy()
        y_bnd = torch.zeros_like(boundary_points).cpu().detach().flatten().numpy()
        plt.scatter(x_bnd, y_bnd,
                    s=100, color='#FF5733', marker='o',
                    edgecolors='black', linewidths=1.5, zorder=5,
                    label='Boundary Points (BCs)')

    # --- Set Plot Properties ---
    x_range = x_max - x_min
    plt.xlim(x_min - 0.1 * x_range, x_max + 0.1 * x_range)
    plt.ylim(-0.5, 0.5)

    plt.title(title, fontsize=14)
    plt.xlabel(f'Dimension 1 (x-axis) in {bounds[0]}', fontsize=12)
    plt.yticks([]) # Hide y-axis ticks/labels
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)
    plt.show()


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






















