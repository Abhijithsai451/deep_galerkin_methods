import network
import torch
import numpy as np
import random

from trainer import DGMTrainer, DGMTrainer_2D
from visualize import *
from heat_equation.data_sampling import *
from utility_functions import *

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)


#%% Deep Galerkin Method in 1D

lx_1d = 2.0
T_max = 2.0
bounds_1d = [[0.0, lx_1d]]

# PDE Constants (for Heat Equation: u_t = alpha * u_xx + f)
ALPHA = 0.01
PDE_CONSTANTS = {'alpha': ALPHA}

# Data Sizes
N_INT = 500
N_IC = 200
N_BC =200


# --- Data Preparation ---

# 1. Domain Data (t in [0, T_max], x in [0, lx_1d])
x_int, t_int = generate_domain_points(N_INT, bounds_1d, T_max)
f_tx = source_term_fn_1D(t_int, x_int, ALPHA).to(device)
interior_data = (t_int, x_int, f_tx)

# 2. Initial Condition Data (t = 0, x in [0, lx_1d])
x_ic = generate_ic_points(N_IC, bounds_1d)
t_ic = torch.zeros_like(x_ic[:, 0:1]).to(device)
u_ic = initial_condition_fn_1D(x_ic).to(device)
ic_data = (t_ic, x_ic, u_ic)

# 3. Boundary Condition Data (t in [0, T_max], x = 0 or 1)
x_bc = generate_boundary_points(N_BC, bounds_1d)
t_bc = torch.rand_like(x_bc[:, 0:1]) * T_max # Sample time for boundary points
u_bc = boundary_condition_fn_1D(t_bc, x_bc).to(device)
bc_data = (t_bc, x_bc, u_bc)

#visualize_points_1d(x_int,x_bc, bounds_1d)

# --- Network Initialization and Training ---
num_layers = 5
nodes_per_layer = 64
learning_rate = 0.001
epochs = 3000
model = network.DGMNet(nodes_per_layer, num_layers, 1).to(device)

trainer = DGMTrainer(
    model=model,
    pde_constants=PDE_CONSTANTS,
    learning_rate=learning_rate
)

trainer.train(
    epochs=epochs,
    domain_data=interior_data,
    ic_data=ic_data,
    bc_data=bc_data,
    lambda_ic=50.0,
    lambda_bc=50.0,
    resample=True,
    sampling_config={
        'n_int': N_INT,
        'n_ic': N_IC,
        'n_bc': N_BC,
        'bounds': bounds_1d,
        't_max': T_max
    }
)
visualize_loss(trainer, title="Training Loss History - Heat Equation 1D")
t_test_time = 2
visualize_solution_1d(
    model=model,
    domain_bound=lx_1d,
    t_test=t_test_time,
    n_test_points=500
)
print("Finished Training for Heat Equation in 1D \n\n")

#%% Deep Galerkin Method in 2D

lx_2d = 2.0
ly_2d = 2.0
T_max = 2.0
bounds_2d = [[0.0, lx_2d],[0.0, ly_2d]]

# PDE Constants (for Heat Equation: u_t = alpha * u_xx + f)
ALPHA = 0.01
PDE_CONSTANTS = {'alpha': ALPHA}

# Data Sizes
N_INT = 800
N_IC = 250
N_BC =250

# Generate points using the corrected method (i_points_2d is a tuple, we take the 0th element)
# --- Data Preparation ---

# 1. Interior Data (t in [0, T_max], x in [0, lx_1d])
spatial_coords, t_int = generate_domain_points(N_INT, bounds_2d, T_max)
x_int = spatial_coords[:, 0:1]
y_int = spatial_coords[:, 1:2]
f_txy = source_term_fn_2D(t_int, x_int, y_int, ALPHA).to(device)
interior_data = (t_int, x_int, y_int,  f_txy)

# 2. Initial Condition Data (t = 0, x in [0, lx_1d])
spatial_coords_ic = generate_ic_points(N_IC, bounds_2d)
x_ic = spatial_coords_ic[:, 0:1]
y_ic = spatial_coords_ic[:, 1:2]
t_ic = torch.zeros_like(x_ic[:, 0:1]).to(device)
u_ic = initial_condition_fn_2D(x_ic,y_ic).to(device)
ic_data = (t_ic, x_ic, y_ic, u_ic)

# 3. Boundary Condition Data (t in [0, T_max], x = 0 or 1)
spatial_coords_bc = generate_boundary_points(N_BC, bounds_2d)
x_bc = spatial_coords_bc[:, 0:1]
y_bc = spatial_coords_bc[:, 1:2]
t_bc = torch.rand_like(x_bc[:, 0:1]) * T_max # Sample time for boundary points
u_bc = boundary_condition_fn_2D(t_bc, x_bc, y_bc).to(device)
bc_data = (t_bc, x_bc,y_bc, u_bc)


#visualize_points_2d(spatial_coords, spatial_coords_bc, bounds_2d)

# --- Network Initialization and Training ---
num_layers = 6
nodes_per_layer = 64
learning_rate = 0.001
epochs = 6000

model = network.DGMNet(nodes_per_layer, num_layers, 2).to(device)

trainer = DGMTrainer_2D(
    model=model,
    pde_constants=PDE_CONSTANTS,
    learning_rate=learning_rate
)

trainer.train(
    epochs=epochs,
    domain_data=interior_data,
    ic_data=ic_data,
    bc_data=bc_data,
    lambda_ic=50.0,
    lambda_bc=50.0,
    resample=True,
    sampling_config={
        'n_int': N_INT,
        'n_ic': N_IC,
        'n_bc': N_BC,
        'bounds': bounds_2d,
        't_max': T_max
    }
)
visualize_loss(trainer, title="Training Loss History - Heat Equation 2D")
t_test_time = 1.00
visualize_2d(
    model=model,
    bounds=bounds_2d,
    t_test=t_test_time,
    n_grid=500
)
visualize_solution_2d(
    model=model,
    bounds=bounds_2d,
    t_test=t_test_time,
    n_grid=500
)
print("Finished Training for Heat Equation Time Dependent in 2D \n\n")
