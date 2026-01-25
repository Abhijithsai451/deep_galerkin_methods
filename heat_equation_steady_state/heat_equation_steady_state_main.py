import torch
import numpy as np
import random
from heat_equation_steady_state.trainer import *
from heat_equation_steady_state.data_sampling import *
from heat_equation_steady_state.visualize import *
from heat_equation_steady_state.utility_functions import *
import heat_equation_steady_state_network as network

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

#%% Deep Galerkin Method with Steady Heat Equation in 1D

lx_1d = 2.0
bounds_1d = [[0.0, lx_1d]]
alpha = 1
# PDE Constants (For Poisson Equation )

# Data Sizes
N_INT = 1500
N_BC =800

# Data Preparation

# 1. Domain Data (x in [0, lx_1d])
x_int = generate_domain_points(N_INT, bounds_1d)
#print("Internal domain points ",x_int)
f_x = source_term_fn_1D(x_int, alpha).to(device)
domain_data = (x_int, f_x)

# 2 Boundary Points
c_bc = generate_bc_points(N_BC, bounds_1d)
u_bc = boundary_condition_fn_1D(c_bc).to(device)
bc_data = (c_bc, u_bc)
#visualize_points_1d(x_int,c_bc, bounds_1d)

# Network Initialization and Training
num_layers = 5
nodes_per_layer = 64
learning_rate = 0.001
epochs = 2000

model = network.DGMNet(nodes_per_layer, num_layers, 1).to(device)

trainer = DGMTrainerSS(
    model=model,
    learning_rate=learning_rate,
    alpha=alpha
)

trainer.train(
    epochs=epochs,
    domain_data=domain_data,
    bc_data=bc_data,
    lambda_pde=500.0,
    lambda_bc=500.0,
    resample=True,
    sampling_config={
        'n_int': N_INT,
        'n_bc': N_BC,
        'bounds': bounds_1d
    }
)
visualize_loss(trainer, title="Training Loss History - Steady-State Heat Equation 1D")
visualize_solution_1d(model, lx_1d,n_test_points=500)

#%% Deep Galerkin Method with Steady Heat Equation in 2D
lx_2d = 2.0
ly_2d = 2.0
T_max = 2.0
bounds_2d = [[0.0, lx_2d],[0.0, ly_2d]]

# PDE Constants (for Heat Equation: u_t = alpha * u_xx + f)
ALPHA = 1

# Data Sizes
N_INT = 2000
N_BC =800

# Generate points using the corrected method (i_points_2d is a tuple, we take the 0th element)
# --- Data Preparation ---

# 1. Interior Data (t in [0, T_max], x in [0, lx_1d])
spatial_coords = generate_domain_points(N_INT, bounds_2d)
x_int = spatial_coords[:, 0:1]
y_int = spatial_coords[:, 1:2]
f_xy = source_term_fn_2D(x_int, y_int,ALPHA).to(device)
interior_data = (x_int, y_int,  f_xy)


# 2. Boundary Condition Data (t in [0, T_max], x = 0 or 1)
spatial_coords_bc = generate_bc_points(N_BC, bounds_2d)
x_bc = spatial_coords_bc[:, 0:1]
y_bc = spatial_coords_bc[:, 1:2]
u_bc = boundary_condition_fn_2D(x_bc, y_bc).to(device)
bc_data = (x_bc,y_bc, u_bc)


#visualize_points_2d(spatial_coords, spatial_coords_bc, bounds_2d)

# --- Network Initialization and Training ---
num_layers = 6
nodes_per_layer = 64
learning_rate = 0.001
epochs = 3000

model = network.DGMNet(nodes_per_layer, num_layers, 2).to(device)

trainer = DGMTrainerSS_2D(
    model=model,
    learning_rate=learning_rate,
    alpha=ALPHA
)
trainer.train(
    epochs=epochs,
    domain_data=interior_data,
    bc_data=bc_data,
    lambda_pde=500.0,
    lambda_bc=500.0,
    resample=True,
    sampling_config={
        'n_int': N_INT,
        'n_bc': N_BC,
        'bounds': bounds_2d
    }
)
visualize_loss(trainer, title="Training Loss History - Steady-State Heat Equation 2D")
visualize_2d(
    model=model,
    bounds=bounds_2d,
    n_grid=500
)
visualize_solution_2d(
    model=model,
    bounds=bounds_2d,
    n_grid=500
)
