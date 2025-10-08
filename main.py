import network
from trainer import DGMTrainer
from visualize import *
from data_sampling import *
#%% Deep Galerkin Method in 1D
"""
lx_1d = 8.0
bounds_1d = [[0.0, lx_1d]]
num_internal_points = 100
num_boundary_points = 10

# Network Parameters
num_layers = 3
nodes_per_layer = 10
learning_rate = 0.001

# Training Parameters
sampling_stages = 500
steps_per_sample = 10


# Sample Generation and Visualization in 1D
spatial_coords, t = generate_domain_points(num_internal_points, bounds_1d)
boundary_coords = generate_boundary_points(100, bounds_1d)

visualize_points_1d(spatial_coords,boundary_coords, bounds_1d)

# Network
model = network.DGMNet(nodes_per_layer, num_layers, 1).to(device)

# Training
print(f"Initiating training on {device}...")

"""
# --- Problem Definition ---
lx_1d = 1.0           # Space domain [0, 1.0]
T_max = 1.0           # Time domain [0, 1.0]
bounds_1d = [[0.0, lx_1d]]

# PDE Constants (for Heat Equation: u_t = alpha * u_xx + f)
ALPHA = 0.01
PDE_CONSTANTS = {'alpha': ALPHA}

# Data Sizes
N_INT = 1000
N_IC = 200
N_BC = 200

# --- Target Functions (Must be defined for your specific problem) ---
# Example for u(x, t) = exp(-t) * sin(pi*x)
def source_term_fn(t, x):
    # f(t,x) = u_t - alpha*u_xx (plug known solution into PDE)
    # u_t = -exp(-t)*sin(pi*x)
    # u_xx = -pi^2 * exp(-t) * sin(pi*x)
    return (-torch.exp(-t) + ALPHA * torch.pi**2 * torch.exp(-t)) * torch.sin(torch.pi * x)

def initial_condition_fn(x):
    # u(x, 0) = sin(pi*x)
    return torch.sin(torch.pi * x)

def boundary_condition_fn(t, x):
    # u(0, t) = u(1, t) = exp(-t)*sin(0 or pi) = 0
    return torch.zeros_like(t)

# --- Data Preparation ---

# 1. Interior Data (t in [0, T_max], x in [0, lx_1d])
x_int, t_int = generate_domain_points(N_INT, bounds_1d, T_max)
f_tx = source_term_fn(t_int, x_int).to(device)
interior_data = (t_int, x_int, f_tx)

# 2. Initial Condition Data (t = 0, x in [0, lx_1d])
x_ic = generate_ic_points(N_IC, bounds_1d)
t_ic = torch.zeros_like(x_ic[:, 0:1]).to(device)
u_ic = initial_condition_fn(x_ic).to(device)
ic_data = (t_ic, x_ic, u_ic)

# 3. Boundary Condition Data (t in [0, T_max], x = 0 or 1)
x_bc = generate_boundary_points(N_BC, bounds_1d)
t_bc = torch.rand_like(x_bc[:, 0:1]) * T_max # Sample time for boundary points
u_bc = boundary_condition_fn(t_bc, x_bc).to(device)
bc_data = (t_bc, x_bc, u_bc)

# --- Network Initialization and Training ---
num_layers = 3
nodes_per_layer = 20
learning_rate = 0.001
epochs = 5000

model = network.DGMNet(nodes_per_layer, num_layers, 1).to(device)

trainer = DGMTrainer(
    model=model,
    pde_constants=PDE_CONSTANTS,
    learning_rate=learning_rate
)

trainer.train(
    epochs=epochs,
    interior_data=interior_data,
    ic_data=ic_data,
    bc_data=bc_data,
    lambda_ic=50.0,
    lambda_bc=50.0
)


"""#%%
#
# Sample Generation for 2D
bounds_2d = [[-2.0, 2.0], [0.0, 4.0]]
num_int = 1500
num_bnd = 200

# Generate points using the corrected method (i_points_2d is a tuple, we take the 0th element)
i_points_2d_tuple = generate_domain_points(num_int, bounds_2d)
i_points_2d = i_points_2d_tuple[0]

b_points_2d = generate_boundary_points(num_bnd, bounds_2d)

visualize_points_2d(i_points_2d, b_points_2d, bounds_2d)
"""