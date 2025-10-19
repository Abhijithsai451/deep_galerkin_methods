import network
from poisson_equation.trainer import DGMTrainerPE, DGMTrainerPE_2D
from poisson_equation.utility_functions import *
from poisson_equation.data_sampling import *

#%%  Deep Galerkin Method with Poisson Equation in 1D

lx_1d = 2.0
bounds_1d = [[0.0, lx_1d]]

# PDE Constants (For Poisson Equation )

# Data Sizes
N_INT = 1500
N_IC = 400
N_BC =400


# Data Preparation

# 1. Domain Data (x in [0, lx_1d])
x_int = generate_domain_points(N_INT, bounds_1d)
f_x = source_term_fn_1D(x_int).to(device)
interior_data = (x_int, f_x)

# 2 Boundary Points
c_bc = generate_bc_points(N_BC, bounds_1d)
u_bc = boundary_condition_fn_1D(c_bc).to(device)
bc_data = (c_bc, u_bc)

# visualize_points(x_int,c_bc, bounds_1d)

# Network Initialization and Training
num_layers = 5
nodes_per_layer = 64
learning_rate = 0.001
epochs = 5000

model = network.DGMNet(nodes_per_layer, num_layers, 1).to(device)

trainer = DGMTrainerPE(
    model=model,
    pde_constants=PDE_CONSTANTS,
    learning_rate=learning_rate
)

trainer.train(
    epochs=epochs,
    interior_data=interior_data,
    bc_data=bc_data,
    lambda_ic=50.0,
    lambda_bc=50.0
)
t_test_time = 2
