from visualize import *
from data_sampling import *

lx_1d = 8.0
bounds_1d = [[0.0, lx_1d]]
num_internal_points = 100
num_boundary_points = 10


spatial_coords, t = generate_domain_points(num_internal_points, bounds_1d)
boundary_coords = generate_boundary_points(100, bounds_1d)


visualize_points_1d(spatial_coords,boundary_coords, bounds_1d)

#%% Sample Generation for 2D
# --- Example Execution ---
bounds_2d = [[-2.0, 2.0], [0.0, 4.0]]
num_int = 1500
num_bnd = 200

# Generate points using the corrected method (i_points_2d is a tuple, we take the 0th element)
i_points_2d_tuple = generate_domain_points(num_int, bounds_2d)
i_points_2d = i_points_2d_tuple[0]

b_points_2d = generate_boundary_points(num_bnd, bounds_2d)

visualize_points_2d(i_points_2d, b_points_2d, bounds_2d)
