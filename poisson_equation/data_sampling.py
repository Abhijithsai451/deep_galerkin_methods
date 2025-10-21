import numpy as np
import torch

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def generate_random_points(num_points, bounds):
    """
    Sample data from the PDE and Boundary conditions
    """
    num_dims = len(bounds)
    if num_dims == 0:
        return torch.empty(num_points, 0)
    points = torch.rand(num_points, num_dims, dtype=torch.float32, device = device)
    # Scale and Shift the points to the defined bounds
    for i in range(num_dims):
        min_val, max_val = bounds[i]
        points[:, i] = (max_val - min_val) * points[:, i] + min_val

    return points

def generate_domain_points(num_points, domain_bounds):
    """
    Sample data from the PDE and Boundary Conditions
    """
    spatial_coords = generate_random_points(num_points,domain_bounds).to(device)
    return spatial_coords

def generate_bc_points(num_points: int, domain_bound: list)-> torch.Tensor:
    """
    Generates a specified number of random points lying on the boundary
    of a rectangular domain defined by the given bounds.
    Args:
        num_points (int): The total number of boundary points to generate.
        bounds (list): A list of tuples/lists, where each inner element
                defines the [min, max] for that dimension.
                E.g., for a 2D square from 0 to 1: [[0, 1], [0, 1]]
    Returns:
        torch.Tensor: A tensor of shape (num_points, num_dimensions)
                containing the boundary points.
    """
    if not domain_bound:
        return torch.empty(0, 0)

    num_dims = len(domain_bound)
    boundary_points = []

    # Calculate the number of faces/boundaries
    num_boundaries = num_dims * 2

    # Determine how many points to allocate to each boundary face
    points_per_boundary = num_points // num_boundaries
    remainder = num_points % num_boundaries

    for dim_idx in range(num_dims):
        min_val, max_val = domain_bound[dim_idx]

        # Boundary Face 1: Fixed at min_val for this dimension
        n_points_face1 = points_per_boundary + (1 if remainder > 0 else 0)
        if remainder >0: remainder -= 1

        if n_points_face1 >0:
            face_points1 = torch.rand(n_points_face1, num_dims)

            face_points1[:, dim_idx] = min_val

            for other_dim_idx in range(num_dims):
                if other_dim_idx != dim_idx:
                    min_o, max_o = domain_bound[other_dim_idx]
                    face_points1[:, other_dim_idx] = (max_o - min_o) * face_points1[:, other_dim_idx] + min_o

            boundary_points.append(face_points1)

        # Boundary Face 2: Fixed at max_val for this dimension

        # Number of points for this face
        n_points_face2 = points_per_boundary + (1 if remainder > 0 else 0)
        if remainder > 0: remainder -= 1

        if n_points_face2 > 0:
            # Initialize a tensor for points on this face
            face_points2 = torch.rand(n_points_face2, num_dims)

            # Set the fixed dimension to its maximum value
            face_points2[:, dim_idx] = max_val

            # Fill the non-fixed dimensions with random values within their bounds
            for other_dim_idx in range(num_dims):
                if other_dim_idx != dim_idx:
                    min_o, max_o = domain_bound[other_dim_idx]
                    # Scale and shift the random numbers
                    face_points2[:, other_dim_idx] = (max_o - min_o) * face_points2[:, other_dim_idx] + min_o

            boundary_points.append(face_points2)

        # Concatenate all generated points
        if boundary_points:
            return torch.cat(boundary_points, dim=0)[:num_points]
        else:
            return torch.empty(0, num_dims)