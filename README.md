# Deep Galerkin Method (DGM) for Solving Partial Differential Equations

## Overview

This project implements a Deep Learning approach, specifically the Deep Galerkin Method (DGM), to solve Partial Differential Equations (PDEs). The current implementation focuses on solving Poisson's equation in both one and two dimensions. The method leverages neural networks to approximate the solution of PDEs by minimizing a custom loss function that incorporates both the PDE residual and boundary conditions.

## Features

*   **1D and 2D Poisson's Equation Solvers**: Contains functions to calculate the PDE residual for Poisson's equation in both 1D and 2D.
*   **Physics-Informed Loss Functions**: Implements loss functions that combine the PDE residual loss with boundary condition losses, guiding the neural network to find solutions that satisfy both the differential equation and its constraints.
*   **Custom Neural Network Architecture**:
    *   `DenseLayer`: A standard fully connected layer with optional activation functions.
    *   `NeuralNetwork`: An LSTM-like layer, inspired by the architecture used in Deep Galerkin Methods.
    *   `DGMNet`: The main network architecture for the Deep Galerkin Method, composed of an initial dense layer, multiple LSTM-like layers, and a final dense output layer.

## Dependencies

*   Python 3.x
*   `torch` (PyTorch)

## Installation

You can install the required dependencies using pip:



## Usage

This repository provides the core components for building and training a Deep Galerkin Method model. To use this project:

1.  **Define your PDE problem**: Specify the domain, boundary conditions, and source term for the Poisson's equation you wish to solve.
2. **Instantiate the `DGMNet` model**:
   ```python
   from dgm_network import DGMNet
   # For a 2D problem, input_dim=2 (e.g., for x and y coordinates)
   # For a 1D problem, input_dim=1 (e.g., for x coordinate)
   model = DGMNet(layer_width=50, n_layers=3, input_dim=2, final_trans=None)
   ```
3.  **Prepare training data**: Generate collocation points within the domain and on the boundaries. These points will be used to compute the PDE residual and enforce boundary conditions.
4. **Train the model**: Use an optimizer (e.g., Adam) and the provided `loss_function` (or `loss_function_2d`) to train the `DGMNet`.

    A sketch of a training loop for a 2D problem:
   ```python
   import torch
   import torch.optim as optim
   from dgm_network import DGMNet
   from losses import loss_function_2d # Use loss_function for 1D problems

   # --- Example: Define dummy data (replace with your actual problem data) ---
   # Interior points (x_int, y_int) and corresponding source term (f_xy)
   x_int = torch.rand(100, 1) * 2 - 1 # Example: x from -1 to 1
   y_int = torch.rand(100, 1) * 2 - 1 # Example: y from -1 to 1
   source_term_domain = -4 * torch.ones_like(x_int) # Example: f(x,y) = -4 for Poisson's equation

   # Boundary points (x_bc, y_bc)
   # For simplicity, let's assume a square boundary from -1 to 1
   x_bc_sides = torch.cat([torch.full((25,1), -1.0), torch.full((25,1), 1.0)], dim=0)
   y_bc_sides = torch.cat([torch.full((25,1), -1.0), torch.full((25,1), 1.0)], dim=0)
   x_bc_top_bottom = torch.cat([torch.rand(25,1)*2-1, torch.rand(25,1)*2-1], dim=0)
   y_bc_top_bottom = torch.cat([torch.full((25,1), -1.0), torch.full((25,1), 1.0)], dim=0)

   x_bc = torch.cat([x_bc_sides, x_bc_top_bottom], dim=0)
   y_bc = torch.cat([y_bc_top_bottom, y_bc_sides], dim=0)
   # ------------------------------------------------------------------------

   # Initialize the model and optimizer
   model = DGMNet(layer_width=50, n_layers=3, input_dim=2, final_trans=None)
   optimizer = optim.Adam(model.parameters(), lr=0.001)

   # Training loop
   num_epochs = 1000
   print("Starting training...")
   for epoch in range(num_epochs):
       optimizer.zero_grad()
       total_loss, pde_loss, bc_loss = loss_function_2d(model, x_int, y_int, source_term_domain, x_bc, y_bc)
       total_loss.backward()
       optimizer.step()

       if (epoch + 1) % 100 == 0:
           print(f"Epoch {epoch+1}, Total Loss: {total_loss.item():.4f}, PDE Loss: {pde_loss:.4f}, BC Loss: {bc_loss:.4f}")

   print("Training complete.")
   # You can now use the 'model' to predict solutions at new points.
   ```
5.  **Evaluate the solution**: Once trained, the `model` can predict the solution `u(x)` or `u(x,y)` at any given input point.

## Project Structure

*   `losses.py`: Contains functions for calculating PDE residuals (e.g., `pde_residual_loss`, `pde_residual_loss_2d`) and the combined loss functions (e.g., `loss_function`, `loss_function_2d`) for 1D and 2D problems.
*   `network.py`: Defines the custom neural network layers (`DenseLayer`, `NeuralNetwork`) and the overall `DGMNet` architecture used in the Deep Galerkin Method.

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

## License

This project is open-sourced under the MIT License.
