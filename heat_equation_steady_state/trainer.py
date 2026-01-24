from pyclbr import Class
from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from heat_equation_steady_state.loss import loss_function, loss_function_2d

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class DGMTrainerSS:
    def __init__(self, model: nn.Module,  learning_rate, alpha: float = 1.0):
        """
        Initializing the DGM Trainer for Steady State Heat Equation
        """
        self.model = model
        self.loss_fn = loss_function
        self.alpha = alpha
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)
        self.loss_history = []
        self.pde_loss_history = []
        self.bc_loss_history = []

    def train(self, epochs: int,
                  domain_data: Tuple[torch.Tensor, torch.Tensor],
                  bc_data: Tuple[torch.Tensor, torch.Tensor],
                  lambda_pde: float = 100.0,
                  lambda_bc: float = 100.0,
                  resample: bool = False,
                  sampling_config: dict = None
                  ):
        self.model.train()

        if not resample:
            # Unpack and move data to device for training
            x_int, f_x = (d.to(self.device) for d in domain_data)
            x_bc, u_bc = (d.to(self.device) for d in bc_data)

        print(f"Starting training on {self.device}. Resampling: {resample}")

        for epoch in range(epochs):
            if resample and sampling_config:
                from heat_equation_steady_state.data_sampling import generate_domain_points, generate_bc_points
                from heat_equation_steady_state.utility_functions import source_term_fn_1D, boundary_condition_fn_1D

                n_int = sampling_config.get('n_int', 1500)
                n_bc = sampling_config.get('n_bc', 800)
                bounds = sampling_config.get('bounds')
                alpha = self.alpha

                # Generate new points
                x_int = generate_domain_points(n_int, bounds).to(self.device)
                f_x = source_term_fn_1D(x_int, alpha).to(self.device)

                x_bc = generate_bc_points(n_bc, bounds).to(self.device)
                u_bc = boundary_condition_fn_1D(x_bc).to(self.device)

            self.optimizer.zero_grad()

            # Calculate the loss function  -> pde_residual_loss + bc_loss + ic_loss(0)
            total_loss, pde_loss , bc_loss = loss_function(self.model,x_int, f_x, x_bc,lambda_pde= lambda_pde, lambda_bc=lambda_bc)

            # Optimization Step
            total_loss.backward()
            self.optimizer.step()

            # Save history
            self.loss_history.append(total_loss.item())
            self.pde_loss_history.append(pde_loss)
            self.bc_loss_history.append(bc_loss)

            # Logging
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs} | Total Loss: {total_loss.item():.4f} "
                          f"| L_pde: {pde_loss:.4f} | L_bc: {bc_loss:.4f}")


class DGMTrainerSS_2D:
    """
    Initializing the DGM Trainer for Steady State Heat Equation in 2D
    """
    def __init__(self, model: nn.Module,  learning_rate, alpha: float = 1.0):
        self.model = model
        self.loss_fn = loss_function_2d
        self.alpha = alpha
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)
        self.loss_history = []
        self.pde_loss_history = []
        self.bc_loss_history = []

    def train(self, epochs: int,
              domain_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
              bc_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
              lambda_pde: float = 100.0,
              lambda_bc: float = 100.0,
              resample: bool = False,
              sampling_config: dict = None
              ):
        self.model.train()

        if not resample:
            # Unpack the data and move to device for training
            x_int, y_int, f_xy = (d.to(self.device) for d in domain_data)
            x_bc, y_bc, u_bc = (d.to(self.device) for d in bc_data)

        print(f"Starting training on {self.device}. Resampling: {resample}")

        for epoch in range(epochs):
            if resample and sampling_config:
                from heat_equation_steady_state.data_sampling import generate_domain_points, generate_bc_points
                from heat_equation_steady_state.utility_functions import source_term_fn_2D, boundary_condition_fn_2D

                n_int = sampling_config.get('n_int', 2000)
                n_bc = sampling_config.get('n_bc', 800)
                bounds = sampling_config.get('bounds')
                alpha = self.alpha

                # Generate new points
                spatial_coords = generate_domain_points(n_int, bounds).to(self.device)
                x_int = spatial_coords[:, 0:1]
                y_int = spatial_coords[:, 1:2]
                f_xy = source_term_fn_2D(x_int, y_int, alpha).to(self.device)

                spatial_coords_bc = generate_bc_points(n_bc, bounds).to(self.device)
                x_bc = spatial_coords_bc[:, 0:1]
                y_bc = spatial_coords_bc[:, 1:2]
                u_bc = boundary_condition_fn_2D(x_bc, y_bc).to(self.device)

            self.optimizer.zero_grad()

            # Calculate the loss function  -> pde_residual_loss + bc_loss + ic_loss(0)
            total_loss, pde_loss, bc_loss = loss_function_2d(self.model,x_int,y_int,f_xy, x_bc,y_bc,u_bc, lambda_pde=lambda_pde,
                                                          lambda_bc=lambda_bc)
            total_loss.backward()
            self.optimizer.step()

            # Save history
            self.loss_history.append(total_loss.item())
            self.pde_loss_history.append(pde_loss)
            self.bc_loss_history.append(bc_loss)

            # logging
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs} | Total Loss: {total_loss.item():.4f} "
                      f"| L_pde: {pde_loss:.4f} | L_bc: {bc_loss:.4f}")










