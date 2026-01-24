from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from heat_equation.loss import loss_function, loss_function_2d
from heat_equation.data_sampling import generate_domain_points, generate_ic_points, generate_boundary_points
from heat_equation.utility_functions import source_term_fn_2D, initial_condition_fn_2D, boundary_condition_fn_2D
from heat_equation.utility_functions import source_term_fn_1D, initial_condition_fn_1D, boundary_condition_fn_1D

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class DGMTrainer:
    def __init__(self,model: nn.Module,  pde_constants: dict, learning_rate):
        """
        Initializes the DGM Trainer.
        """
        self.model = model
        self.loss_fn = loss_function
        self.pde_constants = pde_constants
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)
        self.loss_history = []
        self.pde_loss_history = []
        self.ic_loss_history = []
        self.bc_loss_history = []

    def train(self,
              epochs: int,
              domain_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
              ic_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
              bc_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
              lambda_ic: float = 100.0,
              lambda_bc: float = 100.0,
              resample: bool = False,
              sampling_config: dict = None):

        self.model.train()

        if not resample:
            # Unpack and move data to device for training
            t_int, x_int, f_tx = (d.to(self.device) for d in domain_data)
            t_ic, x_ic, u_ic = (d.to(self.device) for d in ic_data)
            t_bc, x_bc, u_bc = (d.to(self.device) for d in bc_data)

        print(
            f"Starting training on {self.device}. Resampling: {resample}")

        for epoch in range(epochs):
            if resample and sampling_config:
                n_int = sampling_config.get('n_int', 500)
                n_ic = sampling_config.get('n_ic', 200)
                n_bc = sampling_config.get('n_bc', 200)
                bounds = sampling_config.get('bounds')
                t_max = sampling_config.get('t_max')
                alpha = self.pde_constants['alpha']

                # Generate new points
                x_int_new, t_int_new = generate_domain_points(n_int, bounds, t_max)
                f_tx = source_term_fn_1D(t_int_new, x_int_new, alpha).to(self.device)
                t_int, x_int = t_int_new.to(self.device), x_int_new.to(self.device)

                x_ic_new = generate_ic_points(n_ic, bounds)
                t_ic = torch.zeros_like(x_ic_new[:, 0:1]).to(self.device)
                u_ic = initial_condition_fn_1D(x_ic_new).to(self.device)
                x_ic = x_ic_new.to(self.device)

                x_bc_new = generate_boundary_points(n_bc, bounds)
                t_bc = torch.rand_like(x_bc_new[:, 0:1]) * t_max
                u_bc = boundary_condition_fn_1D(t_bc, x_bc_new).to(self.device)
                t_bc, x_bc = t_bc.to(self.device), x_bc_new.to(self.device)

            self.optimizer.zero_grad()

            # Calculate Total Loss
            total_loss, pde_loss_val, ic_loss_val, bc_loss_val = self.loss_fn(
                self.model,
                t_int, x_int, f_tx,  # Interior (PDE)
                t_ic, x_ic, u_ic,  # Initial Condition
                t_bc, x_bc, u_bc,  # Spatial Boundary Condition
                alpha=self.pde_constants['alpha'],
                lambda_ic=lambda_ic,
                lambda_bc=lambda_bc
            )

            # Optimization Step
            total_loss.backward()
            self.optimizer.step()

            # Save history
            self.loss_history.append(total_loss.item())
            self.pde_loss_history.append(pde_loss_val)
            self.ic_loss_history.append(ic_loss_val)
            self.bc_loss_history.append(bc_loss_val)

            # Logging
            if (epoch + 1) % 200 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} | Total Loss: {total_loss.item():.4f} "
                    f"| L_pde: {pde_loss_val:.4f} | L_ic: {ic_loss_val:.4f} | L_bc: {bc_loss_val:.4f}")


class DGMTrainer_2D:
    def __init__(self,model: nn.Module,  pde_constants: dict, learning_rate):
        """
        Initializes the DGM Trainer.
        """
        self.model = model
        self.loss_fn = loss_function_2d
        self.pde_constants = pde_constants
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)
        self.loss_history = []
        self.pde_loss_history = []
        self.ic_loss_history = []
        self.bc_loss_history = []

    def train(self,epochs:int,
              domain_data:Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
              ic_data:Tuple[torch.Tensor, torch.Tensor, torch.Tensor,torch.Tensor],
              bc_data:Tuple[torch.Tensor, torch.Tensor, torch.Tensor,torch.Tensor],
              lambda_ic:float = 100.0,
              lambda_bc:float = 100.0,
              resample: bool = False,
              sampling_config: dict = None):
        self.model.train()

        if not resample:
            t_int, x_int, y_int, f_txy = (d.to(self.device) for d in domain_data)
            t_ic, x_ic, y_ic, u_ic = (d.to(self.device) for d in ic_data)
            t_bc, x_bc, y_bc, u_bc = (d.to(self.device) for d in bc_data)

        print(
            f"Starting training on {self.device}. Resampling: {resample}")

        for epoch in range(epochs):
            if resample and sampling_config:
                n_int = sampling_config.get('n_int', 800)
                n_ic = sampling_config.get('n_ic', 250)
                n_bc = sampling_config.get('n_bc', 250)
                bounds = sampling_config.get('bounds')
                t_max = sampling_config.get('t_max')
                alpha = self.pde_constants['alpha']

                # Generate new points
                spatial_coords, t_int_new = generate_domain_points(n_int, bounds, t_max)
                x_int_new = spatial_coords[:, 0:1]
                y_int_new = spatial_coords[:, 1:2]
                f_txy = source_term_fn_2D(t_int_new, x_int_new, y_int_new, alpha).to(self.device)
                t_int, x_int, y_int = t_int_new.to(self.device), x_int_new.to(self.device), y_int_new.to(self.device)

                spatial_coords_ic = generate_ic_points(n_ic, bounds)
                x_ic_new = spatial_coords_ic[:, 0:1]
                y_ic_new = spatial_coords_ic[:, 1:2]
                t_ic = torch.zeros_like(x_ic_new[:, 0:1]).to(self.device)
                u_ic = initial_condition_fn_2D(x_ic_new, y_ic_new).to(self.device)
                x_ic, y_ic = x_ic_new.to(self.device), y_ic_new.to(self.device)

                spatial_coords_bc = generate_boundary_points(n_bc, bounds)
                x_bc_new = spatial_coords_bc[:, 0:1]
                y_bc_new = spatial_coords_bc[:, 1:2]
                t_bc = torch.rand_like(x_bc_new[:, 0:1]) * t_max
                u_bc = boundary_condition_fn_2D(t_bc, x_bc_new, y_bc_new).to(self.device)
                t_bc, x_bc, y_bc = t_bc.to(self.device), x_bc_new.to(self.device), y_bc_new.to(self.device)

            self.optimizer.zero_grad()

            total_loss, pde_loss, ic_loss, bc_loss = self.loss_fn(self.model,t_int, x_int, y_int, f_txy,
                                                                  t_ic, x_ic, y_ic, u_ic,
                                                                  t_bc, x_bc, y_bc, u_bc,
                                                                  alpha = self.pde_constants['alpha'],
                                                                  lambda_ic = lambda_ic,
                                                                  lambda_bc = lambda_bc
                                                                  )

            # optimization step
            total_loss.backward()
            self.optimizer.step()

            # Save history
            self.loss_history.append(total_loss.item())
            self.pde_loss_history.append(pde_loss)
            self.ic_loss_history.append(ic_loss)
            self.bc_loss_history.append(bc_loss)

            # logging
            if (epoch + 1) % 200 == 0:
                print(f"Epoch {epoch + 1}/{epochs} | Total Loss: {total_loss.item():.4f} | L_pde: {pde_loss:.4f} |"
                      f" L_ic: {ic_loss:.4f} | L_bc: {bc_loss:.4f}")
