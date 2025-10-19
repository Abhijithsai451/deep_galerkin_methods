from pyclbr import Class
from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from poisson_equation.loss import loss_function, loss_function_2d

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class DGMTrainerPE:
    def __init__(self, model: nn.Module, pde_constants: dict, learning_rate):
        """
        Initializing the DGM Trainer for Poisson's Equation
        """
        self.model = model
        self.loss_fn = loss_function
        self.pde_constants = pde_constants
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)

        def train(self, epochs: int,
                  domain_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                  ic_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                  bc_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                  lambda_pde: float = 100.0,
                  lambda_bc: float = 100.0
                  ):
            self.model.train()

            # Unpack and move data to device for training
            t_int, x_int, f_tx = (d.to(self.device) for d in domain_data)
            t_ic, x_ic, u_ic = (d.to(self.device) for d in ic_data)
            t_bc, x_bc, u_bc = (d.to(self.device) for d in bc_data)

            print(f"Starting training on {self.device}. Interior: {t_int.size(0)},"
                  f" IC: {t_ic.size(0)}, BC: {t_bc.size(0)} points.")

            for epoch in range(epochs):
                self.optimizer.zero_grad()

                # Calculate the loss function  -> pde_residual_loss + bc_loss + ic_loss(0)
                total_loss, pde_loss , bc_loss = loss_function(self.model,x_int, x_bc,lambda_pde= lambda_pde, lambda_bc=lambda_bc)

                # Optimization Step
                total_loss.backward()
                self.optimizer.step()

                # Logging
                if (epoch + 1) % 100 == 0:
                    print(f"Epoch {epoch + 1}/{epochs} | Total Loss: {total_loss.item():.4f} "
                          f"| L_pde: {pde_loss:.4f} | L_bc: {bc_loss:.4f}")


class DGMTrainerPE_2D:
    def __init__(self, model: nn.Module, pde_constants: dict, learning_rate):
        self.model = model
        self.loss_fn = loss_function_2d
        self.pde_constants = pde_constants
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)

    def train(self, epochs: int,
              domain_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
              ic_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
              bc_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
              lambda_pde: float = 100.0,
              lambda_bc: float = 100.0
              ):
        self.model.train()

        # Unpack the data and move to device for training
        t_int, x_int, y_int, f_xy = (d.to(self.device) for d in domain_data)
        t_ic , x_ic, y_ic = (d.to(self.device) for d in ic_data)
        t_bc, x_bc, y_bc, f_bc = (d.to(self.device) for d in bc_data)


        for epoch in range(epochs):
            self.optimizer.zero_grad()

            # Calculate the loss function  -> pde_residual_loss + bc_loss + ic_loss(0)
            total_loss, pde_loss, bc_loss = loss_function_2d(self.model,x_int,y_int,f_xy, x_bc,y_bc,f_bc, lambda_pde=lambda_pde,
                                                          lambda_bc=lambda_bc)
            total_loss.backward()
            self.optimizer.step()

            print(f"Epoch {epoch + 1}/{epochs} | Total Loss: {total_loss.item():.4f} "
                  f"| L_pde: {pde_loss:.4f} | L_bc: {bc_loss:.4f}")










