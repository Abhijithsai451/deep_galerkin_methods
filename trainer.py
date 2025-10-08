from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from loss import loss_function

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

    def train(self,
              epochs: int,
              # Data tuples MUST be passed in the correct format: (t, x, target)
              interior_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
              ic_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
              bc_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
              lambda_ic: float = 100.0,
              lambda_bc: float = 100.0):

        self.model.train()

        # Unpack and move data to device once (tensors are already on device but safer to move again)
        t_int, x_int, f_tx = (d.to(self.device) for d in interior_data)
        t_ic, x_ic, u_ic = (d.to(self.device) for d in ic_data)
        t_bc, x_bc, u_bc = (d.to(self.device) for d in bc_data)

        print(
            f"Starting training on {self.device}. Interior: {t_int.size(0)}, IC: {t_ic.size(0)}, BC: {t_bc.size(0)} points.")

        for epoch in range(epochs):
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

            # Logging
            if (epoch + 1) % 100 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} | Total Loss: {total_loss.item():.4f} | L_pde: {pde_loss_val:.4f} | L_ic: {ic_loss_val:.4f} | L_bc: {bc_loss_val:.4f}")


