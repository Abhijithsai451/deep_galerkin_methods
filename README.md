# Deep Galerkin Method (DGM) for Solving Partial Differential Equations

This project implements the Deep Galerkin Method (DGM) to solve Partial Differential Equations (PDEs), specifically focusing on the **Heat Equation** (Time-Dependent) and the **Steady State Heat Equation** (Poisson's Equation).

---

## 1. Case 1: Heat Equation (Time-Dependent)

The Heat Equation describes how heat diffuses through a medium over time:
$$\frac{\partial u}{\partial t} - \alpha \nabla^2 u = f(t, \mathbf{x})$$

### 1.1 The DGM Network and Why LSTMs?
The implementation uses a custom `DGMNet` architecture. Unlike standard feedforward networks, DGM uses **LSTM-like layers**.
- **Architecture**: The network consists of an initial dense layer, followed by several "DGM layers" (similar to LSTM cells), and a final dense output layer.
- **Why LSTMs?**: Solving PDEs involves taking higher-order derivatives of the neural network with respect to its inputs. Standard deep networks often suffer from the **vanishing gradient problem**. The LSTM-like architecture in DGM uses gating mechanisms to maintain information flow and stable gradients, crucial for approximating second-order spatial derivatives.

```python
class NeuralNetwork(nn.Module):
    def forward(self, S: torch.Tensor, X: torch.tensor) -> torch.Tensor:
        # compute components of LSTM layer output
        Z = self.trans1(X @ self.Uz + S @ self.Wz + self.bz)
        G = self.trans1(X @ self.Ug + S @ self.Wg + self.bg)
        R = self.trans1(X @ self.Ur + S @ self.Wr + self.br)
        H = self.trans2(X @ self.Uh + (S * R) @ self.Wh + self.bh)

        # compute LSTM layer output
        S_new = (torch.ones_like(G) - G) * H + Z * S
        return S_new
```

### 1.2 Utility Functions
Utility functions define the "physics" of our specific problem:
- **Source Term ($f$):** Calculates the external heat source/sink.
- **Initial Conditions (IC):** Defines the state of the system at $t=0$.
- **Boundary Conditions (BC):** Defines the constraints at the spatial boundaries.

```python
def source_term_fn_1D(t, x, alpha):
    return (-torch.exp(-t) + alpha * torch.pi**2 * torch.exp(-t)) * torch.sin(torch.pi * x)

def initial_condition_fn_1D(x):
    return torch.sin(torch.pi * x)

def boundary_condition_fn_1D(t, x):
    return torch.zeros_like(t)
```

### 1.3 Loss Functions
The network is trained by minimizing a composite loss function:
$$L_{total} = \lambda_{pde} L_{pde} + \lambda_{ic} L_{ic} + \lambda_{bc} L_{bc}$$

```python
def pde_residual_loss(model, t, x, alpha, f_tx):
    u = model(t, x)
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    
    residual = u_t - alpha * u_xx - f_tx
    return residual
```

### 1.4 Training and Visualization
Training involves sampling random points in space and time and optimizing the network parameters using Adam. We visualize the training loss history and compare predicted results with analytical solutions.

```python
# Training setup
model = dgm_network.DGMNet(nodes_per_layer, num_layers, 1)
trainer = DGMTrainer(model=model, pde_constants={'alpha': ALPHA}, learning_rate=0.0001)
trainer.train(epochs=5500, domain_data=interior_data, ic_data=ic_data, bc_data=bc_data)
```

### 1.5 Achievements
Using DGM, we successfully trained a neural network to solve a second-order time-dependent PDE with a source function. We demonstrated that the network can accurately approximate the heat distribution $u(t, x)$ across the entire domain without needing a mesh (mesh-free method).

---

## 2. Case 2: Steady State Heat Equation (Poisson's Equation)

When the system reaches equilibrium, the time derivative becomes zero, leading to the Steady State Heat Equation:
$$- \alpha \nabla^2 u = f(\mathbf{x})$$

### 2.1 Network Architecture
Similar to the time-dependent case, we use `DGMNet` with LSTM-like layers to handle the second-order spatial derivatives. Time is no longer an input variable.

```python
class DGMNet(nn.Module):
    def forward(self, x):
        S = self.initial_layer(x)
        for layer in self.LSTMLayerList:
            S = layer(S, x)
        result = self.final_layer(S)
        return result
```

### 2.2 Utility Functions
For the steady state, we focus on constant source terms and boundary conditions.

```python
def source_term_fn_1D(x, alpha):
    return -alpha * torch.pi**2 * torch.sin(torch.pi * x)

def boundary_condition_fn_1D(x):
    return torch.zeros_like(x)
```

### 2.3 Loss Functions
The loss function is simplified as there is no Initial Condition:
$$L_{total} = \lambda_{pde} L_{pde} + \lambda_{bc} L_{bc}$$

```python
def pde_residual_loss(model, x, f_x):
    u = model(x)
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    return u_xx - f_x
```

### 2.4 Training and Visualization
Training focuses on reaching a stable spatial distribution. 1D solution curves or 2D heatmaps are generated to show the predicted temperature profile.

### 2.5 Achievements
We achieved a mesh-free solution for the second-order Poisson equation. Using a neural network, we were able to find the solution $u(\mathbf{x})$ given a complex source function $f(\mathbf{x})$, effectively "learning" the physics of the steady-state heat distribution.
