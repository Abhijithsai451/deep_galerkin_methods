# Deep Galerkin Method (DGM) for Solving Partial Differential Equations

This project implements the Deep Galerkin Method (DGM) to solve Partial Differential Equations (PDEs), specifically focusing on the **Heat Equation** (Time-Dependent) and the **Steady State Heat Equation**.

---

## 1. Heat Equation (Time-Dependent)

The Heat Equation describes how heat diffuses through a medium over time:
$$\frac{\partial u}{\partial t} - \alpha \nabla^2 u = f(t, \mathbf{x})$$

### 1.1 The DGM Network and Why LSTMs?
The implementation uses a custom `DGMNet` architecture that replaces standard feedforward layers with a series of LSTM-like units. Unlike standard deep networks, DGM is designed to handle the complexities of approximating high-order derivatives by employing gating mechanisms. These mechanisms, consisting of forget, input, and output-like gates, help maintain stable gradient flow throughout the deep architecture. This is particularly crucial for solving PDEs where we must repeatedly differentiate the network with respect to its inputs, a process that often leads to vanishing gradients in traditional MLP structures.

Within each LSTM-like layer, several "gates" control the flow of information:
- **Z (Update Gate)**: Determines the new candidate information from the current input and previous state.
- **G (Reset Gate)**: Acts as a weighting mechanism that decides how much of the candidate information $H$ is incorporated into the new state.
- **R (Relevance Gate)**: Modulates the influence of the previous state $S$ when calculating the new candidate $H$.
- **H (Candidate State)**: Represents the potential new state calculated using the current input and the modulated previous state.
The final state $S_{new} = (1 - G) \odot H + Z \odot S$ allows the network to learn complex non-linear mappings while ensuring that gradients can propagate effectively even through the high-order partial derivatives required by the PDE residual.

**How the Gates Learn the PDE:**
During training, the backpropagation of the PDE residual loss specifically targets the weights of these gates. As the optimizer minimizes $L_{pde}$, it forces the gates to learn how to selectively combine the spatial and temporal features to satisfy the physical operator. For instance, the gates learn to "remember" the influence of diffusion (the Laplacian) and "gate" the temporal evolution so that the predicted $u(t, \mathbf{x})$ aligns with the provided source term $f(t, \mathbf{x})$ and the physical constraints.

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
Utility functions are essential for defining the specific physical problem and its constraints. They calculate the source term $f(t, \mathbf{x})$, which represents external heat influences, and set the initial state of the system at $t=0$ via Initial Conditions (IC). Additionally, Boundary Conditions (BC) are defined to specify the temperature or flux at the edges of the spatial domain. These functions are used to generate the synthetic data points required to train the mesh-free neural network.

```python
def source_term_fn_1D(t, x, alpha):
    return (-torch.exp(-t) + alpha * torch.pi**2 * torch.exp(-t)) * torch.sin(torch.pi * x)

def initial_condition_fn_1D(x):
    return torch.sin(torch.pi * x)

def boundary_condition_fn_1D(t, x):
    return torch.zeros_like(t)
```

### 1.3 Loss Functions
The network is trained by minimizing a composite loss function that enforces the PDE and its constraints across the domain. The total loss $L_{total} = \lambda_{pde} L_{pde} + \lambda_{ic} L_{ic} + \lambda_{bc} L_{bc}$ weights the contributions from the PDE residual, the initial conditions, and the boundary conditions. By minimizing the PDE residual $L_{pde}$, the network learns to satisfy the underlying physical laws described by the heat equation. The IC and BC losses ensure that the solution adheres to the specific starting state and spatial boundaries of the problem.

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
Training involves an iterative process of sampling random points within the space-time domain and optimizing the network parameters using the Adam optimizer. Unlike traditional methods that require a fixed mesh, DGM samples points stochastically, allowing it to approximate the solution over the entire continuous domain. We visualize the convergence through training loss plots and verify the results by comparing the network's predictions against analytical solutions or known benchmarks. Post-training animations are also generated to observe the heat diffusion process over time.

```python
# Training setup
model = dgm_network.DGMNet(nodes_per_layer, num_layers, 1)
trainer = DGMTrainer(model=model, pde_constants={'alpha': ALPHA}, learning_rate=0.0001)
trainer.train(epochs=5500, domain_data=interior_data, ic_data=ic_data, bc_data=bc_data)
```

### 1.5 Achievements
By utilizing the Deep Galerkin Method, we successfully developed a neural network capable of solving second-order time-dependent PDEs with complex source functions. We demonstrated that the DGM architecture can accurately capture the thermal dynamics of the heat equation without the need for traditional discretization or mesh generation. This mesh-free approach proves highly effective for high-dimensional problems where traditional methods often fail due to the "curse of dimensionality."

---

## 2. Heat Equation in Steady State 

The Steady State Heat Equation describes the temperature distribution of a system that has reached thermal equilibrium, where the temperature no longer changes over time. In this state, the time derivative vanishes, resulting in a balance between spatial diffusion and external source terms:
$$- \alpha \nabla^2 u = f(\mathbf{x})$$

### 2.1 Network Architecture
For the steady-state case, we employ a similar `DGMNet` architecture but adapt it for a purely spatial domain. The network still utilizes LSTM-like layers to robustly compute the second-order spatial derivatives required by the PDE residual. These layers leverage the same gating mechanisms (Update, Reset, and Relevance gates) to prevent gradient degradation while learning the steady-state spatial temperature profile $u(\mathbf{x})$. Since the system is time-independent, time is omitted from the input vector, focusing the network's capacity on the spatial laplacian operator.

**Learning the Physics:**
In the steady-state case, the gates learn to satisfy the equilibrium condition where the spatial laplacian of the temperature matches the external heat source. By backpropagating the spatial residual loss, the gates adjust to capture the stationary curvature of the temperature field required by the second-order derivatives in the PDE, ensuring that the final output $u(\mathbf{x})$ correctly represents the system's thermal balance.

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
In the steady-state scenario, utility functions define the spatial source term $f(\mathbf{x})$ and the fixed boundary conditions that drive the equilibrium state. The source term represents a constant heat production or loss within the domain, while the boundary conditions define the temperature constraints at the edges. These functions provide the ground truth for calculating the residuals and boundary losses during the training process.

```python
def source_term_fn_1D(x, alpha):
    return -alpha * torch.pi**2 * torch.sin(torch.pi * x)

def boundary_condition_fn_1D(x):
    return torch.zeros_like(x)
```

### 2.3 Loss Functions
The loss function for the steady-state heat equation is simplified as the initial condition term is no longer relevant. The optimization objective focuses on minimizing the total loss $L_{total} = \lambda_{pde} L_{pde} + \lambda_{bc} L_{bc}$, which balances the PDE residual and the boundary constraints. The PDE loss ensures that the spatial laplacian of the predicted temperature matches the source term throughout the interior of the domain.

```python
def pde_residual_loss(model, x, f_x):
    u = model(x)
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    return u_xx - f_x
```

### 2.4 Training and Visualization
The training process focuses on achieving a stable spatial distribution that satisfies the steady-state equilibrium. We use stochastic sampling to cover the spatial domain and optimize the network until the residuals are minimized. Visualization typically involves plotting 1D temperature curves or 2D heatmaps to illustrate how the predicted solution aligns with the physical constraints and any available analytical solutions.

### 2.5 Achievements
We successfully implemented a mesh-free solver for the steady-state heat equation, demonstrating the versatility of DGM in solving elliptic PDEs. The neural network effectively "learned" the underlying physics, providing an accurate continuous approximation of the temperature field $u(\mathbf{x})$ given a specific heat source. This achievement highlights the potential of DGM for engineering applications where steady-state thermal analysis is critical for design and safety.
