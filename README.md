# Deep Galerkin Method (DGM) for Heat Equation and Steady State Analysis

This project implements the Deep Galerkin Method (DGM) to solve Partial Differential Equations (PDEs), specifically focusing on the **Heat Equation** (Time-Dependent) and the **Steady State Heat Equation** (Poisson's Equation).

---

## 1. Case 1: Heat Equation (Time-Dependent)

The Heat Equation describes how heat diffuses through a medium over time:
$$\frac{\partial u}{\partial t} - \alpha \nabla^2 u = f(t, \mathbf{x})$$

### 1.1 The DGM Network and Why LSTMs?
The implementation uses a custom `DGMNet` architecture. Unlike standard feedforward networks, DGM uses **LSTM-like layers** (Long Short-Term Memory). 
- **Architecture**: The network consists of an initial dense layer, followed by several "DGM layers" (similar to LSTM cells), and a final dense output layer.
- **Why LSTMs?**: Solving PDEs involves taking higher-order derivatives of the neural network with respect to its inputs (space and time). Standard deep networks often suffer from the **vanishing gradient problem** when these derivatives are propagated through many layers. The LSTM-like architecture in DGM uses gating mechanisms (like forgot and update gates) to maintain information flow and stable gradients, which is crucial for accurately approximating the complex second-order spatial derivatives required by the Heat Equation.

### 1.2 Utility Functions
Utility functions define the "physics" of our specific problem:
- **Source Term ($f$):** Functions like `source_term_fn_1D` and `source_term_fn_2D` calculate the external heat source/sink.
- **Initial Conditions (IC):** `initial_condition_fn_1D` defines the state of the system at $t=0$.
- **Boundary Conditions (BC):** `boundary_condition_fn_1D` defines the constraints at the spatial boundaries (e.g., temperatures at the edges).
- **Analytical Solution:** Functions like `analytical_solution` provide the ground truth for validating the neural network's accuracy.

### 1.3 Loss Functions
The network is trained by minimizing a composite loss function:
$$L_{total} = \lambda_{pde} L_{pde} + \lambda_{ic} L_{ic} + \lambda_{bc} L_{bc}$$
- **PDE Residual Loss ($L_{pde}$):** Measures how well the network satisfies $\frac{\partial u}{\partial t} - \alpha \nabla^2 u - f = 0$ at random collocation points.
- **Initial Condition Loss ($L_{ic}$):** Ensures the network matches the IC at $t=0$.
- **Boundary Condition Loss ($L_{bc}$):** Ensures the network satisfies the BCs at the domain edges.

### 1.4 Training and Visualization
Training involves sampling random points in space and time and optimizing the network parameters using Adam.
- **Plots:** We visualize the training loss history and compare the predicted solution against the analytical solution.
- **Animation:** For time-dependent cases, we generate animations showing how the temperature distribution evolves over time.

### 1.5 Achievements
Using DGM, we successfully trained a neural network to solve a second-order time-dependent PDE with a source function. We demonstrated that the network can accurately approximate the heat distribution $u(t, x)$ across the entire domain without needing a mesh (mesh-free method).

---

## 2. Case 2: Steady State Heat Equation (Poisson's Equation)

When the system reaches equilibrium, the time derivative becomes zero, leading to the Steady State Heat Equation:
$$- \alpha \nabla^2 u = f(\mathbf{x})$$

### 2.1 Network Architecture
Similar to the time-dependent case, we use `DGMNet` with LSTM-like layers to handle the second-order spatial derivatives ($\nabla^2 u$). The input dimension is reduced as time is no longer a variable.

### 2.2 Utility Functions
For the steady state, we focus on:
- **Steady Source Term:** `source_term_fn_1D`/`2D` representing the constant heat source.
- **Boundary Conditions:** Standard Dirichlet conditions (e.g., $u=0$ at boundaries).
- **Analytical Verification:** `analytical_function_1d`/`2d` used to compute relative L2 errors.

### 2.3 Loss Functions
The loss function is simplified as there is no Initial Condition:
$$L_{total} = \lambda_{pde} L_{pde} + \lambda_{bc} L_{bc}$$
The PDE loss ensures $\nabla^2 u - f = 0$ holds throughout the spatial domain.

### 2.4 Training and Visualization
Training focuses on reaching a stable spatial distribution.
- **Plots:** 1D solution curves or 2D heatmaps are generated to show the predicted temperature profile compared to the expected analytical results.

### 2.5 Achievements
We achieved a mesh-free solution for the second-order Poisson equation. By using a neural network, we were able to find the solution $u(\mathbf{x})$ given a complex source function $f(\mathbf{x})$, effectively "learning" the physics of the steady-state heat distribution. The DGM approach proved robust in approximating the solution even with high-frequency source terms.
