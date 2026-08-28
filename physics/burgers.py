"""1D Burgers equation for generalization experiments.

    u_t + u * u_x - nu * u_xx = 0

with Dirichlet BCs u(t,-1)=u(t,1)=0.
"""

import torch
from torch.autograd import grad
import numpy as np


def compute_burgers_residual(model, x, t, nu=0.01 / np.pi):
    """Compute Burgers PDE residual.

    Args:
        model: PINN model with forward(x,t) → (u,)
        x: spatial coordinate (N, 1), requires_grad, in [-1, 1]
        t: time coordinate (N, 1), requires_grad, in [0, 1]

    Returns:
        residual: (N, 1)
    """
    u = model(x, t)

    u_t = grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

    return u_t + u * u_x - nu * u_xx


def burgers_solution(x, t, nu=0.01 / np.pi):
    """Analytical Burgers solution via Cole-Hopf transformation.

    For validation only on small grids.
    """
    import numpy as np
    x_np = x.detach().cpu().numpy().flatten()
    t_np = t.detach().cpu().numpy().flatten()

    u = np.zeros_like(x_np)
    for i in range(len(x_np)):
        xi, ti = x_np[i], t_np[i]
        s = 0.0
        for k in range(1, 100):
            ak = 2 * (-1) ** (k + 1) / (k * np.pi)
            s += ak * np.exp(-nu * k ** 2 * np.pi ** 2 * ti) * np.sin(k * np.pi * xi)
        u[i] = 2 * nu * np.pi * s / (1 + s + 1e-15)
    return torch.tensor(u, dtype=torch.float32).reshape(-1, 1)
