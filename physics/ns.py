"""2D Navier-Stokes equations (vorticity-streamfunction formulation).

For flow past a circular cylinder at Re=100.

Continuity (auto-satisfied via streamfunction psi):
    u = d(psi)/dy,  v = -d(psi)/dx

Momentum (vorticity transport):
    w_t + u w_x + v w_y - nu (w_xx + w_yy) = 0
where w = -laplacian(psi)

Provides both forward and inverse problem interfaces.
"""

import torch
from torch.autograd import grad

NU = 0.01  # kinematic viscosity (1/Re for Re=100)


def compute_ns_residuals(model, x, y, t, nu=NU):
    """Compute NS residuals using streamfunction-vorticity formulation.

    Args:
        model: PINN model with forward(x,y,t) → (psi, p)
        x, y, t: collocation tensors (N,1), requires_grad, in physical units

    Returns:
        res_vorticity: vorticity transport residual  (N, 1)
        psi, p: predicted fields
    """
    psi, p = model(x, y, t)

    # Velocity from streamfunction
    u = grad(psi, y, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
    v = -grad(psi, x, grad_outputs=torch.ones_like(psi), create_graph=True)[0]

    # Vorticity w = v_x - u_y
    v_x = grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    u_y = grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    w = v_x - u_y

    # Vorticity time derivative
    w_t = grad(w, t, grad_outputs=torch.ones_like(w), create_graph=True)[0]

    # Advection
    w_x = grad(w, x, grad_outputs=torch.ones_like(w), create_graph=True)[0]
    w_y = grad(w, y, grad_outputs=torch.ones_like(w), create_graph=True)[0]

    # Diffusion
    w_xx = grad(w_x, x, grad_outputs=torch.ones_like(w_x), create_graph=True)[0]
    w_yy = grad(w_y, y, grad_outputs=torch.ones_like(w_y), create_graph=True)[0]

    # Vorticity transport residual
    res = w_t + u * w_x + v * w_y - nu * (w_xx + w_yy)

    return res, psi, p, u, v
