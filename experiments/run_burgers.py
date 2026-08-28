"""Generalization experiment: CA-PINN for 1D Burgers equation.

u_t + u*u_x - nu*u_xx = 0,  x in [-1,1], t in [0,1]

Uses causal training + ReLoBRaLo + RAR to demonstrate universality.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import grad
from models.pinn_net import PINN
from training.causal import CausalTrainer
from training.loss_balancing import ReLoBRaLo

torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BurgersPINN(nn.Module):
    """1D Burgers PINN: [x,t] → [u]"""

    def __init__(self, use_fourier=True):
        super().__init__()
        self.model = PINN(
            in_dim=2, hidden_layers=[64, 64, 64, 64],
            out_dim=1, use_fourier=use_fourier, fourier_dim=64,
        )

    def forward(self, x, t):
        h, _, _ = self.model(x, t, t)  # reuse h output for u; qx/qy ignored
        return h


class BurgersAdaptiveSampler:
    """RAR for Burgers equation."""

    def __init__(self, n_collocation=10000, n_candidates=30000,
                 resample_every=2000, beta=1.5, nu=0.01 / np.pi):
        self.n_collocation = n_collocation
        self.n_candidates = n_candidates
        self.resample_every = resample_every
        self.beta = beta
        self.nu = nu
        self.device = None

    def should_resample(self, epoch):
        return (epoch > 0) and (epoch % self.resample_every == 0)

    def generate_initial_points(self):
        x = (torch.rand(self.n_collocation, 1) * 2 - 1)
        t = torch.rand(self.n_collocation, 1)
        return {'x': x, 't': t}

    def resample(self, model, epoch, current):
        if not self.should_resample(epoch):
            return current
        if self.device is None:
            self.device = current['x'].device

        cand_x = (torch.rand(self.n_candidates, 1, device=self.device) * 2 - 1)
        cand_t = torch.rand(self.n_candidates, 1, device=self.device)

        with torch.no_grad():
            cand_x.requires_grad_(True)
            cand_t.requires_grad_(True)
            u = model(cand_x, cand_t)
            u_t = grad(u, cand_t, grad_outputs=torch.ones_like(u), create_graph=False)[0]
            u_x = grad(u, cand_x, grad_outputs=torch.ones_like(u), create_graph=False)[0]
            u_xx = grad(u_x, cand_x, grad_outputs=torch.ones_like(u_x), create_graph=False)[0]
            residual = (u_t + u * u_x - self.nu * u_xx).abs().squeeze()

        probs = residual ** self.beta
        probs = probs / (probs.sum() + 1e-8)
        indices = torch.multinomial(probs, self.n_collocation, replacement=True)
        return {'x': cand_x[indices].detach().clone(),
                't': cand_t[indices].detach().clone()}


def burgers_reference(nx=256, nt=100, nu=0.01 / np.pi):
    """Analytical Burgers solution via Cole-Hopf transformation."""
    x = np.linspace(-1, 1, nx)
    t = np.linspace(0, 1, nt)
    X, T = np.meshgrid(x, t)
    u = np.zeros_like(X)

    for i in range(nx):
        for j in range(nt):
            xi, ti = X[j, i], T[j, i]
            s = 0.0
            for k in range(1, 80):
                ak = 2 * (-1) ** (k + 1) / (k * np.pi)
                s += ak * np.exp(-nu * k ** 2 * np.pi ** 2 * ti) * np.sin(k * np.pi * xi)
            u[j, i] = 2 * nu * np.pi * s / (1 + s + 1e-15)

    return x, t, u


def train_burgers(use_causal=True, use_relobralo=True, use_rar=True, n_epochs=15000):
    """Train Burgers PINN with CA modules."""
    model = BurgersPINN(use_fourier=True).to(device)
    n_coll = 10000
    nu = 0.01 / np.pi

    # Data: initial + boundary
    n_data = 100
    x_ic = (torch.rand(n_data, 1) * 2 - 1).to(device)
    t_ic = torch.zeros(n_data, 1).to(device)
    u_ic = -torch.sin(np.pi * x_ic).to(device)

    t_bc = torch.rand(n_data, 1).to(device)
    x_left = -torch.ones(n_data, 1).to(device)
    x_right = torch.ones(n_data, 1).to(device)

    causal = CausalTrainer(n_chunks=50, epsilon=0.1, t_min=0.0, t_max=1.0) if use_causal else None
    balancer = ReLoBRaLo(n_losses=3) if use_relobralo else None
    sampler = BurgersAdaptiveSampler(n_collocation=n_coll, nu=nu) if use_rar else None

    if use_rar:
        coll = sampler.generate_initial_points()
        x_coll, t_coll = coll['x'].to(device), coll['t'].to(device)
    else:
        x_coll = (torch.rand(n_coll, 1, device=device) * 2 - 1)
        t_coll = torch.rand(n_coll, 1, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_loss = float('inf')
    best_state = None

    for epoch in range(n_epochs):
        if use_rar and sampler.should_resample(epoch):
            coll = sampler.resample(model, epoch, coll)
            x_coll, t_coll = coll['x'], coll['t']

        x_coll.requires_grad_(True)
        t_coll.requires_grad_(True)

        u = model(x_coll, t_coll)
        u_t = grad(u, t_coll, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = grad(u, x_coll, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = grad(u_x, x_coll, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        residual = u_t + u * u_x - nu * u_xx

        if use_causal:
            loss_pde, _, _ = causal.weighted_pde_loss(residual.abs().squeeze(), t_coll)
        else:
            loss_pde = torch.mean(residual ** 2)

        # Data loss
        u_pred_ic = model(x_ic, t_ic)
        u_pred_left = model(x_left, t_bc)
        u_pred_right = model(x_right, t_bc)
        loss_data = (torch.mean((u_pred_ic - u_ic) ** 2) +
                     torch.mean(u_pred_left ** 2) + torch.mean(u_pred_right ** 2))

        # BC loss
        loss_bc = torch.mean(u_pred_left ** 2) + torch.mean(u_pred_right ** 2)

        components = [loss_pde, loss_data, loss_bc]
        if use_relobralo:
            weights, loss_total = balancer.update_weights(
                components, model.model.get_last_layer_weights()
            )
        else:
            loss_total = sum(components)

        if torch.isnan(loss_total):
            break

        optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if loss_total.item() < best_loss:
            best_loss = loss_total.item()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 1000 == 0:
            print(f"Epoch {epoch:5d} | Loss: {loss_total.item():.2e}")

    model.load_state_dict(best_state)

    # Evaluate
    x_ref, t_ref, u_ref = burgers_reference()
    x_tensor = torch.tensor(x_ref, dtype=torch.float32, device=device)
    t_tensor = torch.tensor(t_ref, dtype=torch.float32, device=device)
    X_m, T_m = np.meshgrid(x_ref, t_ref)

    model.eval()
    with torch.no_grad():
        x_flat = torch.tensor(X_m.flatten(), dtype=torch.float32, device=device).view(-1, 1)
        t_flat = torch.tensor(T_m.flatten(), dtype=torch.float32, device=device).view(-1, 1)
        u_pred = model(x_flat, t_flat).cpu().numpy().reshape(X_m.shape)

    err = np.linalg.norm(u_pred - u_ref) / (np.linalg.norm(u_ref) + 1e-8)
    print(f"\nBurgers relative L2 error: {err:.4e}")
    return err, model


if __name__ == "__main__":
    print("Training Burgers with CA-PINN...")
    err, model = train_burgers(use_causal=True, use_relobralo=True, use_rar=True)
    print(f"Final relative L2 error: {err:.4e}")
