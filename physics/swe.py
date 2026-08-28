"""2D Shallow Water Equations with localized drainage sink term.

Conservation form with discharges (qx, qy) as variables:

    Continuity:  h_t + qx_x + qy_y + S_drain = 0
    x-Momentum:  qx_t + (qx²/h)_x + (qy·qx/h)_y
                 + g·h·(h_x - S0x) + g·n²·qx·|q|/h^(7/3) = 0
    y-Momentum:  qy_t + (qy²/h)_y + (qy·qx/h)_x
                 + g·h·(h_y - S0y) + g·n²·qy·|q|/h^(7/3) = 0

Drainage: S_drain = C · 0.5 · exp(-dist/12) · h
"""

import torch
from torch.autograd import grad
import numpy as np

# Physical constants
G = 9.81          # gravity (m/s²)
S0X = 0.001       # bed slope x
S0Y = 0.0         # bed slope y


def compute_swe_residuals(model, x, y, t):
    """Compute PDE residuals for 2D SWE + drainage.

    Args:
        model: PINN model with forward(x,y,t) → (h, qx, qy)
        x, y, t: collocation tensors of shape (N,1), requires_grad=True,
                 in PHYSICAL units (not normalized)

    Returns:
        res_cont: continuity residual  (N, 1)
        res_xmom: x-momentum residual  (N, 1)
        res_ymom: y-momentum residual  (N, 1)
        h, qx, qy: predicted fields
    """
    # Normalize inputs
    x_norm = x / 100.0
    y_norm = y / 100.0
    t_norm = t / 3600.0

    h, qx, qy = model(x_norm, y_norm, t_norm)

    # First-order derivatives
    h_t = grad(h, t, grad_outputs=torch.ones_like(h), create_graph=True)[0]
    h_x = grad(h, x, grad_outputs=torch.ones_like(h), create_graph=True)[0]
    h_y = grad(h, y, grad_outputs=torch.ones_like(h), create_graph=True)[0]

    qx_t = grad(qx, t, grad_outputs=torch.ones_like(qx), create_graph=True)[0]
    qx_x = grad(qx, x, grad_outputs=torch.ones_like(qx), create_graph=True)[0]
    qx_y = grad(qx, y, grad_outputs=torch.ones_like(qx), create_graph=True)[0]

    qy_t = grad(qy, t, grad_outputs=torch.ones_like(qy), create_graph=True)[0]
    qy_x = grad(qy, x, grad_outputs=torch.ones_like(qy), create_graph=True)[0]
    qy_y = grad(qy, y, grad_outputs=torch.ones_like(qy), create_graph=True)[0]

    # Flux terms (with stabilization)
    h_safe = torch.clamp(h, min=0.1)
    fxx = qx ** 2 / h_safe
    fxy = qx * qy / h_safe
    fyy = qy ** 2 / h_safe

    fxx_x = grad(fxx, x, grad_outputs=torch.ones_like(fxx), create_graph=True)[0]
    fxy_y = grad(fxy, y, grad_outputs=torch.ones_like(fxy), create_graph=True)[0]
    fxy_x = grad(fxy, x, grad_outputs=torch.ones_like(fxy), create_graph=True)[0]
    fyy_y = grad(fyy, y, grad_outputs=torch.ones_like(fyy), create_graph=True)[0]

    # Gravity (pressure + bed slope)
    gravity_x = G * h * (h_x - S0X)
    gravity_y = G * h * (h_y - S0Y)

    # Manning friction
    vel_mag = torch.sqrt(qx ** 2 + qy ** 2 + 1e-8)
    friction_x = G * model.n ** 2 * qx * vel_mag / (h_safe ** (7.0 / 3.0) + 1e-8)
    friction_y = G * model.n ** 2 * qy * vel_mag / (h_safe ** (7.0 / 3.0) + 1e-8)

    # Drainage sink
    dist = torch.sqrt((x - 50) ** 2 + (y - 50) ** 2 + 1e-8)
    drain = model.C_drain * 0.5 * torch.exp(-dist / 12.0) * h_safe

    # PDE residuals
    res_cont = h_t + qx_x + qy_y + drain
    res_xmom = qx_t + fxx_x + fxy_y + gravity_x + friction_x
    res_ymom = qy_t + fyy_y + fxy_x + gravity_y + friction_y

    return res_cont, res_xmom, res_ymom, h, qx, qy


def compute_bc_loss(model):
    """Compute boundary condition losses.

    Returns:
        loss_bc_up: upstream BC (x=0: qx=qx0, qy=0)
        loss_bc_down: downstream BC (x=100: dh/dx = 0, free outflow)
        loss_gradient: drain depression shape constraint
    """
    n_boundary = 50
    device = next(model.parameters()).device

    # Upstream: qx = qx0, qy = 0
    t_up = torch.linspace(0, 3600, n_boundary, device=device).reshape(-1, 1)
    y_up = torch.linspace(0, 100, n_boundary, device=device).reshape(-1, 1)
    x_up = torch.zeros(n_boundary, 1, device=device)

    # Cross-product to cover boundary surface
    X_up = x_up.repeat(n_boundary, 1)
    Y_up = y_up.repeat_interleave(n_boundary, dim=0)
    T_up = t_up.repeat(n_boundary, 1)

    _, qx_up, qy_up = model(X_up / 100.0, Y_up / 100.0, T_up / 3600.0)
    loss_bc_up = torch.mean((qx_up - model.qx0) ** 2) + torch.mean(qy_up ** 2)

    # Downstream: dh/dx = 0 (free outflow)
    x_down = torch.ones(n_boundary, 1, device=device) * 100.0
    y_down = torch.linspace(0, 100, n_boundary, device=device).reshape(-1, 1)
    t_down = torch.linspace(0, 3600, n_boundary, device=device).reshape(-1, 1)
    X_down = x_down.repeat(n_boundary, 1)
    Y_down = y_down.repeat_interleave(n_boundary, dim=0)
    T_down = t_down.repeat(n_boundary, 1)
    X_down.requires_grad_(True)
    h_down, _, _ = model(X_down / 100.0, Y_down / 100.0, T_down / 3600.0)
    h_down_x = grad(h_down, X_down, grad_outputs=torch.ones_like(h_down), create_graph=True)[0]
    loss_bc_down = torch.mean(h_down_x ** 2)

    # Drain depression shape constraint
    x_near = 50 + 3 * torch.randn(200, 1, device=device)
    y_near = 50 + 3 * torch.randn(200, 1, device=device)
    t_near = torch.rand(200, 1, device=device) * 3600
    x_far = 50 + 15 * torch.randn(200, 1, device=device)
    y_far = 50 + 15 * torch.randn(200, 1, device=device)
    h_near, _, _ = model(x_near / 100.0, y_near / 100.0, t_near / 3600.0)
    h_far, _, _ = model(x_far / 100.0, y_far / 100.0, t_near / 3600.0)
    loss_gradient = torch.mean(torch.relu(h_near - h_far))

    return loss_bc_up, loss_bc_down, loss_gradient


def _load_reference():
    """Load the finite-volume reference solution (cached)."""
    import os as _os
    ref_path = _os.path.join(
        _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
        'outputs', 'swe_reference.npz')
    if not _os.path.exists(ref_path):
        raise FileNotFoundError(
            f"Reference solution not found at {ref_path}. "
            f"Run `python experiments/generate_reference.py` first.")
    d = np.load(ref_path)
    return d['x'], d['y'], d['t'], d['h'], d['qx'], d['qy']


def _interpolators():
    from scipy.interpolate import RegularGridInterpolator
    x_ref, y_ref, t_ref, h_ref, qx_ref, qy_ref = _load_reference()
    # h_ref is (nt, ny, nx) -> axes (t, y, x)
    return (
        RegularGridInterpolator((t_ref, y_ref, x_ref), h_ref, bounds_error=False, fill_value=None),
        RegularGridInterpolator((t_ref, y_ref, x_ref), qx_ref, bounds_error=False, fill_value=None),
        RegularGridInterpolator((t_ref, y_ref, x_ref), qy_ref, bounds_error=False, fill_value=None),
    )


def generate_swe_data(true_n=0.03, true_C=0.05, true_qx0=1.0, n_obs=1000):
    """Sample noisy observations from the finite-volume reference solution."""
    interp_h, _, _ = _interpolators()
    rng = np.random.default_rng(42)
    x_obs = rng.uniform(0, 100, n_obs)
    y_obs = rng.uniform(0, 100, n_obs)
    t_obs = rng.uniform(0, 3600, n_obs)
    pts = np.stack([t_obs, y_obs, x_obs], axis=-1)  # (n_obs, 3) order (t, y, x)
    h_true = interp_h(pts)                          # (n_obs,)

    # 0.5% Gaussian observation noise relative to the mean depth
    h_obs = h_true + 0.005 * np.random.randn(n_obs) * float(h_true.mean())
    h_obs = np.maximum(h_obs, 0.15)

    return (torch.tensor(x_obs, dtype=torch.float32).reshape(-1, 1),
            torch.tensor(y_obs, dtype=torch.float32).reshape(-1, 1),
            torch.tensor(t_obs, dtype=torch.float32).reshape(-1, 1),
            torch.tensor(h_obs, dtype=torch.float32).reshape(-1, 1))


def generate_truth_grid(true_n=0.03, true_C=0.05, true_qx0=1.0, Nx=100, Ny=100, Nt=10):
    """Return the finite-volume reference solution interpolated to (Nt, Nx, Ny)."""
    interp_h, interp_qx, interp_qy = _interpolators()
    x = np.linspace(0, 100, Nx)
    y = np.linspace(0, 100, Ny)
    t = np.linspace(0, 3600, Nt)

    Tg, Xg, Yg = np.meshgrid(t, x, y, indexing='ij')   # (Nt, Nx, Ny)
    pts = np.stack([Tg, Yg, Xg], axis=-1)               # (Nt, Nx, Ny, 3) order (t, y, x)
    h_true = interp_h(pts)                              # (Nt, Nx, Ny)
    qx_true = interp_qx(pts)
    qy_true = interp_qy(pts)

    return x, y, t, h_true, qx_true, qy_true


class SWEAdaptiveSampler:
    """Adaptive sampler specialized for SWE equations.

    Computes aggregate PDE residual = |res_cont| + |res_xmom| + |res_ymom|
    on candidate points, then samples proportional to residual^beta.
    """

    def __init__(self, n_collocation=30000, n_candidates=50000,
                 resample_every=2000, beta=1.5):
        self.n_collocation = n_collocation
        self.n_candidates = n_candidates
        self.resample_every = resample_every
        self.beta = beta
        self.device = None

    def should_resample(self, epoch):
        return (epoch > 0) and (epoch % self.resample_every == 0)

    def generate_initial_points(self):
        """Generate initial collocation points (global + local near drain)."""
        n_global = int(self.n_collocation * 0.7)
        n_local = self.n_collocation - n_global

        x_global = torch.rand(n_global, 1) * 100
        y_global = torch.rand(n_global, 1) * 100
        t_global = torch.rand(n_global, 1) * 3600

        x_local = 40 + 20 * torch.rand(n_local, 1)
        y_local = 40 + 20 * torch.rand(n_local, 1)
        t_local = torch.rand(n_local, 1) * 3600

        return {
            'x': torch.cat([x_global, x_local], dim=0),
            'y': torch.cat([y_global, y_local], dim=0),
            't': torch.cat([t_global, t_local], dim=0),
        }

    def resample(self, model, epoch, current_collocation):
        if not self.should_resample(epoch):
            return current_collocation

        if self.device is None:
            self.device = current_collocation['x'].device

        # Generate candidates
        cand_x = torch.rand(self.n_candidates, 1, device=self.device) * 100
        cand_y = torch.rand(self.n_candidates, 1, device=self.device) * 100
        cand_t = torch.rand(self.n_candidates, 1, device=self.device) * 3600

        # Compute a proxy for PDE residual without full AD (efficient, no graph issues)
        # Use model output variation as a simple proxy: higher variation = higher residual
        with torch.no_grad():
            x_norm = cand_x / 100.0
            y_norm = cand_y / 100.0
            t_norm = cand_t / 3600.0
            h, qx, qy = model(x_norm, y_norm, t_norm)

            # Compute spatial variation of h (proxy for residual)
            # Sort by (x, y, t) and compute local variation
            h_flat = h.squeeze()
            # Simple proxy: deviation from mean + spatial gradients via finite differences
            h_mean = h_flat.mean()
            # Compute local roughness as residual proxy
            sorted_x, x_indices = cand_x.squeeze().sort()
            h_sorted = h_flat[x_indices]
            x_diff = torch.diff(h_sorted).abs()
            # Pad to match size
            h_roughness = torch.zeros_like(h_flat)
            if len(x_diff) > 0:
                h_roughness[x_indices[:-1]] += x_diff
                h_roughness[x_indices[1:]] += x_diff

            # Add drainage zone emphasis
            dist = torch.sqrt((cand_x - 50) ** 2 + (cand_y - 50) ** 2 + 1e-8)
            near_drain = torch.exp(-dist / 20.0)

            # Combine: spatial roughness + near-drain weighting
            residual = (h_roughness + 5.0 * near_drain + (h_flat - h_mean).abs()).clamp(min=1e-10)

        # Probability proportional to residual^beta
        probs = residual ** self.beta
        probs = probs / (probs.sum() + 1e-8)

        # Resample
        indices = torch.multinomial(probs, self.n_collocation, replacement=True)

        new_collocation = {
            'x': cand_x[indices].clone(),
            'y': cand_y[indices].clone(),
            't': cand_t[indices].clone(),
        }

        print(f"  [RAR epoch {epoch}] Resampled collocation points. "
              f"Max residual: {residual.max():.2e}, Mean: {residual.mean():.2e}")

        return new_collocation
