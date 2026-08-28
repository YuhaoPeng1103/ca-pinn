"""Main experiment: CA-PINN for 2D Shallow Water Equations with drainage.

Compares CA-PINN (Causal + ReLoBRaLo + RAR + Fourier) against vanilla PINN
baseline on the SWE + drainage inverse problem.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import time
from physics.swe_model import SWE_PINN
from training.causal import CausalTrainer
from training.loss_balancing import ReLoBRaLo
from physics.swe import (
    compute_swe_residuals, compute_bc_loss,
    generate_swe_data, generate_truth_grid, SWEAdaptiveSampler
)

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def train_ca_pinn(config):
    """Train PINN with all three CA-PINN modules enabled.

    Args:
        config: dict with keys:
            use_causal, use_relobralo, use_rar, use_fourier (bool)
            n_epochs, n_collocation, n_obs
    """
    model = SWE_PINN(
        in_dim=3,
        hidden_layers=[128, 128, 128, 128],
        out_dim=3,
        use_fourier=config.get('use_fourier', True),
        fourier_dim=128,
        fourier_sigma=1.0,
    ).to(device)

    n_epochs = config.get('n_epochs', 30000)
    n_coll = config.get('n_collocation', 30000)
    n_obs = config.get('n_obs', 1000)

    # Generate data
    x_obs, y_obs, t_obs, h_obs = generate_swe_data(n_obs=n_obs)
    x_obs, y_obs, t_obs, h_obs = [v.to(device) for v in [x_obs, y_obs, t_obs, h_obs]]

    # Initialize modules
    use_causal = config.get('use_causal', True)
    use_relobralo = config.get('use_relobralo', True)
    use_rar = config.get('use_rar', True)

    causal = CausalTrainer(n_chunks=72, epsilon=0.1, t_max=3600.0) if use_causal else None
    balancer = ReLoBRaLo(n_losses=4, alpha=0.9, temperature=1.0) if use_relobralo else None
    sampler = SWEAdaptiveSampler(n_collocation=n_coll, resample_every=2000) if use_rar else None

    # Initial collocation points
    if use_rar:
        coll = sampler.generate_initial_points()
        x_coll, y_coll, t_coll = coll['x'], coll['y'], coll['t']
    else:
        n_global = int(n_coll * 0.7)
        n_local = n_coll - n_global
        x_global = torch.rand(n_global, 1) * 100
        y_global = torch.rand(n_global, 1) * 100
        t_global = torch.rand(n_global, 1) * 3600
        x_local = 40 + 20 * torch.rand(n_local, 1)
        y_local = 40 + 20 * torch.rand(n_local, 1)
        t_local = torch.rand(n_local, 1) * 3600
        x_coll = torch.cat([x_global, x_local], dim=0).to(device)
        y_coll = torch.cat([y_global, y_local], dim=0).to(device)
        t_coll = torch.cat([t_global, t_local], dim=0).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=1000, factor=0.5
    )

    history = {
        'loss_total': [], 'loss_pde': [], 'loss_data': [],
        'loss_bc': [], 'n': [], 'C': [], 'qx0': [],
        'active_fraction': [], 'loss_weights': [],
    }
    best_loss = float('inf')
    best_state = None
    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"CA-PINN Training: causal={use_causal}, relobralo={use_relobralo}, "
          f"rar={use_rar}, fourier={config.get('use_fourier', True)}")
    print(f"{'='*60}")

    for epoch in range(n_epochs):
        # Adaptive resampling
        if use_rar and sampler.should_resample(epoch):
            coll = sampler.resample(model, epoch, coll)
            x_coll, y_coll, t_coll = coll['x'], coll['y'], coll['t']

        x_coll.requires_grad_(True)
        y_coll.requires_grad_(True)
        t_coll.requires_grad_(True)

        # Compute PDE residuals
        res_cont, res_xmom, res_ymom, _, _, _ = compute_swe_residuals(
            model, x_coll, y_coll, t_coll
        )

        # PDE loss (causally weighted or uniform)
        if use_causal:
            pde_residual = res_cont.abs() + res_xmom.abs() + res_ymom.abs()
            loss_pde_total, chunk_losses, _ = causal.weighted_pde_loss(
                pde_residual.squeeze(), t_coll
            )
            history['active_fraction'].append(causal.get_active_fraction())
        else:
            loss_pde_total = (
                torch.mean(res_cont**2) + torch.mean(res_xmom**2) + torch.mean(res_ymom**2)
            )

        # Data loss
        x_obs_norm = x_obs / 100.0
        y_obs_norm = y_obs / 100.0
        t_obs_norm = t_obs / 3600.0
        h_pred, _, _ = model(x_obs_norm, y_obs_norm, t_obs_norm)
        loss_data = torch.mean((h_pred - h_obs) ** 2)

        # BC loss
        loss_bc_up, loss_bc_down, loss_gradient = compute_bc_loss(model)
        loss_bc = loss_bc_up + loss_bc_down + loss_gradient

        # Prior loss
        loss_prior = (
            10.0 * (model.n - 0.03) ** 2 +
            5.0 * (model.C_drain - 0.05) ** 2 +
            5.0 * (model.qx0 - 1.0) ** 2
        )

        # Combine losses
        loss_components = [loss_pde_total, loss_data, loss_bc, loss_prior]

        if use_relobralo:
            weights, loss_total = balancer.update_weights(
                loss_components, model.get_last_layer_weights()
            )
            history['loss_weights'].append(weights.cpu().numpy().copy())
        else:
            loss_total = sum(loss_components)

        if torch.isnan(loss_total):
            print(f"Warning: NaN at epoch {epoch}, stopping")
            break

        optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(loss_total.detach() if use_relobralo else loss_total)

        # Record
        history['loss_total'].append(loss_total.item())
        history['loss_pde'].append(loss_pde_total.item() if not use_relobralo else loss_pde_total.detach().item())
        history['loss_data'].append(loss_data.item())
        history['loss_bc'].append(loss_bc.item())
        history['n'].append(model.n.item())
        history['C'].append(model.C_drain.item())
        history['qx0'].append(model.qx0.item())

        if history['loss_total'][-1] < best_loss:
            best_loss = history['loss_total'][-1]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 1000 == 0:
            af = history['active_fraction'][-1] if history['active_fraction'] else 1.0
            w_str = ""
            if use_relobralo and history['loss_weights']:
                w = history['loss_weights'][-1]
                w_str = f" | w=[{w[0]:.2f},{w[1]:.2f},{w[2]:.2f},{w[3]:.2f}]"
            print(f"Epoch {epoch:5d} | Loss: {history['loss_total'][-1]:.2e} "
                  f"| AF: {af:.2f}{w_str} | "
                  f"n={history['n'][-1]:.4f} C={history['C'][-1]:.4f} qx0={history['qx0'][-1]:.4f}")

    model.load_state_dict(best_state)
    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f}s. Best loss: {best_loss:.2e}")
    print(f"Final: n={model.n.item():.4f} (true=0.03), "
          f"C={model.C_drain.item():.4f} (true=0.5), "
          f"qx0={model.qx0.item():.4f} (true=1.0)")

    return model, history


def evaluate_model(model):
    """Evaluate model on truth grid and compute relative L2 errors."""
    x, y, t, h_true, qx_true, qy_true = generate_truth_grid(Nx=100, Ny=100, Nt=10)

    Nt, Nx, Ny = h_true.shape
    h_pred = np.zeros_like(h_true)
    qx_pred = np.zeros_like(qx_true)
    qy_pred = np.zeros_like(qy_true)

    model.eval()
    with torch.no_grad():
        X_mesh, Y_mesh = np.meshgrid(x, y, indexing='ij')
        for it, tt in enumerate(t):
            x_flat = torch.tensor(X_mesh.flatten(), dtype=torch.float32, device=device).view(-1, 1)
            y_flat = torch.tensor(Y_mesh.flatten(), dtype=torch.float32, device=device).view(-1, 1)
            t_flat = torch.ones_like(x_flat) * tt
            h_p, qx_p, qy_p = model(x_flat / 100.0, y_flat / 100.0, t_flat / 3600.0)
            h_pred[it] = h_p.cpu().numpy().reshape(Nx, Ny)
            qx_pred[it] = qx_p.cpu().numpy().reshape(Nx, Ny)
            qy_pred[it] = qy_p.cpu().numpy().reshape(Nx, Ny)

    # Relative L2 errors
    eps = 1e-8
    err_h = np.linalg.norm(h_pred - h_true) / (np.linalg.norm(h_true) + eps)
    err_qx = np.linalg.norm(qx_pred - qx_true) / (np.linalg.norm(qx_true) + eps)
    err_qy = np.linalg.norm(qy_pred - qy_true) / (np.linalg.norm(qy_true) + eps)

    print(f"\nRelative L2 Errors:")
    print(f"  h:  {err_h:.4e}")
    print(f"  qx: {err_qx:.4e}")
    print(f"  qy: {err_qy:.4e}")

    return {'err_h': err_h, 'err_qx': err_qx, 'err_qy': err_qy}, {
        'x': x, 'y': y, 't': t,
        'h_true': h_true, 'qy_true': qy_true, 'qy_true': qy_true,
        'h_pred': h_pred, 'qy_pred': qy_pred, 'qy_pred': qy_pred,
    }


if __name__ == "__main__":
    # Full CA-PINN
    config = {
        'use_causal': True,
        'use_relobralo': True,
        'use_rar': True,
        'use_fourier': True,
        'n_epochs': 30000,
        'n_collocation': 30000,
    }

    model, history = train_ca_pinn(config)
    errors, fields = evaluate_model(model)

    # Save results
    out_dir = 'outputs/ca_pinn_swe'
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, 'model.pth'))
    np.savez(os.path.join(out_dir, 'results.npz'),
             err_h=errors['err_h'], err_qx=errors['err_qx'], err_qy=errors['err_qy'],
             **{k: np.array(v) for k, v in history.items() if k != 'loss_weights'},
             **fields)
    print(f"\nResults saved to {out_dir}/")
