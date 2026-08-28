"""Ablation study: test all 2^3=8 combinations of CA-PINN modules.

Tests every combination of {Causal, ReLoBRaLo, RAR} enabled/disabled
while keeping Fourier features and architecture fixed.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import time
import itertools
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


def train_single_config(use_causal, use_relobralo, use_rar, n_epochs=15000):
    """Train a single configuration and return results."""
    model = SWE_PINN(use_fourier=True, fourier_dim=128, fourier_sigma=1.0).to(device)

    n_coll = 30000
    x_obs, y_obs, t_obs, h_obs = generate_swe_data(n_obs=1000)
    x_obs, y_obs, t_obs, h_obs = [v.to(device) for v in [x_obs, y_obs, t_obs, h_obs]]

    causal = CausalTrainer(n_chunks=72, epsilon=0.1, t_max=3600.0) if use_causal else None
    balancer = ReLoBRaLo(n_losses=4) if use_relobralo else None
    sampler = SWEAdaptiveSampler(n_collocation=n_coll, resample_every=2000) if use_rar else None

    # Init collocation
    if use_rar:
        coll = sampler.generate_initial_points()
        x_coll, y_coll, t_coll = coll['x'].to(device), coll['y'].to(device), coll['t'].to(device)
    else:
        n_global = int(n_coll * 0.7); n_local = n_coll - n_global
        x_coll = torch.cat([torch.rand(n_global, 1) * 100,
                            40 + 20 * torch.rand(n_local, 1)], dim=0).to(device)
        y_coll = torch.cat([torch.rand(n_global, 1) * 100,
                            40 + 20 * torch.rand(n_local, 1)], dim=0).to(device)
        t_coll = torch.cat([torch.rand(n_global, 1) * 3600,
                            torch.rand(n_local, 1) * 3600], dim=0).to(device)
        coll = {'x': x_coll, 'y': y_coll, 't': t_coll}

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1000, factor=0.5)

    best_loss = float('inf')
    best_state = None

    for epoch in range(n_epochs):
        if use_rar and sampler.should_resample(epoch):
            coll = sampler.resample(model, epoch, coll)
            x_coll, y_coll, t_coll = coll['x'], coll['y'], coll['t']

        x_coll.requires_grad_(True)
        y_coll.requires_grad_(True)
        t_coll.requires_grad_(True)

        res_cont, res_xmom, res_ymom, _, _, _ = compute_swe_residuals(model, x_coll, y_coll, t_coll)

        if use_causal:
            pde_residual = res_cont.abs() + res_xmom.abs() + res_ymom.abs()
            loss_pde, _, _ = causal.weighted_pde_loss(pde_residual.squeeze(), t_coll)
        else:
            loss_pde = torch.mean(res_cont**2) + torch.mean(res_xmom**2) + torch.mean(res_ymom**2)

        _, _, _ = model(x_obs / 100.0, y_obs / 100.0, t_obs / 3600.0)
        h_pred, _, _ = model(x_obs / 100.0, y_obs / 100.0, t_obs / 3600.0)
        loss_data = torch.mean((h_pred - h_obs) ** 2)

        loss_bc_up, loss_bc_down, loss_gradient = compute_bc_loss(model)
        loss_bc = loss_bc_up + loss_bc_down + loss_gradient

        loss_prior = (10.0 * (model.n - 0.03)**2 + 5.0 * (model.C_drain - 0.05)**2 +
                      5.0 * (model.qx0 - 1.0)**2)

        components = [loss_pde, loss_data, loss_bc, loss_prior]
        if use_relobralo:
            _, loss_total = balancer.update_weights(components, model.get_last_layer_weights())
        else:
            loss_total = sum(components)

        if torch.isnan(loss_total):
            break

        optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(loss_total.detach() if use_relobralo else loss_total)

        if loss_total.item() < best_loss:
            best_loss = loss_total.item()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    # Evaluate
    x, y, t, h_true, qx_true, qy_true = generate_truth_grid(Nx=50, Ny=50, Nt=5)
    Nt, Nx, Ny = h_true.shape
    h_pred = np.zeros_like(h_true)

    model.eval()
    with torch.no_grad():
        X_mesh, Y_mesh = np.meshgrid(x, y, indexing='ij')
        for it, tt in enumerate(t):
            xf = torch.tensor(X_mesh.flatten(), dtype=torch.float32, device=device).view(-1, 1)
            yf = torch.tensor(Y_mesh.flatten(), dtype=torch.float32, device=device).view(-1, 1)
            tf = torch.ones_like(xf) * tt
            hp, _, _ = model(xf / 100.0, yf / 100.0, tf / 3600.0)
            h_pred[it] = hp.cpu().numpy().reshape(Nx, Ny)

    err_h = np.linalg.norm(h_pred - h_true) / (np.linalg.norm(h_true) + 1e-8)
    err_n = abs(model.n.item() - 0.03) / 0.03
    err_C = abs(model.C_drain.item() - 0.05) / 0.05
    err_qx0 = abs(model.qx0.item() - 1.0) / 1.0

    return {
        'causal': use_causal, 'relobralo': use_relobralo, 'rar': use_rar,
        'err_h': err_h, 'err_n': err_n, 'err_C': err_C, 'err_qx0': err_qx0,
        'best_loss': best_loss,
        'n_final': model.n.item(), 'C_final': model.C_drain.item(),
        'qx0_final': model.qx0.item(),
    }


def run_full_ablation():
    """Run all 8 configurations of the ablation study."""
    configs = list(itertools.product([False, True], repeat=3))
    config_names = ['causal', 'relobralo', 'rar']

    results = []
    print("=" * 70)
    print("ABLATION STUDY: All 8 combinations of {Causal, ReLoBRaLo, RAR}")
    print("=" * 70)

    for i, (use_causal, use_relobralo, use_rar) in enumerate(configs):
        name = f"{'C' if use_causal else '-'}{'R' if use_relobralo else '-'}{'A' if use_rar else '-'}"
        print(f"\n[{i+1}/8] Config: {name} "
              f"(causal={use_causal}, relobralo={use_relobralo}, rar={use_rar})")

        t0 = time.time()
        result = train_single_config(use_causal, use_relobralo, use_rar)
        result['time'] = time.time() - t0
        result['name'] = name
        results.append(result)

        print(f"  err_h={result['err_h']:.4e}, err_n={result['err_n']:.4e}, "
              f"err_C={result['err_C']:.4e}, time={result['time']:.0f}s")

    # Save results
    out_dir = 'outputs/ablation'
    os.makedirs(out_dir, exist_ok=True)

    import json
    with open(os.path.join(out_dir, 'ablation_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=float)

    # Print summary table
    print("\n" + "=" * 70)
    print("ABLATION SUMMARY")
    print(f"{'Config':>8s}  {'err_h':>10s}  {'err_n':>10s}  {'err_C':>10s}  {'err_qx0':>10s}  {'Time':>8s}")
    print("-" * 70)
    for r in sorted(results, key=lambda x: x['err_h']):
        print(f"{r['name']:>8s}  {r['err_h']:10.4e}  {r['err_n']:10.4e}  "
              f"{r['err_C']:10.4e}  {r['err_qx0']:10.4e}  {r['time']:7.0f}s")

    print(f"\nResults saved to {out_dir}/ablation_results.json")
    return results


if __name__ == "__main__":
    run_full_ablation()
