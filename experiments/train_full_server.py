"""Server training script: full 30000-epoch CA-PINN + Vanilla comparison.

Run with: source ~/env.sh && nohup python train_full.py > train_full.log 2>&1 &
"""

import sys, os, time, json
os.environ['MPLBACKEND'] = 'Agg'
# Insert parent directory (ca_pinn root) so imports like `physics.swe_model` work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(10)  # 12-core server, leave 2 for system

device = 'cpu'

from physics.swe_model import SWE_PINN
from training.causal import CausalTrainer
from training.loss_balancing import ReLoBRaLo
from physics.swe import (
    compute_swe_residuals, compute_bc_loss,
    generate_swe_data, generate_truth_grid
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)


def train_model(name, use_causal, use_relobralo, n_epochs, n_coll=8000,
                epsilon=0.01, warmup=0):
    """Train a single model configuration."""
    print(f"\n{'='*60}")
    print(f"Training {name}: {n_epochs} epochs, causal={use_causal}, "
          f"relobralo={use_relobralo}, coll={n_coll}")
    print(f"{'='*60}", flush=True)

    model = SWE_PINN(use_fourier=True).to(device)
    x_obs, y_obs, t_obs, h_obs = generate_swe_data(n_obs=800)
    x_obs, y_obs, t_obs, h_obs = [v.to(device) for v in [x_obs, y_obs, t_obs, h_obs]]

    causal = CausalTrainer(n_chunks=72, epsilon=epsilon, t_max=3600.0) if use_causal else None
    balancer = ReLoBRaLo(n_losses=4) if use_relobralo else None

    # Collocation points
    ng = int(n_coll * 0.7)
    nl = n_coll - ng
    x_coll = torch.cat([torch.rand(ng, 1, device=device) * 100,
                        40 + 20 * torch.rand(nl, 1, device=device)], dim=0)
    y_coll = torch.cat([torch.rand(ng, 1, device=device) * 100,
                        40 + 20 * torch.rand(nl, 1, device=device)], dim=0)
    t_coll = torch.cat([torch.rand(ng, 1, device=device) * 3600,
                        torch.rand(nl, 1, device=device) * 3600], dim=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=2000, factor=0.5)

    history = {
        'loss_total': [], 'loss_pde': [], 'loss_data': [], 'loss_bc': [],
        'n': [], 'C': [], 'qx0': [], 'active_fraction': [],
    }
    best_loss = float('inf')
    best_state = None
    t0 = time.time()

    for epoch in range(n_epochs):
        x_coll.requires_grad_(True)
        y_coll.requires_grad_(True)
        t_coll.requires_grad_(True)

        res_cont, res_xmom, res_ymom, _, _, _ = compute_swe_residuals(
            model, x_coll, y_coll, t_coll)

        # PDE loss
        if use_causal and epoch >= warmup:
            pde_res = (res_cont.abs() + res_xmom.abs() + res_ymom.abs()).squeeze()
            loss_pde, _, _ = causal.weighted_pde_loss(pde_res, t_coll)
            history['active_fraction'].append(causal.get_active_fraction())
        else:
            loss_pde = (torch.mean(res_cont**2) + torch.mean(res_xmom**2) +
                        torch.mean(res_ymom**2))
            history['active_fraction'].append(1.0)

        # Data loss
        h_pred, _, _ = model(x_obs / 100.0, y_obs / 100.0, t_obs / 3600.0)
        loss_data = torch.mean((h_pred - h_obs) ** 2)

        # BC loss
        lbu, lbd, lbg = compute_bc_loss(model)
        loss_bc = lbu + lbd + lbg

        # Prior loss
        loss_prior = (10.0 * (model.n - 0.03) ** 2 +
                      5.0 * (model.C_drain - 0.05) ** 2 +
                      5.0 * (model.qx0 - 1.0) ** 2)

        # Combine
        components = [loss_pde, loss_data, loss_bc, loss_prior]
        if use_relobralo:
            _, loss_total = balancer.update_weights(
                components, model.get_last_layer_weights())
        else:
            loss_total = sum(components)

        if torch.isnan(loss_total):
            print(f"NaN at epoch {epoch}", flush=True)
            break

        optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(loss_total.detach())

        # Record
        history['loss_total'].append(loss_total.item())
        history['loss_pde'].append(loss_pde.item() if isinstance(loss_pde, torch.Tensor) and loss_pde.numel() == 1 else (loss_pde.detach().item() if isinstance(loss_pde, torch.Tensor) else float(loss_pde)))
        history['loss_data'].append(loss_data.item())
        history['loss_bc'].append(loss_bc.item())
        history['n'].append(model.n.item())
        history['C'].append(model.C_drain.item())
        history['qx0'].append(model.qx0.item())

        if history['loss_total'][-1] < best_loss:
            best_loss = history['loss_total'][-1]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 1000 == 0:
            elapsed = time.time() - t0
            af = history['active_fraction'][-1]
            print(f"E{epoch:5d}/{n_epochs} | Loss={history['loss_total'][-1]:.3e} "
                  f"AF={af:.2f} n={model.n.item():.4f} C={model.C_drain.item():.4f} "
                  f"qx0={model.qx0.item():.4f} t={elapsed/60:.0f}min", flush=True)

    # Restore best
    if best_state:
        model.load_state_dict(best_state)

    elapsed = time.time() - t0

    # Evaluate
    xg, yg, tg, h_true, qx_true, qy_true = generate_truth_grid(Nx=60, Ny=60, Nt=10)
    h_pred = np.zeros_like(h_true)
    model.eval()
    with torch.no_grad():
        X_m, Y_m = np.meshgrid(xg, yg, indexing='ij')
        for it, tt in enumerate(tg):
            xf = torch.tensor(X_m.flatten(), dtype=torch.float32, device=device).view(-1, 1)
            yf = torch.tensor(Y_m.flatten(), dtype=torch.float32, device=device).view(-1, 1)
            tf = torch.ones_like(xf) * tt
            hp, _, _ = model(xf / 100.0, yf / 100.0, tf / 3600.0)
            h_pred[it] = hp.cpu().numpy().reshape(60, 60)

    err_h = np.linalg.norm(h_pred - h_true) / (np.linalg.norm(h_true) + 1e-8)
    err_n = abs(model.n.item() - 0.03) / 0.03
    err_C = abs(model.C_drain.item() - 0.05) / 0.05
    err_qx0 = abs(model.qx0.item() - 1.0)

    result = {
        'name': name,
        'err_h': float(err_h),
        'err_n': float(err_n),
        'err_C': float(err_C),
        'err_qx0': float(err_qx0),
        'n_final': float(model.n.item()),
        'C_final': float(model.C_drain.item()),
        'qx0_final': float(model.qx0.item()),
        'best_loss': float(best_loss),
        'time_s': float(elapsed),
    }

    # Save
    out_dir = os.path.join(OUT_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, 'model.pth'))
    np.savez(os.path.join(out_dir, 'results.npz'),
             err_h=err_h, err_n=err_n, err_C=err_C, err_qx0=err_qx0,
             n=model.n.item(), C=model.C_drain.item(), qx0=model.qx0.item(),
             h_true=h_true, h_pred=h_pred,
             loss_total=np.array(history['loss_total']),
             loss_pde=np.array(history['loss_pde']),
             loss_data=np.array(history['loss_data']),
             loss_bc=np.array(history['loss_bc']),
             n_hist=np.array(history['n']),
             C_hist=np.array(history['C']),
             qx0_hist=np.array(history['qx0']),
             active_fraction=np.array(history['active_fraction']))

    with open(os.path.join(out_dir, 'result.json'), 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n{name} DONE: err_h={err_h:.4e} err_n={err_n:.4e} "
          f"err_C={err_C:.4e} err_qx0={err_qx0:.4e} "
          f"time={elapsed/60:.0f}min", flush=True)
    return result


if __name__ == "__main__":
    results = []

    # 1. Vanilla baseline (30000 epochs for full convergence)
    r = train_model('vanilla', use_causal=False, use_relobralo=False,
                    n_epochs=30000, n_coll=8000)
    results.append(r)

    # 2. CA-PINN full (30000 epochs)
    r = train_model('ca_pinn', use_causal=True, use_relobralo=True,
                    n_epochs=30000, n_coll=8000, epsilon=0.01, warmup=1000)
    results.append(r)

    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    for r in results:
        print(f"{r['name']:>10s}: err_h={r['err_h']:.4e} err_n={r['err_n']:.4e} "
              f"err_C={r['err_C']:.4e} err_qx0={r['err_qx0']:.4e} "
              f"time={r['time_s']/3600:.1f}h")

    with open(os.path.join(OUT_DIR, 'summary.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to {OUT_DIR}/summary.json")
