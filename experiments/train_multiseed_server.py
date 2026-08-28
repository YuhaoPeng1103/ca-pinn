"""Multi-seed statistical study for the CMAME submission.

Runs the SWE inverse-problem experiment (the paper's main result) over
multiple random seeds and reports mean +/- std for the headline metrics,
which numerical journals require to establish statistical significance.

Configs (30,000 epochs each, n_obs=800, n_coll=8000):
  vanilla, causal, relobralo (fixed prior), both (causal + relobralo)

Seeds: 0, 1, 2 (three additional seeds; combined with the earlier seed-42
run this yields four independent repetitions per configuration).

Only the final metrics are saved (per-seed), to keep disk usage bounded.
"""

import sys, os, time, json
os.environ['MPLBACKEND'] = 'Agg'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

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


def _prior_loss(model):
    return (10.0 * (model.n - 0.03) ** 2 +
            5.0 * (model.C_drain - 0.05) ** 2 +
            5.0 * (model.qx0 - 1.0) ** 2)


def train_one(name, use_causal, use_relobralo, n_epochs, seed,
              n_obs=800, n_coll=8000, epsilon_init=0.1, epsilon_max=50.0,
              warmup=1000):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(10)

    print(f"\n[{'='*56}]")
    print(f"[{name}] seed={seed} causal={use_causal} relobralo={use_relobralo} "
          f"n_obs={n_obs}", flush=True)

    model = SWE_PINN(use_fourier=True).to(device)
    x_obs, y_obs, t_obs, h_obs = generate_swe_data(n_obs=n_obs)
    x_obs, y_obs, t_obs, h_obs = [v.to(device) for v in [x_obs, y_obs, t_obs, h_obs]]

    causal = CausalTrainer(n_chunks=72, epsilon=epsilon_init, t_max=3600.0,
                           epsilon_max=epsilon_max) if use_causal else None
    balancer = ReLoBRaLo(n_losses=3) if use_relobralo else None  # fixed prior

    ng = int(n_coll * 0.7)
    nl = n_coll - ng
    x_coll = torch.cat([torch.rand(ng, 1, device=device) * 100,
                        40 + 20 * torch.rand(nl, 1, device=device)], dim=0)
    y_coll = torch.cat([torch.rand(ng, 1, device=device) * 100,
                        40 + 20 * torch.rand(nl, 1, device=device)], dim=0)
    t_coll = torch.cat([torch.rand(ng, 1, device=device) * 3600,
                        torch.rand(nl, 1, device=device) * 3600], dim=0)

    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2000, factor=0.5)

    t0 = time.time()
    for epoch in range(n_epochs):
        if causal is not None:
            causal.set_epoch(epoch, n_epochs)

        x_coll.requires_grad_(True)
        y_coll.requires_grad_(True)
        t_coll.requires_grad_(True)

        res_cont, res_xmom, res_ymom, _, _, _ = compute_swe_residuals(model, x_coll, y_coll, t_coll)

        if use_causal and epoch >= warmup:
            pde_res = (res_cont.abs() + res_xmom.abs() + res_ymom.abs()).squeeze()
            loss_pde, _, _ = causal.weighted_pde_loss(pde_res, t_coll)
        else:
            loss_pde = (torch.mean(res_cont**2) + torch.mean(res_xmom**2) +
                        torch.mean(res_ymom**2))

        h_pred, _, _ = model(x_obs / 100.0, y_obs / 100.0, t_obs / 3600.0)
        loss_data = torch.mean((h_pred - h_obs) ** 2)

        lbu, lbd, lbg = compute_bc_loss(model)
        loss_bc = lbu + lbd + lbg

        loss_prior = _prior_loss(model)

        components = [loss_pde, loss_data, loss_bc]
        if use_relobralo:
            _, loss_balanced = balancer.update_weights(components, model.get_last_layer_weights())
        else:
            loss_balanced = sum(components)
        loss_total = loss_balanced + loss_prior  # fixed-weight prior

        if torch.isnan(loss_total):
            print(f"NaN at epoch {epoch}", flush=True)
            break

        opt.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        scheduler.step(loss_total.detach())

        if epoch % 2000 == 0:
            print(f"E{epoch:5d} | Loss={loss_total.item():.3e} "
                  f"n={model.n.item():.4f} t={time.time()-t0:.0f}s", flush=True)

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
        'name': name, 'seed': seed,
        'err_h': float(err_h), 'err_n': float(err_n),
        'err_C': float(err_C), 'err_qx0': float(err_qx0),
        'n_final': float(model.n.item()),
        'C_final': float(model.C_drain.item()),
        'qx0_final': float(model.qx0.item()),
        'time_s': float(elapsed),
    }
    print(f"[{name}] DONE seed={seed}: err_h={err_h:.4e} err_n={err_n:.4e} "
          f"time={elapsed/3600:.1f}h", flush=True)
    return result


if __name__ == "__main__":
    configs = [
        ('vanilla', False, False),
        ('causal', True, False),
        ('relobralo', False, True),
        ('both', True, True),
    ]
    seeds = [0, 1, 2]  # three additional seeds (seed 42 from earlier run)

    summary = {}
    for name, use_causal, use_relobralo in configs:
        per_seed = []
        for seed in seeds:
            r = train_one(f'{name}_s{seed}', use_causal, use_relobralo, 30000, seed)
            per_seed.append(r)
            # Save incrementally so a crash does not lose completed runs
            with open(os.path.join(OUT_DIR, 'multiseed_summary.json'), 'w') as f:
                json.dump(summary, f, indent=2)
        summary[name] = {
            'err_h_mean': float(np.mean([r['err_h'] for r in per_seed])),
            'err_h_std': float(np.std([r['err_h'] for r in per_seed])),
            'err_n_mean': float(np.mean([r['err_n'] for r in per_seed])),
            'err_n_std': float(np.std([r['err_n'] for r in per_seed])),
            'err_C_mean': float(np.mean([r['err_C'] for r in per_seed])),
            'err_qx0_mean': float(np.mean([r['err_qx0'] for r in per_seed])),
            'per_seed': per_seed,
        }
        with open(os.path.join(OUT_DIR, 'multiseed_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n[{name}] MEAN: err_h={summary[name]['err_h_mean']:.4e}±"
              f"{summary[name]['err_h_std']:.4e} err_n={summary[name]['err_n_mean']:.4e}±"
              f"{summary[name]['err_n_std']:.4e}", flush=True)

    print("\n" + "=" * 70)
    print("MULTISEED SUMMARY")
    print("=" * 70)
    for name in summary:
        s = summary[name]
        print(f"{name:>12s}: err_h={s['err_h_mean']:.4e}±{s['err_h_std']:.4e} "
              f"err_n={s['err_n_mean']:.4e}±{s['err_n_std']:.4e}")
