"""Supplementary server experiments for CMAME submission.

Two experiments that directly strengthen the paper's core claims:

  1. relobralo_balprior_30k — ReLoBRaLo balancing ALL FOUR losses
     (PDE / data / BC / prior). This is the full-scale (30,000-epoch)
     direct evidence for the paper's central design principle: including
     the parameter prior among the balanced losses re-ill-poses the
     inverse problem (its weight decays to zero and parameters drift).
     Previously this was only shown at 1,500 epochs (short training).

  2. lr_anneal_30k — Wang et al. (2021) learning-rate annealing baseline
     (gradient-norm-based loss weighting), applied to the 3 multi-task
     losses with a fixed-weight prior. This is the standard gradient-
     pathology baseline that reviewers expect for comparison.

Both reuse the SWE inverse-problem setup from train_matrix_server.py.
"""

import sys, os, time, json
os.environ['MPLBACKEND'] = 'Agg'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from torch.autograd import grad

torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(10)

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


class LRAnnealing:
    """Wang et al. (2021) learning-rate annealing, last-layer anchor.

    lambda_i = max_j ||grad_j|| / ||grad_i||  (gradient-norm-inverse weights),
    followed by an EMA. Uses the last-layer weights as the anchor so the
    comparison with ReLoBRaLo is fair (same anchor).
    """

    def __init__(self, n_losses=3, alpha=0.9):
        self.n_losses = n_losses
        self.alpha = alpha
        self.weights = torch.ones(n_losses)

    def update_weights(self, losses, last_layer_weights):
        norms = torch.zeros(self.n_losses, device=last_layer_weights.device)
        for i, loss in enumerate(losses):
            if loss.item() == 0.0:
                norms[i] = 0.0
                continue
            g = grad(loss, last_layer_weights, retain_graph=True,
                     create_graph=False, allow_unused=True)[0]
            norms[i] = 0.0 if g is None else g.norm()
        norms = norms.clamp(min=1e-8)
        lam_hat = norms.max() / norms
        self.weights = self.alpha * self.weights + (1 - self.alpha) * lam_hat
        balanced = sum(w * l for w, l in zip(self.weights, losses))
        return self.weights.detach(), balanced


def _prior_loss(model):
    return (10.0 * (model.n - 0.03) ** 2 +
            5.0 * (model.C_drain - 0.05) ** 2 +
            5.0 * (model.qx0 - 1.0) ** 2)


def _setup(n_obs, n_coll=8000):
    model = SWE_PINN(use_fourier=True).to(device)
    x_obs, y_obs, t_obs, h_obs = generate_swe_data(n_obs=n_obs)
    x_obs, y_obs, t_obs, h_obs = [v.to(device) for v in [x_obs, y_obs, t_obs, h_obs]]

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
    return model, (x_obs, y_obs, t_obs, h_obs), (x_coll, y_coll, t_coll), opt, scheduler


def _evaluate(model, name, elapsed):
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
        'name': name, 'err_h': float(err_h), 'err_n': float(err_n),
        'err_C': float(err_C), 'err_qx0': float(err_qx0),
        'n_final': float(model.n.item()),
        'C_final': float(model.C_drain.item()),
        'qx0_final': float(model.qx0.item()),
        'time_s': float(elapsed),
    }
    out_dir = os.path.join(OUT_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, 'model.pth'))
    np.savez(os.path.join(out_dir, 'results.npz'),
             err_h=err_h, err_n=err_n, err_C=err_C, err_qx0=err_qx0,
             n=model.n.item(), C=model.C_drain.item(), qx0=model.qx0.item(),
             h_true=h_true, h_pred=h_pred)
    with open(os.path.join(out_dir, 'result.json'), 'w') as f:
        json.dump(result, f, indent=2)
    return result


def train_balprior(name, n_epochs=30000, n_obs=800, n_coll=8000):
    """ReLoBRaLo balances all 4 losses (PDE/data/BC/prior)."""
    print(f"\n{'='*60}")
    print(f"[{name}] {n_epochs} epochs | ReLoBRaLo(4 losses incl. prior) | n_obs={n_obs}")
    print(f"{'='*60}", flush=True)

    model, (x_obs, y_obs, t_obs, h_obs), (x_coll, y_coll, t_coll), opt, scheduler = _setup(n_obs, n_coll)
    balancer = ReLoBRaLo(n_losses=4)  # 4 losses INCLUDING prior

    best_state = None
    t0 = time.time()
    n_hist, C_hist, qx0_hist = [], [], []

    for epoch in range(n_epochs):
        x_coll.requires_grad_(True)
        y_coll.requires_grad_(True)
        t_coll.requires_grad_(True)

        res_cont, res_xmom, res_ymom, _, _, _ = compute_swe_residuals(model, x_coll, y_coll, t_coll)
        loss_pde = torch.mean(res_cont**2) + torch.mean(res_xmom**2) + torch.mean(res_ymom**2)

        h_pred, _, _ = model(x_obs / 100.0, y_obs / 100.0, t_obs / 3600.0)
        loss_data = torch.mean((h_pred - h_obs) ** 2)

        lbu, lbd, lbg = compute_bc_loss(model)
        loss_bc = lbu + lbd + lbg

        loss_prior = _prior_loss(model)

        components = [loss_pde, loss_data, loss_bc, loss_prior]
        _, loss_total = balancer.update_weights(components, model.get_last_layer_weights())

        if torch.isnan(loss_total):
            print(f"NaN at epoch {epoch}", flush=True)
            break

        opt.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        scheduler.step(loss_total.detach())

        n_hist.append(model.n.item())
        C_hist.append(model.C_drain.item())
        qx0_hist.append(model.qx0.item())

        if epoch % 1000 == 0:
            print(f"E{epoch:5d} | Loss={loss_total.item():.3e} "
                  f"n={model.n.item():.4f} C={model.C_drain.item():.4f} "
                  f"qx0={model.qx0.item():.4f} t={time.time()-t0:.0f}s", flush=True)

    result = _evaluate(model, name, time.time() - t0)
    print(f"\n[{name}] DONE: err_h={result['err_h']:.4e} err_n={result['err_n']:.4e} "
          f"err_C={result['err_C']:.4e} err_qx0={result['err_qx0']:.4e} "
          f"time={result['time_s']/3600:.1f}h", flush=True)
    return result


def train_lr_anneal(name, n_epochs=30000, n_obs=800, n_coll=8000):
    """Wang et al. (2021) LR-annealing baseline, fixed prior."""
    print(f"\n{'='*60}")
    print(f"[{name}] {n_epochs} epochs | LR-annealing(3 losses) + fixed prior | n_obs={n_obs}")
    print(f"{'='*60}", flush=True)

    model, (x_obs, y_obs, t_obs, h_obs), (x_coll, y_coll, t_coll), opt, scheduler = _setup(n_obs, n_coll)
    balancer = LRAnnealing(n_losses=3)

    best_state = None
    t0 = time.time()

    for epoch in range(n_epochs):
        x_coll.requires_grad_(True)
        y_coll.requires_grad_(True)
        t_coll.requires_grad_(True)

        res_cont, res_xmom, res_ymom, _, _, _ = compute_swe_residuals(model, x_coll, y_coll, t_coll)
        loss_pde = torch.mean(res_cont**2) + torch.mean(res_xmom**2) + torch.mean(res_ymom**2)

        h_pred, _, _ = model(x_obs / 100.0, y_obs / 100.0, t_obs / 3600.0)
        loss_data = torch.mean((h_pred - h_obs) ** 2)

        lbu, lbd, lbg = compute_bc_loss(model)
        loss_bc = lbu + lbd + lbg

        loss_prior = _prior_loss(model)

        components = [loss_pde, loss_data, loss_bc]
        _, loss_balanced = balancer.update_weights(components, model.get_last_layer_weights())
        loss_total = loss_balanced + loss_prior  # fixed-weight prior

        if torch.isnan(loss_total):
            print(f"NaN at epoch {epoch}", flush=True)
            break

        opt.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        scheduler.step(loss_total.detach())

        if epoch % 1000 == 0:
            print(f"E{epoch:5d} | Loss={loss_total.item():.3e} "
                  f"n={model.n.item():.4f} C={model.C_drain.item():.4f} "
                  f"qx0={model.qx0.item():.4f} t={time.time()-t0:.0f}s", flush=True)

    result = _evaluate(model, name, time.time() - t0)
    print(f"\n[{name}] DONE: err_h={result['err_h']:.4e} err_n={result['err_n']:.4e} "
          f"err_C={result['err_C']:.4e} err_qx0={result['err_qx0']:.4e} "
          f"time={result['time_s']/3600:.1f}h", flush=True)
    return result


if __name__ == "__main__":
    results = []
    results.append(train_balprior('relobralo_balprior_30k', 30000))
    results.append(train_lr_anneal('lr_anneal_30k', 30000))

    print("\n" + "=" * 70)
    print("EXTRA SUMMARY")
    print("=" * 70)
    for r in results:
        print(f"{r['name']:>26s}: err_h={r['err_h']:.4e} err_n={r['err_n']:.4e} "
              f"err_C={r['err_C']:.4e} err_qx0={r['err_qx0']:.4e} "
              f"time={r['time_s']/3600:.1f}h")
    with open(os.path.join(OUT_DIR, 'extra_summary.json'), 'w') as f:
        json.dump(results, f, indent=2)
