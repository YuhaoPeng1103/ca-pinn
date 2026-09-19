"""Head-to-head comparison of loss balancers on the SWE inverse problem.

Question under test
-------------------
The manuscript's headline claim is that "balancing a parameter prior re-ill-poses
the inverse problem" (err_C 0.047% -> 37.2%).  That was measured with
`training/loss_balancing.py::ReLoBRaLo`, which is a *last-layer-anchored
gradient-norm* balancer -- NOT the published ReLoBRaLo, which uses loss
statistics and no gradients at all.

This script runs all three balancers against the same problem:

  ll_ema       last-layer-anchored gradient-norm EMA   (training/loss_balancing.py)
  lra_full     full-parameter gradient-norm annealing  (Wang et al. 2021)
  true_relo    loss-statistics ReLoBRaLo               (Bischof & Kraus 2025)

each with the prior either balanced (4 terms) or held at a fixed weight (3
terms + prior added separately).

Controlled experiment (toy model, 300 iterations) showed the three behave
qualitatively differently on an anchor-decoupled loss:

  ll_ema       weight -> 0        (loss silently switched off)
  lra_full     weight -> +inf     (loss dominates)
  true_relo    weight bounded     (unaffected by decoupling)

This script tests whether that translates into different parameter-recovery
outcomes on a real inverse problem.

Usage
-----
    python experiments/train_balancer_compare.py --n-coll 2000 --epochs 30000
    python experiments/train_balancer_compare.py --only true_relo_balprior
"""

import sys, os, time, json, argparse
os.environ['MPLBACKEND'] = 'Agg'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

device = 'cpu'

from physics.swe_model import SWE_PINN
from training.loss_balancing import ReLoBRaLo
from training.balancers import ReLoBRaLoTrue, LRAnnealingFull
from training.anchor_robust import AnchorRobustBalancer
from physics.swe import (
    compute_swe_residuals, compute_bc_loss,
    generate_swe_data, generate_truth_grid
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

# name -> (balancer kind, balance the prior?)
CONFIGS = {
    'll_ema_fixed':        ('ll_ema',    False),
    'll_ema_balprior':     ('ll_ema',    True),
    'lra_full_fixed':      ('lra_full',  False),
    'lra_full_balprior':   ('lra_full',  True),
    'true_relo_fixed':     ('true_relo', False),
    'true_relo_balprior':  ('true_relo', True),
    # Same two gradient-based balancers, wrapped with the anchor-robust fix.
    # These test whether the fix recovers parameter accuracy with no manual
    # intervention and no knowledge of which loss is decoupled.
    'll_ema_arb_balprior':   ('ll_ema_arb',   True),
    'lra_full_arb_balprior': ('lra_full_arb', True),
}


def prior_centers(shift=0.0):
    """Prior centres, optionally misspecified by a relative `shift`.

    shift=0    -> prior centre equals the true parameter values
    shift=0.2  -> prior centre is 20% above the truth

    Used to test whether a balancer follows the DATA or merely the PRIOR:
    if the prior centre is wrong and the recovered parameters track it, the
    balancer has turned the prior into a hard constraint and the inverse
    problem is no longer being solved.
    """
    return (0.03 * (1.0 + shift), 0.05 * (1.0 + shift), 1.0 * (1.0 + shift))


def _prior_loss(model, shift=0.0, scale=1.0):
    """Parameter prior. `scale` controls its overall strength.

    scale=1.0  -> the weights used in the paper (10, 5, 5)
    scale=0.0  -> the prior is omitted entirely (see `run`)

    A weak/absent prior is used to test whether the DATA can constrain the
    parameters at all, or whether the reported parameter recovery is simply
    the prior being echoed back.
    """
    n0, C0, q0 = prior_centers(shift)
    return scale * (10.0 * (model.n - n0) ** 2 +
                    5.0 * (model.C_drain - C0) ** 2 +
                    5.0 * (model.qx0 - q0) ** 2)


def _build_balancer(kind, balance_prior, model, include_prior=True):
    """Build a balancer. Kinds ending in `_arb` wrap the base balancer with the
    anchor-robust fix from `training/anchor_robust.py`."""
    m = 4 if (balance_prior and include_prior) else 3
    arb = kind.endswith('_arb')
    base_kind = kind[:-4] if arb else kind

    if base_kind == 'll_ema':
        base = ReLoBRaLo(n_losses=m)
    elif base_kind == 'lra_full':
        base = LRAnnealingFull(n_losses=m, params=list(model.parameters()))
    elif base_kind == 'true_relo':
        base = ReLoBRaLoTrue(n_losses=m)
    else:
        raise ValueError(kind)

    if arb:
        return AnchorRobustBalancer(
            base,
            anchor=model.get_last_layer_weights(),
            params=list(model.parameters()),
            tau=1e-3, probe_every=200, warmup=100)
    return base


def _apply(balancer, kind, components, model):
    """Call the balancer with the signature appropriate to its family.

    The anchor-robust wrapper holds the anchor itself, so it is driven like a
    plain `update_weights(losses)` balancer.
    """
    family = kind[:-4] if kind.endswith('_arb') else kind
    if family == 'll_ema':
        return balancer.update_weights(components, model.get_last_layer_weights())
    return balancer.update_weights(components)


def _evaluate(model, name, elapsed, n_eval=60, prior_shift=0.0):
    xg, yg, tg, h_true, qx_true, qy_true = generate_truth_grid(
        Nx=n_eval, Ny=n_eval, Nt=10)
    h_pred = np.zeros_like(h_true)
    model.eval()
    with torch.no_grad():
        X_m, Y_m = np.meshgrid(xg, yg, indexing='ij')
        for it, tt in enumerate(tg):
            xf = torch.tensor(X_m.flatten(), dtype=torch.float32,
                              device=device).view(-1, 1)
            yf = torch.tensor(Y_m.flatten(), dtype=torch.float32,
                              device=device).view(-1, 1)
            tf = torch.ones_like(xf) * tt
            hp, _, _ = model(xf / 100.0, yf / 100.0, tf / 3600.0)
            h_pred[it] = hp.cpu().numpy().reshape(n_eval, n_eval)

    err_h = np.linalg.norm(h_pred - h_true) / (np.linalg.norm(h_true) + 1e-8)
    err_n = abs(model.n.item() - 0.03) / 0.03
    err_C = abs(model.C_drain.item() - 0.05) / 0.05
    err_qx0 = abs(model.qx0.item() - 1.0)

    # How far the recovered parameters sit from the (possibly wrong) prior
    # centre: a small value means the balancer simply adopted the prior.
    n0, C0, q0 = prior_centers(prior_shift)
    err_C_to_prior = abs(model.C_drain.item() - C0) / C0
    err_n_to_prior = abs(model.n.item() - n0) / n0

    result = {
        'name': name, 'err_h': float(err_h), 'err_n': float(err_n),
        'err_C': float(err_C), 'err_qx0': float(err_qx0),
        'prior_shift': float(prior_shift),
        'prior_C_center': float(C0), 'prior_n_center': float(n0),
        'err_C_to_prior': float(err_C_to_prior),
        'err_n_to_prior': float(err_n_to_prior),
        'n_final': float(model.n.item()),
        'C_final': float(model.C_drain.item()),
        'qx0_final': float(model.qx0.item()),
        'time_s': float(elapsed),
    }
    out_dir = os.path.join(OUT_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, 'results.npz'),
             err_h=err_h, err_n=err_n, err_C=err_C, err_qx0=err_qx0,
             n=model.n.item(), C=model.C_drain.item(), qx0=model.qx0.item(),
             h_true=h_true, h_pred=h_pred)
    with open(os.path.join(out_dir, 'result.json'), 'w') as f:
        json.dump(result, f, indent=2)
    return result


def run(name, kind, balance_prior, n_epochs, n_coll, n_obs, prior_shift=0.0,
        prior_scale=1.0):
    print(f"\n{'='*64}")
    print(f"[{name}] {n_epochs} ep | {kind} | balance_prior={balance_prior} | "
          f"n_coll={n_coll}")
    print(f"{'='*64}", flush=True)

    model = SWE_PINN(use_fourier=True).to(device)
    x_obs, y_obs, t_obs, h_obs = generate_swe_data(n_obs=n_obs)
    x_obs, y_obs, t_obs, h_obs = [v.to(device) for v in
                                  [x_obs, y_obs, t_obs, h_obs]]

    include_prior = prior_scale > 0.0
    balancer = _build_balancer(kind, balance_prior, model, include_prior)

    ng = int(n_coll * 0.7)
    nl = n_coll - ng
    x_coll = torch.cat([torch.rand(ng, 1, device=device) * 100,
                        40 + 20 * torch.rand(nl, 1, device=device)], dim=0)
    y_coll = torch.cat([torch.rand(ng, 1, device=device) * 100,
                        40 + 20 * torch.rand(nl, 1, device=device)], dim=0)
    t_coll = torch.cat([torch.rand(ng, 1, device=device) * 3600,
                        torch.rand(nl, 1, device=device) * 3600], dim=0)

    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=2000, factor=0.5)

    t0 = time.time()
    w_hist = []
    for epoch in range(n_epochs):
        x_coll.requires_grad_(True)
        y_coll.requires_grad_(True)
        t_coll.requires_grad_(True)

        res_cont, res_xmom, res_ymom, _, _, _ = compute_swe_residuals(
            model, x_coll, y_coll, t_coll)
        loss_pde = (torch.mean(res_cont**2) + torch.mean(res_xmom**2) +
                    torch.mean(res_ymom**2))

        h_pred, _, _ = model(x_obs / 100.0, y_obs / 100.0, t_obs / 3600.0)
        loss_data = torch.mean((h_pred - h_obs) ** 2)

        lbu, lbd, lbg = compute_bc_loss(model)
        loss_bc = lbu + lbd + lbg

        if not include_prior:
            # No prior at all: tests whether the data constrains the
            # parameters on its own.
            components = [loss_pde, loss_data, loss_bc]
            w, loss_total = _apply(balancer, kind, components, model)
            w_hist.append(0.0)
        else:
            loss_prior = _prior_loss(model, prior_shift, prior_scale)
            if balance_prior:
                components = [loss_pde, loss_data, loss_bc, loss_prior]
                w, loss_total = _apply(balancer, kind, components, model)
                w_hist.append(float(w[-1]))
            else:
                components = [loss_pde, loss_data, loss_bc]
                w, loss_balanced = _apply(balancer, kind, components, model)
                loss_total = loss_balanced + loss_prior
                w_hist.append(float(loss_prior.item()))

        if torch.isnan(loss_total):
            print(f"NaN at epoch {epoch}", flush=True)
            break

        opt.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        scheduler.step(loss_total.detach())

        if epoch % 1000 == 0:
            extra = f"w_prior={w_hist[-1]:.3e} " if balance_prior else ""
            print(f"E{epoch:5d} | Loss={loss_total.item():.3e} "
                  f"n={model.n.item():.4f} C={model.C_drain.item():.4f} "
                  f"{extra}t={time.time()-t0:.0f}s", flush=True)

    elapsed = time.time() - t0
    result = _evaluate(model, name, elapsed, prior_shift=prior_shift)
    result['w_prior_final'] = w_hist[-1] if w_hist else None
    result['w_prior_trace'] = w_hist[::max(1, len(w_hist)//200)]
    print(f"\n[{name}] DONE: err_h={result['err_h']:.4e} "
          f"err_n={result['err_n']:.4e} err_C={result['err_C']:.4e} "
          f"w_prior_final={result['w_prior_final']} "
          f"time={elapsed/3600:.2f}h", flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-coll', type=int, default=2000)
    ap.add_argument('--epochs', type=int, default=30000)
    ap.add_argument('--n-obs', type=int, default=800)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--only', type=str, default=None,
                    help='comma-separated config names, or omit for all')
    ap.add_argument('--prior-shift', type=float, default=0.0,
                    help='relative misspecification of the prior centre '
                         '(0 = correct, 0.2 = 20%% above truth). Used for the '
                         'prior-misspecification test.')
    ap.add_argument('--prior-scale', type=float, default=1.0,
                    help='overall strength of the prior. 1.0 = paper weights '
                         '(10, 5, 5); 0.0 = no prior at all (tests whether the '
                         'data alone constrains the parameters).')
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    print(f"threads={args.threads} n_coll={args.n_coll} epochs={args.epochs} "
          f"n_obs={args.n_obs}", flush=True)

    names = (args.only.split(',') if args.only else list(CONFIGS))
    suffix = ''
    if args.prior_shift != 0:
        suffix += f"_shift{int(round(args.prior_shift * 100))}"
    if args.prior_scale != 1.0:
        suffix += f"_pscale{args.prior_scale:g}".replace('.', 'p')
    summ_name = ('balancer_compare_summary.json' if not suffix
                 else f'balancer_compare_summary{suffix}.json')
    results = []
    for nm in names:
        kind, bal = CONFIGS[nm]
        full = nm + suffix
        try:
            results.append(run(full, kind, bal, args.epochs, args.n_coll,
                               args.n_obs, args.prior_shift,
                               args.prior_scale))
        except RuntimeError as e:
            print(f"[{full}] FAILED: {e}", flush=True)
            results.append({'name': full, 'error': str(e)})
        # persist incrementally so a crash does not lose completed runs
        with open(os.path.join(OUT_DIR, summ_name), 'w') as f:
            json.dump(results, f, indent=2)

    print("\n" + "=" * 72)
    print("BALANCER COMPARISON SUMMARY")
    print("=" * 72)
    for r in results:
        if 'error' in r:
            print(f"{r['name']:>22s}: ERROR {r['error'][:50]}")
        else:
            print(f"{r['name']:>22s}: err_h={r['err_h']:.4e} "
                  f"err_n={r['err_n']:.4e} err_C={r['err_C']:.4e} "
                  f"w_prior={r.get('w_prior_final')}")
