"""Validate the quantitative anchor-bias theory inside a real MTL training loop.

The theory is an *instantaneous* statement about the weights a balancer assigns
at a given step.  We therefore test it per step: at every training iteration we
compute the theory's predicted weight for the auxiliary loss from that step's
coupling ratio c_aux and gradient norms, and compare it with the weight the
balancer actually assigned.  Averaging weights, couplings and norms separately
before comparing (a ratio of averages) would not test the formula.

Predictions
    relative-norm :  lambda_aux / lambda_aux^0 = c * S / S_c
    inverse-ratio :  lambda_aux / lambda_aux^0 = 1 / c

with lambda_aux^0 the weight the balancer would give the *same* loss if it were
fully coupled, computed from the same logged norms (K*G_aux/S and max_j G_j /
G_aux respectively).

Usage:
    python experiments/train_mtl_anchor.py
"""

import os, sys, json
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.mtl_anchor_benchmark import (
    make_mtl_data, MTLNet, task_losses, aux_loss, measure_norms)

torch.manual_seed(0)
np.random.seed(0)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(OUT, exist_ok=True)

N_TASKS = 4
K = N_TASKS + 1


class RelativeNorm:
    """lambda_k = K * g_k / sum_j g_j  (instantaneous; matches the theory)."""

    def __init__(self, m):
        self.m = m

    def weights(self, anchor_norms):
        g = torch.tensor(anchor_norms, dtype=torch.float32).clamp(min=1e-12)
        return g / g.sum() * self.m


class InverseRatio:
    """lambda_k = max_j g_j / g_k, refresh every `every` steps (Wang et al.)."""

    def __init__(self, m, alpha=0.1, every=5):
        self.m, self.alpha, self.every = m, alpha, every
        self.w = torch.ones(m)
        self._step = 0

    def weights(self, full_norms):
        self._step += 1
        if self._step % self.every == 0:
            g = torch.tensor(full_norms, dtype=torch.float32).clamp(min=1e-12)
            self.w = (1 - self.alpha) * self.w + self.alpha * (g.max() / g)
        return self.w


def run(balancer_name, gamma, n_epochs=300):
    X, Y = make_mtl_data(seed=0)
    model = MTLNet(n_tasks=N_TASKS)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    bal = RelativeNorm(K) if balancer_name == 'rel' else InverseRatio(K, alpha=1.0, every=1)

    errs, cs, ws, ratios = [], [], [], []
    n_switched_off = 0          # steps where the loss was fully decoupled
    for ep in range(n_epochs):
        opt.zero_grad()
        preds = model(X)
        losses = task_losses(preds, Y) + [aux_loss(model, gamma)]

        ga, gf = measure_norms(losses, model)
        # BOTH balancers use the same anchor (trunk last layer); they differ
        # only in the update rule, which isolates the failure direction.
        w = bal.weights(ga)
        total = sum(wi * l for wi, l in zip(w, losses))
        total.backward()
        opt.step()

        if ep < int(0.25 * n_epochs):          # let the model settle
            continue

        c = np.array([a / f if f > 1e-12 else 0.0 for a, f in zip(ga, gf)])
        G = np.array(gf)
        S, Sc = G.sum(), (c * G).sum()
        c_aux, G_aux, g_aux, w_aux = c[-1], G[-1], ga[-1], float(w[-1])

        # exact decoupling: the formula predicts lambda -> 0 (relative-norm) or
        # lambda -> inf (inverse-ratio); this is the qualitative regime and is
        # reported separately rather than as a relative error.
        if c_aux < 1e-9:
            n_switched_off += 1
            cs.append(c_aux); ws.append(w_aux); continue

        Ga = np.array(ga)
        M = float((c * G).max())
        M_G = float(G.max())
        if balancer_name == 'rel':
            lam0 = K * G_aux / S          # fully-coupled relative-norm reference
            pred = c_aux * S / Sc
        else:
            # General inverse-ratio law: the reference maximum is attained by
            # whichever loss has the largest *full* norm, and that loss need not
            # be fully coupled.  Writing M = max_j (c_j G_j) and M_G = max_j G_j,
            #     lambda_k / lambda_k^0 = M / (c_k * M_G),
            # which reduces to 1/c_k when the maximum is fully coupled.
            lam0 = M_G / G_aux
            pred = M / (c_aux * M_G)
        if lam0 <= 0 or pred <= 0 or not np.isfinite(pred):
            continue
        measured = w_aux / lam0
        errs.append(abs(measured - pred) / pred * 100)
        cs.append(c_aux); ws.append(w_aux); ratios.append(measured / pred)

    n_total = len(cs) + 0
    return {'balancer': balancer_name, 'gamma': gamma,
            'c_aux_mean': float(np.mean(cs)) if cs else float('nan'),
            'c_aux_min': float(np.min(cs)) if cs else float('nan'),
            'w_aux_mean': float(np.mean(ws)) if ws else float('nan'),
            'frac_steps_decoupled': n_switched_off / max(n_total, 1),
            'median_err_pct': float(np.median(errs)) if errs else float('nan'),
            'p90_err_pct': float(np.percentile(errs, 90)) if errs else float('nan'),
            'max_err_pct': float(np.max(errs)) if errs else float('nan'),
            'ratio_mean': float(np.mean(ratios)) if ratios else float('nan'),
            'n_compared': len(errs)}


if __name__ == "__main__":
    gammas = [0.0, 0.02, 0.1, 0.5, 2.0, 10.0]
    results = []
    for name, label in [('rel', 'relative-norm (anchor = trunk last layer)'),
                        ('inv', 'inverse-ratio (anchor = full parameters)')]:
        print(f"\n{'='*78}\n{label}\n{'='*78}")
        print(f"{'gamma':>7} {'c_aux mean':>11} {'w_aux':>9} {'decoupled':>10} "
              f"{'w_aux':>9} {'med err':>9} {'p90 err':>9} {'max err':>9}")
        for gm in gammas:
            r = run(name, gm)
            results.append(r)
            print(f"{gm:>7.2f} {r['c_aux_mean']:>11.4f} {r['w_aux_mean']:>9.4f} {r['frac_steps_decoupled']*100:>9.1f}% "
                  f"{r['w_aux_mean']:>9.4f} {r['median_err_pct']:>8.2f}% "
                  f"{r['p90_err_pct']:>8.2f}% {r['max_err_pct']:>8.2f}%")

    with open(os.path.join(OUT, 'mtl_anchor_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT}/mtl_anchor_results.json")
