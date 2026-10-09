"""Does the anchor-robust wrapper repair the MTL failure without harming tasks?

Setup as in `train_mtl_anchor.py`: a shared trunk with four task heads plus an
auxiliary regulariser whose coupling to the trunk's last layer is set by gamma
(gamma = 0 gives exact decoupling).  Both balancers use the trunk's last layer
as their anchor.

For each balancer we compare three arms:

    plain        the balancer alone
    + ARB        the same balancer wrapped by AnchorRobustBalancer
    reference    the auxiliary loss made fully coupled (gamma -> 1)

What we report
    w_aux        the weight the balancer ends up giving the auxiliary loss
    L_aux        the auxiliary loss value actually attained (is it applied?)
    task         mean task loss (does the wrapper damage the real tasks?)

Success criterion: with ARB, the auxiliary loss is actually driven down
(comparable to the fully-coupled reference) while the task loss stays close to
the un-wrapped balancer.  Without ARB at gamma = 0 the auxiliary loss is either
ignored (relative-norm) or dominates (inverse-ratio).
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
N_TASKS, K = 4, 5


class RelativeNorm:
    def __init__(self, m):
        self.m = m

    def weights(self, anchor_norms):
        g = torch.tensor(anchor_norms, dtype=torch.float32).clamp(min=1e-12)
        return g / g.sum() * self.m


class InverseRatio:
    def __init__(self, m):
        self.m = m

    def weights(self, anchor_norms):
        g = torch.tensor(anchor_norms, dtype=torch.float32).clamp(min=1e-12)
        return (g.max() / g)


class ARB:
    """Anchor-robust wrapper: neutralise losses the anchor cannot see.

    Probes the coupling ratio c_k = ||grad_anchor L_k|| / ||grad_theta L_k||
    every `probe_every` steps; any loss below `tau` has its weight replaced by
    the mean weight of the visible losses, then all weights are renormalised to
    sum to K.
    """

    def __init__(self, base, model, tau=1e-3, probe_every=10, warmup=20):
        self.base, self.model = base, model
        self.tau, self.probe_every, self.warmup = tau, probe_every, warmup
        self._step, self.mask = 0, None

    def weights(self, anchor_norms, full_norms):
        self._step += 1
        K = len(anchor_norms)
        if self._step >= self.warmup and self._step % self.probe_every == 0:
            c = np.array([a / f if f > 1e-12 else 0.0
                          for a, f in zip(anchor_norms, full_norms)])
            self.mask = c < self.tau
        w = self.base.weights(anchor_norms)
        if self.mask is not None and self.mask.any():
            vis = ~self.mask
            if vis.any():
                w = w.clone()
                w[self.mask] = w[vis].mean()
                w = w * (K / w.sum().clamp(min=1e-12))
        return w


def run(base_name, use_arb, gamma, n_epochs=300):
    X, Y = make_mtl_data(seed=0)
    model = MTLNet(n_tasks=N_TASKS)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    base = RelativeNorm(K) if base_name == 'rel' else InverseRatio(K)
    arb = ARB(base, model) if use_arb else None

    w_hist, aux_hist, task_hist = [], [], []
    for ep in range(n_epochs):
        opt.zero_grad()
        losses = task_losses(model(X), Y) + [aux_loss(model, gamma)]
        ga, gf = measure_norms(losses, model)
        w = arb.weights(ga, gf) if arb is not None else base.weights(ga)
        total = sum(wi * l for wi, l in zip(w, losses))
        total.backward()
        opt.step()

        if ep >= int(0.75 * n_epochs):
            w_hist.append(float(w[-1]))
            aux_hist.append(losses[-1].detach().item())
            task_hist.append(float(sum(l.item() for l in losses[:-1]) / N_TASKS))

    return {'balancer': base_name, 'arb': use_arb, 'gamma': gamma,
            'w_aux': float(np.mean(w_hist)),
            'L_aux': float(np.mean(aux_hist)),
            'task_loss': float(np.mean(task_hist))}


if __name__ == "__main__":
    rows = []
    for bname, label in [('rel', 'relative-norm'), ('inv', 'inverse-ratio')]:
        print(f"\n{'='*76}")
        print(f"{label}  (anchor = trunk last layer)")
        print(f"{'='*76}")
        print(f"{'arm':<34} {'w_aux':>12} {'L_aux':>12} {'task loss':>12}")
        # decoupled auxiliary (the failure case)
        for arb, lab in [(False, 'plain (decoupled aux)'),
                         (True,  '+ ARB (decoupled aux)')]:
            r = run(bname, arb, 0.0)
            rows.append(r)
            print(f"{lab:<34} {r['w_aux']:>12.4g} {r['L_aux']:>12.4g} "
                  f"{r['task_loss']:>12.4g}")
        # fully coupled reference: the auxiliary loss as it should behave
        r = run(bname, False, 1.0)
        rows.append(r)
        print(f"{'reference (aux coupled)':<34} {r['w_aux']:>12.4g} "
              f"{r['L_aux']:>12.4g} {r['task_loss']:>12.4g}")

    with open(os.path.join(OUT, 'mtl_arb_results.json'), 'w') as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved to {OUT}/mtl_arb_results.json")
