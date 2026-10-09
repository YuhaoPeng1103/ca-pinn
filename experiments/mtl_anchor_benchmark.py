"""Multi-task benchmark for validating the quantitative anchor-bias theory.

Setting
-------
A shared trunk feeds T task heads. Training minimises

    L = sum_k lambda_k L_k^task  +  lambda_aux L_aux

where L_aux is an auxiliary (regularisation) loss.  Its coupling to the
balancing anchor is controlled by a single knob `gamma`:

    L_aux = ||W_head||^2  +  gamma * ||W_trunk_last||^2

  gamma = 0     -> L_aux depends only on the heads, so it is exactly decoupled
                   from any degree of freedom in the trunk (c = 0)
  gamma large   -> L_aux is progressively more coupled to the trunk (c -> 1)

The anchor convention studied here is the trunk's last layer, which is the
"last layer" shortcut that efficiency-motivated implementations adopt.

Predictions under test (see paper/theory_anchor_bias.md)
--------------------------------------------------------
With c = ||grad_phi L_aux|| / ||grad_theta L_aux||, G_k = ||grad_theta L_k||,
S = sum_j G_j and S_c = sum_j c_j G_j, and lambda^0 the weight the balancer
would assign to the same loss if it were fully coupled:

    relative-norm balancer :  lambda/lambda^0 = c * S / S_c
    inverse-ratio balancer :  lambda/lambda^0 = 1 / c

Both lambda^0 values are computable from the logged gradient norms, so the
experiment compares a measured weight ratio against a prediction derived from
quantities logged in the same step.
"""

import torch
import torch.nn as nn


# ----------------------------------------------------------------------
# Data: T synthetic regression tasks with differing difficulty
# ----------------------------------------------------------------------
def make_mtl_data(n=512, d=16, n_tasks=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g)
    Y = []
    for k in range(n_tasks):
        W = torch.randn(d, 1, generator=g) / (k + 1) ** 0.5
        f = X @ W
        if k % 2 == 0:
            f = torch.sin(2.0 * f)          # harder, oscillatory
        else:
            f = torch.tanh(f)               # easier, smooth
        Y.append(f + 0.01 * torch.randn(f.shape, generator=g))
    return X, Y


# ----------------------------------------------------------------------
# Model: shared trunk + per-task heads
# ----------------------------------------------------------------------
class MTLNet(nn.Module):
    def __init__(self, d=16, hidden=64, n_tasks=4):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(d, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(n_tasks)])

    def forward(self, x):
        h = self.trunk(x)
        return [head(h) for head in self.heads]

    def trunk_last(self):
        """The anchor: last layer of the shared trunk."""
        return self.trunk[-2].weight

    def trunk_params(self):
        return list(self.trunk.parameters())


# ----------------------------------------------------------------------
# Losses
# ----------------------------------------------------------------------
def task_losses(preds, targets):
    return [torch.mean((p - y) ** 2) for p, y in zip(preds, targets)]


def aux_loss(model, gamma, scale=1e-3):
    """Auxiliary regularisation with tunable coupling to the trunk anchor.

    `scale` keeps the auxiliary loss's gradient norm below that of the task
    losses, so that an inverse-ratio balancer's maximum is attained by a
    coupled loss — the premise of the prediction lambda/lambda^0 = 1/c.
    `gamma` controls the coupling: 0 gives exact decoupling from the trunk.
    """
    head_reg = sum(head.weight.pow(2).sum() for head in model.heads)
    trunk_reg = model.trunk[-2].weight.pow(2).sum()
    return scale * (head_reg + gamma * trunk_reg)


# ----------------------------------------------------------------------
# Gradient-norm utilities (anchor = trunk last layer, full = all params)
# ----------------------------------------------------------------------
def grad_norm(loss, params):
    gs = torch.autograd.grad(loss, params, retain_graph=True,
                             create_graph=False, allow_unused=True)
    sq = None
    for g in gs:
        if g is not None:
            t = g.detach().pow(2).sum()
            sq = t if sq is None else sq + t
    return float(torch.sqrt(sq)) if sq is not None else 0.0


def measure_norms(losses, model):
    """Return (anchor norms, full norms) for a list of losses."""
    anchor = [model.trunk_last()]
    full = list(model.parameters())
    ga = [grad_norm(l, anchor) for l in losses]
    gf = [grad_norm(l, full) for l in losses]
    return ga, gf
