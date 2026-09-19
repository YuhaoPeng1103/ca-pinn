"""Loss balancers, faithfully implemented per their original papers.

Three balancers are provided so they can be compared head-to-head:

1. ReLoBRaLoTrue   — Bischof & Kraus, "Multi-Objective Loss Balancing for
                     Physics-Informed Deep Learning", CMAME 439 (2025);
                     arXiv:2110.09813.  Uses **loss statistics only** (ratio of
                     current to reference loss, passed through a softmax with a
                     random lookback).  **Computes no gradients.**

2. LRAnnealingFull — Wang, Teng & Perdikaris, "Understanding and Mitigating
                     Gradient Flow Pathologies in PINNs", SIAM J. Sci.
                     Comput. 43(5), 2021.  Uses gradient norms w.r.t. the
                     **full parameter vector theta**.

3. (See training/loss_balancing.py for the last-layer-anchored EMA variant used
   in the earlier experiments.  It matches neither of the above.)

IMPORTANT NOTE ON THE EARLIER CODE
----------------------------------
`training/loss_balancing.py::ReLoBRaLo` was described in the manuscript as
ReLoBRaLo (Bischof & Kraus).  It is **not** that method: it updates weights from
gradient norms of the last layer, whereas the published ReLoBRaLo uses loss
values and computes no gradients at all.  The published method is implemented
here as `ReLoBRaLoTrue`, so the two can be compared directly.
"""

import torch


class ReLoBRaLoTrue:
    """ReLoBRaLo, implemented from the published update rule.

    The scaling update (Bischof & Kraus 2025):

        lambda_i^bal(t, t') = m * exp( L_i(t) / (T * L_i(t')) )
                              / sum_j exp( L_j(t) / (T * L_j(t')) )

        lambda_i^hist(t)    = rho * lambda_i(t-1) + (1 - rho) * lambda_i^bal(t, 0)

        lambda_i(t)         = alpha * lambda_i^hist(t)
                              + (1 - alpha) * lambda_i^bal(t, t-1)

    where
        L_i(t) : value of loss term i at iteration t
        m      : number of loss terms
        T      : temperature (small T -> sharper softmax)
        rho    : Bernoulli "saudade" variable; E[rho] chosen close to 1
        alpha  : exponential decay rate, in [0.9, 0.999]

    Gradients are never computed for the scalings, and the scalings are detached
    before back-propagation.

    Note the direction of the softmax argument: a term whose loss has *stopped*
    decreasing (L_i(t)/L_i(t') ~ 1, the largest ratio while other terms keep
    falling) receives the *largest* scaling.  That is the opposite of what a
    last-layer gradient-norm balancer does to a decoupled loss, whose relative
    gradient norm decays to zero — which is precisely why the two must be
    compared empirically rather than assumed equivalent.
    """

    def __init__(self, n_losses, alpha=0.999, temperature=0.1,
                 rho_prob=0.999, eps=1e-12):
        """
        Args:
            n_losses: number of loss terms, m.
            alpha: decay rate blending history with the current-step balance.
            temperature: softmax temperature T.
            rho_prob: probability that rho = 1 (expected value of the "saudade"
                variable; the paper recommends a value close to 1).
            eps: floor used when forming ratios, to keep them finite.
        """
        self.m = n_losses
        self.alpha = alpha
        self.temperature = temperature
        self.rho_prob = rho_prob
        self.eps = eps

        self.lambda_prev = torch.ones(n_losses)
        self.L0 = None        # losses at t = 0
        self.L_prev = None    # losses at t - 1

    def _bal(self, L_t, L_ref):
        """lambda^bal(t, t'): softmax over the loss ratios L(t)/L(t')."""
        ratio = L_t / L_ref.clamp(min=self.eps)
        logits = ratio / max(self.temperature, self.eps)
        logits = logits - logits.max()          # softmax is shift-invariant
        w = torch.exp(logits)
        return self.m * w / w.sum().clamp(min=self.eps)

    def update_weights(self, losses, last_layer_weights=None):
        """Update scalings; return (weights, balanced_loss).

        Args:
            losses: sequence of scalar loss tensors (expected positive).
            last_layer_weights: unused; accepted so this class is a drop-in
                replacement for the gradient-based balancers.

        Returns:
            weights: detached tensor of shape (m,) summing to m.
            balanced_loss: sum_i weights_i * losses_i.
        """
        L_t = torch.stack([l.detach().reshape(()) for l in losses]).float()

        if self.L0 is None:
            # No history yet: return unit scalings for the first step.
            self.L0 = L_t.clone()
            self.L_prev = L_t.clone()
            return self.lambda_prev.clone(), sum(losses)

        bal_prev = self._bal(L_t, self.L_prev)   # lambda^bal(t, t-1)
        bal_init = self._bal(L_t, self.L0)       # lambda^bal(t, 0)

        # Random lookback: rho = 1 keeps the previous scaling; rho = 0 re-anchors
        # to the balance against the initial losses.
        rho = 1.0 if torch.rand(()) < self.rho_prob else 0.0
        lam_hist = rho * self.lambda_prev + (1.0 - rho) * bal_init

        lam = self.alpha * lam_hist + (1.0 - self.alpha) * bal_prev
        lam = torch.nan_to_num(lam, nan=1.0, posinf=self.m, neginf=0.0)
        lam = lam.clamp(min=0.0)

        self.lambda_prev = lam.clone()
        self.L_prev = L_t.clone()

        balanced = sum(w * l for w, l in zip(lam, losses))
        return lam, balanced


class LRAnnealingFull:
    """Learning-rate annealing over the full parameter vector (Wang et al. 2021).

        lambda_hat_i = max_theta |grad_theta L_r| / mean_theta |grad_theta L_i|

    where L_r is the residual loss (taken here as the first entry of `losses`)
    and the gradient norms are taken w.r.t. **all** network parameters.  Target
    weights are smoothed by a moving average,

        lambda_i <- (1 - alpha) lambda_i + alpha lambda_hat_i,

    and refreshed every `update_every` iterations (the paper uses alpha = 0.1
    and updates every 10 Adam steps).

    This is the faithful counterpart to the last-layer-anchored variant in
    `loss_balancing.py`; the two differ *only* in the anchor, which is exactly
    the design choice under study.
    """

    def __init__(self, n_losses, params, alpha=0.1, update_every=10, eps=1e-8):
        """
        Args:
            n_losses: number of loss terms, m.
            params: iterable of parameters to differentiate against. Pass
                `model.parameters()` for the full parameter vector.
            alpha: smoothing factor for the moving average.
            update_every: refresh the scalings every this many calls.
            eps: floor on gradient norms, to avoid division by zero.
        """
        self.m = n_losses
        self.params = [p for p in params if p.requires_grad]
        self.alpha = alpha
        self.update_every = update_every
        self.eps = eps
        self.weights = torch.ones(n_losses)
        self._step = 0

    def _grad_norms(self, losses):
        """L2 norm of each loss's gradient w.r.t. the full parameter vector."""
        norms = torch.zeros(len(losses))
        for i, loss in enumerate(losses):
            grads = torch.autograd.grad(
                loss, self.params, retain_graph=True, create_graph=False,
                allow_unused=True,
            )
            sq = None
            for g in grads:
                if g is not None:
                    term = g.detach().pow(2).sum()
                    sq = term if sq is None else sq + term
            norms[i] = torch.sqrt(sq) if sq is not None else 0.0
        return norms

    def update_weights(self, losses, last_layer_weights=None):
        """Update scalings; return (weights, balanced_loss).

        The gradient computation is performed every call (so that
        `retain_graph=True` bookkeeping stays simple) but the scalings are only
        refreshed every `update_every` calls, matching the paper.
        """
        self._step += 1
        if self._step % self.update_every == 0 and self.params:
            norms = self._grad_norms(losses).clamp(min=self.eps)
            lam_hat = norms.max() / norms                      # Eq. (11)
            self.weights = ((1 - self.alpha) * self.weights
                            + self.alpha * lam_hat.detach())

        balanced = sum(w * l for w, l in zip(self.weights, losses))
        return self.weights.detach(), balanced
