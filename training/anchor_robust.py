"""Anchor-robust loss balancing: diagnosis and fix for silent loss failure.

Background
----------
Gradient-norm loss balancers estimate each loss's "size" from its gradient with
respect to an *anchor* — typically a subset of the parameters, chosen for
cheapness (e.g. ReLoBRaLo-style implementations use the last layer).  That
estimate is only meaningful if the loss actually depends on the anchor.

If a loss L_k is decoupled from the anchor phi, then grad_phi L_k = 0, and the
balancer's estimate of the loss's magnitude collapses to zero.  The consequence
depends on the functional form of the balancer:

  * *relative-norm* balancers (lambda_k ∝ ||grad_phi L_k|| / max_j ||grad_phi L_j||)
    drive lambda_k -> 0: the loss is **silently switched off**.
  * *inverse-ratio* balancers (lambda_k = max_j ||grad_theta L_j|| / ||grad_theta L_k||,
    i.e. Wang et al. 2021 learning-rate annealing) divide by ~0 and drive
    lambda_k -> infinity: the loss **dominates everything else**.

Neither failure raises an error, and neither is visible in the training loss.
In an inverse problem the decoupled loss is usually the parameter prior, so the
recovered parameters silently stop being constrained.

This module provides:

  `estimate_anchor_coupling(losses, anchor, params)`
      The diagnostic.  Returns c_k = ||grad_phi L_k|| / ||grad_theta L_k||, the
      fraction of a loss's gradient energy that lies on the anchor.  c_k ~ 1
      means the loss is fully visible to the anchor; c_k ~ 0 means it is
      invisible and must not be balanced by anchor-gradient signals.

  `AnchorRobustBalancer(base, anchor, params)`
      The fix.  A wrapper around *any* balancer.  It periodically probes the
      coupling ratios and, for losses whose coupling falls below a threshold,
      replaces the balancer's (meaningless) weight with a neutral one — the
      mean weight of the visible losses — renormalising so the weights still
      sum to the number of terms.  Decoupled losses are therefore neither
      discarded nor allowed to dominate.

      The wrapper is agnostic to the base balancer, so existing pipelines
      (learning-rate annealing, GradNorm, ReLoBRaLo, ...) can adopt it without
      changing anything else.
"""

import inspect

import torch


def _grad_norm(loss, params):
    """L2 norm of d(loss)/d(params), summed over all parameter tensors."""
    grads = torch.autograd.grad(
        loss, params, retain_graph=True, create_graph=False, allow_unused=True,
    )
    sq = None
    for g in grads:
        if g is not None:
            term = g.detach().pow(2).sum()
            sq = term if sq is None else sq + term
    if sq is None:
        return torch.zeros(())
    return torch.sqrt(sq)


def estimate_anchor_coupling(losses, anchor, params, eps=1e-12):
    """Coupling ratio c_k = ||grad_anchor L_k|| / ||grad_full L_k|| for each loss.

    Args:
        losses: sequence of scalar loss tensors.
        anchor: the parameter subset the balancer differentiates against
            (e.g. the last layer's weight matrix).
        params: the full parameter set (e.g. list(model.parameters())).
        eps: floor to avoid division by zero.

    Returns:
        Tensor of shape (K,) with c_k in [0, 1].  c_k ~ 0 flags a loss that the
        anchor cannot see, i.e. one whose balanced weight is meaningless.
    """
    out = torch.zeros(len(losses))
    for i, loss in enumerate(losses):
        a = _grad_norm(loss, [anchor])
        f = _grad_norm(loss, list(params))
        out[i] = (a / f.clamp(min=eps)).detach()
    return out


class AnchorRobustBalancer:
    """Wrap a base balancer so anchor-decoupled losses are handled safely.

    Usage:
        base = LRAnnealingFull(n_losses=4, params=list(model.parameters()))
        arb = AnchorRobustBalancer(base, anchor=model.get_last_layer_weights(),
                                   params=list(model.parameters()))
        weights, total = arb.update_weights(components)

    Every `probe_every` calls the coupling ratios are re-estimated.  Losses with
    c_k < `tau` are marked decoupled and receive the mean weight of the visible
    losses instead of the base balancer's estimate.

    Args:
        base: any object exposing update_weights(losses) -> (weights, loss).
        anchor: anchor parameter(s) the base balancer differentiates against.
        params: full parameter set, used for the diagnostic.
        tau: coupling threshold below which a loss counts as decoupled.
        probe_every: how often to recompute the coupling ratios (the probe costs
            one extra backward pass per loss, so it is amortised).
        warmup: number of calls before the first probe (lets the losses settle).
        verbose: print the coupling ratios when they are probed.
    """

    def __init__(self, base, anchor, params, tau=1e-3, probe_every=200,
                 warmup=50, verbose=False):
        self.base = base
        self.anchor = anchor
        self.params = list(params)
        self.tau = tau
        self.probe_every = probe_every
        self.warmup = warmup
        self.verbose = verbose

        self._step = 0
        self.coupling = None      # (K,) latest ratios
        self.is_decoupled = None  # (K,) bool mask
        self.last_probe_step = None

        # Some balancers take the anchor explicitly (e.g. the last-layer EMA
        # variant), others do not. Detect which so the wrapper can drive either.
        try:
            n_params = len(inspect.signature(base.update_weights).parameters)
        except (TypeError, ValueError):
            n_params = 1
        self._base_takes_anchor = n_params >= 2

    def _call_base(self, losses):
        if self._base_takes_anchor:
            return self.base.update_weights(losses, self.anchor)
        return self.base.update_weights(losses)

    def update_weights(self, losses, last_layer_weights=None):
        """Return (weights, balanced_loss) with decoupled losses made safe."""
        self._step += 1

        # ---- periodic diagnostic ----
        if (self.coupling is None) or (
                self._step >= self.warmup
                and (self._step - (self.last_probe_step or 0)) >= self.probe_every):
            with torch.enable_grad():
                self.coupling = estimate_anchor_coupling(
                    losses, self.anchor, self.params)
            self.is_decoupled = self.coupling < self.tau
            self.last_probe_step = self._step
            if self.verbose:
                print(f"  [ARB] step {self._step} coupling="
                      f"{[f'{c:.2e}' for c in self.coupling.tolist()]} "
                      f"decoupled={self.is_decoupled.tolist()}", flush=True)

        # ---- base balancer, then correct the decoupled entries ----
        # The base is always driven with the full loss list: a decoupled loss
        # contributes a ~0 gradient norm, so it cannot perturb the weights of
        # the visible losses (the max/normalisation is set by the visible
        # ones).  We then overwrite its weight, which is the meaningless one.
        w, _ = self._call_base(losses)

        if self.is_decoupled is not None and self.is_decoupled.any():
            visible = ~self.is_decoupled
            if visible.any():
                K = len(losses)
                w = w.clone()
                # Decoupled losses get the mean visible weight: neither
                # discarded (which would silently drop the loss) nor allowed
                # to dominate (which an inverse-ratio balancer would do).
                w[self.is_decoupled] = w[visible].mean()
                # Renormalise so the weights sum to K, matching the
                # convention used by the balancers.
                w = w * (K / w.sum().clamp(min=1e-12))

        balanced = sum(wi * l for wi, l in zip(w, losses))
        return w.detach(), balanced
