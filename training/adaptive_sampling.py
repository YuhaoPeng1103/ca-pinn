"""Residual-based Adaptive Refinement (RAR) for collocation point sampling.

Periodically evaluates PDE residual on a fine candidate grid and resamples
collocation points proportional to residual magnitude, focusing compute on
regions where the PDE is least satisfied.

Reference: Lu et al., "DeepXDE: A deep learning library for solving
differential equations", 2021.
"""

import torch
import numpy as np


class AdaptiveSampler:
    """Manages adaptive collocation point resampling."""

    def __init__(
        self,
        n_collocation: int = 30000,
        n_candidates: int = 50000,
        resample_every: int = 2000,
        beta: float = 1.5,
        bounds: dict = None,
    ):
        """
        Args:
            n_collocation: number of collocation points to maintain
            n_candidates: number of candidate points to evaluate
            resample_every: resample every K epochs
            beta: sharpness of residual-weighted sampling (higher = more focused)
            bounds: dict of {dim: (min, max)}, e.g. {'x': (0,100), 'y': (0,100)}
        """
        self.n_collocation = n_collocation
        self.n_candidates = n_candidates
        self.resample_every = resample_every
        self.beta = beta
        self.bounds = bounds or {'x': (0, 100), 'y': (0, 100), 't': (0, 3600)}
        self.device = None

    def _generate_candidates(self):
        """Generate random candidate points within bounds."""
        candidates = {}
        for dim, (lo, hi) in self.bounds.items():
            candidates[dim] = torch.rand(self.n_candidates, 1, device=self.device) * (hi - lo) + lo
        return candidates

    def should_resample(self, epoch):
        """Check if resampling should occur at this epoch."""
        return (epoch > 0) and (epoch % self.resample_every == 0)

    def resample(self, model, epoch, current_collocation):
        """Evaluate residual on candidates and resample collocation points.

        Args:
            model: PINN model
            epoch: current epoch number
            current_collocation: dict of current collocation tensors
                                 {'x': (N,1), 'y': (N,1), 't': (N,1)} on device

        Returns:
            new_collocation: dict of new collocation tensors on device
        """
        if self.device is None:
            self.device = current_collocation['x'].device

        if not self.should_resample(epoch):
            return current_collocation

        # Generate candidates and compute residual
        cand = self._generate_candidates()
        residuals = self._compute_residuals(model, cand)

        # Probability proportional to residual^beta
        probs = residuals ** self.beta
        probs = probs / (probs.sum() + 1e-8)

        # Sample n_collocation points according to probs
        indices = torch.multinomial(probs, self.n_collocation, replacement=True)

        new_collocation = {
            dim: cand[dim][indices].detach().requires_grad_(True)
            for dim in cand
        }

        return new_collocation

    def _compute_residuals(self, model, cand):
        """Compute aggregate PDE residual for each candidate point.

        Subclasses override this for specific PDE systems.
        Returns tensor of shape (n_candidates,).
        """
        raise NotImplementedError("Subclass must implement _compute_residuals")
