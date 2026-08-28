"""Causal training: exponential time-domain weighting for PINNs.

Core idea: weight PDE loss temporally so that early times must be solved
before later times receive significant weight. This respects the causal
structure of time-dependent PDEs.

Reference: Wang et al. "Respecting causality is all you need for
training physics-informed neural networks", 2022.
"""

import torch


class CausalTrainer:
    """Manages causal time weights for PDE loss."""

    def __init__(
        self,
        n_chunks: int = 72,
        epsilon: float = 0.1,
        t_min: float = 0.0,
        t_max: float = 3600.0,
        epsilon_max: float = None,
    ):
        """
        Args:
            n_chunks: number of time chunks
            epsilon: initial causality strength (larger = stricter causal gate)
            t_min, t_max: time domain bounds
            epsilon_max: if set, epsilon grows linearly from `epsilon` to
                `epsilon_max` over training (call `set_epoch` each epoch).
                Wang et al. 2022 use this schedule to progressively enforce
                causality as training proceeds.
        """
        self.n_chunks = n_chunks
        self.epsilon = epsilon
        self.epsilon_init = epsilon
        self.epsilon_max = epsilon_max
        self.t_min = t_min
        self.t_max = t_max
        self.chunk_edges = torch.linspace(t_min, t_max, n_chunks + 1)
        self.total_epochs = None
        self.reset()

    def set_epoch(self, epoch, total_epochs=None):
        """Update epsilon according to linear schedule.

        epsilon(epoch) = epsilon_init + (epsilon_max - epsilon_init) * epoch/total
        """
        if total_epochs is not None:
            self.total_epochs = total_epochs
        if self.epsilon_max is not None and self.total_epochs is not None:
            frac = min(1.0, epoch / max(self.total_epochs - 1, 1))
            self.epsilon = self.epsilon_init + (self.epsilon_max - self.epsilon_init) * frac

    def reset(self):
        """Reset accumulated losses for all chunks."""
        self.accumulated_loss = torch.zeros(self.n_chunks)
        self.current_weights = torch.ones(self.n_chunks)

    def assign_chunks(self, t):
        """Assign each collocation point to a time chunk.

        Args:
            t: time tensor of shape (N, 1) in physical units

        Returns:
            chunk_ids: LongTensor of shape (N,) with chunk indices [0, n_chunks-1]
        """
        chunk_size = (self.t_max - self.t_min) / self.n_chunks
        t_flat = t.squeeze()
        chunk_ids = ((t_flat - self.t_min) / chunk_size).long().clamp(0, self.n_chunks - 1)
        return chunk_ids

    def compute_residual_by_chunk(self, residual, chunk_ids):
        """Compute mean squared residual per time chunk.

        Args:
            residual: PDE residual tensor of shape (N,)
            chunk_ids: chunk assignment of shape (N,)

        Returns:
            chunk_losses: tensor of shape (n_chunks,) with mean residual per chunk
        """
        residual_sq = residual ** 2
        chunk_losses = torch.zeros(self.n_chunks, device=residual.device)
        for c in range(self.n_chunks):
            mask = (chunk_ids == c)
            if mask.sum() > 0:
                chunk_losses[c] = residual_sq[mask].mean()
        return chunk_losses

    def update_weights(self, chunk_losses):
        """Update causal weights based on current-iteration losses.

        w_i = exp(-epsilon * sum_{k=1}^{i-1} L_k)

        Uses current-iteration losses (not accumulated history) so the
        causal gate opens as earlier chunks are solved.

        Args:
            chunk_losses: mean PDE residual per chunk at current iteration

        Returns:
            weights: causal weights for each chunk (n_chunks,)
        """
        chunk_losses_cpu = chunk_losses.detach().cpu()

        # EMA of chunk losses for monitoring
        if self.accumulated_loss.sum() == 0:
            self.accumulated_loss = chunk_losses_cpu
        else:
            alpha = 0.9
            self.accumulated_loss = (
                alpha * self.accumulated_loss + (1 - alpha) * chunk_losses_cpu
            )

        # Causal weights: use current losses for gate (not accumulated)
        cumsum = torch.zeros(self.n_chunks)
        cumsum[1:] = torch.cumsum(chunk_losses_cpu[:-1], dim=0)

        weights = torch.exp(-self.epsilon * cumsum)
        self.current_weights = weights
        return weights.to(chunk_losses.device)

    def weighted_pde_loss(self, residual, t):
        """Compute causally-weighted PDE loss in one call.

        Args:
            residual: PDE residual tensor of shape (N,)
            t: time tensor of shape (N, 1) in physical units

        Returns:
            weighted_loss: scalar causally-weighted PDE loss
            chunk_losses: per-chunk losses for monitoring (n_chunks,)
            weights: per-chunk weights for monitoring (n_chunks,)
        """
        chunk_ids = self.assign_chunks(t)
        chunk_losses = self.compute_residual_by_chunk(residual, chunk_ids)
        weights = self.update_weights(chunk_losses)
        weighted_loss = (weights * chunk_losses).sum()
        return weighted_loss, chunk_losses, weights

    def get_active_fraction(self):
        """Return fraction of chunks that are 'active' (weight > 0.5)."""
        n_active = (self.current_weights > 0.5).sum().item()
        return n_active / self.n_chunks
