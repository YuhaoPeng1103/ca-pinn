"""ReLoBRaLo: Relative Loss Balancing with Random Lookback.

Automatically balances multiple loss terms (PDE, data, BC, prior) by
tracking gradient statistics and applying exponential moving average
weights with random exploration.

Reference: Bischof & Kraus, "Multi-Objective Loss Balancing for
Physics-Informed Deep Learning", 2021.
"""

import torch
import torch.nn as nn


class ReLoBRaLo:
    """Dynamic loss balancing using gradient statistics."""

    def __init__(
        self,
        n_losses: int,
        alpha: float = 0.9,
        temperature: float = 1.0,
        random_sigma: float = 0.01,
    ):
        """
        Args:
            n_losses: number of loss terms to balance
            alpha: EMA smoothing factor (higher = smoother)
            temperature: sharpness of relative weighting
            random_sigma: std of log-normal noise for exploration
        """
        self.n_losses = n_losses
        self.alpha = alpha
        self.temperature = temperature
        self.random_sigma = random_sigma
        # Exponential moving average of gradient norms
        self.ema_grad_norms = None

    def compute_grad_norms(self, losses, last_layer_weights):
        """Compute gradient norm of each loss w.r.t. last layer weights.

        Args:
            losses: list of scalar loss tensors [L1, L2, ..., Lk]
            last_layer_weights: nn.Parameter (weight matrix of last layer)

        Returns:
            grad_norms: tensor of shape (n_losses,)
        """
        grad_norms = torch.zeros(len(losses), device=last_layer_weights.device)
        for i, loss in enumerate(losses):
            if loss.item() == 0.0:
                grad_norms[i] = 0.0
                continue
            grads = torch.autograd.grad(
                loss, last_layer_weights, retain_graph=True, create_graph=False,
                allow_unused=True
            )[0]
            if grads is None:
                grad_norms[i] = 0.0
                continue
            grad_norms[i] = grads.norm()

        # Avoid division by zero
        grad_norms = grad_norms.clamp(min=1e-8)
        return grad_norms

    def update_weights(self, losses, last_layer_weights):
        """Update per-loss weights using ReLoBRaLo.

        Args:
            losses: list of scalar loss tensors [L1, ..., Lk]
            last_layer_weights: weight parameter from last linear layer

        Returns:
            weights: tensor of shape (n_losses,) with updated balance weights
            balanced_loss: scalar = sum(weight_i * loss_i)
        """
        if self.ema_grad_norms is None:
            self.ema_grad_norms = torch.ones(self.n_losses, device=last_layer_weights.device)

        grad_norms = self.compute_grad_norms(losses, last_layer_weights)
        max_grad = grad_norms.max()

        # Relative gradient norms
        rel_norms = grad_norms / (max_grad + 1e-8)

        # EMA update
        self.ema_grad_norms = (
            self.alpha * self.ema_grad_norms + (1 - self.alpha) * rel_norms
        )

        # Apply temperature scaling and normalize
        lambda_hat = self.ema_grad_norms ** (1.0 / self.temperature)
        lambda_hat = lambda_hat / (lambda_hat.sum() + 1e-8) * self.n_losses

        # Random perturbation
        noise = torch.randn(self.n_losses, device=lambda_hat.device) * self.random_sigma
        lambda_final = lambda_hat * torch.exp(noise)

        # Compute balanced loss
        balanced_loss = sum(w * loss for w, loss in zip(lambda_final, losses))

        return lambda_final.detach(), balanced_loss
