"""Random Fourier Feature encoding for mitigating spectral bias.

Tancik et al. "Fourier Features Let Networks Learn High Frequency
Functions in Low Dimensional Domains", NeurIPS 2020.
"""

import torch
import torch.nn as nn
import math


class FourierFeatureEncoding(nn.Module):
    """Maps low-dim input to high-dim Fourier features via random projection.

    gamma(v) = [cos(2π B v), sin(2π B v)]

    where B ~ N(0, sigma^2) is a fixed (non-trainable) random matrix.
    """

    def __init__(self, in_dim: int, out_dim: int, sigma: float = 1.0):
        """
        Args:
            in_dim: input dimension (e.g., 3 for x,y,t)
            out_dim: number of Fourier features (must be even)
            sigma: standard deviation of random projection matrix.
                   Larger sigma → higher frequency features.
        """
        super().__init__()
        assert out_dim % 2 == 0, "out_dim must be even"
        self.m = out_dim // 2
        self.sigma = sigma
        # Fixed random projection matrix
        B = torch.randn(self.m, in_dim) * sigma
        self.register_buffer('B', B)

    def forward(self, x):
        """Project input through Fourier features.

        Args:
            x: input tensor of shape (N, in_dim), normalized to [0, 1]

        Returns:
            tensor of shape (N, out_dim)
        """
        proj = 2.0 * math.pi * (x @ self.B.T)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
