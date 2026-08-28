"""MLP backbone for PINN with Fourier feature encoding."""

import torch
import torch.nn as nn
from .fourier_features import FourierFeatureEncoding


class PINN(nn.Module):
    """Physics-Informed Neural Network with optional Fourier features.

    Architecture:
        Input → [Fourier Features] → MLP → [h, qx, qy]

    h uses softplus to ensure positivity (water depth > 0).
    """

    def __init__(
        self,
        in_dim: int = 3,
        hidden_layers: list = None,
        out_dim: int = 3,
        activation: nn.Module = None,
        use_fourier: bool = True,
        fourier_dim: int = 128,
        fourier_sigma: float = 1.0,
    ):
        """
        Args:
            in_dim: input dimension (3 for x, y, t)
            hidden_layers: list of hidden layer widths, e.g. [128,128,128,128]
            out_dim: output dimension (3 for h, qx, qy)
            activation: activation function (default: Tanh)
            use_fourier: whether to use Fourier feature encoding
            fourier_dim: Fourier feature output dimension
            fourier_sigma: Fourier feature frequency scale
        """
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [128, 128, 128, 128]
        if activation is None:
            activation = nn.Tanh()

        self.use_fourier = use_fourier

        if use_fourier:
            self.fourier = FourierFeatureEncoding(in_dim, fourier_dim, fourier_sigma)
            first_dim = fourier_dim
        else:
            first_dim = in_dim

        layers = [first_dim] + hidden_layers + [out_dim]
        self.linears = nn.ModuleList()
        for i in range(len(layers) - 1):
            layer = nn.Linear(layers[i], layers[i + 1])
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            self.linears.append(layer)

        self.activation = activation
        self.n_layers = len(self.linears)

    def forward(self, x, y, t):
        """Forward pass.

        Args:
            x, y, t: input tensors of shape (N, 1), already normalized to [0,1]

        Returns:
            h: water depth (positive via softplus)  shape (N, 1)
            qx: x-discharge  shape (N, 1)
            qy: y-discharge  shape (N, 1)
        """
        inputs = torch.cat([x, y, t], dim=1)

        if self.use_fourier:
            u = self.fourier(inputs)
        else:
            u = inputs

        for i, linear in enumerate(self.linears):
            u = linear(u)
            if i != len(self.linears) - 1:
                u = self.activation(u)

        h = torch.nn.functional.softplus(u[:, 0:1])
        qx = u[:, 1:2]
        qy = u[:, 2:3]
        return h, qx, qy

    def get_last_layer_weights(self):
        """Return last layer weight for gradient-based loss balancing."""
        return self.linears[-1].weight
