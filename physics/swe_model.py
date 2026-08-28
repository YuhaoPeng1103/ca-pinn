"""SWE-specific PINN model with trainable physics parameters.

Extends the base PINN with Manning's n, drainage C, and inflow qx0
as bounded trainable parameters for the inverse problem.
"""

import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.pinn_net import PINN


class SWE_PINN(PINN):
    """PINN for 2D Shallow Water Equations inverse problem.

    Inherits full PINN architecture and adds trainable physics parameters:
    - Manning's n (roughness coefficient)
    - C_drain (drainage coefficient)
    - qx0 (inflow discharge)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Raw parameters (transformed via sigmoid for boundedness)
        self.n_raw = nn.Parameter(torch.tensor(0.3, dtype=torch.float32))
        self.C_raw = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.qx0_raw = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    @property
    def n(self):
        """Manning's n, bounded to [0.025, 0.045]."""
        return 0.025 + 0.02 * torch.sigmoid(self.n_raw)

    @property
    def C_drain(self):
        """Drainage coefficient, bounded to [0.02, 0.12]."""
        return 0.02 + 0.10 * torch.sigmoid(self.C_raw)

    @property
    def qx0(self):
        """Inflow discharge, bounded to [0.5, 2.0]."""
        return 0.5 + 1.5 * torch.sigmoid(self.qx0_raw)
