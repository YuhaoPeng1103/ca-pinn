"""Evaluation metrics for PINN experiments."""

import numpy as np


def relative_l2_error(pred, true, eps=1e-8):
    """Compute relative L2 error: ||pred - true||_2 / ||true||_2."""
    return np.linalg.norm(pred - true) / (np.linalg.norm(true) + eps)


def relative_l1_error(pred, true, eps=1e-8):
    """Compute relative L1 error: ||pred - true||_1 / ||true||_1."""
    return np.linalg.norm((pred - true).flatten(), 1) / (np.linalg.norm(true.flatten(), 1) + eps)


def max_absolute_error(pred, true):
    """Compute maximum absolute error."""
    return np.max(np.abs(pred - true))


def relative_parameter_error(pred_val, true_val):
    """Compute relative error of a scalar parameter."""
    return abs(pred_val - true_val) / (abs(true_val) + 1e-8)


def convergence_rate(loss_history, window=1000):
    """Estimate convergence rate from loss history.

    Returns exponent alpha such that loss ~ epoch^(-alpha).
    """
    if len(loss_history) < 2 * window:
        return 0.0

    # Use last portion of training
    x = np.arange(window)
    y = np.array(loss_history[-window:])
    y = np.maximum(y, 1e-15)

    # Log-log linear fit
    coeffs = np.polyfit(np.log(x + 1), np.log(y), 1)
    return -coeffs[0]
