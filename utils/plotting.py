"""Publication-quality plotting for CA-PINN experiments.

Uses matplotlib with consistent styling suitable for journal figures.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


# NeurIPS-friendly style
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
    'mathtext.fontset': 'stix',
})


def plot_training_curves(histories, labels, colors, save_path):
    """Plot loss and parameter convergence curves for multiple runs.

    Args:
        histories: list of dicts with keys 'loss_total', 'loss_pde',
                   'loss_data', 'n', 'C', 'qx0'
        labels: list of strings for legend
        colors: list of color strings
        save_path: output file path
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Parameter convergence
    for ax, param, true_val, ylabel in [
        (axes[0, 0], 'n', 0.03, "Manning's n"),
        (axes[0, 1], 'C', 0.5, 'Drain coefficient C'),
        (axes[0, 2], 'qx0', 1.0, 'Inflow $q_{x0}$ (m$^2$/s)'),
    ]:
        for hist, label, color in zip(histories, labels, colors):
            ax.plot(hist[param], color=color, linewidth=1.5, label=label, alpha=0.8)
        ax.axhline(y=true_val, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel} convergence (true={true_val})')
        ax.legend()
        ax.grid(True, alpha=0.2)

    # Row 2: Loss curves
    ax_loss = axes[1, 0]
    for hist, label, color in zip(histories, labels, colors):
        ax_loss.semilogy(hist['loss_total'], color=color, linewidth=1.5, label=label, alpha=0.8)
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Total Loss')
    ax_loss.set_title('Training Loss')
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.2)

    # PDE vs Data loss for first history
    ax_detail = axes[1, 1]
    hist0 = histories[0]
    ax_detail.semilogy(hist0['loss_pde'], color='#2196F3', linewidth=1, alpha=0.7, label='PDE')
    ax_detail.semilogy(hist0['loss_data'], color='#FF5722', linewidth=1, alpha=0.7, label='Data')
    if 'loss_bc' in hist0:
        ax_detail.semilogy(hist0['loss_bc'], color='#4CAF50', linewidth=1, alpha=0.7, label='BC')
    ax_detail.set_xlabel('Epoch')
    ax_detail.set_ylabel('Loss')
    ax_detail.set_title(f'{labels[0]}: Loss Components')
    ax_detail.legend()
    ax_detail.grid(True, alpha=0.2)

    # Active fraction (causal) or placeholder
    ax_af = axes[1, 2]
    if 'active_fraction' in hist0 and len(hist0['active_fraction']) > 0:
        af = hist0['active_fraction']
        ax_af.plot(af, color='#9C27B0', linewidth=1.5)
        ax_af.set_xlabel('Epoch')
        ax_af.set_ylabel('Active time fraction')
        ax_af.set_title('Causal Gate Progress')
        ax_af.set_ylim(0, 1.05)
    else:
        ax_af.text(0.5, 0.5, 'No causal training', ha='center', va='center',
                   transform=ax_af.transAxes, fontsize=12)
    ax_af.grid(True, alpha=0.2)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Training curves saved to {save_path}")


def plot_water_depth_comparison(model, x_grid, y_grid, t_snapshots, save_path):
    """Plot water depth contours at multiple time snapshots.

    Args:
        model: trained PINN model (on correct device)
        x_grid, y_grid: 1D spatial coordinate arrays
        t_snapshots: list of time values to plot
        save_path: output file path
    """
    import torch
    device = next(model.parameters()).device
    n_t = len(t_snapshots)
    n_cols = min(n_t, 4)
    n_rows = (n_t + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows * n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')

    model.eval()
    with torch.no_grad():
        for idx, tt in enumerate(t_snapshots):
            ax = axes[idx]
            xf = torch.tensor(X.flatten(), dtype=torch.float32, device=device).view(-1, 1)
            yf = torch.tensor(Y.flatten(), dtype=torch.float32, device=device).view(-1, 1)
            tf = torch.ones_like(xf) * tt
            h_pred, _, _ = model(xf / 100.0, yf / 100.0, tf / 3600.0)
            H = h_pred.cpu().numpy().reshape(X.shape)

            im = ax.contourf(X, Y, H, 20, cmap='viridis')
            ax.plot(50, 50, 'r*', markersize=12, label='Drain')
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.set_title(f't = {tt:.0f} s')
            ax.legend(loc='upper right')
            plt.colorbar(im, ax=ax, label='h (m)')

    # Hide unused axes
    for idx in range(len(t_snapshots), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Water depth comparison saved to {save_path}")


def plot_ablation_summary(results, save_path):
    """Plot ablation study results as a grouped bar chart.

    Args:
        results: list of dicts from run_ablation
        save_path: output file path
    """
    names = [r['name'] for r in results]
    err_h = [r['err_h'] for r in results]
    err_n = [r['err_n'] for r in results]
    err_C = [r['err_C'] for r in results]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    bars1 = ax.bar(x - width, err_h, width, label='Rel. L2 error (h)', color='#2196F3')
    bars2 = ax.bar(x, err_n, width, label='Rel. error (n)', color='#FF5722')
    bars3 = ax.bar(x + width, err_C, width, label='Rel. error (C)', color='#4CAF50')

    ax.set_xlabel('Configuration')
    ax.set_ylabel('Relative Error')
    ax.set_title('Ablation Study: Effect of CA-PINN Modules')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Ablation summary saved to {save_path}")


def plot_spectral_analysis(model, save_path):
    """Plot Fourier spectrum of model predictions to check spectral bias.

    Takes a 1D profile along y=50 at t=1800s and plots its FFT.
    """
    import torch
    device = next(model.parameters()).device

    x_line = torch.linspace(0, 100, 256, device=device).reshape(-1, 1)
    y_fixed = torch.ones_like(x_line) * 50
    t_fixed = torch.ones_like(x_line) * 1800

    model.eval()
    with torch.no_grad():
        h_pred, _, _ = model(x_line / 100.0, y_fixed / 100.0, t_fixed / 3600.0)
    h_np = h_pred.cpu().numpy().flatten()

    # FFT
    fft = np.abs(np.fft.rfft(h_np))
    freqs = np.fft.rfftfreq(len(h_np), d=100.0 / 256.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(np.linspace(0, 100, 256), h_np, 'b-', linewidth=1.5)
    ax1.axvline(x=50, color='r', linestyle='--', alpha=0.5, label='Drain')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('h (m)')
    ax1.set_title('Water depth along y=50 at t=1800s')
    ax1.legend()
    ax1.grid(True, alpha=0.2)

    ax2.semilogy(freqs[1:], fft[1:], 'b-', linewidth=1.5)
    ax2.set_xlabel('Spatial frequency (1/m)')
    ax2.set_ylabel('|FFT|')
    ax2.set_title('Fourier spectrum of h(x)')
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Spectral analysis saved to {save_path}")
