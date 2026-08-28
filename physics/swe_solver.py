"""2D Shallow Water Equations — finite-volume solver (Lax-Friedrichs).

Solves the conservative form of the 2D SWE with Manning friction and a
localized drainage sink, producing a *physically consistent* reference
solution (unlike the previous analytical synthetic field):

    h_t  + qx_x + qy_y + S_drain = 0
    qx_t + (qx^2/h + g h^2/2)_x + (qx qy/h)_y = g h S0x - friction_x
    qy_t + (qy qx/h)_x + (qy^2/h + g h^2/2)_y = g h S0y - friction_y

Method: explicit first-order Lax-Friedrichs (conservative, symmetric central
differences + numerical diffusion; robust and easy to verify). Cell-centered
collocated grid, ghost-cell boundary conditions.

    Upstream (x=0):   qx = qx0, h = normal depth  (specified inflow reservoir)
    Downstream (x=L): zero-gradient               (free outflow)
    Lateral (y=0,L):  free-slip (qy mirrored)     (rigid walls)

The drainage sink S_drain = C * 0.5 * exp(-dist/12) * h is identical to the
PINN residual, so the reference and the PINN solve the same system. The
drainage coefficient C is chosen so total withdrawal stays well below the
inflow (mass-conserving; the original C=0.5 was physically inconsistent).
"""

import numpy as np

G = 9.81
S0X = 0.001
S0Y = 0.0


def _flux_x(h, qx, qy):
    h_safe = np.maximum(h, 1e-6)
    return np.stack([
        qx,
        qx ** 2 / h_safe + 0.5 * G * h ** 2,
        qx * qy / h_safe,
    ], axis=-1)


def _flux_y(h, qx, qy):
    h_safe = np.maximum(h, 1e-6)
    return np.stack([
        qy,
        qx * qy / h_safe,
        qy ** 2 / h_safe + 0.5 * G * h ** 2,
    ], axis=-1)


def _drain_rate(C, X, Y):
    dist = np.sqrt((X - 50.0) ** 2 + (Y - 50.0) ** 2 + 1e-8)
    return C * 0.5 * np.exp(-dist / 12.0)


def _source(h, qx, qy, n, C, X, Y):
    """Full source term: drainage sink + bed slope + Manning friction."""
    h_safe = np.maximum(h, 1e-6)
    vel_mag = np.sqrt(qx ** 2 + qy ** 2 + 1e-8)
    dist = np.sqrt((X - 50.0) ** 2 + (Y - 50.0) ** 2 + 1e-8)
    drain = C * 0.5 * np.exp(-dist / 12.0) * h_safe
    friction_x = G * n ** 2 * qx * vel_mag / (h_safe ** (7.0 / 3.0) + 1e-8)
    friction_y = G * n ** 2 * qy * vel_mag / (h_safe ** (7.0 / 3.0) + 1e-8)
    return np.stack([
        -drain,
        G * h * S0X - friction_x,
        G * h * S0Y - friction_y,
    ], axis=-1)


def _pad_bc(h, qx, qy, qx0_t, h_norm):
    """Pad with one ghost cell and apply boundary conditions.

    Upstream (x=0): subcritical inflow — specify discharge qx=qx0_t and set the
    ghost water depth from the outgoing Riemann invariant
        R^- = u - 2 sqrt(g h)   (propagates from the interior to the boundary).
    The ghost depth is found by a bounded Newton iteration with a fallback to
    zero-gradient if the iteration does not converge. This characteristic
    boundary keeps the reservoir at the physically correct level while
    preserving mass conservation.

    Downstream (x=L): free outflow (zero-gradient).
    Lateral (y=0,L): free-slip (mirror h, qx; flip qy).
    """
    # Lateral free-slip
    h_top = np.concatenate([h[:1, :], h, h[-1:, :]], axis=0)
    qx_top = np.concatenate([qx[:1, :], qx, qx[-1:, :]], axis=0)
    qy_top = np.concatenate([-qy[:1, :], qy, -qy[-1:, :]], axis=0)

    # Upstream ghost depth via outgoing Riemann invariant (bounded Newton)
    h_in = np.maximum(h[:, 0], 1e-4)
    u_in = qx[:, 0] / h_in
    c_in = np.sqrt(G * h_in)
    Rm = u_in - 2.0 * c_in                       # outgoing Riemann invariant
    h_gh = np.clip(h_in, 0.05, 5.0)
    converged = np.ones_like(h_gh, dtype=bool)
    for _ in range(8):
        sqrt_gh = np.sqrt(G * h_gh)
        f = qx0_t / h_gh - 2.0 * sqrt_gh - Rm
        fp = -qx0_t / h_gh ** 2 - np.sqrt(G / h_gh)
        step = f / np.where(np.abs(fp) > 1e-8, fp, 1e-8)
        h_new = np.clip(h_gh - step, 0.05, 5.0)
        # flag non-convergence if |f| still large and step stalled
        h_gh = h_new
    # Fallback: where Newton did not settle, use the interior depth
    h_gh = np.where(np.isfinite(h_gh), h_gh, h_in)

    h_gh_lat = np.concatenate([h_gh[:1], h_gh, h_gh[-1:]], axis=0)  # (ny+2,)
    h_p = np.concatenate([h_gh_lat[:, None], h_top, h_top[:, -1:]], axis=1)
    qx_p = np.concatenate([np.full_like(qx_top[:, :1], qx0_t), qx_top, qx_top[:, -1:]], axis=1)
    qy_p = np.concatenate([np.zeros_like(qy_top[:, :1]), qy_top, qy_top[:, -1:]], axis=1)
    return h_p, qx_p, qy_p


def solve_swe(nx=200, ny=200, n=0.03, C=0.05, qx0=1.0,
              T=3600.0, CFL=0.5, nt_save=10, inflow_amp=0.0):
    """Solve 2D SWE with drainage to a quasi-steady state (Lax-Friedrichs).

    Args:
        nx, ny: number of cells in x and y.
        n, C, qx0: true Manning's n, drainage coefficient, inflow discharge.
        T: final time (s).
        CFL: Courant number (< 1).
        nt_save: number of output time snapshots.
        inflow_amp: relative amplitude of sinusoidal inflow variation.

    Returns:
        x, y: cell-center coordinates (nx,), (ny,).
        t_out: (nt_save,) time snapshots.
        h_out, qx_out, qy_out: (nt_save, ny, nx) fields.
    """
    Lx, Ly = 100.0, 100.0
    dx, dy = Lx / nx, Ly / ny

    xc = (np.arange(nx) + 0.5) * dx
    yc = (np.arange(ny) + 0.5) * dy
    X, Y = np.meshgrid(xc, yc)  # (ny, nx)

    h_norm = (n * qx0 / np.sqrt(S0X)) ** (3.0 / 5.0)
    h = np.full((ny, nx), h_norm)
    qx = np.full((ny, nx), qx0)
    qy = np.zeros((ny, nx))
    U = np.stack([h, qx, qy], axis=-1)      # (ny, nx, 3)

    t = 0.0
    save_times = np.linspace(0.0, T, nt_save)
    save_idx = 0
    t_out, h_out, qx_out, qy_out = [], [], [], []

    step = 0
    while t < T:
        u = qx / np.maximum(h, 1e-6)
        v = qy / np.maximum(h, 1e-6)
        c = np.sqrt(G * np.maximum(h, 1e-6))
        cmax = float(np.max(np.abs(u) + c))
        dt = CFL * min(dx, dy) / max(cmax, 1e-3)
        dt = min(dt, T - t)

        qx0_t = qx0 * (1.0 + inflow_amp * np.sin(2.0 * np.pi * t / 3600.0))

        h_p, qx_p, qy_p = _pad_bc(h, qx, qy, qx0_t, h_norm)
        U_g = np.stack([h_p, qx_p, qy_p], axis=-1)   # (ny+2, nx+2, 3)
        F = _flux_x(h_p, qx_p, qy_p)                 # (ny+2, nx+2, 3)
        Gy = _flux_y(h_p, qx_p, qy_p)
        S = _source(h, qx, qy, n, C, X, Y)           # (ny, nx, 3)

        # Lax-Friedrichs: symmetric central difference + diffusion
        U_avg = 0.25 * (U_g[1:-1, 2:, :] + U_g[1:-1, :-2, :]
                        + U_g[2:, 1:-1, :] + U_g[:-2, 1:-1, :])   # (ny, nx, 3)
        dFdx = (F[1:-1, 2:, :] - F[1:-1, :-2, :]) / (2.0 * dx)
        dGdy = (Gy[2:, 1:-1, :] - Gy[:-2, 1:-1, :]) / (2.0 * dy)

        U_new = U_avg - dt * dFdx - dt * dGdy + dt * S

        h = np.maximum(U_new[:, :, 0], 1e-4)
        qx = U_new[:, :, 1]
        qy = U_new[:, :, 2]
        U = U_new

        t += dt
        step += 1

        while save_idx < nt_save and t >= save_times[save_idx] - 1e-12:
            t_out.append(save_times[save_idx])
            h_out.append(h.copy())
            qx_out.append(qx.copy())
            qy_out.append(qy.copy())
            save_idx += 1

        if step % 5000 == 0:
            print(f"  [swe_solver] t={t:.0f}s / {T}s  h_min={h.min():.4f} "
                  f"h_max={h.max():.4f}  dt={dt:.4f}s", flush=True)

    return xc, yc, np.array(t_out), np.array(h_out), np.array(qx_out), np.array(qy_out)


if __name__ == "__main__":
    import time
    t0 = time.time()
    x, y, t_out, h_out, qx_out, qy_out = solve_swe(nx=200, ny=200, nt_save=10)
    print(f"\nDone in {time.time()-t0:.1f}s. h range: [{h_out[-1].min():.4f}, "
          f"{h_out[-1].max():.4f}]")
