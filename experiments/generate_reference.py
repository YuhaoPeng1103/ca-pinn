"""Generate the SWE reference solution (finite-volume) and save it.

Replaces the previous analytical synthetic field with a physically consistent
reference obtained by the Lax-Friedrichs finite-volume solver.

Usage:
    python experiments/generate_reference.py
"""

import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.swe_solver import solve_swe, _drain_rate

# Reference parameters (true values for the inverse problem)
TRUE_N = 0.03
TRUE_C = 0.05
TRUE_QX0 = 1.0

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'outputs', 'swe_reference.npz')


def main(nx=100, ny=100, T=10000.0, nt_save=10):
    t0 = time.time()
    x, y, t, h, qx, qy = solve_swe(nx=nx, ny=ny, n=TRUE_N, C=TRUE_C,
                                   qx0=TRUE_QX0, T=T, nt_save=nt_save)
    elapsed = time.time() - t0

    # Mass conservation check at final time
    q_in = qx[-1, :, 0].mean()
    q_out = qx[-1, :, -1].mean()
    X, Y = np.meshgrid(x, y)
    drain = (_drain_rate(TRUE_C, X, Y) * h[-1]).sum() * (100.0 / nx) ** 2
    conserv_err = q_out * 100.0 + drain - q_in * 100.0

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, x=x, y=y, t=t, h=h, qx=qx, qy=qy,
             true_n=TRUE_N, true_C=TRUE_C, true_qx0=TRUE_QX0)
    print(f"\nSaved reference to {OUT}")
    print(f"  solve time = {elapsed:.1f}s, grid = {nx}x{ny}, T = {T}s")
    print(f"  h range at final time = [{h[-1].min():.4f}, {h[-1].max():.4f}]")
    print(f"  q_in={q_in:.4f}, q_out={q_out:.4f}, drain={drain:.1f} m3/s")
    print(f"  mass conservation error = {conserv_err:.2f} m3/s "
          f"({100*abs(conserv_err)/(q_in*100):.3f}% of inflow)")
    # drainage depression relative to the no-drainage normal depth
    h_norm = (TRUE_N * TRUE_QX0 / np.sqrt(0.001)) ** (3.0 / 5.0)
    print(f"  depression vs normal depth = {h_norm - h[-1].min():.4f} m "
          f"({100*(h_norm - h[-1].min())/h_norm:.1f}%)")


if __name__ == "__main__":
    main()
