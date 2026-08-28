"""Run all experiments: SWE main, ablation, Burgers, and generate figures.

Usage:
    python run_all.py              # Run all experiments
    python run_all.py --swe-only   # Run only SWE main experiment
    python run_all.py --quick      # Quick test with reduced epochs
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='CA-PINN Experiment Suite')
    parser.add_argument('--swe-only', action='store_true', help='Run only SWE main')
    parser.add_argument('--ablation-only', action='store_true', help='Run only ablation')
    parser.add_argument('--quick', action='store_true', help='Quick test with fewer epochs')
    args = parser.parse_args()

    quick = args.quick
    n_epochs_main = 3000 if quick else 30000
    n_epochs_abl = 2000 if quick else 15000

    print("=" * 70)
    print("CA-PINN Experiment Suite")
    print(f"Mode: {'QUICK (reduced epochs)' if quick else 'FULL'}")
    print("=" * 70)

    if not args.ablation_only:
        print("\n" + "=" * 70)
        print("EXPERIMENT 1: SWE Main (CA-PINN vs Vanilla)")
        print("=" * 70)

        from experiments.run_swe import train_ca_pinn, evaluate_model
        config = {
            'use_causal': True, 'use_relobralo': True,
            'use_rar': True, 'use_fourier': True,
            'n_epochs': n_epochs_main, 'n_collocation': 30000,
        }
        model, history = train_ca_pinn(config)
        errors, fields = evaluate_model(model)

        out_dir = 'outputs/ca_pinn_swe'
        os.makedirs(out_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(out_dir, 'model.pth'))
        np.savez(os.path.join(out_dir, 'results.npz'),
                 err_h=errors['err_h'], err_qx=errors['err_qx'], err_qy=errors['err_qy'],
                 **{k: np.array(v) for k, v in history.items() if k != 'loss_weights'},
                 **fields)
        print(f"SWE results saved to {out_dir}/")

        # Generate figures
        from utils.plotting import plot_training_curves, plot_water_depth_comparison
        plot_training_curves([history], ['CA-PINN'], ['#2196F3'],
                             os.path.join(out_dir, 'figs', 'training_curves.pdf'))

    if not args.swe_only:
        print("\n" + "=" * 70)
        print("EXPERIMENT 2: Ablation Study")
        print("=" * 70)

        from experiments.run_ablation import run_full_ablation
        results = run_full_ablation()

        from utils.plotting import plot_ablation_summary
        plot_ablation_summary(results, 'outputs/ablation/figs/ablation_summary.pdf')

        print("\n" + "=" * 70)
        print("EXPERIMENT 3: Burgers Generalization")
        print("=" * 70)

        from experiments.run_burgers import train_burgers
        err, _ = train_burgers(use_causal=True, use_relobralo=True, use_rar=True,
                               n_epochs=n_epochs_main)
        print(f"Burgers L2 error: {err:.4e}")

    print("\n" + "=" * 70)
    print("All experiments completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
