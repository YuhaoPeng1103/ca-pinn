"""Resume the multi-seed study after a server restart.

The first run completed vanilla (3 seeds) and causal (3 seeds), which are
already saved in multiseed_summary.json. This script runs the remaining
configs (relobralo, both) and merges them into the summary, then runs the
missing vanilla_30k (seed 42) baseline.
"""

import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_multiseed_server import train_one, OUT_DIR


def run_config(name, use_causal, use_relobralo, seeds=(0, 1, 2)):
    per_seed = []
    for seed in seeds:
        r = train_one(f'{name}_s{seed}', use_causal, use_relobralo, 30000, seed)
        per_seed.append(r)
    summary = {
        'err_h_mean': float(np.mean([r['err_h'] for r in per_seed])),
        'err_h_std': float(np.std([r['err_h'] for r in per_seed])),
        'err_n_mean': float(np.mean([r['err_n'] for r in per_seed])),
        'err_n_std': float(np.std([r['err_n'] for r in per_seed])),
        'err_C_mean': float(np.mean([r['err_C'] for r in per_seed])),
        'err_qx0_mean': float(np.mean([r['err_qx0'] for r in per_seed])),
        'per_seed': per_seed,
    }
    return name, summary


if __name__ == "__main__":
    summary_path = os.path.join(OUT_DIR, 'multiseed_summary.json')
    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
    print(f"已有配置: {list(summary.keys())}", flush=True)

    for name, use_causal, use_relobralo in [('relobralo', False, True),
                                            ('both', True, True)]:
        n, s = run_config(name, use_causal, use_relobralo)
        summary[n] = s
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n[{n}] MEAN: err_h={s['err_h_mean']:.4e}±{s['err_h_std']:.4e} "
              f"err_n={s['err_n_mean']:.4e}±{s['err_n_std']:.4e}", flush=True)

    print("\n" + "=" * 70)
    print("MULTISEED SUMMARY (resume complete)")
    print("=" * 70)
    for name in summary:
        s = summary[name]
        print(f"{name:>12s}: err_h={s['err_h_mean']:.4e}±{s['err_h_std']:.4e} "
              f"err_n={s['err_n_mean']:.4e}±{s['err_n_std']:.4e}")

    # Run the missing vanilla_30k (seed 42) baseline
    print("\n开始 vanilla_30k 补跑", flush=True)
    from train_vanilla_server import train_model
    train_model('vanilla_30k', False, False, 30000)
    print("\n全部完成", flush=True)
