# Adaptive Loss Balancing for Physics-Informed Neural Networks in Hydrodynamic Inverse Problems

This repository contains the code for the paper *"Adaptive Loss Balancing for Physics-Informed Neural Networks in Hydrodynamic Inverse Problems: Design Principles from Full-Convergence Experiments"*.

## Overview

We study how gradient-based loss balancing (ReLoBRaLo) and causal time-domain weighting interact with **parameter priors** in physics-informed neural network (PINN) inverse problems. The central finding is a design principle: *adaptive balancing must apply only to conflicting multi-task losses (PDE / data / BC), never to parameter priors* — including the prior among the balanced losses drives its weight to zero and re-ill-poses the inverse problem.

## Requirements

- Python 3.9
- PyTorch 2.0
- NumPy < 2.0
- SciPy
- Matplotlib

```bash
pip install torch numpy scipy matplotlib
```

All experiments run on CPU (no GPU required).

## Reproducing the experiments

### 1. Generate the reference solution (finite-volume solver)

The shallow-water reference solution is computed by a Lax–Friedrichs finite-volume solver (`physics/swe_solver.py`):

```bash
python experiments/generate_reference.py
```

This writes the reference fields to `outputs/swe_reference.npz`.

### 2. Main experiments (ablation + sparse-observation)

```bash
python experiments/train_matrix_server.py     # causal / relobralo / both + sparse
python experiments/train_vanilla_server.py    # vanilla baseline (seed 42)
```

### 3. Prior-fixing + baseline comparison

```bash
python experiments/train_extra_server.py      # balance-including-prior + LR annealing
```

### 4. Multi-seed statistics

```bash
python experiments/train_multiseed_server.py  # 4 configs x 3 seeds (mean +/- std)
```

### 5. Generalization benchmarks

```bash
python experiments/train_burgers_server.py    # Burgers equation (Cole-Hopf reference)
python experiments/train_ns_server.py         # Navier-Stokes cylinder wake (CFD data)
```

### 6. Figures

```bash
python make_figures.py
```

## Code structure

```
ca_pinn/
├── physics/          # Governing equations + finite-volume solver
│   ├── swe.py            # SWE residuals + observation sampling
│   ├── swe_solver.py     # Lax-Friedrichs finite-volume reference solver
│   ├── swe_model.py      # SWE PINN with trainable physical parameters
│   ├── burgers.py        # Burgers equation
│   └── ns.py             # Navier-Stokes (vorticity-streamfunction)
├── training/         # Training strategies
│   ├── loss_balancing.py # ReLoBRaLo
│   ├── causal.py         # causal time-domain weighting
│   └── adaptive_sampling.py
├── models/           # Network architectures (Fourier features + MLP)
├── experiments/      # Training scripts
├── utils/            # Metrics and plotting helpers
└── make_figures.py   # Generate paper figures
```

## Note on the reference solution

The original synthetic field (an analytically constructed, physically inconsistent ``h``) has been replaced by a mass-conserving finite-volume solution of the full 2D shallow-water equations. The solver is verified by (i) recovering the Manning normal depth to machine precision in the absence of drainage, and (ii) mass conservation (downstream outflow + integrated drainage sink = upstream inflow) to within 0.1%.
