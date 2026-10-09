# Quantitative theory of anchor bias in gradient-based loss balancing

Draft derivation. Numerically verified to machine precision for K = 2 and K = 4
(see `experiments/verify_theory.py`). To be integrated into the manuscript.

## 1. Setting

Training objective with K terms,
L(θ) = Σ_k λ_k L_k(θ), and a gradient-norm balancer that sets the weights λ
from the per-loss gradient norms measured on an **anchor** φ ⊆ θ.

For each loss define

- full-parameter gradient norm   G_k = ‖∇_θ L_k‖
- anchor gradient norm           g_k = ‖∇_φ L_k‖
- **anchor-coupling ratio**      c_k = g_k / G_k  ∈ [0, 1]

c_k = 1 means the loss's entire gradient lies on the anchor; c_k = 0 means the
loss is invisible to it. Write S = Σ_j G_j and S_c = Σ_j c_j G_j.

**Reference weights.** Let λ_k⁰ denote the weights the balancer assigns to a
hypothetical set of losses that are *identical except fully coupled* — that is,
with the same total gradient norms G_k but with g_k = G_k for every k. λ_k⁰ is
the weight the balancer's designer implicitly expects, since the update rules
are derived under the assumption that the anchor measures each loss faithfully.

## 2. Relative-norm balancers

Update rule (normalised so that Σ_k λ_k = K):

    λ_k = K · g_k / Σ_j g_j .

Substituting g_k = c_k G_k gives the **absolute** weight

    λ_k = K · c_k G_k / S_c .                                              (1)

Dividing by the reference weight λ_k⁰ = K · G_k / S yields the bias factor

    ┌──────────────────────────────────────────────┐
    │  λ_k / λ_k⁰  =  c_k · S / S_c                 │              (2)
    └──────────────────────────────────────────────┘

Two consequences, both non-obvious:

- **The bias is not simply c_k.** The factor S/S_c ≥ 1 (since c_j ≤ 1)
  partially compensates: because the decoupled loss also shrinks the
  normalisation, it is under-weighted by *less* than its coupling ratio
  suggests. With a single decoupled loss of coupling c and gradient share
  f = G_k/S, (2) reads λ_k/λ_k⁰ = c / (1 − (1−c) f).
- **Nevertheless λ_k/λ_k⁰ → 0 as c_k → 0**, so an exactly decoupled loss is
  switched off, in agreement with the binary statement.

## 3. Inverse-ratio balancers

Update rule (learning-rate annealing):

    λ_k = max_j g_j / g_k .

Substituting g_k = c_k G_k,

    λ_k = max_j (c_j G_j) / (c_k G_k) .                                    (3)

If the maximum is attained by a fully coupled loss (c_j = 1, the typical case
when the decoupled loss is the small one), max_j (c_j G_j) = max_j G_j, and
dividing by λ_k⁰ = max_j G_j / G_k gives

    ┌──────────────────────────────────────────────┐
    │  λ_k / λ_k⁰  =  1 / c_k                       │              (4)
    └──────────────────────────────────────────────┘

Remarkably, (4) is **independent of every other loss**: the over-weighting of a
partially coupled loss depends only on its own coupling ratio. As c_k → 0 the
weight diverges, so an exactly decoupled loss dominates the objective.

## 4. Summary and interpretation

| balancer | bias  λ_k/λ_k⁰ | limit as c_k → 0 | depends on |
|---|---|---|---|
| relative-norm | c_k · S/S_c | 0 (loss switched off) | c_k and the loss's gradient share |
| inverse-ratio | 1/c_k | ∞ (loss dominates) | c_k only |

The same coupling ratio c_k therefore produces **opposite and unbounded**
distortions in the two families. Both are silent: neither the objective value
nor any standard diagnostic reveals them.

## 5. Numerical verification

`experiments/verify_theory.py` constructs losses with prescribed coupling and
compares (2) and (4) with the measured weight ratio.

Relative-norm (K = 2), Eq. (2):

| c (measured) | measured λ/λ⁰ | predicted | rel. error |
|---|---|---|---|
| 1.0000 | 1.0000 | 1.0000 | 0.00 % |
| 0.8000 | 0.8084 | 0.8084 | 0.00 % |
| 0.5000 | 0.5133 | 0.5133 | 0.00 % |
| 0.2000 | 0.2086 | 0.2086 | 0.00 % |
| 0.1000 | 0.1049 | 0.1049 | 0.00 % |
| 0.0200 | 0.0211 | 0.0211 | 0.00 % |

Inverse-ratio (K = 2), Eq. (4):

| c (measured) | measured λ/λ⁰ | predicted | rel. error |
|---|---|---|---|
| 1.0000 | 1.0000 | 1.0000 | 0.00 % |
| 0.8000 | 1.2500 | 1.2500 | 0.00 % |
| 0.5000 | 2.0000 | 2.0000 | 0.00 % |
| 0.2000 | 5.0000 | 5.0000 | 0.00 % |
| 0.1000 | 10.0000 | 10.0000 | 0.00 % |
| 0.0200 | 50.0000 | 50.0000 | 0.00 % |

General K (four losses with c = 1.00, 0.70, 0.30, 0.05), Eq. (2):

| loss | c | measured | predicted | rel. error |
|---|---|---|---|---|
| 0 | 1.00 | 1.5634 | 1.5634 | 0.000 % |
| 1 | 0.70 | 1.0944 | 1.0944 | 0.000 % |
| 2 | 0.30 | 0.4690 | 0.4690 | 0.000 % |
| 3 | 0.05 | 0.0782 | 0.0782 | 0.000 % |

## 6. Scope and a related phenomenon

Equations (2) and (4) describe the bias induced by **partial or complete
decoupling from the balancer's own anchor**.

A distinct but related issue arises when the anchor covers all parameters
(φ = θ, as in learning-rate annealing) but a loss's gradient is *sparse* in θ —
supported on a handful of coordinates out of many. Such a loss has a small G_k
for a reason unrelated to the loss being well satisfied, and an inverse-ratio
balancer still assigns it a large weight through (3). This is the situation in
the shallow-water experiment, where the parameter prior touches three
coordinates out of roughly fifty thousand: its coupling to the last layer is
exactly zero, while its coupling to the full parameter vector is one and its
gradient norm is nonetheless tiny.

The two effects therefore reinforce each other in that setting: decoupling
drives λ → 0 under a relative-norm balancer and λ → ∞ under an inverse-ratio
balancer, and sparse support additionally inflates the inverse-ratio weight.
