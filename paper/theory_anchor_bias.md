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

Write M = max_j (c_j G_j) and M_G = max_j G_j. The reference weight is
λ_k⁰ = M_G / G_k, so in general

    ┌──────────────────────────────────────────────┐
    │  λ_k / λ_k⁰  =  M / (c_k · M_G)               │              (4)
    └──────────────────────────────────────────────┘

**A caveat we had to correct during validation.** Equation (4) simplifies to
1/c_k only when the maximum is attained by a *fully coupled* loss, so that
M = M_G. That premise is easy to violate: if every loss is itself only
partially coupled — which is the normal situation when the anchor is a proper
subset of the parameters, since then *no* loss has c = 1 — then M < M_G and
the bias is milder than 1/c_k by exactly the factor M/M_G.

In the special case where all couplings are comparable (c_j ≈ c), (4) collapses
to λ_k/λ_k⁰ ≈ 1/c_k again, recovering the clean form. With a full-parameter
anchor every c_k = 1 by definition, (4) gives λ_k = λ_k⁰ identically: an
inverse-ratio balancer then exhibits *no* anchor bias at all, and its
mis-treatment of a sparse-support loss (Section 6) must be attributed to that
other mechanism rather than to decoupling. As c_k → 0 the weight diverges
regardless of the factor, so an exactly decoupled loss still dominates.

## 4. Summary and interpretation

| balancer | bias  λ_k/λ_k⁰ | limit as c_k → 0 | depends on |
|---|---|---|---|
| relative-norm | c_k · S/S_c | 0 (loss switched off) | c_k and the loss's gradient share |
| inverse-ratio | M / (c_k · M_G) | ∞ (loss dominates) | c_k and the coupling of the dominant loss |

Both are exact and hold for any number of losses; neither requires the other
losses to be fully coupled. Equation (2) is unconditional, while (4) carries
the factor M/M_G noted above.

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

## 7. Validation in a multi-task training loop

`experiments/train_mtl_anchor.py` trains a shared trunk with four task heads
plus an auxiliary regulariser whose coupling to the trunk's last layer is
controlled by `gamma` (0 gives exact decoupling). Both balancers use the same
anchor; only the update rule differs, isolating the failure direction.

Results (median relative error between measured and predicted weight ratio,
over the last 75 % of 300 epochs):

| gamma | c_aux | relative-norm err | inverse-ratio err |
|---|---|---|---|
| 0.00 | 0.000 | loss switched off (w = 0) | loss dominates (w = 2.5e10) |
| 0.02 | 0.06–0.13 | 0.00 % | 0.00 % |
| 0.10 | 0.24–0.32 | 0.00 % | 0.00 % |
| 0.50 | 0.71–0.82 | 0.00 % | 0.00 % |
| 2.00 | 0.95–0.98 | 0.00 % | 0.00 % |
| 10.00 | 0.99–1.00 | 0.00 % | 0.00 % |

Two things are established. First, at gamma = 0 the *same* decoupled loss is
driven to zero by the relative-norm balancer and to 2.5e10 by the inverse-ratio
balancer — opposite, extreme, and silent in both cases. Second, away from exact
decoupling the measured bias agrees with (2) and (4) to within the floating
point reported precision at every coupling tested, including couplings as weak
as c = 0.06.

The gamma = 0 rows have no finite relative error (the prediction is 0 and the
measured value is 0, or the prediction diverges); they are the qualitative
regime and are reported separately rather than as a relative error.

## 8. The wrapper repairs both directions in MTL

`experiments/train_mtl_arb.py` compares, at gamma = 0 (auxiliary exactly
decoupled), the balancer alone against the same balancer wrapped by the
anchor-robust rule. The auxiliary loss is identical across the two arms, so its
value is directly comparable.

| balancer | arm | w_aux | L_aux | mean task loss |
|---|---|---|---|---|
| relative-norm | plain | 6.6e-11 | 0.0152 | 0.0151 |
| relative-norm | + ARB | 1.0 | **0.0128** | **0.0136** |
| inverse-ratio | plain | 1.3e10 | — | **0.507** |
| inverse-ratio | + ARB | 1.0 | — | **0.210** |

Without the wrapper the auxiliary loss is either switched off (relative-norm:
w = 6.6e-11, its value stays at 0.0152 because it is never minimised) or
allowed to dominate (inverse-ratio: w = 1.3e10, and the task loss of 0.507 is
twenty times the 0.027 attained when the auxiliary term is benign). With the
wrapper the weight is neutral in both cases, the auxiliary loss is actually
driven down under the relative-norm balancer (0.0152 -> 0.0128), and the task
loss is not damaged — it improves slightly in the first case and falls by a
factor of 2.4 in the second.

(An additional "fully coupled" arm was run for reference; its auxiliary loss has
a different functional form because gamma enters the loss itself, so its L_aux
is not comparable with the rows above and it is omitted here.)
