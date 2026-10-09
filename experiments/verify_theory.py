"""Numerically verify the quantitative anchor-bias theory.

Derivation being tested
-----------------------
Each loss k has full-parameter gradient norm G_k = ||grad_theta L_k|| and anchor
gradient norm g_k = ||grad_phi L_k||, with coupling ratio c_k = g_k / G_k.

Reference: the SAME loss with c = 1 (all of its gradient on the anchor, so
G is held fixed and only the coupling changes). Let lambda^0 be the weight the
balancer assigns in that reference case.

Prediction (relative-norm balancer, lambda_k proportional to g_k):
    lambda / lambda^0  =  c / (1 - (1 - c) f),   f = G / (G_A + G)      ... (R)

Prediction (inverse-ratio balancer, lambda = max_j g_j / g_k), with loss A
dominating the maximum in both cases:
    lambda / lambda^0  =  1 / c                                          ... (I)

Construction: loss A depends only on the anchor W (so c_A = 1). Loss B has a
W-part and a mu-part, scaled so that ||grad_theta L_B|| = G_target for EVERY c
and ||grad_W L_B|| = c * G_target. G_target is chosen below G_A so that loss A
attains the maximum, matching the premise of (I).
"""

import torch

torch.manual_seed(0)


def build(c):
    W = torch.nn.Parameter(torch.tensor([[0.7, -0.4], [0.2, 0.9]]))
    mu = torch.nn.Parameter(torch.tensor(1.3))
    x = torch.tensor([[1.0, 0.5]])

    alpha = 10.0                      # loss A dominates
    G_target = 1.0                    # fixed total norm of loss B

    def wpart():
        return ((x @ W.T) ** 2).sum()

    def mupart():
        return (mu - 0.5) ** 2

    gw = float(torch.autograd.grad(wpart(), [W], retain_graph=True)[0]
               .pow(2).sum()) ** 0.5
    gm = float(torch.autograd.grad(mupart(), [mu], retain_graph=True)[0]
               .pow(2).sum()) ** 0.5

    # ||grad_W L_B|| = c*G_target  and  ||grad_theta L_B|| = G_target
    a = c * G_target / gw
    b = G_target * (1 - c ** 2) ** 0.5 / gm

    def lA():
        return alpha * wpart()

    def lB():
        return a * wpart() + b * mupart()

    return lA, lB, W, [W, mu]


def norms(fn, W, params):
    gf = sum(float(g.pow(2).sum()) for g in
             torch.autograd.grad(fn(), params, retain_graph=True,
                                 allow_unused=True) if g is not None) ** 0.5
    ga = float(torch.autograd.grad(fn(), [W], retain_graph=True)[0]
               .pow(2).sum()) ** 0.5
    return ga, gf


def measure(c, kind):
    lA, lB, W, params = build(c)
    gA, GA = norms(lA, W, params)
    gB, GB = norms(lB, W, params)
    c_meas = gB / GB
    f = GB / (GA + GB)

    if kind == 'rel':
        lam = 2 * gB / (gA + gB)
        lam0 = 2 * GB / (GA + GB)
        pred = c_meas / (1 - (1 - c_meas) * f)
    else:
        lam = max(gA, gB) / gB
        lam0 = max(GA, GB) / GB
        pred = 1.0 / c_meas
    return c_meas, lam / lam0, pred


print("=" * 74)
print("(R) relative-norm:  lambda/lambda0 = c / (1 - (1-c) f)")
print("=" * 74)
print(f"{'c (measured)':>14} {'measured':>12} {'predicted':>12} {'rel.err':>10}")
for ct in [1.0, 0.8, 0.5, 0.2, 0.1, 0.02]:
    c, rm, pr = measure(ct, 'rel')
    print(f"{c:>14.4f} {rm:>12.4f} {pr:>12.4f} {abs(rm-pr)/pr*100:>9.2f}%")

print()
print("=" * 74)
print("(I) inverse-ratio:  lambda/lambda0 = 1 / c")
print("=" * 74)
print(f"{'c (measured)':>14} {'measured':>12} {'predicted':>12} {'rel.err':>10}")
for ct in [1.0, 0.8, 0.5, 0.2, 0.1, 0.02]:
    c, rm, pr = measure(ct, 'inv')
    print(f"{c:>14.4f} {rm:>12.4f} {pr:>12.4f} {abs(rm-pr)/pr*100:>9.2f}%")
