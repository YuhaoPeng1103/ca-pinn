"""Server experiment: Burgers equation (strongly time-dependent).

Directly tests the conditional-module hypothesis:
    causal training helps strongly time-dependent problems.

Configs (20,000 epochs each, n_coll=10,000):
  1. burgers_vanilla   — no modules
  2. burgers_causal    — causal (eps 0.1->50 schedule)
  3. burgers_relo      — ReLoBRaLo on PDE/IC/BC (no priors here)
  4. burgers_both      — causal + ReLoBRaLo

Reference: analytical Cole-Hopf solution with 200 Fourier terms.
"""

import sys, os, time, json, math
os.environ['MPLBACKEND'] = 'Agg'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import grad

torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(10)

device = 'cpu'
NU = 0.01 / np.pi

from training.causal import CausalTrainer
from training.loss_balancing import ReLoBRaLo

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)


class BurgersNet(nn.Module):
    """MLP with Fourier features: (x, t) -> u"""

    def __init__(self, fourier_dim=64, hidden=(64, 64, 64, 64)):
        super().__init__()
        B = torch.randn(fourier_dim // 2, 2) * 1.0
        self.register_buffer('B', B)
        layers = [fourier_dim] + list(hidden) + [1]
        self.linears = nn.ModuleList()
        for i in range(len(layers) - 1):
            l = nn.Linear(layers[i], layers[i + 1])
            nn.init.xavier_normal_(l.weight)
            nn.init.zeros_(l.bias)
            self.linears.append(l)
        self.act = nn.Tanh()

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        proj = 2.0 * math.pi * (inputs @ self.B.T)
        u = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
        for i, l in enumerate(self.linears):
            u = l(u)
            if i != len(self.linears) - 1:
                u = self.act(u)
        return u


def burgers_reference(nx=256, nt=100, n_terms=200):
    """Cole-Hopf analytical solution with n_terms Fourier modes."""
    x = np.linspace(-1, 1, nx)
    t = np.linspace(0, 1, nt)
    X, T = np.meshgrid(x, t)
    u = np.zeros_like(X)

    k = np.arange(1, n_terms + 1)[:, None, None]  # (K,1,1)
    # ak * exp(-nu k^2 pi^2 t) * sin(k pi x), summed over k
    ak = 2 * (-1.0) ** (k[:, 0, 0] + 1) / (k[:, 0, 0] * np.pi)  # (K,)
    s = np.zeros_like(X)
    for i in range(0, n_terms, 20):  # batched to save memory
        kk = k[i:i + 20]
        akk = ak[i:i + 20][:, None, None]
        s += (akk * np.exp(-NU * kk ** 2 * np.pi ** 2 * T[None]) *
              np.sin(kk * np.pi * X[None])).sum(axis=0)

    u = 2 * NU * np.pi * s / (1 + s + 1e-15)
    return x, t, u


def train_burgers(name, use_causal, use_relobralo, n_epochs=20000,
                  n_coll=10000, n_data=100):
    print(f"\n{'='*60}")
    print(f"[{name}] {n_epochs} epochs | causal={use_causal} | relobralo={use_relobralo}")
    print(f"{'='*60}", flush=True)

    model = BurgersNet().to(device)
    causal = CausalTrainer(n_chunks=50, epsilon=0.1, t_min=0.0, t_max=1.0,
                           epsilon_max=50.0) if use_causal else None
    balancer = ReLoBRaLo(n_losses=3) if use_relobralo else None

    # IC + BC data
    x_ic = (torch.rand(n_data, 1, device=device) * 2 - 1)
    t_ic = torch.zeros(n_data, 1, device=device)
    u_ic = -torch.sin(np.pi * x_ic)
    t_bc = torch.rand(n_data, 1, device=device)
    x_l = -torch.ones(n_data, 1, device=device)
    x_r = torch.ones(n_data, 1, device=device)

    # Collocation
    x_coll = (torch.rand(n_coll, 1, device=device) * 2 - 1)
    t_coll = torch.rand(n_coll, 1, device=device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=2000, factor=0.5)

    best_loss = float('inf')
    best_state = None
    t0 = time.time()

    for epoch in range(n_epochs):
        if causal is not None:
            causal.set_epoch(epoch, n_epochs)

        x_coll.requires_grad_(True)
        t_coll.requires_grad_(True)

        u = model(x_coll, t_coll)
        u_t = grad(u, t_coll, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = grad(u, x_coll, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = grad(u_x, x_coll, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        residual = u_t + u * u_x - NU * u_xx

        if use_causal:
            loss_pde, _, _ = causal.weighted_pde_loss(residual.abs().squeeze(), t_coll)
        else:
            loss_pde = torch.mean(residual ** 2)

        u_p_ic = model(x_ic, t_ic)
        u_p_l = model(x_l, t_bc)
        u_p_r = model(x_r, t_bc)
        loss_data = torch.mean((u_p_ic - u_ic) ** 2) + torch.mean(u_p_l ** 2) + torch.mean(u_p_r ** 2)
        loss_bc = torch.mean(u_p_l ** 2) + torch.mean(u_p_r ** 2)

        components = [loss_pde, loss_data, loss_bc]
        if use_relobralo:
            _, loss_total = balancer.update_weights(components, model.linears[-1].weight)
        else:
            loss_total = sum(components)

        if torch.isnan(loss_total):
            print(f"NaN at epoch {epoch}", flush=True)
            break

        opt.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        scheduler.step(loss_total.detach())

        if loss_total.item() < best_loss:
            best_loss = loss_total.item()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 1000 == 0:
            eps = f"{causal.epsilon:.1f}" if causal else "-"
            print(f"E{epoch:5d} | Loss={loss_total.item():.3e} eps={eps} "
                  f"t={time.time()-t0:.0f}s", flush=True)

    model.load_state_dict(best_state)
    elapsed = time.time() - t0

    # Evaluate on reference grid
    x_ref, t_ref, u_ref = burgers_reference()
    X_m, T_m = np.meshgrid(x_ref, t_ref)
    model.eval()
    with torch.no_grad():
        xf = torch.tensor(X_m.flatten(), dtype=torch.float32, device=device).view(-1, 1)
        tf = torch.tensor(T_m.flatten(), dtype=torch.float32, device=device).view(-1, 1)
        # Evaluate in batches to limit memory
        u_pred = np.zeros(X_m.size)
        bs = 5000
        for i in range(0, X_m.size, bs):
            u_pred[i:i + bs] = model(xf[i:i + bs], tf[i:i + bs]).cpu().numpy().flatten()
    u_pred = u_pred.reshape(X_m.shape)

    err = np.linalg.norm(u_pred - u_ref) / (np.linalg.norm(u_ref) + 1e-8)

    result = {'name': name, 'err_u': float(err), 'best_loss': float(best_loss),
              'time_s': float(elapsed)}

    out_dir = os.path.join(OUT_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, 'model.pth'))
    np.savez(os.path.join(out_dir, 'results.npz'),
             err=err, u_pred=u_pred, u_ref=u_ref,
             x=x_ref, t=t_ref)
    with open(os.path.join(out_dir, 'result.json'), 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n[{name}] DONE: err_u={err:.4e} time={elapsed/3600:.1f}h", flush=True)
    return result


if __name__ == "__main__":
    results = []
    results.append(train_burgers('burgers_vanilla', False, False))
    results.append(train_burgers('burgers_causal', True, False))
    results.append(train_burgers('burgers_relo', False, True))
    results.append(train_burgers('burgers_both', True, True))

    print("\n" + "=" * 60)
    print("BURGERS SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"{r['name']:>18s}: err_u={r['err_u']:.4e} time={r['time_s']/3600:.1f}h")
    with open(os.path.join(OUT_DIR, 'burgers_summary.json'), 'w') as f:
        json.dump(results, f, indent=2)
