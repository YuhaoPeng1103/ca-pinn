"""Server experiment: 2D Navier-Stokes cylinder wake (strongly time-dependent).

Double purpose:
  1. Generalization validation of ReLoBRaLo (fixed-prior principle not needed;
     this is a forward problem with no unknown parameters).
  2. Second test of the conditional-module hypothesis: causal training should
     HELP vortex shedding (strongly time-dependent), unlike the quasi-steady SWE.

Data: cylinder_nektar_wake.mat (Raissi et al. 2019 CFD data)
Configs (15,000 epochs each):
  ns_vanilla, ns_causal, ns_relo, ns_both
"""

import sys, os, time, json
os.environ['MPLBACKEND'] = 'Agg'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import grad
import scipy.io as sio

torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(10)

device = 'cpu'
NU = 0.01  # Re = 100

from training.causal import CausalTrainer
from training.loss_balancing import ReLoBRaLo

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)
DATA_PATH = '/home/student01/pngyuo/ca_pinn/cylinder_nektar_wake.mat'


def load_cylinder_data(path=DATA_PATH, n_u=5000):
    """Load CFD data.

    Actual mat structure:
        X_star: (5000, 2)   x, y spatial points
        t:      (200, 1)    time steps (t in [0, ~20])
        U_star: (5000, 2, 200)  u ([:,0,:]) and v ([:,1,:])
        p_star: (5000, 200) pressure
    """
    data = sio.loadmat(path)
    X = data['X_star']   # (5000, 2)
    t_vec = data['t']    # (200, 1)
    U = data['U_star']   # (5000, 2, 200)

    x = X[:, 0:1]  # (5000, 1)
    y = X[:, 1:2]
    u_all = U[:, 0, :]  # (5000, 200)
    v_all = U[:, 1, :]
    t_all = t_vec       # (200, 1)
    nt = t_all.shape[0]
    n_pts = x.shape[0]

    # Subsample training points
    idx = np.random.choice(n_pts, n_u, replace=False)
    x_tr = x[idx]
    y_tr = y[idx]
    u_tr = u_all[idx]  # (n_u, 200)
    v_tr = v_all[idx]

    return (x_tr, y_tr, t_all, u_tr, v_tr,
            x, y, t_all, u_all, v_all, n_pts, nt)


class NSNet(nn.Module):
    """PINN for NS in streamfunction-vorticity form: (x,y,t) -> (psi, p)"""

    def __init__(self, hidden=(64, 64, 64, 64)):
        super().__init__()
        layers = [3] + list(hidden) + [2]
        self.linears = nn.ModuleList()
        for i in range(len(layers) - 1):
            l = nn.Linear(layers[i], layers[i + 1])
            nn.init.xavier_normal_(l.weight)
            nn.init.zeros_(l.bias)
            self.linears.append(l)
        self.act = nn.Tanh()

    def forward(self, x, y, t):
        inputs = torch.cat([x, y, t], dim=1)
        u = inputs
        for i, l in enumerate(self.linears):
            u = l(u)
            if i != len(self.linears) - 1:
                u = self.act(u)
        psi = u[:, 0:1]
        p = u[:, 1:2]
        return psi, p


def train_ns(name, use_causal, use_relobralo, n_epochs=8000, n_coll=8000):
    print(f"\n{'='*60}")
    print(f"[{name}] {n_epochs} epochs | causal={use_causal} | relobralo={use_relobralo}")
    print(f"{'='*60}", flush=True)

    x_tr, y_tr, t_all, u_tr, v_tr, x_full, y_full, t_f, u_all, v_all, n_pts, nt = \
        load_cylinder_data(n_u=3000)

    x_tr = torch.tensor(x_tr, dtype=torch.float32, device=device)
    y_tr = torch.tensor(y_tr, dtype=torch.float32, device=device)
    t_all_t = torch.tensor(t_all, dtype=torch.float32, device=device)
    u_tr_t = torch.tensor(u_tr, dtype=torch.float32, device=device)
    v_tr_t = torch.tensor(v_tr, dtype=torch.float32, device=device)

    model = NSNet().to(device)
    causal = CausalTrainer(n_chunks=50, epsilon=0.1, t_min=0.0, t_max=20.0,
                           epsilon_max=50.0) if use_causal else None
    balancer = ReLoBRaLo(n_losses=3) if use_relobralo else None

    # Collocation: domain bounds from CFD (x in [1,8], y in [-2,2], t in [0,20])
    x_coll = (torch.rand(n_coll, 1, device=device) * 7 + 1)
    y_coll = (torch.rand(n_coll, 1, device=device) * 4 - 2)
    t_coll = torch.rand(n_coll, 1, device=device) * 20

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=1000, factor=0.5)

    best_loss = float('inf')
    best_state = None
    t0 = time.time()

    for epoch in range(n_epochs):
        if causal is not None:
            causal.set_epoch(epoch, n_epochs)

        x_coll.requires_grad_(True)
        y_coll.requires_grad_(True)
        t_coll.requires_grad_(True)

        psi, p = model(x_coll, y_coll, t_coll)

        # Velocity from streamfunction
        u = grad(psi, y_coll, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
        v = -grad(psi, x_coll, grad_outputs=torch.ones_like(psi), create_graph=True)[0]

        # Vorticity w = v_x - u_y
        v_x = grad(v, x_coll, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        u_y = grad(u, y_coll, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        w = v_x - u_y

        w_t = grad(w, t_coll, grad_outputs=torch.ones_like(w), create_graph=True)[0]
        w_x = grad(w, x_coll, grad_outputs=torch.ones_like(w), create_graph=True)[0]
        w_y = grad(w, y_coll, grad_outputs=torch.ones_like(w), create_graph=True)[0]
        w_xx = grad(w_x, x_coll, grad_outputs=torch.ones_like(w_x), create_graph=True)[0]
        w_yy = grad(w_y, y_coll, grad_outputs=torch.ones_like(w_y), create_graph=True)[0]

        residual = w_t + u * w_x + v * w_y - NU * (w_xx + w_yy)

        if use_causal:
            loss_pde, _, _ = causal.weighted_pde_loss(residual.abs().squeeze(), t_coll)
        else:
            loss_pde = torch.mean(residual ** 2)

        # Data loss on (u, v)
        # Sample random times per epoch for efficiency
        nt_samp = 5
        t_idx = np.random.choice(nt, nt_samp, replace=False)
        x_data = x_tr.repeat(nt_samp, 1).requires_grad_(True)
        y_data = y_tr.repeat(nt_samp, 1).requires_grad_(True)
        t_data = t_all_t[t_idx].repeat_interleave(x_tr.shape[0], dim=0)
        u_data = u_tr_t[:, t_idx].t().reshape(-1, 1)
        v_data = v_tr_t[:, t_idx].t().reshape(-1, 1)

        psi_d, _ = model(x_data, y_data, t_data)
        u_pred = grad(psi_d, y_data, grad_outputs=torch.ones_like(psi_d), create_graph=True)[0]
        v_pred = -grad(psi_d, x_data, grad_outputs=torch.ones_like(psi_d), create_graph=True)[0]
        loss_data = torch.mean((u_pred - u_data) ** 2) + torch.mean((v_pred - v_data) ** 2)

        # BC loss: periodic top/bottom approx via zero velocity far field
        loss_bc = torch.tensor(0.0, device=device)

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

    # Evaluate: u,v at final time on all 5000 points
    # IMPORTANT: create fresh detached tensors to avoid graph reuse errors
    # after the training loop's gradient operations
    x_full_t = torch.tensor(np.asarray(x_full, dtype=np.float64).copy(), dtype=torch.float32, device=device)
    y_full_t = torch.tensor(np.asarray(y_full, dtype=np.float64).copy(), dtype=torch.float32, device=device)
    t_final_val = float(t_f[-1, 0])
    t_full_t = torch.full_like(x_full_t, t_final_val)
    model.eval()
    x_eval = x_full_t.clone().detach().requires_grad_(True)
    y_eval = y_full_t.clone().detach().requires_grad_(True)
    t_eval = torch.full_like(x_eval, t_final_val)
    psi_eval, _ = model(x_eval, y_eval, t_eval)
    # Compute BOTH partials in a single backward pass to avoid graph reuse error
    dpsi_dy, dpsi_dx = grad(
        psi_eval, [y_eval, x_eval],
        grad_outputs=torch.ones_like(psi_eval), create_graph=False)
    u_f = dpsi_dy
    v_f = -dpsi_dx
    u_pred = u_f.detach().cpu().numpy().flatten()
    v_pred = v_f.detach().cpu().numpy().flatten()

    u_true = u_all[:, -1]
    v_true = v_all[:, -1]
    err_u = np.linalg.norm(u_pred - u_true) / (np.linalg.norm(u_true) + 1e-8)
    err_v = np.linalg.norm(v_pred - v_true) / (np.linalg.norm(v_true) + 1e-8)

    result = {'name': name, 'err_u': float(err_u), 'err_v': float(err_v),
              'best_loss': float(best_loss), 'time_s': float(elapsed)}

    out_dir = os.path.join(OUT_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, 'model.pth'))
    np.savez(os.path.join(out_dir, 'results.npz'),
             err_u=err_u, err_v=err_v, u_pred=u_pred, v_pred=v_pred,
             u_true=u_true, v_true=v_true)
    with open(os.path.join(out_dir, 'result.json'), 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n[{name}] DONE: err_u={err_u:.4e} err_v={err_v:.4e} "
          f"time={elapsed/3600:.1f}h", flush=True)
    return result


if __name__ == "__main__":
    results = []
    results.append(train_ns('ns_vanilla', False, False))
    results.append(train_ns('ns_causal', True, False))
    results.append(train_ns('ns_relo', False, True))
    results.append(train_ns('ns_both', True, True))

    print("\n" + "=" * 60)
    print("NS SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"{r['name']:>14s}: err_u={r['err_u']:.4e} err_v={r['err_v']:.4e} "
              f"time={r['time_s']/3600:.1f}h")
    with open(os.path.join(OUT_DIR, 'ns_summary.json'), 'w') as f:
        json.dump(results, f, indent=2)
