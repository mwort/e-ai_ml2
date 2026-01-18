#!/usr/bin/env python
# coding: utf-8

# In[1]:


# (1) FUNCTIONS: Lorenz-63 in torch, observation helpers, neural update (DeepSets-style)

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----- Lorenz-63 in torch -----
class Lorenz63Params:
    def __init__(self, sigma=10.0, rho=28.0, beta=8.0/3.0):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

def lorenz63_rhs(x, p: Lorenz63Params):
    if x.ndim == 1:  # single state (3,)
        dx = p.sigma * (x[1] - x[0])
        dy = x[0] * (p.rho - x[2]) - x[1]
        dz = x[0] * x[1] - p.beta * x[2]
        return torch.tensor([dx, dy, dz])
    else:            # batch of states (N,3)
        dx = p.sigma * (x[:,1] - x[:,0])
        dy = x[:,0] * (p.rho - x[:,2]) - x[:,1]
        dz = x[:,0] * x[:,1] - p.beta * x[:,2]
        return torch.stack([dx, dy, dz], dim=1)


def rk4_step(x, dt, f, params):
    k1 = f(x, params)
    k2 = f(x + 0.5*dt*k1, params)
    k3 = f(x + 0.5*dt*k2, params)
    k4 = f(x + dt*k3, params)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def integrate_l63(x0, params, dt, steps):
    X = [x0]
    x = x0
    for _ in range(steps):
        x = rk4_step(x, dt, lorenz63_rhs, params)
        X.append(x)
    return torch.stack(X)  # (steps+1,3)

# ----- Observation helpers -----
def make_obs_operator(select="xy"):
    if select == "xy":
        H = torch.tensor([[1.,0.,0.],[0.,1.,0.]])
    elif select == "full":
        H = torch.eye(3)
    else:
        H = torch.tensor([[1.,0.,0.],[0.,0.,1.]])  # x and z
    return H

def add_obs_noise(H, X, R_std, rng=torch):
    """
    X: (T,3), H:(m,3)
    returns Y: (T,m)
    """
    m = H.shape[0]
    Y = (H @ X.T).T
    noise = R_std * rng.randn(*Y.shape)
    return Y + noise

# ----- DeepSets neural update -----
class ParticleUpdateNN(nn.Module):
    def __init__(self, d_state=3, d_obs=2, hidden=64):
        super().__init__()
        # phi maps particle+obs -> embedding
        self.phi = nn.Sequential(
            nn.Linear(d_state+d_obs, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        # rho maps pooled embedding -> global context
        self.rho = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        # psi maps (particle+context+obs) -> particle increment
        self.psi = nn.Sequential(
            nn.Linear(d_state+hidden+d_obs, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_state)
        )
    def forward(self, Xb, y):
        """
        Xb: (N,d_state), y:(d_obs,)
        returns Xa: (N,d_state)
        """
        N = Xb.shape[0]
        y_rep = y.expand(N,-1)
        inp = torch.cat([Xb,y_rep], dim=1)  # (N,d_state+d_obs)
        emb = self.phi(inp)                 # (N,hidden)
        pooled = emb.mean(dim=0, keepdim=True)  # (1,hidden)
        context = self.rho(pooled).expand(N,-1) # (N,hidden)
        out_inp = torch.cat([Xb,context,y_rep], dim=1)
        dX = self.psi(out_inp)
        Xa = Xb + dX
        return Xa


# In[2]:


# (2) TRUTH & OBSERVATIONS

torch.manual_seed(1)

params_true = Lorenz63Params()
dt_assim = 0.1
Nt = 150

# truth trajectory
x0_true = torch.tensor([1.0,1.0,20.0])
X_truth = integrate_l63(x0_true, params_true, dt_assim, Nt)  # (Nt+1,3)

# obs operator: observe x,y
H = make_obs_operator("xy")
R_std = 0.1
Y = add_obs_noise(H, X_truth, R_std)

print("Truth:", X_truth.shape, " Obs:", Y.shape)


# In[ ]:


# (3) TRAINING LOOP — background via Lorenz-63 forecast with sigma=11

# model and optimizer
d_state = 3; d_obs = H.shape[0]
net = ParticleUpdateNN(d_state, d_obs, hidden=64)
opt = torch.optim.Adam(net.parameters(), lr=1e-3)

def obs_likelihood(y, Xa, H, R_std):
    """
    y: (m,), Xa: (N,d)  -> returns loglik per particle (N,)
    """
    Yp = (H @ Xa.T).T
    innov = y - Yp
    quad = (innov**2).sum(dim=1) / (R_std**2)
    const = torch.tensor(2*torch.pi*(R_std**2), dtype=Xa.dtype, device=Xa.device)
    loglik = -0.5 * (quad + y.shape[0] * torch.log(const))
    return loglik

# ensemble size and forecast settings
Npart = 20
nsub = 5                 # substeps per assimilation interval (numerical accuracy)
model_std = 0.20         # optional additive model noise after forecast (0.0..0.5)

# approximate/wrong model parameters for forecast
params_model = Lorenz63Params(sigma=11.0, rho=28.0, beta=8/3)

# initial analysis ensemble (biased start, like Gaussian PF)
x0_bias = X_truth[0] + torch.tensor([2.0, -2.0, 3.0], dtype=X_truth.dtype)
Xa_prev = x0_bias.expand(Npart, 3) + 2.0 * torch.randn(Npart, 3, dtype=X_truth.dtype)

losses = []

for epoch in range(501):
    total_loss = 0.0

    # reset ensemble at start of each epoch (optional; comment out if you prefer continuity across epochs)
    Xa_prev = x0_bias.expand(Npart, 3) + 2.0 * torch.randn(Npart, 3, dtype=X_truth.dtype)

    for n in range(Nt):
        # ---- Forecast: propagate Xa_prev -> Xb with wrong model (sigma=11)
        Xb = Xa_prev
        h = dt_assim / nsub
        for _ in range(nsub):
            Xb = rk4_step(Xb, h, lorenz63_rhs, params_model)

        # optional additive model noise to maintain spread
        if model_std > 0:
            Xb = Xb + model_std * torch.randn_like(Xb)

        # ---- Analysis via neural update
        y = Y[n]                     # observation at time n
        Xa = net(Xb, y)              # updated particles

        # ---- Likelihood-based loss (no Xa target)
        loglik = obs_likelihood(y, Xa, H, R_std)
        # negative log average likelihood
        loss = -(torch.logsumexp(loglik, dim=0) - torch.log(torch.tensor(Npart, dtype=X_truth.dtype)))

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item()

        # detach analysis for next cycle to avoid backprop through time
        Xa_prev = Xa.detach()

    losses.append(total_loss / Nt)
    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d}, avg loss {losses[-1]:.3f}")


# In[ ]:


# (4) EVALUATION & VISUALIZATION: forecast-based xb, NN-based xa, errors
# Uses Npart, params_model, nsub, model_std, dt_assim, net, X_truth, Y from above.

import matplotlib.pyplot as plt
import torch

Xb_means = []
Xa_means = []

# initial analysis ensemble (same style as training)
device = X_truth.device
dtype  = X_truth.dtype
x0_bias = X_truth[0] + torch.tensor([2.0, -2.0, 3.0], device=device, dtype=dtype)
Xa_prev = x0_bias.expand(Npart, 3) + 2.0 * torch.randn(Npart, 3, device=device, dtype=dtype)

for n in range(Nt):
    # ---- Forecast: Xa_prev -> Xb with wrong model (sigma=11)
    Xb = Xa_prev
    h = dt_assim / nsub
    for _ in range(nsub):
        Xb = rk4_step(Xb, h, lorenz63_rhs, params_model)

    if model_std > 0:
        Xb = Xb + model_std * torch.randn_like(Xb)

    Xb_means.append(Xb.mean(dim=0))

    # ---- Analysis via neural update
    y = Y[n]                # observation at time n
    Xa = net(Xb, y)
    #Xa = Xb
    Xa_means.append(Xa.mean(dim=0))

    # next cycle
    Xa_prev = Xa.detach()

Xb_means = torch.stack(Xb_means)   # (Nt,3)
Xa_means = torch.stack(Xa_means)   # (Nt,3)
Xt = X_truth[:-1]                  # (Nt,3)

# ---- component time series ----
labels = [r"$x_1$", r"$x_2$", r"$x_3$"]
fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
for i in range(3):
    axs[i].plot(Xt[:, i].detach().cpu().numpy(), 'k-', label="truth")
    axs[i].plot(Xb_means[:, i].detach().cpu().numpy(), 'r--', label="background")
    axs[i].plot(Xa_means[:, i].detach().cpu().numpy(), 'b-', label="analysis")
    axs[i].set_ylabel(labels[i])
    axs[i].grid(alpha=0.3)
    axs[i].legend(loc="upper right")
axs[-1].set_xlabel("time step")
fig.suptitle("Neural PF (torch): Truth vs Background vs Analysis (means)")
plt.tight_layout()
plt.savefig("lorenz63_aipf.png")
plt.show()

# ---- error norms over time ----
err_b = torch.norm(Xb_means - Xt, dim=1)   # ||xb - xt||
err_a = torch.norm(Xa_means - Xt, dim=1)   # ||xa - xt||

plt.figure(figsize=(8, 4))
plt.plot(err_b.detach().cpu().numpy(), 'r--', label="||xb - xt||")
plt.plot(err_a.detach().cpu().numpy(), 'b-', label="||xa - xt||")
plt.xlabel("time step")
plt.ylabel("error norm")
plt.title("Error norms: background vs analysis")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("lorenz53_aipf_error.png")
plt.show()

# ---- average errors (mean over time) ----
avg_err_b = err_b.mean().item()
avg_err_a = err_a.mean().item()
print(f"Average background error: {avg_err_b:.3f}")
print(f"Average analysis error:   {avg_err_a:.3f}")


# In[ ]:


# (5) ENSEMBLE SCATTER: stacked subplots, each with its own legend, incl. xb/xa means

import numpy as np
import matplotlib.pyplot as plt
import torch

# --- settings ---
num_panels = 8
plane = "xy"  # {"xy","xz","yz"}
idx_map = {"xy": (0,1), "xz": (0,2), "yz": (1,2)}
i, j = idx_map[plane]

# pick evenly spaced assimilation indices (exclude 0)
sample_idx = np.linspace(1, Nt-1, num_panels, dtype=int)
sample_idx = np.unique(sample_idx)

# --- capture ensembles ---
device = X_truth.device
dtype  = X_truth.dtype

x0_bias = X_truth[0] + torch.tensor([2.0, -2.0, 3.0], device=device, dtype=dtype)
Xa_prev = x0_bias.expand(Npart, 3) + 2.0 * torch.randn(Npart, 3, device=device, dtype=dtype)

captured = []
for n in range(Nt):
    # Forecast
    Xb = Xa_prev
    h = dt_assim / nsub
    for _ in range(nsub):
        Xb = rk4_step(Xb, h, lorenz63_rhs, params_model)
    if model_std > 0:
        Xb = Xb + model_std * torch.randn_like(Xb)

    y = Y[n]
    Xa = net(Xb, y)

    if n in sample_idx:
        xt = X_truth[n]
        err_b = torch.norm(Xb.mean(dim=0) - xt).item()
        err_a = torch.norm(Xa.mean(dim=0) - xt).item()
        captured.append({
            "n": n,
            "xt": xt.detach().cpu(),
            "Xb": Xb.detach().cpu(),
            "Xa": Xa.detach().cpu(),
            "err_b": err_b,
            "err_a": err_a,
        })

    Xa_prev = Xa.detach()

# --- plotting & saving ---
rows = len(captured)
for snap in captured:
    n = snap["n"]
    xt = snap["xt"].numpy()
    Xb = snap["Xb"].numpy()
    Xa = snap["Xa"].numpy()

    fig, ax = plt.subplots(figsize=(6, 4))

    # ensemble members
    ax.scatter(Xb[:, i], Xb[:, j], s=15, alpha=0.4, c="red", label="forecast ens")
    ax.scatter(Xa[:, i], Xa[:, j], s=15, alpha=0.4, marker="^", c="blue", label="analysis ens")
    ax.scatter([xt[i]], [xt[j]], s=90, marker="*", c="gold", edgecolor="k", label="truth")

    # means
    mu_b = Xb.mean(axis=0); mu_a = Xa.mean(axis=0)
    ax.scatter([mu_b[i]], [mu_b[j]], s=80, edgecolor="k", facecolor="none", label="xb mean")
    ax.scatter([mu_a[i]], [mu_a[j]], s=80, edgecolor="k", facecolor="none", marker="s", label="xa mean")

    ax.set_title(f"step n={n} | ||xb-xt||={snap['err_b']:.2f}, ||xa-xt||={snap['err_a']:.2f}")
    ax.set_xlabel(f"x_{i+1}")
    ax.set_ylabel(f"x_{j+1}")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", frameon=False)

    fig.suptitle(f"Ensemble scatter in $x_{{{i+1}}}$–$x_{{{j+1}}}$ plane", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # save figure with time step in filename
    fname = f"l63_ensemble_scatter_step_{n:04d}.png"
    fig.savefig(fname, dpi=150)
    print(f"Saved {fname}")

    plt.close(fig)  # close to avoid memory buildup

# error table
print("Step |  ||xb-xt||   ||xa-xt||")
for snap in captured:
    print(f"{snap['n']:4d} |  {snap['err_b']:9.3f}  {snap['err_a']:9.3f}")



# In[ ]:




