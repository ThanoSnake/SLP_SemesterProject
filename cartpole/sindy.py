"""
sindy.py — SINDYc (Sparse Identification of Nonlinear Dynamics with control) on the
VAE latent's physical dimensions, as a lightweight, interpretable alternative to the LSTM.

Idea: if Principles 1/3 succeed, the supervised latent dims (z[:, :N_SUP]) ARE the
physical state (x, x_dot, theta, theta_dot). A black-box LSTM is then overkill — the
dynamics should be expressible as a SHORT, readable difference equation:

    z_{t+1} = z_t + Theta(z_t, u_t) @ Xi

where Theta is a library of candidate functions (polynomials, trig, control couplings)
and Xi is a SPARSE coefficient matrix found by sequentially thresholded least squares.

"c" (control): CartPole is a controlled system, so the action enters the library as an
exogenous input u_t. Without it, SINDy would attribute the action's effect to the wrong
state terms.

This is both:
  * a predictive benchmark (same per-horizon physical MSE the LSTM is judged on), and
  * an interpretability probe (the recovered equations are printed and can be read).

The fitted model exposes the SAME interface as LatentPredictor (init_hidden / step), so
it drops straight into the existing free_run / test_p*.py evaluation harness.

Note: equations are in the STANDARDIZED latent space (the space the VAE was supervised in).
Multiply through by the state std to recover physical units.
"""
import os
import copy
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from loader import list_npz, precompute_latents, LatentSequenceDataset, load_norm_stats
from vae import VAE
from vae_p1 import VAE_P1
from lstm import LatentPredictor

#
#  Config
#
DATA_ROOT = "<cartpole-dataset>"
NORM_STATS = os.path.join(DATA_ROOT, "norm_stats.npz")
LATENT_ROOT = "/kaggle/working/cartpole_sindy_latents"
SAVE_DIR = "/kaggle/working/cartpole_sindy"

LATENT_SIZE = 64
SHIFT = 0

N_SUP = 4
N_IMG = LATENT_SIZE - N_SUP
N_ACTIONS = 2
HIDDEN = 64          # LSTM hidden size (must match the trained checkpoints)
LAYERS = 2           # LSTM layer count (must match the trained checkpoints)
STATE_NAMES = ["x", "x_dot", "theta", "theta_dot"]
ANGLE_DIMS = (2,)   # which physical dims are angles (-> get sin/cos features)

# Latents are encoded fresh for every model. For each we compare two dynamics models on the
# SAME clean latents: the trained LSTM vs SINDy. Question: does Principle 1's structure make
# the latent dynamics sparsely-identifiable (SINDy) and does it match/beat the black-box LSTM?
# Both VAEs supervise the first N_SUP dims toward physical state.
MODELS = [
    {"label": "Baseline",
     "make_vae": lambda: VAE(latent_size=LATENT_SIZE),
     "vae_ckpt": "<cartpole-baseline-vae>",
     "lstm_ckpt": "<cartpole-baseline-lstm>",
     "latent_root": os.path.join(LATENT_ROOT, "baseline")},
    {"label": "Principle 1",
     "make_vae": lambda: VAE_P1(n_sup=N_SUP, n_img=N_IMG),
     "vae_ckpt": "<cartpole-p1-vae>",
     "lstm_ckpt": "<cartpole-p1-lstm>",
     "latent_root": os.path.join(LATENT_ROOT, "p1")},
]

SEQ_LEN = 30
TEST_STRIDE = 1
BATCH = 128
LOG_Y = True             # log-scale the MSE curves (errors grow fast over the horizon)

# Library
POLY_DEGREE = 2          # state monomials up to this degree (cross terms included)
INCLUDE_TRIG = True      # add sin/cos on ANGLE_DIMS (captures gravity term sin(theta))
INCLUDE_COUPLING = True  # add state*control products (force couples with state)
INCLUDE_BIAS = True
CONTROL_MODE = "signed"  # "signed" (binary -> +/-1) or "onehot_drop" (n_actions-1 cols)

# Sparse regression
THRESHOLD = 0.02         # STLSQ cutoff: coeffs below this are pruned to zero
RIDGE_ALPHA = 1e-3       # small L2 for numerical stability
MAX_SAMPLES = 500_000    # subsample transitions if more than this (memory cap)
SEED = 0

# Hybrid training (residual LSTM on top of the frozen SINDy core) — mirrors lstm_p1.py
EPOCHS = 50
LR = 1e-3
CLIP = 1.0
W_PHYS = 1.0
P_START, P_END, P_DECAY_EPOCHS = 1.0, 0.3, 40   # scheduled sampling
L_START, CURRICULUM_EPOCHS = 5, 15              # horizon curriculum
EARLY_STOP_PATIENCE = 6
SCHED_PATIENCE = 3
TRAIN_STRIDE = 5
TRAIN_BATCH = 64
NUM_WORKERS = 2
CORR_BOUND = None        # None -> unbounded residual; float -> CORR_BOUND*tanh(fc) for stability



#
#  Control encoding
#
def action_to_control(a_idx, n_actions, mode):
    """Map integer actions (B,) to control features (B, n_ctrl).
    signed: binary action -> +/-1 (symmetric, physical for a left/right push).
    onehot_drop: drop the first action as reference -> avoids collinearity with the bias."""
    a_idx = a_idx.long()
    if mode == "signed" and n_actions == 2:
        return (2.0 * a_idx.float() - 1.0).unsqueeze(1)
    oh = F.one_hot(a_idx, n_actions).float()
    return oh[:, 1:]


def control_dim(n_actions, mode):
    return 1 if (mode == "signed" and n_actions == 2) else (n_actions - 1)


#
#  Feature library Theta(z, u)
#
class FeatureLibrary:
    """Builds the candidate-function matrix used by SINDy.

    Columns (in fixed order so fit and rollout stay consistent):
      bias -> state monomials (deg 1..POLY_DEGREE) -> sin/cos(angle dims)
           -> linear control -> state*control couplings
    """
    def __init__(self, n_state, n_ctrl, poly_degree=2, include_trig=False,
                 angle_dims=(), include_coupling=True, include_bias=True):
        self.n_state = n_state
        self.n_ctrl = n_ctrl
        self.include_bias = include_bias
        self.angle_dims = tuple(angle_dims) if include_trig else ()
        self.include_coupling = include_coupling

        # Precompute the index tuples for every term once.
        self.poly_terms = []
        for deg in range(1, poly_degree + 1):
            self.poly_terms += list(itertools.combinations_with_replacement(range(n_state), deg))
        self.coupling_terms = ([(i, k) for i in range(n_state) for k in range(n_ctrl)]
                               if include_coupling else [])

    def n_features(self):
        return (int(self.include_bias) + len(self.poly_terms)
                + 2 * len(self.angle_dims) + self.n_ctrl + len(self.coupling_terms))

    def transform(self, Z, U):
        """Z (N, n_state), U (N, n_ctrl) -> Theta (N, P). Pure torch (works on CPU or GPU)."""
        cols = []
        if self.include_bias:
            cols.append(torch.ones(Z.shape[0], 1, device=Z.device, dtype=Z.dtype))
        for term in self.poly_terms:
            c = torch.ones(Z.shape[0], device=Z.device, dtype=Z.dtype)
            for idx in term:
                c = c * Z[:, idx]
            cols.append(c.unsqueeze(1))
        for d in self.angle_dims:
            cols.append(torch.sin(Z[:, d]).unsqueeze(1))
            cols.append(torch.cos(Z[:, d]).unsqueeze(1))
        for k in range(self.n_ctrl):
            cols.append(U[:, k].unsqueeze(1))
        for (i, k) in self.coupling_terms:
            cols.append((Z[:, i] * U[:, k]).unsqueeze(1))
        return torch.cat(cols, dim=1)

    def names(self, state_names, ctrl_names):
        out = []
        if self.include_bias:
            out.append("1")
        for term in self.poly_terms:
            out.append("*".join(state_names[i] for i in term))
        for d in self.angle_dims:
            out.append(f"sin({state_names[d]})")
            out.append(f"cos({state_names[d]})")
        out += list(ctrl_names)
        for (i, k) in self.coupling_terms:
            out.append(f"{state_names[i]}*{ctrl_names[k]}")
        return out


#
#  Sparse regression — sequentially thresholded least squares
#
def _ridge_lstsq(Phi, Y, alpha):
    """Solve min ||Phi @ Xi - Y||^2 + alpha||Xi||^2 via stacked least squares (stable)."""
    P = Phi.shape[1]
    if alpha > 0:
        aug = torch.sqrt(torch.tensor(alpha, dtype=Phi.dtype))
        Phi = torch.cat([Phi, aug * torch.eye(P, dtype=Phi.dtype)], dim=0)
        Y = torch.cat([Y, torch.zeros(P, Y.shape[1], dtype=Y.dtype)], dim=0)
    return torch.linalg.lstsq(Phi, Y).solution


def stlsq(Phi, Y, threshold, alpha, max_iter=20):
    """Phi (N,P), Y (N,m). Returns sparse Xi (P,m). Prune small coeffs, refit survivors, repeat."""
    Xi = _ridge_lstsq(Phi, Y, alpha)
    P, m = Xi.shape
    for _ in range(max_iter):
        prev = Xi.clone()
        for j in range(m):
            keep = Xi[:, j].abs() >= threshold
            Xi[~keep, j] = 0.0
            if keep.any():
                sol = _ridge_lstsq(Phi[:, keep], Y[:, j:j + 1], alpha)
                Xi[keep, j] = sol[:, 0]
        if torch.allclose(Xi, prev):
            break
    return Xi


#
#  Predictor — drop-in replacement for LatentPredictor
#
class SINDyPredictor(nn.Module):
    """Same (init_hidden, step) interface as the LSTM, so free_run / test_p*.py work unchanged.
    SINDy is Markov: hidden state is unused. Only the physical dims are modeled; style dims
    are carried through untouched (the metric never reads them)."""
    def __init__(self, library, Xi, n_sup, n_actions, control_mode="signed", residual=True):
        super().__init__()
        self.library = library
        self.n_sup = n_sup
        self.n_actions = n_actions
        self.control_mode = control_mode
        self.residual = residual
        self.register_buffer("Xi", Xi if torch.is_tensor(Xi) else torch.as_tensor(Xi))

    def init_hidden(self, b, device):
        return None

    def step(self, z, a_onehot, hidden):
        zp = z[:, :self.n_sup]
        u = action_to_control(a_onehot.argmax(dim=-1), self.n_actions, self.control_mode)
        phi = self.library.transform(zp, u.to(zp.dtype))
        delta = phi @ self.Xi.to(zp.dtype)
        zp_next = zp + delta if self.residual else delta
        out = z.clone()
        out[:, :self.n_sup] = zp_next
        return out, hidden


class HybridPredictor(nn.Module):
    """Hybrid Predictor that combines a frozen SINDy core with a trainable residual LSTM.
    For physical dimensions (0..n_sup-1), the prediction is:
        z_{t+1} = z_t + SINDy(z_t, u_t) + LSTM_residual(z_t, u_t, h_t)
    For style/image dimensions (n_sup..), SINDy is not applied, so the prediction is:
        z_{t+1} = z_t + LSTM_residual(z_t, u_t, h_t)
    """
    def __init__(self, library, Xi, latent=64, action_dim=2, hidden=64, layers=2,
                 n_sup=4, control_mode="signed", corr_bound=None):
        super().__init__()
        self.library = library
        self.n_sup = n_sup
        self.latent = latent
        self.action_dim = action_dim
        self.control_mode = control_mode
        self.corr_bound = corr_bound
        
        # SINDy Xi is frozen
        self.register_buffer("Xi", Xi if torch.is_tensor(Xi) else torch.as_tensor(Xi))
        
        # Embed the exact same LatentPredictor class from lstm.py
        self.lstm_model = LatentPredictor(latent, action_dim, hidden, layers)
        
        self.reset_tracking()

    def init_hidden(self, b, device):
        return self.lstm_model.init_hidden(b, device)

    def reset_tracking(self):
        self._last_corrs = []
        self._last_sindys = []

    def get_tracked(self):
        corrs = torch.stack(self._last_corrs, dim=1) if self._last_corrs else torch.tensor(0.0)
        sindys = torch.stack(self._last_sindys, dim=1) if self._last_sindys else torch.tensor(0.0)
        return corrs, sindys

    def step(self, z, a_onehot, hidden):
        # 1. Residual LSTM correction from the standard LatentPredictor
        z_lstm_next, hidden = self.lstm_model.step(z, a_onehot, hidden)
        corr = z_lstm_next - z
        if self.corr_bound is not None:
            corr = self.corr_bound * torch.tanh(corr)
            
        if hasattr(self, "_last_corrs") and self._last_corrs is not None:
            self._last_corrs.append(corr)
            
        # 2. SINDy physics delta on physical dims
        zp = z[:, :self.n_sup]
        u = action_to_control(a_onehot.argmax(dim=-1), self.action_dim, self.control_mode)
        phi = self.library.transform(zp, u.to(zp.dtype))
        sindy_delta = phi @ self.Xi.to(zp.dtype)
        
        if hasattr(self, "_last_sindys") and self._last_sindys is not None:
            self._last_sindys.append(sindy_delta)
            
        # 3. Combine: z_{t+1} = z_t + corr + SINDy_delta (only on physical dims)
        z_next = z + corr
        z_next[:, :self.n_sup] = z_next[:, :self.n_sup] + sindy_delta
        return z_next, hidden



#
#  Build transition pairs from precomputed latents
#
def load_transitions(latent_dir, n_sup, n_actions, control_mode, max_samples=None, seed=0):
    """Returns Z (M, n_sup), U (M, n_ctrl), dZ (M, n_sup) over all episodes in latent_dir.
    A transition is (z[k][:n_sup], acts[k]) -> z[k+1][:n_sup]; dZ is the residual target."""
    Zs, Us, dZs = [], [], []
    for f in tqdm(list_npz(latent_dir), desc="building transitions"):
        with np.load(f) as d:
            z = d["z"].astype(np.float32)
            acts = d["acts"].astype(np.float32)
        if z.shape[0] < 2:
            continue
        zp = torch.from_numpy(z[:, :n_sup])
        a = torch.from_numpy(acts)
        u = action_to_control(a[:-1], n_actions, control_mode)
        Zs.append(zp[:-1])
        Us.append(u)
        dZs.append(zp[1:] - zp[:-1])
    Z = torch.cat(Zs, 0)
    U = torch.cat(Us, 0)
    dZ = torch.cat(dZs, 0)

    if max_samples is not None and Z.shape[0] > max_samples:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(Z.shape[0], generator=g)[:max_samples]
        Z, U, dZ = Z[idx], U[idx], dZ[idx]
    return Z, U, dZ


def print_equations(library, Xi, state_names, ctrl_names):
    """Print the recovered difference equations: z_i[t+1] = z_i[t] + (sparse terms)."""
    names = library.names(state_names, ctrl_names)
    print("\nDiscovered dynamics (standardized latent space):")
    for j, sname in enumerate(state_names):
        terms = [(names[p], Xi[p, j].item()) for p in range(len(names))
                 if abs(Xi[p, j].item()) > 0.0]
        rhs = " ".join(f"{c:+.4f}*{nm}" for nm, c in terms) if terms else "+0"
        print(f"  {sname}[t+1] = {sname}[t] {rhs}")


def save_model(path, library, Xi, cfg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, Xi=Xi.cpu().numpy(), **cfg)
    print("saved SINDy model ->", path)


#
#  Encode a model's latents fresh from the dataset
#
def encode_latents(cfg, device):
    """Load the model's VAE and write z latents to cfg['latent_root']/{train,val,test}.
    A single local encoder works for any VAE variant (both expose .encode -> (mu, logvar))."""
    vae = cfg["make_vae"]().to(device)
    vae.load_state_dict(torch.load(cfg["vae_ckpt"], map_location=device))
    vae.eval()

    @torch.no_grad()
    def _encode(img_t, img_tp1):
        x = torch.cat([img_t, img_tp1], dim=1).to(device)
        mu, _ = vae.encode(x)
        return mu

    for split in ("train", "val", "test"):
        src = os.path.join(DATA_ROOT, split)
        if os.path.isdir(src):
            print(f"[{cfg['label']}] pre-encoding {split} ...")
            precompute_latents(_encode, src, os.path.join(cfg["latent_root"], split),
                               shift=SHIFT, device=device)
    del vae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


#
#  Multi-step evaluation (mirrors test_p*.py free_run / per-horizon physical MSE)
#
@torch.no_grad()
def free_run(model, batch, n_actions, n_sup):
    z_t, action, z_tp1, state_t, state_tp1 = batch
    B, L, _ = z_t.shape
    z_in = z_t[:, 0]
    hidden = model.init_hidden(B, z_t.device)
    preds = []
    for k in range(L):
        a = F.one_hot(action[:, k].long(), n_actions).float()
        z_pred, hidden = model.step(z_in, a, hidden)
        preds.append(z_pred)
        z_in = z_pred
    return torch.stack(preds, dim=1), state_tp1


@torch.no_grad()
def eval_per_horizon(model, loader, device, std4, n_actions, n_sup):
    """Free-running rollout -> physical MSE per horizon (de-standardized to real units)."""
    se, n = None, 0
    for batch in tqdm(loader, desc="eval", leave=False):
        batch = [b.to(device, non_blocking=True) for b in batch]
        preds, state_tp1 = free_run(model, batch, n_actions, n_sup)
        err = (preds[..., :n_sup] - state_tp1) * std4
        s = (err ** 2).sum(dim=0)
        se = s if se is None else se + s
        n += preds.size(0)
    return (se / n).mean(dim=1).cpu().numpy()


#
#  Per-model: build test loader, fit SINDy, load the trained LSTM
#
def make_test_loader(cfg, mean, std):
    test_ds = LatentSequenceDataset(os.path.join(cfg["latent_root"], "test"),
                                    seq_len=SEQ_LEN, stride=TEST_STRIDE,
                                    state_mean=mean, state_std=std)
    return DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)


def fit_sindy(cfg, library, ctrl_names, device):
    """Fit SINDYc on the model's train latents; print equations; save; return predictor, Xi."""
    Z, U, dZ = load_transitions(os.path.join(cfg["latent_root"], "train"), N_SUP, N_ACTIONS,
                                CONTROL_MODE, max_samples=MAX_SAMPLES, seed=SEED)
    Phi = library.transform(Z, U)
    Xi = stlsq(Phi, dZ, THRESHOLD, RIDGE_ALPHA)
    nnz = int((Xi != 0).sum().item())
    print(f"\n=== {cfg['label']} ===  transitions={Z.shape[0]}  nonzero={nnz}/{Xi.numel()}")
    print_equations(library, Xi, STATE_NAMES, ctrl_names)

    # Compute per-dim train R^2 as an informative printout
    dZ_pred = Phi @ Xi
    ss_res = ((dZ - dZ_pred) ** 2).sum(dim=0)
    ss_tot = ((dZ - dZ.mean(dim=0)) ** 2).sum(dim=0)
    r2 = 1.0 - ss_res / ss_tot
    print("\nPer-dimension train R^2 (SINDy fit quality):")
    for name, val in zip(STATE_NAMES, r2):
        print(f"  {name}: {val.item():.4f}")

    slug = cfg["label"].lower().replace(" ", "_")
    save_model(os.path.join(SAVE_DIR, f"sindy_{slug}.npz"), library, Xi, cfg={
        "n_sup": N_SUP, "n_actions": N_ACTIONS, "control_mode": CONTROL_MODE,
        "poly_degree": POLY_DEGREE, "include_trig": INCLUDE_TRIG,
        "angle_dims": np.array(ANGLE_DIMS), "include_coupling": INCLUDE_COUPLING,
        "include_bias": INCLUDE_BIAS,
    })
    predictor = SINDyPredictor(library, Xi, N_SUP, N_ACTIONS, CONTROL_MODE).to(device)
    return predictor, Xi



def load_lstm(cfg, device):
    """Load the trained residual-LSTM predictor (same interface as SINDyPredictor)."""
    model = LatentPredictor(LATENT_SIZE, N_ACTIONS, HIDDEN, LAYERS).to(device)
    model.load_state_dict(torch.load(cfg["lstm_ckpt"], map_location=device))
    model.eval()
    return model


def load_hybrid(cfg, library, Xi, device):
    """Load the HybridPredictor by initializing SINDy and loading LSTM weights from cfg['lstm_ckpt']."""
    model = HybridPredictor(
        library=library,
        Xi=Xi,
        latent=LATENT_SIZE,
        action_dim=N_ACTIONS,
        hidden=HIDDEN,
        layers=LAYERS,
        n_sup=N_SUP,
        control_mode=CONTROL_MODE,
        corr_bound=CORR_BOUND
    ).to(device)
    
    # Load the pretrained LSTM weights into the embedded lstm_model
    model.lstm_model.load_state_dict(torch.load(cfg["lstm_ckpt"], map_location=device))
    model.eval()
    return model



@torch.no_grad()
def eval_hybrid_residual_norm(model, loader, device):
    """Evaluate average norm of LSTM correction and SINDy delta on the given dataset.
    Only computes norms over the physical dimensions (0..n_sup-1).
    """
    model.eval()
    corr_norms = []
    sindy_norms = []
    
    for batch in tqdm(loader, desc="eval residual norm", leave=False):
        batch = [b.to(device, non_blocking=True) for b in batch]
        z_t, action, z_tp1, _, _ = batch
        B, L, _ = z_t.shape
        
        z_in = z_t[:, 0]
        hidden = model.init_hidden(B, device)
        
        model.reset_tracking()
        
        for k in range(L):
            a = F.one_hot(action[:, k].long(), N_ACTIONS).float()
            z_in, hidden = model.step(z_in, a, hidden)
            
        corrs, sindys = model.get_tracked() # corrs: (B, L, D), sindys: (B, L, n_sup)
        
        # Calculate norms over physical dimensions (0..n_sup-1)
        corr_phys = corrs[..., :model.n_sup]
        c_norm = torch.linalg.norm(corr_phys, dim=-1) # (B, L)
        s_norm = torch.linalg.norm(sindys, dim=-1)    # (B, L)
        
        corr_norms.append(c_norm.cpu().numpy())
        sindy_norms.append(s_norm.cpu().numpy())
        
    all_corr = np.concatenate(corr_norms, axis=0)   # (N, L)
    all_sindy = np.concatenate(sindy_norms, axis=0) # (N, L)
    
    mean_corr = float(all_corr.mean())
    mean_sindy = float(all_sindy.mean())
    
    return mean_corr, mean_sindy


#
#  Plot
#
def plot_lstm_vs_sindy(results, save_dir):
    """One subplot per model (Baseline, P1): per-horizon physical MSE, LSTM vs SINDy vs Hybrid."""
    os.makedirs(save_dir, exist_ok=True)
    horizons = np.arange(1, SEQ_LEN + 1)
    labels = list(results.keys())
    styles = {
        "LSTM": ("C0", "-"), 
        "SINDy": ("C3", "--"),
        "Hybrid": ("C2", "-.")
    }

    fig, axes = plt.subplots(1, len(labels), figsize=(6.0 * len(labels), 4.6), squeeze=False)
    for j, label in enumerate(labels):
        ax = axes[0][j]
        for method in results[label].keys():
            mse_h = results[label][method]
            color, ls = styles.get(method, ("C7", "-"))
            ax.plot(horizons, mse_h, color=color, ls=ls, lw=2,
                    label=f"{method} (mean={mse_h.mean():.4f})")
        if LOG_Y:
            ax.set_yscale("log")
        ax.set_title(f"{label}: Dynamics Comparison")
        ax.set_xlabel("Prediction horizon")
        ax.set_ylabel("State MSE (physical units)")
        ax.set_xlim(1, SEQ_LEN)
        ax.grid(alpha=0.3, which="both")
        ax.legend()
    plt.tight_layout()
    path = os.path.join(save_dir, "lstm_vs_sindy.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print("saved figure ->", path)


#
#  Main
#
if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    mean, std = load_norm_stats(NORM_STATS)
    std4 = torch.tensor(std[:N_SUP], device=device)

    n_ctrl = control_dim(N_ACTIONS, CONTROL_MODE)
    ctrl_names = (["u"] if n_ctrl == 1 else [f"u{k}" for k in range(n_ctrl)])
    library = FeatureLibrary(N_SUP, n_ctrl, poly_degree=POLY_DEGREE,
                             include_trig=INCLUDE_TRIG, angle_dims=ANGLE_DIMS,
                             include_coupling=INCLUDE_COUPLING, include_bias=INCLUDE_BIAS)
    print(f"library features: {library.n_features()}")

    # For each model: encode latents, then eval THREE dynamics models (LSTM, SINDy, Hybrid)
    # on the SAME clean test latents -> fully matched comparison.
    results = {}   # results[label] = {"LSTM": mse_h, "SINDy": mse_h, "Hybrid": mse_h}
    for cfg in MODELS:
        encode_latents(cfg, device)
        test_dl = make_test_loader(cfg, mean, std)
        sindy_model, Xi = fit_sindy(cfg, library, ctrl_names, device)
        lstm_model = load_lstm(cfg, device)
        hybrid_model = load_hybrid(cfg, library, Xi, device)


        
        # Compute test MSE
        results[cfg["label"]] = {
            "LSTM":   eval_per_horizon(lstm_model, test_dl, device, std4, N_ACTIONS, N_SUP),
            "SINDy":  eval_per_horizon(sindy_model, test_dl, device, std4, N_ACTIONS, N_SUP),
            "Hybrid": eval_per_horizon(hybrid_model, test_dl, device, std4, N_ACTIONS, N_SUP),
        }
        
        # Compute and report residual norm metrics on test set
        mean_corr, mean_sindy = eval_hybrid_residual_norm(hybrid_model, test_dl, device)
        print(f"\n[{cfg['label']}] Hybrid residual norm metrics (physical dims):")
        print(f"  Mean LSTM correction ||corr||: {mean_corr:.6f}")
        print(f"  Mean SINDy delta ||sindy_delta||: {mean_sindy:.6f}")
        if mean_sindy > 0:
            print(f"  Relative residual magnitude (||corr|| / ||sindy_delta||): {mean_corr / mean_sindy:.6f}")

    # Side-by-side comparison: model x method
    HS = [h for h in (1, 10, 20, SEQ_LEN) if h <= SEQ_LEN]
    print(f"\n{'='*78}")
    print("Physical MSE per horizon (de-standardized) — lower is better")
    print(f"{'='*78}")
    header = f"{'model':<14}{'method':<8}" + "".join(f"{'h'+str(h):>11}" for h in HS) + f"{'mean':>11}"
    print(header)
    print("-" * len(header))
    for label, by_method in results.items():
        for method in ("LSTM", "SINDy", "Hybrid"):
            mse_h = by_method[method]
            row = f"{label:<14}{method:<8}" + "".join(f"{mse_h[h-1]:>11.4f}" for h in HS)
            print(row + f"{mse_h.mean():>11.4f}")

    plot_lstm_vs_sindy(results, SAVE_DIR)

