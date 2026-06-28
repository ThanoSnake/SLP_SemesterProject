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

# Trained hybrid checkpoints (absolute, configurable per model)
HYBRID_BASELINE_CKPT = "<hybrid-baseline>"
HYBRID_P1_CKPT = "<hybrid-p1>"

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
     "latent_root": os.path.join(LATENT_ROOT, "baseline"),
     "hybrid_ckpt": HYBRID_BASELINE_CKPT},
    {"label": "Principle 1",
     "make_vae": lambda: VAE_P1(n_sup=N_SUP, n_img=N_IMG),
     "vae_ckpt": "<cartpole-p1-vae>",
     "lstm_ckpt": "<cartpole-p1-lstm>",
     "latent_root": os.path.join(LATENT_ROOT, "p1"),
     "hybrid_ckpt": HYBRID_P1_CKPT},
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
EPOCHS = 40
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
CORR_BOUND = 0.4        # None -> unbounded residual; float -> CORR_BOUND*tanh(fc) for stability
LSTM_DELTA_BOUND = None    # None -> raw LSTM; float -> clamp the LSTM's per-step delta to +/- value
TRAIN_LSTM = False         # True -> train a fresh (bounded) LSTM here (+save); False -> load it
TRAIN_HYBRID = True     # True -> train a fresh hybrid (+save); False -> load from cfg['hybrid_ckpt']
SHOW_PROGRESS = False    # tqdm bars render as line-spam under Kaggle's non-TTY logs -> off by default

# Visual-noise sweep on TEST images before encoding (mirrors test_p1.py). Models are fit/trained
# on CLEAN train/val latents; the test split is re-encoded at each level to measure robustness.
NOISE_TYPE = "gaussian"                       # "gaussian" | "salt_pepper"
NOISE_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30]  # std on [0,1] image; 0.0 = clean
NOISE_SEED = 42


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
    """Frozen SINDy physics core + trainable residual LSTM.

        physical dims:  z_{t+1} = z_t + LSTM_resid(z_t, u_t, h_t) + SINDy(z_t, u_t)
        style dims:     z_{t+1} = z_t + LSTM_resid(z_t, u_t, h_t)

    The embedded LatentPredictor zero-inits its output head, so the hybrid STARTS as pure SINDy
    and training teaches the LSTM only the residual (Δz_true - SINDy). SINDy (Xi) stays frozen.
    Per-step corr/sindy tracking is OPT-IN (self.track) so normal eval stays leak-free.
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
        self.track = False
        self.register_buffer("Xi", Xi.detach().clone() if torch.is_tensor(Xi) else torch.as_tensor(Xi))
        # Same residual LSTM as lstm.py; its zero-init head means corr=0 at start -> pure SINDy.
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

    def _sindy_delta(self, zp, a_onehot):
        u = action_to_control(a_onehot.argmax(dim=-1), self.action_dim, self.control_mode)
        return self.library.transform(zp, u.to(zp.dtype)) @ self.Xi.to(zp.dtype)

    def step(self, z, a_onehot, hidden):
        # LSTM residual correction (corr = the learned delta from the embedded predictor)
        z_lstm_next, hidden = self.lstm_model.step(z, a_onehot, hidden)
        corr = z_lstm_next - z
        if self.corr_bound is not None:
            corr = self.corr_bound * torch.tanh(corr)
        # Frozen SINDy increment on the physical dims
        sindy_delta = self._sindy_delta(z[:, :self.n_sup], a_onehot)
        # Combine without in-place ops (safe for autograd during training)
        phys = z[:, :self.n_sup] + corr[:, :self.n_sup] + sindy_delta
        rest = z[:, self.n_sup:] + corr[:, self.n_sup:]
        z_next = torch.cat([phys, rest], dim=-1)
        if self.track:
            self._last_corrs.append(corr[:, :self.n_sup].detach())
            self._last_sindys.append(sindy_delta.detach())
        return z_next, hidden


class BoundedPredictor(nn.Module):
    """Wrap a predictor and clamp its per-step delta to +/- bound — a 'threshold' on the
    prediction that caps runaway latent jumps under noise. Same step/init_hidden interface."""
    def __init__(self, model, bound):
        super().__init__()
        self.model = model
        self.bound = bound

    def init_hidden(self, b, device):
        return self.model.init_hidden(b, device)

    def step(self, z, a_onehot, hidden):
        z_next, hidden = self.model.step(z, a_onehot, hidden)
        delta = torch.clamp(z_next - z, -self.bound, self.bound)
        return z + delta, hidden


#
#  Build transition pairs from precomputed latents
#
def load_transitions(latent_dir, n_sup, n_actions, control_mode, max_samples=None, seed=0):
    """Returns Z (M, n_sup), U (M, n_ctrl), dZ (M, n_sup) over all episodes in latent_dir.
    A transition is (z[k][:n_sup], acts[k]) -> z[k+1][:n_sup]; dZ is the residual target."""
    Zs, Us, dZs = [], [], []
    for f in tqdm(list_npz(latent_dir), desc="building transitions", disable=not SHOW_PROGRESS):
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
def add_gaussian_noise(img, std, gen):
    noise = torch.randn(img.shape, generator=gen, device=img.device) * std
    return torch.clamp(img + noise, 0.0, 1.0)


def add_salt_pepper_noise(img, amount, gen):
    mask = torch.rand(img.shape, generator=gen, device=img.device)
    out = img.clone()
    out[mask < amount / 2] = 0.0
    out[mask > 1 - amount / 2] = 1.0
    return out


def make_noise_fn(noise_type, level, seed, device):
    """Returns (img_tensor) -> noisy_img_tensor with a fixed seed (reproducible)."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    if level == 0.0:
        return lambda x: x
    if noise_type == "gaussian":
        return lambda x: add_gaussian_noise(x, level, gen)
    if noise_type == "salt_pepper":
        return lambda x: add_salt_pepper_noise(x, level, gen)
    raise ValueError(f"Unknown noise type: {noise_type}")


def noise_tag(level):
    return f"{NOISE_TYPE}_{level:.2f}".replace(".", "p")


@torch.no_grad()
def precompute_latents_noisy(encode_fn, root, out_root, noise_fn, shift=0, batch=256, device="cuda"):
    """Like loader.precompute_latents but applies noise_fn to every frame BEFORE encoding.
    The whole image sequence is noised once, then sliced into (t, t+1) pairs, so each physical
    frame gets a single noise realization (consistent whether seen as t or t+1)."""
    from os.path import join, basename
    from os import makedirs
    makedirs(out_root, exist_ok=True)
    for f in tqdm(list_npz(root), desc="encoding (noisy)", disable=not SHOW_PROGRESS):
        with np.load(f) as d:
            imgs = torch.from_numpy(d["imgs"].astype(np.float32) / 255.0).permute(0, 3, 1, 2)
            acts = d["acts"].astype(np.float32)
            states = d["states"].astype(np.float32)
            x = (d[f"noisy_states_{shift}"] if shift in (2, 5, 10) else d["states"]).astype(np.float32)
        imgs = noise_fn(imgs.to(device))
        img_t, img_tp1 = imgs[:-1], imgs[1:]
        zs = []
        for b in range(0, img_t.shape[0], batch):
            zb = encode_fn(img_t[b:b + batch], img_tp1[b:b + batch])
            zs.append(zb.cpu().numpy())
        z = np.concatenate(zs, 0).astype(np.float32) if zs else np.empty((0, 0), np.float32)
        np.savez_compressed(join(out_root, basename(f)),
                            z=z, acts=acts[:-1], states=states[:-1], x=x[:-1])


def encode_latents(cfg, device):
    """Encode CLEAN train/val latents (for SINDy fit + hybrid training) and the TEST split at
    each NOISE_LEVELS (for the robustness sweep). The VAE class (VAE / VAE_P1) + checkpoint are
    used to produce mu; output goes to cfg['latent_root']/{train,val,test_<tag>}."""
    vae = cfg["make_vae"]()
    vae.load_state_dict(torch.load(cfg["vae_ckpt"], map_location=device))
    vae.to(device)
    vae.eval()

    @torch.no_grad()
    def _encode(img_t, img_tp1):
        x = torch.cat([img_t, img_tp1], dim=1).to(device)
        mu, _ = vae.encode(x)
        return mu

    # Clean train/val (models are fit/trained on these)
    for split in ("train", "val"):
        src = os.path.join(DATA_ROOT, split)
        if os.path.isdir(src):
            print(f"[{cfg['label']}] encoding {split} (clean) ...")
            precompute_latents(_encode, src, os.path.join(cfg["latent_root"], split),
                               shift=SHIFT, device=device)

    # Test split at every noise level
    src = os.path.join(DATA_ROOT, "test")
    if os.path.isdir(src):
        for nl in NOISE_LEVELS:
            out = os.path.join(cfg["latent_root"], f"test_{noise_tag(nl)}")
            print(f"[{cfg['label']}] encoding test ({noise_tag(nl)}) ...")
            if nl > 0.0:
                noise_fn = make_noise_fn(NOISE_TYPE, nl, NOISE_SEED, device)
                precompute_latents_noisy(_encode, src, out, noise_fn, shift=SHIFT, device=device)
            else:
                precompute_latents(_encode, src, out, shift=SHIFT, device=device)

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
    for batch in tqdm(loader, desc="eval", leave=False, disable=not SHOW_PROGRESS):
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
def make_test_loader(cfg, mean, std, level):
    test_ds = LatentSequenceDataset(os.path.join(cfg["latent_root"], f"test_{noise_tag(level)}"),
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



def _lstm_ckpt_path(cfg):
    slug = cfg["label"].lower().replace(" ", "_")
    return os.path.join(SAVE_DIR, f"lstm_bounded_{slug}.pth")


def build_lstm(device):
    """Residual LSTM, optionally wrapped so its per-step delta is clamped to +/- LSTM_DELTA_BOUND."""
    m = LatentPredictor(LATENT_SIZE, N_ACTIONS, HIDDEN, LAYERS)
    if LSTM_DELTA_BOUND is not None:
        m = BoundedPredictor(m, LSTM_DELTA_BOUND)
    return m.to(device)


def load_lstm(cfg, device):
    """Load the pre-saved baseline LSTM from its checkpoint."""
    model = LatentPredictor(LATENT_SIZE, N_ACTIONS, HIDDEN, LAYERS)
    model.load_state_dict(torch.load(cfg["lstm_ckpt"], map_location=device))
    model.to(device)
    model.eval()
    print(f"    [{cfg['label']}] LSTM loaded from pre-existing: {cfg['lstm_ckpt']}")
    return model


def make_loader(latent_dir, mean, std, stride, batch, shuffle):
    ds = LatentSequenceDataset(latent_dir, seq_len=SEQ_LEN, stride=stride,
                               state_mean=mean, state_std=std)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, drop_last=shuffle,
                      num_workers=NUM_WORKERS, pin_memory=True)


def _train_rollout(model, batch, p_tf, cur_len, device):
    """Free-running rollout with scheduled sampling (per-sample teacher forcing)."""
    z_t, action, z_tp1, state_t, state_tp1 = batch
    L = min(cur_len, z_t.shape[1])
    B = z_t.shape[0]
    z_gt = z_tp1[:, :L]
    hidden = model.init_hidden(B, device)
    z_in = z_t[:, 0]
    preds = []
    for k in range(L):
        a = F.one_hot(action[:, k].long(), N_ACTIONS).float()
        z_pred, hidden = model.step(z_in, a, hidden)
        preds.append(z_pred)
        if k < L - 1:
            use_tf = (torch.rand(B, 1, device=device) < p_tf).float()
            z_in = use_tf * z_gt[:, k] + (1.0 - use_tf) * z_pred.detach()
    return torch.stack(preds, dim=1), z_gt


def train_hybrid(model, cfg, mean, std, std4, device):
    """Train ONLY the residual LSTM (SINDy Xi is a frozen buffer -> no grad). Zero-init head means
    the hybrid starts as pure SINDy and learns Δz - SINDy. Scheduled sampling + horizon curriculum,
    early-stopped on val per-horizon MSE. Returns the model with its best weights loaded."""
    train_dl = make_loader(os.path.join(cfg["latent_root"], "train"), mean, std,
                           TRAIN_STRIDE, TRAIN_BATCH, shuffle=True)
    val_dl = make_loader(os.path.join(cfg["latent_root"], "val"), mean, std,
                         TRAIN_STRIDE, BATCH, shuffle=False)
    opt = optim.Adam(model.parameters(), lr=LR)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=SCHED_PATIENCE)

    best, bad, best_state = float("inf"), 0, None
    print(f"  training {cfg['label']} hybrid (max {EPOCHS} epochs) ...")
    for epoch in range(1, EPOCHS + 1):
        p_tf = max(P_END, P_START - (P_START - P_END) * (epoch - 1) / max(P_DECAY_EPOCHS, 1))
        cur_len = int(round(min(SEQ_LEN, L_START + (SEQ_LEN - L_START)
                                * (epoch - 1) / max(CURRICULUM_EPOCHS, 1))))
        model.train()
        for batch in train_dl:
            batch = [b.to(device, non_blocking=True) for b in batch]
            preds, z_gt = _train_rollout(model, batch, p_tf, cur_len, device)
            loss = (F.mse_loss(preds, z_gt)
                    + W_PHYS * F.mse_loss(preds[..., :N_SUP], z_gt[..., :N_SUP]))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()

        mse_h = eval_per_horizon(model, val_dl, device, std4, N_ACTIONS, N_SUP)
        val_mean = float(mse_h.mean())
        sched.step(val_mean)
        improved = val_mean < best - 1e-6
        if improved:
            best, bad, best_state = val_mean, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
        print(f"  E{epoch:02d}/{EPOCHS} | p_tf={p_tf:.2f} H={cur_len:2d} | "
              f"val phys-MSE={val_mean:.4f}{'  *best' if improved else ''}")
        if bad >= EARLY_STOP_PATIENCE:
            print(f"  early stop at epoch {epoch} (no val improvement for {EARLY_STOP_PATIENCE} epochs)")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    # Persist the trained hybrid (residual LSTM weights + frozen Xi buffer).
    slug = cfg["label"].lower().replace(" ", "_")
    save_path = os.path.join(SAVE_DIR, f"hybrid_{slug}.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"    [{cfg['label']}] hybrid trained: best val phys-MSE = {best:.4f}  -> {save_path}")
    return model


def load_hybrid(cfg, library, Xi, device):
    """Reconstruct the hybrid shell (frozen SINDy + residual LSTM) and load trained weights."""
    model = HybridPredictor(library, Xi, latent=LATENT_SIZE, action_dim=N_ACTIONS,
                            hidden=HIDDEN, layers=LAYERS, n_sup=N_SUP,
                            control_mode=CONTROL_MODE, corr_bound=CORR_BOUND)
    if TRAIN_HYBRID:
        slug = cfg["label"].lower().replace(" ", "_")
        path = os.path.join(SAVE_DIR, f"hybrid_{slug}.pth")
    else:
        path = cfg["hybrid_ckpt"]
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print(f"    [{cfg['label']}] hybrid loaded from {path}")
    return model


def train_lstm(cfg, mean, std, std4, device):
    """Train a fresh (delta-bounded) LSTM with the same scheduled-sampling + curriculum recipe,
    so the bound is baked into the learned model. Saves to _lstm_ckpt_path(cfg)."""
    model = build_lstm(device)
    train_dl = make_loader(os.path.join(cfg["latent_root"], "train"), mean, std,
                           TRAIN_STRIDE, TRAIN_BATCH, shuffle=True)
    val_dl = make_loader(os.path.join(cfg["latent_root"], "val"), mean, std,
                         TRAIN_STRIDE, BATCH, shuffle=False)
    opt = optim.Adam(model.parameters(), lr=LR)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=SCHED_PATIENCE)

    best, bad, best_state = float("inf"), 0, None
    print(f"  training {cfg['label']} LSTM (delta bound={LSTM_DELTA_BOUND}, max {EPOCHS} epochs) ...")
    for epoch in range(1, EPOCHS + 1):
        p_tf = max(P_END, P_START - (P_START - P_END) * (epoch - 1) / max(P_DECAY_EPOCHS, 1))
        cur_len = int(round(min(SEQ_LEN, L_START + (SEQ_LEN - L_START)
                                * (epoch - 1) / max(CURRICULUM_EPOCHS, 1))))
        model.train()
        for batch in train_dl:
            batch = [b.to(device, non_blocking=True) for b in batch]
            preds, z_gt = _train_rollout(model, batch, p_tf, cur_len, device)
            loss = (F.mse_loss(preds, z_gt)
                    + W_PHYS * F.mse_loss(preds[..., :N_SUP], z_gt[..., :N_SUP]))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()

        mse_h = eval_per_horizon(model, val_dl, device, std4, N_ACTIONS, N_SUP)
        val_mean = float(mse_h.mean())
        sched.step(val_mean)
        improved = val_mean < best - 1e-6
        if improved:
            best, bad, best_state = val_mean, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
        print(f"  E{epoch:02d}/{EPOCHS} | p_tf={p_tf:.2f} H={cur_len:2d} | "
              f"val phys-MSE={val_mean:.4f}{'  *best' if improved else ''}")
        if bad >= EARLY_STOP_PATIENCE:
            print(f"  early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    path = _lstm_ckpt_path(cfg)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"  [{cfg['label']}] LSTM trained: best val phys-MSE={best:.4f} -> {path}")
    return model



@torch.no_grad()
def eval_hybrid_residual_norm(model, loader, device):
    """Average per-step norms of the LSTM correction vs the SINDy delta (physical dims only).
    Small ||corr|| relative to ||sindy|| => physics explains most of the dynamics (more physical
    latent). Tracking is enabled only for this pass, then turned back off."""
    model.eval()
    model.track = True
    corr_norms, sindy_norms = [], []
    try:
        for batch in tqdm(loader, desc="residual norm", leave=False, disable=not SHOW_PROGRESS):
            batch = [b.to(device, non_blocking=True) for b in batch]
            z_t, action, z_tp1, _, _ = batch
            B, L, _ = z_t.shape
            z_in = z_t[:, 0]
            hidden = model.init_hidden(B, device)
            model.reset_tracking()
            for k in range(L):
                a = F.one_hot(action[:, k].long(), N_ACTIONS).float()
                z_in, hidden = model.step(z_in, a, hidden)
            corrs, sindys = model.get_tracked()   # each (B, L, n_sup)
            corr_norms.append(torch.linalg.norm(corrs, dim=-1).cpu().numpy())
            sindy_norms.append(torch.linalg.norm(sindys, dim=-1).cpu().numpy())
    finally:
        model.track = False
        model.reset_tracking()

    mean_corr = float(np.concatenate(corr_norms, axis=0).mean())
    mean_sindy = float(np.concatenate(sindy_norms, axis=0).mean())
    return mean_corr, mean_sindy


#
#  Plots
#
METHODS = ("LSTM", "SINDy", "Hybrid")
METHOD_STYLE = {"LSTM": ("C0", "-"), "SINDy": ("C3", "--"), "Hybrid": ("C2", "-.")}


def plot_degradation(results, save_dir):
    """One subplot per model: mean state-MSE (over horizon) vs noise level, one line per method.
    Shows how each dynamics model degrades as the encoded latent gets noisier."""
    os.makedirs(save_dir, exist_ok=True)
    labels = list(results.keys())
    fig, axes = plt.subplots(1, len(labels), figsize=(6.0 * len(labels), 4.6), squeeze=False)
    for j, label in enumerate(labels):
        ax = axes[0][j]
        for method in METHODS:
            ys = [results[label][nl][method].mean() for nl in NOISE_LEVELS]
            color, ls = METHOD_STYLE[method]
            ax.plot(NOISE_LEVELS, ys, color=color, ls=ls, lw=2, marker="o", label=method)
        if LOG_Y:
            ax.set_yscale("log")
        ax.set_title(f"{label}: robustness to {NOISE_TYPE} noise")
        ax.set_xlabel(f"noise level ({NOISE_TYPE})")
        ax.set_ylabel("mean state MSE (over horizon)")
        ax.grid(alpha=0.3, which="both")
        ax.legend()
    plt.tight_layout()
    path = os.path.join(save_dir, "noise_degradation.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print("saved figure ->", path)


def plot_horizon_at_level(results, save_dir, level):
    """One subplot per model: per-horizon physical MSE at a single noise level, LSTM/SINDy/Hybrid."""
    os.makedirs(save_dir, exist_ok=True)
    horizons = np.arange(1, SEQ_LEN + 1)
    labels = list(results.keys())
    fig, axes = plt.subplots(1, len(labels), figsize=(6.0 * len(labels), 4.6), squeeze=False)
    for j, label in enumerate(labels):
        ax = axes[0][j]
        for method in METHODS:
            mse_h = results[label][level][method]
            color, ls = METHOD_STYLE[method]
            ax.plot(horizons, mse_h, color=color, ls=ls, lw=2,
                    label=f"{method} (mean={mse_h.mean():.4f})")
        if LOG_Y:
            ax.set_yscale("log")
        ax.set_title(f"{label}: per-horizon @ {noise_tag(level)}")
        ax.set_xlabel("Prediction horizon")
        ax.set_ylabel("State MSE (physical units)")
        ax.set_xlim(1, SEQ_LEN)
        ax.grid(alpha=0.3, which="both")
        ax.legend()
    plt.tight_layout()
    path = os.path.join(save_dir, f"per_horizon_{noise_tag(level)}.png")
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

    # For each model: encode clean train/val + noisy test sweep, fit/train on clean, then eval
    # all THREE dynamics models (LSTM, SINDy, Hybrid) at every noise level.
    results = {}   # results[label][noise_level] = {"LSTM": mse_h, "SINDy": mse_h, "Hybrid": mse_h}
    for cfg in MODELS:
        encode_latents(cfg, device)
        sindy_model, Xi = fit_sindy(cfg, library, ctrl_names, device)
        if TRAIN_LSTM:
            lstm_model = train_lstm(cfg, mean, std, std4, device)
        else:
            lstm_model = load_lstm(cfg, device)

        # Hybrid (frozen SINDy + residual LSTM): train fresh or load a previous run.
        if TRAIN_HYBRID:
            hybrid_model = HybridPredictor(library, Xi, latent=LATENT_SIZE, action_dim=N_ACTIONS,
                                           hidden=HIDDEN, layers=LAYERS, n_sup=N_SUP,
                                           control_mode=CONTROL_MODE, corr_bound=CORR_BOUND).to(device)
            train_hybrid(hybrid_model, cfg, mean, std, std4, device)
        else:
            hybrid_model = load_hybrid(cfg, library, Xi, device)

        models = {"LSTM": lstm_model, "SINDy": sindy_model, "Hybrid": hybrid_model}
        results[cfg["label"]] = {}
        for nl in NOISE_LEVELS:
            test_dl = make_test_loader(cfg, mean, std, nl)
            results[cfg["label"]][nl] = {
                name: eval_per_horizon(m, test_dl, device, std4, N_ACTIONS, N_SUP)
                for name, m in models.items()
            }

        # Residual-norm metric on the CLEAN test (how much work the LSTM does beyond physics)
        clean_nl = 0.0 if 0.0 in NOISE_LEVELS else min(NOISE_LEVELS)
        clean_dl = make_test_loader(cfg, mean, std, clean_nl)
        mean_corr, mean_sindy = eval_hybrid_residual_norm(hybrid_model, clean_dl, device)
        print(f"\n[{cfg['label']}] Hybrid residual norm (clean test, physical dims):")
        print(f"  ||corr||={mean_corr:.6f}  ||sindy||={mean_sindy:.6f}"
              + (f"  ratio={mean_corr / mean_sindy:.4f}" if mean_sindy > 0 else ""))

    # Robustness table: mean state-MSE (over horizon) per model x noise level x method
    print(f"\n{'='*70}")
    print(f"Mean state-MSE (over horizon) vs {NOISE_TYPE} noise level — lower is better")
    print(f"{'='*70}")
    for label in results:
        print(f"\n{label}")
        header = f"{'noise':<8}" + "".join(f"{m:>11}" for m in METHODS)
        print(header)
        print("-" * len(header))
        for nl in NOISE_LEVELS:
            row = f"{nl:<8.2f}" + "".join(f"{results[label][nl][m].mean():>11.4f}" for m in METHODS)
            print(row)

    plot_degradation(results, SAVE_DIR)
    for nl in NOISE_LEVELS:
        plot_horizon_at_level(results, SAVE_DIR, nl)

