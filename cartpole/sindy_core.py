"""
sindy_core.py — Shared SINDy backbone (from-scratch, numpy-only) for the CartPole world model.

What it does: learns INTERPRETABLE dynamics on the 4-dim physical state [x, ẋ, θ, θ̇] with sparse
regression (STLSQ), in DISCRETE next-state / DELTA form:
        x_{t+1} = x_t + Θ(x_t, F_t) · Ξ          (Ξ sparse -> few, readable equations)
This is the dyn_phys: ℝ⁴ → ℝ⁴ of the paper's Definition 2 — the physical counterpart of the LSTM.

KEY design points (consistent with the methodology discussion):
  * We work in RAW physical units (sin θ needs rad) — the encoded dims z[:4] are
    standardized, so de-standardize before the fit and re-standardize only for the metric.
  * DELTA form (residual) like the residual LSTM -> better conditioning, identity = Ξ=0.
  * F_t = signed force from the action: action∈{0,1} -> {-1,+1} (the magnitude is absorbed into Ξ).
  * Discrete next-state (not finite-difference derivatives) -> matches the LSTM's next-step
    prediction and avoids derivative noise.

numpy-only (no torch/pysindy) -> runs anywhere, fully transparent.
"""
import os
import sys

import numpy as np
from itertools import combinations_with_replacement

# --- path bootstrap: this file lives in the SAME folder (cartpole/) as vae/lstm/loader and the
#     SINDy siblings; make sure that folder is on sys.path (covers a different cwd) ---
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


N_SUP = 4
DT = 0.02                                   # CartPole tau (documentation only; dt is absorbed into Ξ)
DIM_NAMES = ["x", "x_dot", "theta", "theta_dot"]
BASE_NAMES = ["x", "x_dot", "theta", "theta_dot", "F"]


# ---------------------------------------------------------------------------
# IO helpers (numpy-only copies of the loader, to keep the core dependency-light)
# ---------------------------------------------------------------------------
def load_norm_stats(path):
    z = np.load(path)
    return z["mean"].astype(np.float64), z["std"].astype(np.float64)


def list_npz(root):
    out = []
    for sd in sorted(os.listdir(root)):
        p = os.path.join(root, sd)
        if os.path.isdir(p):
            out += [os.path.join(p, f) for f in sorted(os.listdir(p)) if f.endswith(".npz")]
        elif p.endswith(".npz"):
            out.append(p)
    return sorted(out)


def action_to_force(u):
    """action {0,1} -> signed direction {-1,+1}. The force magnitude is learned in Ξ."""
    return 2.0 * np.asarray(u, np.float64) - 1.0


# ---------------------------------------------------------------------------
# Feature library  Θ(x, F)
# ---------------------------------------------------------------------------
def feature_library(X, F, mode="physics"):
    """X: (N,4) raw [x, ẋ, θ, θ̇]; F: (N,) signed force. -> (Θ (N,n_feat), names)."""
    X = np.atleast_2d(np.asarray(X, np.float64))
    F = np.asarray(F, np.float64).reshape(-1)
    x, xd, th, thd = X[:, 0], X[:, 1], X[:, 2], X[:, 3]

    if mode == "physics":
        s, c = np.sin(th), np.cos(th)
        feats = {
            "1": np.ones_like(x),
            "x": x, "x_dot": xd, "theta": th, "theta_dot": thd, "F": F,
            "sin": s, "cos": c,
            "theta_dot^2": thd ** 2,
            "theta_dot^2*sin": thd ** 2 * s,
            "theta_dot^2*cos": thd ** 2 * c,
            "sin*cos": s * c,
            "F*cos": F * c,
            "F*sin": F * s,
            "sin*cos^2": s * c * c,
            "theta_dot^2*sin*cos": thd ** 2 * s * c,
        }
        names = list(feats.keys())
        Theta = np.stack([feats[n] for n in names], axis=1)
        return Theta, names

    if mode == "poly2":
        base = np.stack([x, xd, th, thd, F], axis=1)            # (N,5)
        cols, names = [np.ones_like(x)], ["1"]
        for i in range(5):
            cols.append(base[:, i]); names.append(BASE_NAMES[i])
        for i, j in combinations_with_replacement(range(5), 2):
            cols.append(base[:, i] * base[:, j])
            names.append(f"{BASE_NAMES[i]}*{BASE_NAMES[j]}")
        return np.stack(cols, axis=1), names

    raise ValueError(f"Unknown feature mode: {mode}")


# ---------------------------------------------------------------------------
# STLSQ — sequentially thresholded least squares (with column normalization)
# ---------------------------------------------------------------------------
def _ridge_solve(A, b, ridge):
    AtA = A.T @ A + ridge * np.eye(A.shape[1])
    return np.linalg.solve(AtA, A.T @ b)


def stlsq(Theta, dX, threshold=0.02, ridge=1e-6, n_iter=10):
    """dX: (N,4) = x_{t+1}-x_t. -> Ξ (n_feat,4). Threshold on NORMALIZED columns (scale-invariant)."""
    scale = np.linalg.norm(Theta, axis=0)
    scale[scale == 0] = 1.0
    Tn = Theta / scale
    n_feat, n_tgt = Tn.shape[1], dX.shape[1]
    Xi = _ridge_solve(Tn, dX, ridge)
    for _ in range(n_iter):
        small = np.abs(Xi) < threshold
        Xi[small] = 0.0
        for j in range(n_tgt):
            big = ~small[:, j]
            if big.any():
                Xi[big, j] = _ridge_solve(Tn[:, big], dX[:, j], ridge)
    return Xi / scale[:, None]                                   # back to original units


def fit_sindy(X, F, X_next, mode="physics", threshold=0.02, ridge=1e-6):
    """ -> (Ξ (n_feat,4), names). Learns the DELTA dX = X_next - X."""
    Theta, names = feature_library(X, F, mode)
    Xi = stlsq(Theta, (X_next - X).astype(np.float64), threshold, ridge)
    return Xi, names


# ---------------------------------------------------------------------------
# Rollout (discrete map)
# ---------------------------------------------------------------------------
def sindy_step(x, u, Xi, mode="physics"):
    """x: (B,4); u: (B,) action. -> x_next (B,4) = x + Θ(x,F)·Ξ."""
    Theta, _ = feature_library(x, action_to_force(u), mode)
    return x + Theta @ Xi


def sindy_rollout(x0, U_seq, Xi, mode="physics"):
    """x0: (B,4)· U_seq: (B,L) actions. -> preds (B,L,4) (free-running)."""
    x0 = np.atleast_2d(np.asarray(x0, np.float64))
    U_seq = np.atleast_2d(np.asarray(U_seq))
    B, L = U_seq.shape
    preds = np.empty((B, L, N_SUP), np.float64)
    x = x0.copy()
    for k in range(L):
        x = sindy_step(x, U_seq[:, k], Xi, mode)
        preds[:, k] = x
    return preds


# ---------------------------------------------------------------------------
# Data assembly from precomputed latent dirs (z, acts, states[RAW], x)
# ---------------------------------------------------------------------------
def assemble_fit_data(latent_dir, which, mean, std):
    """which ∈ {'encoded','gt'}. -> (X (M,4), F (M,), X_next (M,4)) in RAW physical units.
       encoded: z[:, :4]*std+mean (raw)·  gt: states (raw clean GT)."""
    mean4 = np.asarray(mean[:N_SUP], np.float64)
    std4 = np.asarray(std[:N_SUP], np.float64)
    Xs, Fs, Xns = [], [], []
    for f in list_npz(latent_dir):
        d = np.load(f)
        phys = (d["z"][:, :N_SUP] * std4 + mean4) if which == "encoded" else d["states"].astype(np.float64)
        acts = d["acts"]
        if phys.shape[0] < 2:
            continue
        Xs.append(phys[:-1]); Fs.append(action_to_force(acts[:-1])); Xns.append(phys[1:])
    if not Xs:
        raise RuntimeError(f"No usable episodes in {latent_dir}")
    return np.concatenate(Xs), np.concatenate(Fs), np.concatenate(Xns)


def assemble_windows(latent_dir, mean, std, seq_len=30, stride=1, which_seed="encoded"):
    """Windows with the SAME LOGIC as LatentSequenceDataset (same count/indexing).
       -> (seed_raw (N,4), U (N,L), gt_raw (N,L,4)) in RAW physical units.
       seed: encoded z[s,:4] (raw) or GT states[s];  gt: RAW states[s+1 : s+L+1]."""
    mean4 = np.asarray(mean[:N_SUP], np.float64)
    std4 = np.asarray(std[:N_SUP], np.float64)
    seeds, Us, gts = [], [], []
    for f in list_npz(latent_dir):
        d = np.load(f)
        z, acts, states = d["z"], d["acts"], d["states"].astype(np.float64)
        n = z.shape[0] - (seq_len + 1) + 1
        for s in range(0, max(n, 0), stride):
            seed = (z[s, :N_SUP] * std4 + mean4) if which_seed == "encoded" else states[s]
            seeds.append(seed)
            Us.append(acts[s:s + seq_len])
            gts.append(states[s + 1:s + seq_len + 1])
    if not seeds:
        raise RuntimeError(f"No windows from {latent_dir} (seq_len={seq_len} too large?)")
    return np.asarray(seeds), np.asarray(Us), np.asarray(gts)


# ---------------------------------------------------------------------------
# Metric helpers (standardized, same convention as the test_pX scripts)
# ---------------------------------------------------------------------------
def standardized_sq_err(pred_raw, gt_raw, std):
    """pred_raw, gt_raw: (N,L,4) RAW. -> (N,L,4) STANDARDIZED squared error (÷ std4)."""
    std4 = np.asarray(std[:N_SUP], np.float64)
    return (((pred_raw - gt_raw) / std4) ** 2)


def median_iqr(arr):
    return (np.median(arr, axis=0), np.percentile(arr, 25, axis=0), np.percentile(arr, 75, axis=0))


def bootstrap_paired(diff, n_boot, rng):
    """diff (N,L) -> median + 95% bootstrap CI (resample windows)."""
    N, L = diff.shape
    med = np.median(diff, axis=0)
    boots = np.empty((n_boot, L), np.float64)
    for b in range(n_boot):
        boots[b] = np.median(diff[rng.integers(0, N, size=N)], axis=0)
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return med, lo, hi


# ---------------------------------------------------------------------------
# Pretty-print discovered equations (interpretability!)
# ---------------------------------------------------------------------------
def format_equations(Xi, names, tol=1e-8):
    """ -> list of strings: 'Δ<dim> = c1*feat1 + ...' (non-zero terms only)."""
    lines = []
    for j, dim in enumerate(DIM_NAMES):
        terms = [f"{Xi[i, j]:+.4g}*{names[i]}" for i in range(len(names)) if abs(Xi[i, j]) > tol]
        lines.append(f"Δ{dim} = " + (" ".join(terms) if terms else "0"))
    return lines
