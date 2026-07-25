"""
sindy_core.py — Shared SINDy backbone (from-scratch, numpy-only) for the LunarLander world model.

What it does: learns INTERPRETABLE dynamics on the 8-dim physical state [x, y, vx, vy, θ, ω, leg1, leg2]
with sparse regression (STLSQ), in DISCRETE next-state / DELTA form:
        x_{t+1} = x_t + Θ(x_t, u_t) · Ξ          (Ξ sparse -> few, readable equations)
This is the dyn_phys: ℝ⁸ → ℝ⁸ of the paper's Definition 2 — the physical counterpart of the LSTM.

DIFFERENCES from the CartPole core (why it needs its own file):
  * N_SUP = 8 (instead of 4) — the full LunarLander state.
  * 4 DISCRETE actions (instead of 2): {0:noop, 1:left, 2:main, 3:right}. There is no signed "force";
    instead we use thrust indicators (a_main, a_left, a_right) as features.
  * A "physics" feature library specific to LunarLander: kinematics (vx,vy,ω), gravity (constant),
    main-engine thrust projected onto the axes (a_main·sinθ, a_main·cosθ), side engines.

The generic parts (STLSQ, fit, rollout, windowing, metrics) follow the same logic as the CartPole
core — just parameterized on this environment's N_SUP/feature_library/actions.

numpy-only (no torch/pysindy) -> runs anywhere, fully transparent.
"""
import os
import sys

import numpy as np
from itertools import combinations_with_replacement

# --- path bootstrap (robust): put on sys.path both the SINDy siblings' folder AND the folder that
#     contains vae/lstm/loader — works either flat (vae alongside) or in a subfolder (vae in the parent,
#     e.g. lunarlander/extra/ -> vae in lunarlander/) ---
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
_d = _HERE
for _ in range(4):
    if _d not in sys.path:
        sys.path.insert(0, _d)
    if os.path.exists(os.path.join(_d, "vae.py")):
        break
    _parent = os.path.dirname(_d)
    if _parent == _d:
        break
    _d = _parent


N_SUP = 8
N_ACTIONS = 4
DT = 0.02                                   # LunarLander tau (dt is absorbed into Ξ)
DIM_NAMES = ["x", "y", "vx", "vy", "theta", "omega", "leg1", "leg2"]
BASE_NAMES = ["x", "y", "vx", "vy", "theta", "omega", "leg1", "leg2", "a_main", "a_left", "a_right"]


# ---------------------------------------------------------------------------
# IO helpers (numpy-only — same as the CartPole core)
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


def action_indicators(u):
    """action {0:noop,1:left,2:main,3:right} -> (a_main, a_left, a_right), each (N,) ∈ {0,1}.
    The thrust magnitude is learned in Ξ; the actions enter as indicators (not a signed scalar)."""
    u = np.asarray(u, np.float64).reshape(-1)
    return (u == 2).astype(np.float64), (u == 1).astype(np.float64), (u == 3).astype(np.float64)


# ---------------------------------------------------------------------------
# Feature library  Θ(x, u)  — LunarLander-aware
# ---------------------------------------------------------------------------
def feature_library(X, U, mode="physics"):
    """X: (N,8) raw [x,y,vx,vy,θ,ω,leg1,leg2]; U: (N,) raw action. -> (Θ (N,n_feat), names)."""
    X = np.atleast_2d(np.asarray(X, np.float64))
    x, y, vx, vy = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    th, om, l1, l2 = X[:, 4], X[:, 5], X[:, 6], X[:, 7]
    am, al, ar = action_indicators(U)

    if mode == "physics":
        s, c = np.sin(th), np.cos(th)
        feats = {
            "1": np.ones_like(x),
            "x": x, "y": y, "vx": vx, "vy": vy,
            "theta": th, "omega": om, "leg1": l1, "leg2": l2,
            "sin": s, "cos": c,
            "a_main": am, "a_left": al, "a_right": ar,
            "a_main*sin": am * s, "a_main*cos": am * c,   # main thrust projected onto the axes
        }
        names = list(feats.keys())
        Theta = np.stack([feats[n] for n in names], axis=1)
        return Theta, names

    if mode == "poly2":
        base = np.stack([x, y, vx, vy, th, om, l1, l2, am, al, ar], axis=1)   # (N,11)
        cols, names = [np.ones_like(x)], ["1"]
        for i in range(len(BASE_NAMES)):
            cols.append(base[:, i]); names.append(BASE_NAMES[i])
        for i, j in combinations_with_replacement(range(len(BASE_NAMES)), 2):
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
    """dX: (N,8) = x_{t+1}-x_t. -> Ξ (n_feat,8). Threshold on NORMALIZED columns (scale-invariant)."""
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


def fit_sindy(X, U, X_next, mode="physics", threshold=0.02, ridge=1e-6):
    """ -> (Ξ (n_feat,8), names). Learns the DELTA dX = X_next - X."""
    Theta, names = feature_library(X, U, mode)
    Xi = stlsq(Theta, (X_next - X).astype(np.float64), threshold, ridge)
    return Xi, names


# ---------------------------------------------------------------------------
# Rollout (discrete map)
# ---------------------------------------------------------------------------
def sindy_step(x, u, Xi, mode="physics"):
    """x: (B,8); u: (B,) raw action. -> x_next (B,8) = x + Θ(x,u)·Ξ."""
    Theta, _ = feature_library(x, u, mode)
    return x + Theta @ Xi


def sindy_rollout(x0, U_seq, Xi, mode="physics"):
    """x0: (B,8)· U_seq: (B,L) actions. -> preds (B,L,8) (free-running)."""
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
    """which ∈ {'encoded','gt'}. -> (X (M,8), U (M,), X_next (M,8)) in RAW physical units.
       U = RAW actions (not a signed force — the thrust indicators are built in feature_library).
       encoded: z[:, :8]*std+mean (raw)·  gt: states (raw clean GT)."""
    mean8 = np.asarray(mean[:N_SUP], np.float64)
    std8 = np.asarray(std[:N_SUP], np.float64)
    Xs, Us, Xns = [], [], []
    for f in list_npz(latent_dir):
        d = np.load(f)
        phys = (d["z"][:, :N_SUP] * std8 + mean8) if which == "encoded" else d["states"].astype(np.float64)
        acts = d["acts"]
        if phys.shape[0] < 2:
            continue
        Xs.append(phys[:-1]); Us.append(acts[:-1]); Xns.append(phys[1:])
    if not Xs:
        raise RuntimeError(f"No usable episodes in {latent_dir}")
    return np.concatenate(Xs), np.concatenate(Us), np.concatenate(Xns)


def assemble_windows(latent_dir, mean, std, seq_len=30, stride=1, which_seed="encoded"):
    """Windows with the SAME LOGIC as LatentSequenceDataset (same count/indexing).
       -> (seed_raw (N,8), U (N,L) RAW actions, gt_raw (N,L,8)) in RAW physical units.
       seed: encoded z[s,:8] (raw) or GT states[s];  gt: RAW states[s+1 : s+L+1]."""
    mean8 = np.asarray(mean[:N_SUP], np.float64)
    std8 = np.asarray(std[:N_SUP], np.float64)
    seeds, Us, gts = [], [], []
    for f in list_npz(latent_dir):
        d = np.load(f)
        z, acts, states = d["z"], d["acts"], d["states"].astype(np.float64)
        n = z.shape[0] - (seq_len + 1) + 1
        for s in range(0, max(n, 0), stride):
            seed = (z[s, :N_SUP] * std8 + mean8) if which_seed == "encoded" else states[s]
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
    """pred_raw, gt_raw: (N,L,8) RAW. -> (N,L,8) STANDARDIZED squared error (÷ std8)."""
    std8 = np.asarray(std[:N_SUP], np.float64)
    return (((pred_raw - gt_raw) / std8) ** 2)


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
