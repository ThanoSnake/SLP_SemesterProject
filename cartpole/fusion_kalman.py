"""
fusion_kalman.py — (E) Kalman/Bayesian fusion of SINDy <-> data (NO re-training).

Two variants (as requested):

  E1) FILTERING / STATE ESTIMATION  [process = SINDy, measurement = encoded z[:4] every frame]
      At each step: predict (SINDy EKF) -> update with the observation (de-std encoded z[:4]).
      GOAL: a denoised physical-state estimate combining physics + data. This is an ESTIMATION
      task (we have per-frame observations) -> shows SINDy+VAE as a state estimator/denoiser
      (answers Future Directions C & E of the paper). It shines on noisy images.

  E2) PREDICTIVE PSEUDO-MEASUREMENT  [process = SINDy, "measurement" = the LSTM prediction]
      Stays on the PURE-ROLLOUT task: SINDy predicts, and the LSTM's prediction is used as a
      noisy observation for correction. R_k (the LSTM's reliability) grows with the horizon
      (compounding) -> the filter automatically down-weights the LSTM far out: a probabilistic version
      of the horizon gate (C), but with covariance bookkeeping.

CALIBRATION (on val, covariances ONLY — no network retraining):
  * Q  = diag Var( SINDy one-step residual against GT )       (process-model error)
  * R_enc = diag Var( encoded z[:4] − GT )                   (encoder measurement noise)  [E1]
  * R_k   = diag Var( LSTM_pred(h=k) − GT ) per horizon k     (LSTM reliability)            [E2]

EKF: discrete map x_{k}=x_{k-1}+Θ(x_{k-1},F)·Ξ, with a NUMERICAL Jacobian (batched, vectorized).

Location: flat inside cartpole/. Run: !python3 cartpole/fusion_kalman.py
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from sindy_core import *          # SINDy core (numpy) + path bootstrap for vae/lstm/loader
from sindy_eval_utils import *    # VAE encode / LSTM rollout / noise / measurement helpers

from paths import BASELINE_LSTM, BASELINE_VAE, DATA_ROOT, outputs

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
NORM_STATS = os.path.join(DATA_ROOT, "norm_stats.npz")
VAE_CKPT = BASELINE_VAE
LSTM_CKPT = BASELINE_LSTM
LATENT_ROOT = outputs("sindy_latents")
SAVE_DIR = outputs("sindy_kalman_out")

SEQ_LEN, TEST_STRIDE, N_SUP = 30, 1, 4

FEATURE_MODE = "physics"
THRESHOLD, RIDGE = 0.02, 1e-6
JAC_EPS = 1e-5
VAR_FLOOR = 1e-8

NOISE_CONDS = [("gaussian", 0.0), ("gaussian", 0.05), ("gaussian", 0.10)]
NOISE_SEED = 42
N_BOOT, BOOT_SEED = 1000, 0

DIM_NAMES = ["x", "x_dot", "theta", "theta_dot"]
DIM_LABELS = ["x", r"$\dot{x}$", r"$\theta$", r"$\dot{\theta}$"]
DIM_UNITS = ["(cart pos)", "(cart vel)", "[rad]", "[rad/s]"]
HS = [h for h in (1, 10, 20, 30) if h <= SEQ_LEN]
LOG_Y = True


# ---------------------------------------------------------------------------
# EKF machinery (batched over windows)
# ---------------------------------------------------------------------------
def sindy_jacobian(x, u, Xi):
    """ -> (F (N,4,4) with F[:,i,j]=∂f_i/∂x_j, xpred (N,4))."""
    base = sindy_step(x, u, Xi, FEATURE_MODE)
    N = x.shape[0]
    Fj = np.empty((N, N_SUP, N_SUP))
    for j in range(N_SUP):
        xp = x.copy(); xp[:, j] += JAC_EPS
        Fj[:, :, j] = (sindy_step(xp, u, Xi, FEATURE_MODE) - base) / JAC_EPS
    return Fj, base


def ekf_run(seed, U, meas, Xi, Q, R, P0):
    """EKF: process = SINDy, measurement model H=I.
       R: (4,4) constant (E1) or (L,4,4) per step (E2). -> est (N,L,4)."""
    N, L, _ = meas.shape
    x = seed.astype(np.float64).copy()
    P = np.broadcast_to(np.asarray(P0, np.float64), (N, N_SUP, N_SUP)).copy()
    Qn = Q[None]
    I4 = np.eye(N_SUP)[None]
    est = np.empty((N, L, N_SUP))
    per_step_R = (R.ndim == 3)
    for k in range(L):
        Fk, xpred = sindy_jacobian(x, U[:, k], Xi)
        P = Fk @ P @ Fk.transpose(0, 2, 1) + Qn               # P⁻ = F P Fᵀ + Q
        Rk = R[k][None] if per_step_R else R[None]
        K = P @ np.linalg.inv(P + Rk)                         # H=I -> S=P+R
        innov = meas[:, k] - xpred
        x = xpred + np.einsum("nij,nj->ni", K, innov)
        P = (I4 - K) @ P
        est[:, k] = x
    return est


def diag_cov(resid):
    return np.diag(np.maximum(resid.reshape(-1, N_SUP).var(axis=0), VAR_FLOOR))


# ---------------------------------------------------------------------------
# Calibration of the covariances on val (GT used for calibration only)
# ---------------------------------------------------------------------------
def calibrate_Q(val_dir, Xi, mean, std):
    X, Fsig, Xn = assemble_fit_data(val_dir, "gt", mean, std)
    Theta, _ = feature_library(X, Fsig, FEATURE_MODE)
    resid = Xn - (X + Theta @ Xi)
    return diag_cov(resid)


def calibrate_R_enc(val_dir, mean, std):
    meas, _ = encoded_measurements(val_dir, mean, std)
    _, _, gt = assemble_windows(val_dir, mean, std, seq_len=SEQ_LEN,
                                   stride=TEST_STRIDE, which_seed="encoded")
    n = min(meas.shape[0], gt.shape[0])
    return diag_cov(meas[:n] - gt[:n])


def calibrate_R_lstm(lstm, val_dir, mean, std, device):
    """ -> R_k (L,4,4): per-horizon diag covariance of the LSTM error on val (RAW units)."""
    lstm_std, gt_std = lstm_free_run_dir(lstm, val_dir, mean, std, device)
    lstm_raw = destandardize(lstm_std, mean, std)
    gt_raw = destandardize(gt_std, mean, std)
    err = lstm_raw - gt_raw                                    # (N,L,4)
    L = err.shape[1]
    R = np.zeros((L, N_SUP, N_SUP))
    for k in range(L):
        R[k] = np.diag(np.maximum(err[:, k].var(axis=0), VAR_FLOOR))
    return R


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def _overall(curves, colors, tag, title, fname, save_dir, xlabel):
    horizons = np.arange(1, SEQ_LEN + 1)
    plt.figure(figsize=(6.8, 4.8))
    for name in curves:
        med, q25, q75 = median_iqr(curves[name].mean(axis=2))
        plt.plot(horizons, med, color=colors[name], lw=2, label=name)
        plt.fill_between(horizons, q25, q75, color=colors[name], alpha=0.12)
    if LOG_Y:
        plt.yscale("log")
    plt.title(title); plt.xlabel(xlabel)
    plt.ylabel("State MSE (median, IQR) [standardized]")
    plt.xlim(1, SEQ_LEN); plt.grid(alpha=0.3, which="both"); plt.legend()
    plt.tight_layout()
    p = os.path.join(save_dir, fname)
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print("saved:", p)


def _perdim(curves, colors, tag, title, fname, save_dir, xlabel):
    horizons = np.arange(1, SEQ_LEN + 1)
    fig, axes = plt.subplots(1, N_SUP, figsize=(4.2 * N_SUP, 4.0), squeeze=False)
    for d in range(N_SUP):
        ax = axes[0][d]
        for name in curves:
            med, q25, q75 = median_iqr(curves[name][:, :, d])
            ax.plot(horizons, med, color=colors[name], lw=2, label=name)
            ax.fill_between(horizons, q25, q75, color=colors[name], alpha=0.10)
        if LOG_Y:
            ax.set_yscale("log")
        ax.set_title(f"{DIM_LABELS[d]} {DIM_UNITS[d]}")
        ax.set_xlabel(xlabel); ax.set_xlim(1, SEQ_LEN); ax.grid(alpha=0.3, which="both")
        if d == 0:
            ax.set_ylabel("MSE (median, standardized)"); ax.legend(fontsize=8)
    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    p = os.path.join(save_dir, fname)
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("saved:", p)


def _summary(curves, label):
    for name in curves:
        m = np.median(curves[name].mean(axis=2), axis=0)
        print(f"  {name:<22} " + "  ".join(f"h{h}={m[h-1]:.5f}" for h in HS))


# ---------------------------------------------------------------------------
# E1: filtering / estimation
# ---------------------------------------------------------------------------
def run_filtering(test_dir, Xi, Q, R_enc, mean, std, tag, label, save_dir):
    meas, seed = encoded_measurements(test_dir, mean, std)
    seed_enc, U, gt = assemble_windows(test_dir, mean, std, seq_len=SEQ_LEN,
                                          stride=TEST_STRIDE, which_seed="encoded")
    n = min(meas.shape[0], gt.shape[0])
    meas, seed_enc, U, gt = meas[:n], seed_enc[:n], U[:n], gt[:n]

    est = ekf_run(seed_enc, U, meas, Xi, Q, R_enc, P0=R_enc)
    sindy_ol = sindy_rollout(seed_enc, U, Xi, mode=FEATURE_MODE)

    curves = {
        "raw encoder": sq_err_standardized(meas, gt, std),
        "SINDy open-loop": sq_err_standardized(sindy_ol, gt, std),
        "EKF (SINDy+encoder)": sq_err_standardized(est, gt, std),
    }
    colors = {"raw encoder": "C0", "SINDy open-loop": "C2", "EKF (SINDy+encoder)": "C3"}
    _overall(curves, colors, tag, f"E1 filtering — {label}",
             f"kf1_overall_{tag}.png", save_dir, "Step (within window)")
    _perdim(curves, colors, tag, f"E1 filtering (per-dim) — {label}",
            f"kf1_perdim_{tag}.png", save_dir, "Step")
    print(f"\n  [E1 filtering] median standardized MSE | {label}")
    _summary(curves, label)
    return {k: np.median(v.mean(axis=2), axis=0) for k, v in curves.items()}


# ---------------------------------------------------------------------------
# E2: predictive pseudo-measurement
# ---------------------------------------------------------------------------
def run_predictive(lstm, test_dir, Xi, Q, R_enc, R_lstm, mean, std, device, tag, label, save_dir, rng):
    lstm_std, gt_std = lstm_free_run_dir(lstm, test_dir, mean, std, device)
    lstm_raw = destandardize(lstm_std, mean, std)
    seed_enc, U, gt = assemble_windows(test_dir, mean, std, seq_len=SEQ_LEN,
                                          stride=TEST_STRIDE, which_seed="encoded")
    n = min(lstm_raw.shape[0], gt.shape[0])
    lstm_raw, lstm_std, gt_std = lstm_raw[:n], lstm_std[:n], gt_std[:n]
    seed_enc, U, gt = seed_enc[:n], U[:n], gt[:n]

    est = ekf_run(seed_enc, U, lstm_raw, Xi, Q, R_lstm, P0=R_enc)
    sindy_ol = sindy_rollout(seed_enc, U, Xi, mode=FEATURE_MODE)

    curves = {
        "LSTM": (lstm_std - gt_std) ** 2,
        "SINDy": sq_err_standardized(sindy_ol, gt, std),
        "Fused (KF)": sq_err_standardized(est, gt, std),
    }
    colors = {"LSTM": "C0", "SINDy": "C2", "Fused (KF)": "C1"}
    _overall(curves, colors, tag, f"E2 predictive fusion — {label}",
             f"kf2_overall_{tag}.png", save_dir, "Prediction Horizon")
    _perdim(curves, colors, tag, f"E2 predictive fusion (per-dim) — {label}",
            f"kf2_perdim_{tag}.png", save_dir, "Horizon")

    # paired Δ (best single − Fused)
    lstm_m = np.median(curves["LSTM"].mean(axis=2), axis=0).mean()
    sindy_m = np.median(curves["SINDy"].mean(axis=2), axis=0).mean()
    best = "LSTM" if lstm_m < sindy_m else "SINDy"
    diff = curves[best].mean(axis=2) - curves["Fused (KF)"].mean(axis=2)
    med_d, lo_d, hi_d = bootstrap_paired(diff, N_BOOT, rng)
    horizons = np.arange(1, SEQ_LEN + 1)
    plt.figure(figsize=(7.5, 4.8)); plt.axhline(0, color="k", lw=1)
    plt.plot(horizons, med_d, color="C1", lw=2, label=f"{best} − Fused")
    plt.fill_between(horizons, lo_d, hi_d, color="C1", alpha=0.18)
    plt.title(f"Paired Δ (>0 ⇒ Fused better than {best}) | {label}")
    plt.xlabel("Prediction Horizon"); plt.ylabel("Δ state-MSE")
    plt.xlim(1, SEQ_LEN); plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    p = os.path.join(save_dir, f"kf2_paired_{tag}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print("saved:", p)

    print(f"\n  [E2 predictive] median standardized MSE | {label}")
    _summary(curves, label)
    print(f"  Δ({best}−Fused)  " +
          "  ".join(f"h{h}={med_d[h-1]:+.5f}[{lo_d[h-1]:+.5f},{hi_d[h-1]:+.5f}]" for h in HS))
    out = {k: np.median(v.mean(axis=2), axis=0) for k, v in curves.items()}
    out["paired_best"] = best
    out["paired_median"] = med_d
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    mean, std = load_norm_stats(NORM_STATS)
    rng = np.random.default_rng(BOOT_SEED)

    train_dir = ensure_encoded(VAE_CKPT, DATA_ROOT,
                                   os.path.join(LATENT_ROOT, "clean"),
                                   device, noise_fn=None, splits=("train",))["train"]
    Xe, Fe, Xne = assemble_fit_data(train_dir, "encoded", mean, std)
    Xi_e, names = fit_sindy(Xe, Fe, Xne, mode=FEATURE_MODE, threshold=THRESHOLD, ridge=RIDGE)
    print("\n[SINDy enc-fit] discovered dynamics:")
    for line in format_equations(Xi_e, names):
        print("   " + line)

    lstm = load_lstm(LSTM_CKPT, device)
    save = {"horizons": np.arange(1, SEQ_LEN + 1)}

    for ntype, level in NOISE_CONDS:
        tag = noise_tag(ntype, level)
        label = "clean" if level == 0.0 else f"{ntype} sigma={level:.2f}"
        print(f"\n{'='*60}\n  CONDITION: {label}\n{'='*60}")

        nf = make_noise_fn(ntype, level, NOISE_SEED, device)
        val_dir = ensure_encoded(VAE_CKPT, DATA_ROOT, os.path.join(LATENT_ROOT, tag),
                                     device, noise_fn=nf, splits=("val",),
                                     force=(level != 0.0))["val"]
        test_dir = ensure_encoded(VAE_CKPT, DATA_ROOT, os.path.join(LATENT_ROOT, tag),
                                      device, noise_fn=nf, splits=("test",),
                                      force=(level != 0.0))["test"]

        # ---- calibrate the covariances on val ----
        Q = calibrate_Q(val_dir, Xi_e, mean, std)
        R_enc = calibrate_R_enc(val_dir, mean, std)
        R_lstm = calibrate_R_lstm(lstm, val_dir, mean, std, device)
        print("  Q diag   :", np.round(np.diag(Q), 6))
        print("  R_enc diag:", np.round(np.diag(R_enc), 6))

        e1 = run_filtering(test_dir, Xi_e, Q, R_enc, mean, std, tag, label, SAVE_DIR)
        e2 = run_predictive(lstm, test_dir, Xi_e, Q, R_enc, R_lstm, mean, std, device,
                            tag, label, SAVE_DIR, rng)

        for k, v in e1.items():
            save[f"{tag}__E1__{k}"] = v
        for k, v in e2.items():
            save[f"{tag}__E2__{k}"] = v

    np.savez(os.path.join(SAVE_DIR, "kalman_curves.npz"), **save)
    print("\nsaved figures + kalman_curves.npz ->", SAVE_DIR)


if __name__ == "__main__":
    main()
