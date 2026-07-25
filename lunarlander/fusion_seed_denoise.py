"""
fusion_seed_denoise.py — (F) SINDy seed-denoising BEFORE the LSTM rollout (NO re-training).

IDEA: the encoded seed z_0[:4] is noisy (especially the velocities — see test_baseline:
low R² on ẋ,θ̇). We use SINDy's ONE-STEP CONSISTENCY to "clean" it
BEFORE handing it to the (unchanged) baseline LSTM:

    physics_seed = sindy_step(z[s-1], u[s-1])          # predict the seed from the PREVIOUS frame
    seed_denoised = (1-α)·z[s][:4]  +  α·physics_seed   # convex blend (past information only!)

α∈[0,1] is tuned with ONE small fit on val (no network retraining). Then a normal
ENCODED rollout runs with seed[:4] replaced (the style dims z[4:] are left as they are).

It is essentially the minimal 1-step special case of (E) Kalman — targeted at the seed.

OUTPUTS per noise condition:
  (0) Per-dim seed RMSE: raw-seed vs denoised-seed (how much the seed was cleaned)
  (1) Overall median+IQR rollout MSE: LSTM(raw seed) vs LSTM(denoised seed)
  (2) Per-dim median+IQR; (3) Paired Δ (raw − denoised, >0 => denoising helps)

ENV-AGNOSTIC: the SAME file for cartpole & lunarlander; the environment comes from sindy_core/sindy_eval_utils.
Run: !python3 <env-folder>/fusion_seed_denoise.py
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from sindy_core import *          # SINDy core (numpy) + path bootstrap for vae/lstm/loader
from sindy_eval_utils import *    # VAE encode / LSTM rollout / noise / measurement helpers

from paths import outputs

# ---------------------------------------------------------------------------
# CONFIG — env-agnostic; the per-environment config comes from sindy_eval_utils/sindy_core (import *).
# ---------------------------------------------------------------------------
SAVE_DIR = outputs(f"sindy_{ENV_TAG}_seed_denoise_out")
ALPHA_GRID = np.round(np.linspace(0.0, 1.0, 11), 3)     # tuning of α on val
LOG_Y = True
C_RAW, C_DEN = "C0", "C3"


# ---------------------------------------------------------------------------
# Denoised seed (standardized) from context + Ξ + α
# ---------------------------------------------------------------------------
def denoised_seed_std(ctx, Xi, alpha, mean, std):
    mean4 = np.asarray(mean[:N_SUP], np.float64)
    std4 = np.asarray(std[:N_SUP], np.float64)
    physics = sindy_step(ctx["prev_raw"], ctx["prev_act"], Xi, mode=FEATURE_MODE)  # (N,4) raw
    den = ctx["enc_seed_raw"].copy()
    use = ctx["has_prev"]
    den[use] = (1.0 - alpha) * ctx["enc_seed_raw"][use] + alpha * physics[use]
    return ((den - mean4) / std4).astype(np.float32)


def tune_alpha(lstm, val_dir, Xi, mean, std, device):
    """Picks the α minimizing the mean standardized rollout MSE on val."""
    ctx = seed_context(val_dir, mean, std)
    best_a, best_mse = 0.0, np.inf
    scores = []
    for a in ALPHA_GRID:
        seed_std = denoised_seed_std(ctx, Xi, a, mean, std)
        pred, gt = lstm_free_run_dir(lstm, val_dir, mean, std, device, seed_phys_std=seed_std)
        mse = float(((pred - gt) ** 2).mean())
        scores.append(mse)
        if mse < best_mse:
            best_mse, best_a = mse, float(a)
    print("  val alpha sweep:", "  ".join(f"{a:.1f}->{s:.5f}" for a, s in zip(ALPHA_GRID, scores)))
    print(f"  -> best alpha = {best_a:.2f} (val MSE {best_mse:.5f})")
    return best_a


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_overall(err, tag, label, save_dir):
    horizons = np.arange(1, SEQ_LEN + 1)
    plt.figure(figsize=(6.8, 4.8))
    for name, e, c in [("LSTM (raw seed)", err["raw"], C_RAW),
                       ("LSTM (SINDy-denoised seed)", err["den"], C_DEN)]:
        med, q25, q75 = median_iqr(e.mean(axis=2))
        plt.plot(horizons, med, color=c, lw=2, label=name)
        plt.fill_between(horizons, q25, q75, color=c, alpha=0.12)
    if LOG_Y:
        plt.yscale("log")
    plt.title(f"Seed-denoise (F) — median state-MSE | {label}")
    plt.xlabel("Prediction Horizon"); plt.ylabel("State MSE (median, IQR) [standardized]")
    plt.xlim(1, SEQ_LEN); plt.grid(alpha=0.3, which="both"); plt.legend()
    plt.tight_layout()
    p = os.path.join(save_dir, f"seed_overall_{tag}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print("saved:", p)


def plot_perdim(err, tag, label, save_dir):
    horizons = np.arange(1, SEQ_LEN + 1)
    fig, axes = plt.subplots(1, N_SUP, figsize=(4.2 * N_SUP, 4.0), squeeze=False)
    for d in range(N_SUP):
        ax = axes[0][d]
        for name, e, c in [("raw seed", err["raw"], C_RAW),
                           ("denoised seed", err["den"], C_DEN)]:
            med, q25, q75 = median_iqr(e[:, :, d])
            ax.plot(horizons, med, color=c, lw=2, label=name)
            ax.fill_between(horizons, q25, q75, color=c, alpha=0.12)
        if LOG_Y:
            ax.set_yscale("log")
        ax.set_title(f"{DIM_LABELS[d]} {DIM_UNITS[d]}")
        ax.set_xlabel("Horizon"); ax.set_xlim(1, SEQ_LEN); ax.grid(alpha=0.3, which="both")
        if d == 0:
            ax.set_ylabel("MSE (median, standardized)"); ax.legend(fontsize=8)
    plt.suptitle(f"Seed-denoise (F) per-dim | {label}", y=1.02)
    plt.tight_layout()
    p = os.path.join(save_dir, f"seed_perdim_{tag}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("saved:", p)


def plot_paired(err, tag, label, save_dir, rng):
    diff = err["raw"].mean(axis=2) - err["den"].mean(axis=2)
    med, lo, hi = bootstrap_paired(diff, N_BOOT, rng)
    horizons = np.arange(1, SEQ_LEN + 1)
    plt.figure(figsize=(7.5, 4.8))
    plt.axhline(0, color="k", lw=1)
    plt.plot(horizons, med, color=C_DEN, lw=2, label="raw − denoised")
    plt.fill_between(horizons, lo, hi, color=C_DEN, alpha=0.18)
    plt.title(f"Paired Δ (>0 ⇒ denoise helps) | {label}")
    plt.xlabel("Prediction Horizon"); plt.ylabel("Δ state-MSE")
    plt.xlim(1, SEQ_LEN); plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout()
    p = os.path.join(save_dir, f"seed_paired_{tag}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print("saved:", p)
    return med, lo, hi


def seed_rmse(seed_std_raw, seed_gt_raw, denoised_std, mean, std):
    """Per-dim standardized RMSE seed-vs-GT (raw vs denoised)."""
    mean4 = np.asarray(mean[:N_SUP], np.float64)
    std4 = np.asarray(std[:N_SUP], np.float64)
    gt_std = (seed_gt_raw - mean4) / std4
    raw_std = (seed_std_raw - mean4) / std4
    raw_rmse = np.sqrt(((raw_std - gt_std) ** 2).mean(0))
    den_rmse = np.sqrt(((denoised_std - gt_std) ** 2).mean(0))
    return raw_rmse, den_rmse


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

    # ---- fit SINDy (encoded) on the CLEAN train split ----
    train_dir = ensure_encoded(VAE_CKPT, DATA_ROOT,
                                   os.path.join(LATENT_ROOT, "clean"),
                                   device, noise_fn=None, splits=("train",))["train"]
    Xe, Fe, Xne = assemble_fit_data(train_dir, "encoded", mean, std)
    Xi_e, names = fit_sindy(Xe, Fe, Xne, mode=FEATURE_MODE, threshold=THRESHOLD, ridge=RIDGE)

    lstm = load_lstm(LSTM_CKPT, device)
    save = {"horizons": np.arange(1, SEQ_LEN + 1)}

    for ntype, level in NOISE_CONDS:
        tag = noise_tag(ntype, level)
        label = "clean" if level == 0.0 else f"{ntype} σ={level:.2f}"
        print(f"\n{'='*60}\n  CONDITION: {label}\n{'='*60}")

        nf = make_noise_fn(ntype, level, NOISE_SEED, device)
        # noisy encode of val (for the α tuning) + test (for evaluation)
        val_dir = ensure_encoded(VAE_CKPT, DATA_ROOT, os.path.join(LATENT_ROOT, tag),
                                     device, noise_fn=nf, splits=("val",),
                                     force=(level != 0.0))["val"]
        test_dir = ensure_encoded(VAE_CKPT, DATA_ROOT, os.path.join(LATENT_ROOT, tag),
                                      device, noise_fn=nf, splits=("test",),
                                      force=(level != 0.0))["test"]

        alpha = tune_alpha(lstm, val_dir, Xi_e, mean, std, device)

        # ---- evaluate on test ----
        ctx = seed_context(test_dir, mean, std)
        seed_std = denoised_seed_std(ctx, Xi_e, alpha, mean, std)
        pred_raw, gt = lstm_free_run_dir(lstm, test_dir, mean, std, device)            # raw seed
        pred_den, _ = lstm_free_run_dir(lstm, test_dir, mean, std, device,
                                            seed_phys_std=seed_std)                        # denoised
        err = {"raw": (pred_raw - gt) ** 2, "den": (pred_den - gt) ** 2}

        # seed quality (vs GT seed = states[s])
        seed_gt_raw, _, _ = assemble_windows(test_dir, mean, std, seq_len=SEQ_LEN,
                                                stride=TEST_STRIDE, which_seed="gt")
        raw_rmse, den_rmse = seed_rmse(ctx["enc_seed_raw"], seed_gt_raw, seed_std, mean, std)

        plot_overall(err, tag, label, SAVE_DIR)
        plot_perdim(err, tag, label, SAVE_DIR)
        med_d, lo_d, hi_d = plot_paired(err, tag, label, SAVE_DIR, rng)

        print(f"  alpha*={alpha:.2f}  seed RMSE(std) per-dim  raw -> denoised:")
        for d in range(N_SUP):
            print(f"    {DIM_NAMES[d]:<10} {raw_rmse[d]:.4f} -> {den_rmse[d]:.4f}")
        for name in ("raw", "den"):
            m = np.median(err[name].mean(axis=2), axis=0)
            print(f"  {('raw' if name=='raw' else 'denoised'):<10} " +
                  "  ".join(f"h{h}={m[h-1]:.5f}" for h in HS))
        print("  Δ(raw−den)  " +
              "  ".join(f"h{h}={med_d[h-1]:+.5f}[{lo_d[h-1]:+.5f},{hi_d[h-1]:+.5f}]" for h in HS))

        save[f"{tag}__alpha"] = alpha
        save[f"{tag}__raw_median"] = np.median(err["raw"].mean(axis=2), axis=0)
        save[f"{tag}__den_median"] = np.median(err["den"].mean(axis=2), axis=0)
        save[f"{tag}__paired_median"] = med_d
        save[f"{tag}__seed_rmse_raw"] = raw_rmse
        save[f"{tag}__seed_rmse_den"] = den_rmse

    np.savez(os.path.join(SAVE_DIR, "seed_denoise_curves.npz"), **save)
    print("\nsaved figures + seed_denoise_curves.npz ->", SAVE_DIR)


if __name__ == "__main__":
    main()
