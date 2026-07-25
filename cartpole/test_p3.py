"""
test_p3.py — Evaluation of Baseline vs P3-semi vs P3-weak on NOISY images (CartPole).

FOCUSED VERSION (single setting) — same structure/plots as test_p1/test_p2, but with THREE
models (Baseline, P3 semi, P3 weak; same monolithic VAE, only the SUPERVISION differs):
  * ONLY a small gaussian noise σ=0.05.
  * ONLY "encoded" seed mode (z_0 from the VAE -> LSTM rollout).
  * Noise is applied EXCLUSIVELY at the encoding stage (precompute_latents), before the encoder.

OUTPUTS:
  (1) Overall median+IQR state-MSE per horizon (mean over dims) — 3 curves   [standardized]
  (2) Per-dim median+IQR state-MSE per horizon — 3 curves                    [standardized]
  (3) Paired Δ (Baseline − semi) and (Baseline − weak) + 95% bootstrap CI
  (4) PHYSICAL QUANTITIES of a RANDOM test window: GT vs baseline vs semi vs weak  [physical units]

NOTE on units: the MSE curves (1–3) are STANDARDIZED (as in the other test_pX). The
trajectory (4) is in PHYSICAL units (de-standardized).
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from vae import VAE
from lstm import LatentPredictor
from loader import LatentSequenceDataset, load_norm_stats, list_npz

from paths import BASELINE_LSTM, BASELINE_VAE, DATA_ROOT, P3_SEMI_LSTM, P3_SEMI_VAE, P3_WEAK_LSTM, P3_WEAK_VAE, outputs

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
NORM_STATS = os.path.join(DATA_ROOT, "norm_stats.npz")
SAVE_DIR = outputs("cartpole_p3_out")

SHIFT = 0
LATENT_SIZE, N_SUP, N_IMG = 64, 4, 60
N_ACTIONS, HIDDEN, LAYERS = 2, 64, 2
SEQ_LEN, TEST_STRIDE, BATCH = 30, 1, 128

DIM_NAMES = ["x", "x_dot", "theta", "theta_dot"]
DIM_LABELS = ["x", r"$\dot{x}$", r"$\theta$", r"$\dot{\theta}$"]
DIM_UNITS = ["(cart pos)", "(cart vel)", "[rad]", "[rad/s]"]
N_BOOT = 1000
BOOT_SEED = 0
LOG_Y = True

# ---------------------------------------------------------------------------
# NOISE CONFIG — a single setting: small gaussian σ=0.05
# ---------------------------------------------------------------------------
NOISE_TYPE = "gaussian"           # "gaussian" | "salt_pepper"
NOISE_SIGMA = 0.05                # std (gaussian) on a [0,1] image
NOISE_SEED = 42                   # reproducible noise

# Trajectory plot (4): a random test window
TRAJ_SEED = None                  # None -> GENUINELY random (a different window every run); int -> reproducible
TRAJ_WINDOW = None                # None -> random; or an integer index for a specific window
N_TRAJ_WINDOWS = 1

# ---------------------------------------------------------------------------
# Model definitions — Baseline vs P3 semi vs P3 weak
# (the SAME monolithic VAE class; only the supervision differs -> different checkpoints)
# ---------------------------------------------------------------------------
MODELS = [
    {"label": "Baseline", "color": "C0",
     "make_vae": lambda: VAE(latent_size=LATENT_SIZE),
     "vae_ckpt": BASELINE_VAE,
     "lstm_ckpt": BASELINE_LSTM,
     "latent_root": outputs("cartpole_p3_latents/baseline")},
    {"label": "P3 semi", "color": "C1",
     "make_vae": lambda: VAE(latent_size=LATENT_SIZE),
     "vae_ckpt": P3_SEMI_VAE,
     "lstm_ckpt": P3_SEMI_LSTM,
     "latent_root": outputs("cartpole_p3_latents/p3_semi")},
    {"label": "P3 weak", "color": "C3",
     "make_vae": lambda: VAE(latent_size=LATENT_SIZE),
     "vae_ckpt": P3_WEAK_VAE,
     "lstm_ckpt": P3_WEAK_LSTM,
     "latent_root": outputs("cartpole_p3_latents/p3_weak")},
]


# ---------------------------------------------------------------------------
# Noise injection — float [0,1] image tensors
# ---------------------------------------------------------------------------
def add_gaussian_noise(img_tensor, std, rng_gen):
    noise = torch.randn(img_tensor.shape, generator=rng_gen, device=img_tensor.device) * std
    return torch.clamp(img_tensor + noise, 0.0, 1.0)


def add_salt_pepper_noise(img_tensor, amount, rng_gen):
    mask = torch.rand(img_tensor.shape, generator=rng_gen, device=img_tensor.device)
    out = img_tensor.clone()
    out[mask < amount / 2] = 0.0
    out[mask > 1 - amount / 2] = 1.0
    return out


def make_noise_fn(noise_type, level, seed, device):
    rng_gen = torch.Generator(device=device)
    rng_gen.manual_seed(seed)
    if level == 0.0:
        return lambda x: x
    if noise_type == "gaussian":
        return lambda x: add_gaussian_noise(x, level, rng_gen)
    elif noise_type == "salt_pepper":
        return lambda x: add_salt_pepper_noise(x, level, rng_gen)
    raise ValueError(f"Unknown noise type: {noise_type}")


# ---------------------------------------------------------------------------
# NOISY precompute_latents — applies the noise BEFORE encoding
# ---------------------------------------------------------------------------
@torch.no_grad()
def precompute_latents_noisy(encode_fn, root, out_root, noise_fn,
                             shift=0, batch=256, device="cuda"):
    from os.path import join, basename
    from os import makedirs
    makedirs(out_root, exist_ok=True)
    for f in list_npz(root):
        with np.load(f) as d:
            imgs = torch.from_numpy(d["imgs"].astype(np.float32) / 255.0).permute(0, 3, 1, 2)
            acts = d["acts"].astype(np.float32)
            states = d["states"].astype(np.float32)
            x = (d[f"noisy_states_{shift}"] if shift in (2, 5, 10)
                 else d["states"]).astype(np.float32)

        imgs = noise_fn(imgs.to(device))                       # noise on ALL frames before encoding
        img_t, img_tp1 = imgs[:-1], imgs[1:]
        zs = []
        for b in range(0, img_t.shape[0], batch):
            zb = encode_fn(img_t[b:b + batch], img_tp1[b:b + batch])
            zs.append(zb.cpu().numpy())
        z = np.concatenate(zs, 0).astype(np.float32) if zs else np.empty((0, 0), np.float32)
        np.savez_compressed(join(out_root, basename(f)),
                            z=z, acts=acts[:-1], states=states[:-1], x=x[:-1])


# ---------------------------------------------------------------------------
# Rollout (ENCODED) + collection of predicted physical dims & GT
# ---------------------------------------------------------------------------
@torch.no_grad()
def free_run(model, batch):
    """ENCODED free-running rollout: seed z_0 = z_t[:,0], own prediction fed back."""
    z_t, action, z_tp1, state_t, state_tp1 = batch
    B, L, _ = z_t.shape
    z_in = z_t[:, 0]
    hidden = model.init_hidden(B, z_t.device)
    preds = []
    for k in range(L):
        a = F.one_hot(action[:, k].long(), N_ACTIONS).float()
        z_pred, hidden = model.step(z_in, a, hidden)
        preds.append(z_pred)
        z_in = z_pred
    return torch.stack(preds, dim=1), state_tp1


@torch.no_grad()
def collect_preds_gt(model, loader, device):
    """ -> (preds (N,L,N_SUP), gt (N,L,N_SUP)), STANDARDIZED physical dims."""
    model.eval()
    P, G = [], []
    for batch in loader:
        batch = [b.to(device, non_blocking=True) for b in batch]
        preds, state_tp1 = free_run(model, batch)
        P.append(preds[..., :N_SUP].cpu().numpy())
        G.append(state_tp1.cpu().numpy())
    return np.concatenate(P, 0), np.concatenate(G, 0)


# ---------------------------------------------------------------------------
# Robust statistics
# ---------------------------------------------------------------------------
def median_iqr(arr):
    """arr (N,L) -> median, q25, q75 per horizon."""
    return (np.median(arr, axis=0),
            np.percentile(arr, 25, axis=0),
            np.percentile(arr, 75, axis=0))


def bootstrap_paired(diff, n_boot, rng):
    """diff (N,L) = mse_base − mse_variant per window/horizon (>0 => variant better).
    -> median(diff) per horizon + 95% bootstrap CI (resample windows)."""
    N, L = diff.shape
    med = np.median(diff, axis=0)
    boots = np.empty((n_boot, L), dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)
        boots[b] = np.median(diff[idx], axis=0)
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return med, lo, hi


# ---------------------------------------------------------------------------
# Model evaluation at the single noise setting
# ---------------------------------------------------------------------------
def evaluate_model_noisy(m, device, mean_s, std_s):
    """Load VAE, encode test images WITH gaussian σ=NOISE_SIGMA, rollout LSTM (encoded)
    -> {"pred": (N,L,4), "gt": (N,L,4)} standardized."""
    noise_fn = make_noise_fn(NOISE_TYPE, NOISE_SIGMA, NOISE_SEED, device)
    noise_tag = f"{NOISE_TYPE}_{NOISE_SIGMA:.2f}".replace(".", "p")

    print(f"\n[{m['label']}] noise={NOISE_TYPE} σ={NOISE_SIGMA:.2f}")
    print(f"  VAE ({m['vae_ckpt']}) -> precompute test latents (noisy)")
    vae = m["make_vae"]().to(device)
    vae.load_state_dict(torch.load(m["vae_ckpt"], map_location=device)); vae.eval()

    @torch.no_grad()
    def _encode(img_t, img_tp1):
        vae.eval()
        x = torch.cat([img_t, img_tp1], dim=1).to(device)
        mu, _ = vae.encode(x)
        return mu

    out_test = os.path.join(m["latent_root"], noise_tag, "test")
    precompute_latents_noisy(_encode, os.path.join(DATA_ROOT, "test"),
                             out_test, noise_fn=noise_fn, shift=SHIFT, device=device)
    del vae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    test_ds = LatentSequenceDataset(out_test, seq_len=SEQ_LEN, stride=TEST_STRIDE,
                                    state_mean=mean_s, state_std=std_s)
    test_dl = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)
    print(f"  test windows: {len(test_ds)}")

    lstm = LatentPredictor(LATENT_SIZE, N_ACTIONS, HIDDEN, LAYERS).to(device)
    lstm.load_state_dict(torch.load(m["lstm_ckpt"], map_location=device))
    pred, gt = collect_preds_gt(lstm, test_dl, device)
    print(f"  encoded rollout -> pred {pred.shape}")
    del lstm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"pred": pred, "gt": gt}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_median_iqr(err, save_dir):
    """(1) Overall median+IQR state-MSE (mean over dims) — all models."""
    horizons = np.arange(1, SEQ_LEN + 1)
    plt.figure(figsize=(6.8, 4.8))
    for m in MODELS:
        arr = err[m["label"]].mean(axis=2)
        med, q25, q75 = median_iqr(arr)
        plt.plot(horizons, med, color=m["color"], lw=2, label=m["label"])
        plt.fill_between(horizons, q25, q75, color=m["color"], alpha=0.15)
    if LOG_Y:
        plt.yscale("log")
    plt.title(f"median state-MSE (encoded) | {NOISE_TYPE} σ={NOISE_SIGMA:.2f}")
    plt.xlabel("Prediction Horizon"); plt.ylabel("State MSE (median, IQR band)")
    plt.xlim(1, SEQ_LEN); plt.grid(alpha=0.3, which="both"); plt.legend()
    plt.tight_layout()
    p = os.path.join(save_dir, "p3_median_iqr_encoded.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print("saved:", p)


def plot_perdim(err, save_dir):
    """(2) Per-dim median+IQR state-MSE — all models."""
    horizons = np.arange(1, SEQ_LEN + 1)
    fig, axes = plt.subplots(1, N_SUP, figsize=(4.2 * N_SUP, 4.0), squeeze=False)
    for d in range(N_SUP):
        ax = axes[0][d]
        for m in MODELS:
            med, q25, q75 = median_iqr(err[m["label"]][:, :, d])
            ax.plot(horizons, med, color=m["color"], lw=2, label=m["label"])
            ax.fill_between(horizons, q25, q75, color=m["color"], alpha=0.15)
        if LOG_Y:
            ax.set_yscale("log")
        ax.set_title(f"{DIM_LABELS[d]} {DIM_UNITS[d]}")
        ax.set_xlabel("Horizon"); ax.set_xlim(1, SEQ_LEN); ax.grid(alpha=0.3, which="both")
        if d == 0:
            ax.set_ylabel("MSE (median, standardized)"); ax.legend()
    plt.suptitle(f"Per-dim state-MSE (encoded) | {NOISE_TYPE} σ={NOISE_SIGMA:.2f}", y=1.02)
    plt.tight_layout()
    p = os.path.join(save_dir, "p3_perdim_encoded.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("saved:", p)


def plot_paired(err, save_dir, rng):
    """(3) Paired Δ (Baseline − variant) median + 95% bootstrap CI, for each variant."""
    base = MODELS[0]["label"]
    variants = MODELS[1:]
    horizons = np.arange(1, SEQ_LEN + 1)
    paired = {}

    plt.figure(figsize=(7.5, 4.8))
    plt.axhline(0, color="k", lw=1)
    for v in variants:
        diff = err[base].mean(axis=2) - err[v["label"]].mean(axis=2)
        med, lo, hi = bootstrap_paired(diff, N_BOOT, rng)
        paired[v["label"]] = (med, lo, hi)
        plt.plot(horizons, med, color=v["color"], lw=2, label=f"{base} − {v['label']}")
        plt.fill_between(horizons, lo, hi, color=v["color"], alpha=0.18)
    plt.title(f"Paired Δ (>0 ⇒ variant better) — encoded | {NOISE_TYPE} σ={NOISE_SIGMA:.2f}")
    plt.xlabel("Prediction Horizon"); plt.ylabel("Δ state-MSE")
    plt.xlim(1, SEQ_LEN); plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout()
    p = os.path.join(save_dir, "p3_paired_encoded.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print("saved:", p)
    return paired


def plot_trajectory(data, mean_s, std_s, save_dir, rng):
    """(4) Physical trajectory of a RANDOM window: GT vs baseline vs semi vs weak."""
    base = MODELS[0]["label"]
    mean4 = np.asarray(mean_s[:N_SUP], np.float64)
    std4 = np.asarray(std_s[:N_SUP], np.float64)
    gt_all = data[base]["gt"]                                  # (N,L,4) standardized (shared GT)
    N, L, _ = gt_all.shape
    horizons = np.arange(1, L + 1)

    for wi in range(N_TRAJ_WINDOWS):
        w = TRAJ_WINDOW if TRAJ_WINDOW is not None else int(rng.integers(0, N))
        for m in MODELS[1:]:                                   # GT must be identical across models
            if not np.allclose(data[base]["gt"][w], data[m["label"]]["gt"][w], atol=1e-4):
                print(f"[warn] window {w}: GT differs ({base} vs {m['label']}) — window alignment?")

        gt_phys = gt_all[w] * std4 + mean4
        preds_phys = {m["label"]: data[m["label"]]["pred"][w] * std4 + mean4 for m in MODELS}

        fig, axes = plt.subplots(2, 2, figsize=(12, 6))
        for d in range(N_SUP):
            ax = axes[d // 2][d % 2]
            ax.plot(horizons, gt_phys[:, d], color="k", lw=2.0, label="GT")
            for m in MODELS:
                ax.plot(horizons, preds_phys[m["label"]][:, d], color=m["color"],
                        lw=1.5, ls="--", label=m["label"])
            ax.set_title(f"{DIM_LABELS[d]} {DIM_UNITS[d]}")
            ax.set_xlabel("Prediction Horizon"); ax.set_xlim(1, L); ax.grid(alpha=0.3)
            if d == 0:
                ax.legend(fontsize=8)
        plt.suptitle(f"Physical trajectory — test window #{w} | "
                     f"{NOISE_TYPE} σ={NOISE_SIGMA:.2f} (physical units)")
        plt.tight_layout()
        p = os.path.join(save_dir, f"p3_trajectory_window{w}.png")
        plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
        print("saved:", p)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean_s, std_s = load_norm_stats(NORM_STATS)
    rng = np.random.default_rng(BOOT_SEED)
    traj_rng = np.random.default_rng(TRAJ_SEED)
    assert len(MODELS) >= 2, "Paired analysis expects a baseline + ≥1 variant."
    base = MODELS[0]["label"]
    variants = MODELS[1:]

    print(f"\n{'='*60}\n  NOISE: {NOISE_TYPE} σ={NOISE_SIGMA:.2f} | encoded mode\n{'='*60}")
    data = {m["label"]: evaluate_model_noisy(m, device, mean_s, std_s) for m in MODELS}

    # Align window counts across ALL models (same windows -> shared GT)
    n = min(data[m["label"]]["pred"].shape[0] for m in MODELS)
    if any(data[m["label"]]["pred"].shape[0] != n for m in MODELS):
        counts = {m["label"]: data[m["label"]]["pred"].shape[0] for m in MODELS}
        print(f"[WARN] #windows differ {counts}; truncating to {n}.")
    for m in MODELS:
        data[m["label"]]["pred"] = data[m["label"]]["pred"][:n]
        data[m["label"]]["gt"] = data[m["label"]]["gt"][:n]

    err = {m["label"]: (data[m["label"]]["pred"] - data[m["label"]]["gt"]) ** 2 for m in MODELS}

    # ---- plots ----
    plot_median_iqr(err, SAVE_DIR)
    plot_perdim(err, SAVE_DIR)
    paired = plot_paired(err, SAVE_DIR, rng)
    plot_trajectory(data, mean_s, std_s, SAVE_DIR, traj_rng)

    # ---- summary table ----
    HS = [h for h in (1, 10, 20, 30) if h <= SEQ_LEN]
    print(f"\n{'='*80}")
    print(f"=== SUMMARY: median state-MSE (standardized) | {NOISE_TYPE} σ={NOISE_SIGMA:.2f}, encoded ===")
    print(f"{'='*80}")
    for m in MODELS:
        med = np.median(err[m["label"]].mean(axis=2), axis=0)
        print(f"  {m['label']:<12} " + "  ".join(f"h{h}={med[h-1]:.5f}" for h in HS))
    for v in variants:
        med_d, lo_d, hi_d = paired[v["label"]]
        print(f"  Δ(base−{v['label']}) " +
              "  ".join(f"h{h}={med_d[h-1]:+.5f}[{lo_d[h-1]:+.5f},{hi_d[h-1]:+.5f}]" for h in HS))

    # ---- save ----
    save_dict = {"horizons": np.arange(1, SEQ_LEN + 1),
                 "noise_type": NOISE_TYPE, "noise_sigma": NOISE_SIGMA}
    for m in MODELS:
        save_dict[f"{m['label']}_err_median"] = np.median(err[m["label"]].mean(axis=2), axis=0)
    for v in variants:
        med_d, lo_d, hi_d = paired[v["label"]]
        save_dict[f"paired_{v['label']}_median"] = med_d
        save_dict[f"paired_{v['label']}_lo"] = lo_d
        save_dict[f"paired_{v['label']}_hi"] = hi_d
    np.savez(os.path.join(SAVE_DIR, "cmp_p3_noise005_curves.npz"), **save_dict)
    print("\nsaved figures + cmp_p3_noise005_curves.npz ->", SAVE_DIR)


if __name__ == "__main__":
    main()
