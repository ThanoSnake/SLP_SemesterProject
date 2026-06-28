"""
test_p3.py — Noisy evaluation comparing Baseline, P3-semi, and P3-weak (LunarLander).

Evaluates robustness of the three pipelines under visual noise (Gaussian or Salt & Pepper).
P3 uses the same monolithic VAE architecture as Baseline — only supervision differs.
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Depending on environment, adjust imports:
from vae import VAE
from vae_p3 import VAE_P3
from lstm import LatentPredictor
from loader import LatentSequenceDataset, load_norm_stats, list_npz

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATA_ROOT = "<lunarlander-dataset>"
NORM_STATS = os.path.join(DATA_ROOT, "norm_stats.npz")
SAVE_DIR = "/kaggle/working/lunarlander_p3_out"
SHIFT = 0

LATENT_SIZE, N_SUP = 64, 8
N_ACTIONS, HIDDEN, LAYERS = 4, 64, 2
SEQ_LEN, TEST_STRIDE, BATCH = 30, 1, 128

SEED_MODES = ["hybrid", "encoded"]
DIM_NAMES = ["x", "y", "vx", "vy", "theta", "omega", "leg1", "leg2"]
N_BOOT = 1000
BOOT_SEED = 0
LOG_Y = True

# ---------------------------------------------------------------------------
# NOISE CONFIG
# ---------------------------------------------------------------------------
NOISE_TYPE = "gaussian"           # "gaussian" or "salt_pepper"
NOISE_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30]
NOISE_SEED = 42

# P3 uses same architecture layout as Baseline (different supervision only)
MODELS = [
    {"label": "Baseline", "color": "C0",
     "make_vae": lambda: VAE(latent_size=LATENT_SIZE),
     "vae_ckpt": "<lunarlander-baseline-vae>",
     "lstm_ckpt": {
         "hybrid":  "<lunarlander-baseline-lstm>",
         "encoded": "<lunarlander-baseline-lstm>",
     },
     "latent_root": "/kaggle/working/lunarlander_p3_latents/baseline"},
    {"label": "P3 semi", "color": "C1",
     "make_vae": lambda: VAE_P3(latent_size=LATENT_SIZE),
     "vae_ckpt": "<lunarlander-p3-semi-vae>",
     "lstm_ckpt": {
         "hybrid":  "<lunarlander-p3-semi-lstm>",
         "encoded": "<lunarlander-p3-semi-lstm>",
     },
     "latent_root": "/kaggle/working/lunarlander_p3_latents/p3_semi"},
    {"label": "P3 weak", "color": "C3",
     "make_vae": lambda: VAE_P3(latent_size=LATENT_SIZE),
     "vae_ckpt": "<lunarlander-p3-weak-vae>",
     "lstm_ckpt": {
         "hybrid":  "<lunarlander-p3-weak-lstm>",
         "encoded": "<lunarlander-p3-weak-lstm>",
     },
     "latent_root": "/kaggle/working/lunarlander_p3_latents/p3_weak"},
]


# ---------------------------------------------------------------------------
# Noise Injection Functions
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
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")


# ---------------------------------------------------------------------------
# Noisy Precomputation
# ---------------------------------------------------------------------------
@torch.no_grad()
def precompute_latents_noisy(encode_fn, root, out_root, noise_fn,
                              shift=0, batch=256, device="cuda"):
    from os.path import join, basename
    from os import makedirs
    makedirs(out_root, exist_ok=True)
    for f in tqdm(list_npz(root), desc="encoding (noisy)"):
        with np.load(f) as d:
            imgs = torch.from_numpy(d["imgs"].astype(np.float32) / 255.0).permute(0, 3, 1, 2)
            acts = d["acts"].astype(np.float32)
            states = d["states"].astype(np.float32)
            x = (d[f"noisy_states_{shift}"] if shift in (2, 5, 10)
                 else d["states"]).astype(np.float32)

        imgs = noise_fn(imgs.to(device))
        img_t, img_tp1 = imgs[:-1], imgs[1:]
        zs = []
        for b in range(0, img_t.shape[0], batch):
            zb = encode_fn(img_t[b:b + batch], img_tp1[b:b + batch])
            zs.append(zb.cpu().numpy())
        z = np.concatenate(zs, 0).astype(np.float32) if zs else np.empty((0, 0), np.float32)

        np.savez_compressed(join(out_root, basename(f)),
                            z=z, acts=acts[:-1], states=states[:-1], x=x[:-1])


# ---------------------------------------------------------------------------
# Rollout and Evaluation Functions
# ---------------------------------------------------------------------------
def _hybrid(z, state, n_sup):
    h = z.clone(); h[..., :n_sup] = state
    return h


@torch.no_grad()
def free_run(model, batch, encoded):
    z_t, action, z_tp1, state_t, state_tp1 = batch
    B, L, _ = z_t.shape
    z_in = z_t[:, 0] if encoded else _hybrid(z_t[:, :1], state_t[:, :1], N_SUP)[:, 0]
    hidden = model.init_hidden(B, z_t.device)
    preds = []
    for k in range(L):
        a = F.one_hot(action[:, k].long(), N_ACTIONS).float()
        z_pred, hidden = model.step(z_in, a, hidden)
        preds.append(z_pred)
        z_in = z_pred
    return torch.stack(preds, dim=1), state_tp1


@torch.no_grad()
def collect_sq_err(model, loader, device, encoded):
    model.eval()
    chunks = []
    for batch in loader:
        batch = [b.to(device, non_blocking=True) for b in batch]
        preds, state_tp1 = free_run(model, batch, encoded)
        err2 = (preds[..., :N_SUP] - state_tp1) ** 2
        chunks.append(err2.cpu().numpy())
    return np.concatenate(chunks, axis=0)


# ---------------------------------------------------------------------------
# Robust Statistics
# ---------------------------------------------------------------------------
def median_iqr(arr):
    return (np.median(arr, axis=0),
            np.percentile(arr, 25, axis=0),
            np.percentile(arr, 75, axis=0))


def bootstrap_paired(diff, n_boot, rng):
    N, L = diff.shape
    med = np.median(diff, axis=0)
    boots = np.empty((n_boot, L), dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)
        boots[b] = np.median(diff[idx], axis=0)
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return med, lo, hi


# ---------------------------------------------------------------------------
# Evaluation Routine
# ---------------------------------------------------------------------------
def evaluate_model_noisy(m, device, mean_s, std_s, noise_level, noise_type):
    noise_fn = make_noise_fn(noise_type, noise_level, NOISE_SEED, device)
    noise_tag = f"{noise_type}_{noise_level:.2f}".replace(".", "p")

    print(f"\n[{m['label']}] noise={noise_type} level={noise_level:.2f}")
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

    ckpts = m["lstm_ckpt"] if isinstance(m["lstm_ckpt"], dict) else {md: m["lstm_ckpt"] for md in SEED_MODES}
    out = {}
    for mode in SEED_MODES:
        lstm = LatentPredictor(LATENT_SIZE, N_ACTIONS, HIDDEN, LAYERS).to(device)
        lstm.load_state_dict(torch.load(ckpts[mode], map_location=device))
        out[mode] = collect_sq_err(lstm, test_dl, device, encoded=(mode == "encoded"))
        print(f"  mode={mode:<7} ckpt={os.path.basename(ckpts[mode])} -> err {out[mode].shape}")
        del lstm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------------------
# Plotting Functions
# ---------------------------------------------------------------------------
def plot_median_iqr_per_level(all_err, noise_levels, mode, save_dir, noise_type):
    horizons = np.arange(1, SEQ_LEN + 1)
    for nl in noise_levels:
        plt.figure(figsize=(6.8, 4.8))
        for m in MODELS:
            label, color = m["label"], m["color"]
            arr = all_err[nl][label][mode].mean(axis=2)
            med, q25, q75 = median_iqr(arr)
            plt.plot(horizons, med, color=color, lw=2, label=label)
            plt.fill_between(horizons, q25, q75, color=color, alpha=0.15)
        if LOG_Y:
            plt.yscale("log")
        plt.title(f"median state-MSE — {mode} | {noise_type} σ={nl:.2f}")
        plt.xlabel("Prediction Horizon"); plt.ylabel("State MSE (median, IQR band)")
        plt.xlim(1, SEQ_LEN); plt.grid(alpha=0.3, which="both"); plt.legend()
        plt.tight_layout()
        tag = f"{nl:.2f}".replace(".", "p")
        plt.savefig(os.path.join(save_dir, f"noise_median_iqr_{mode}_{noise_type}_{tag}.png"), dpi=150)
        plt.show()


def plot_paired_per_level(all_err, noise_levels, save_dir, noise_type, rng):
    base = MODELS[0]["label"]
    variants = MODELS[1:]
    horizons = np.arange(1, SEQ_LEN + 1)
    paired_all = {}

    for mode in SEED_MODES:
        fig, axes = plt.subplots(1, len(noise_levels),
                                 figsize=(5.5 * len(noise_levels), 4.8), squeeze=False)
        for j, nl in enumerate(noise_levels):
            ax = axes[0][j]
            ax.axhline(0, color="k", lw=1)
            for v in variants:
                diff = (all_err[nl][base][mode].mean(axis=2)
                        - all_err[nl][v["label"]][mode].mean(axis=2))
                med, lo, hi = bootstrap_paired(diff, N_BOOT, rng)
                paired_all[(mode, v["label"], nl)] = (med, lo, hi)

                ax.plot(horizons, med, color=v["color"], lw=2, label=f"{base} − {v['label']}")
                ax.fill_between(horizons, lo, hi, color=v["color"], alpha=0.18)
            ax.set_title(f"paired Δ — {mode} | σ={nl:.2f}")
            ax.set_xlabel("Horizon"); ax.set_ylabel("Δ state-MSE")
            ax.set_xlim(1, SEQ_LEN); ax.grid(alpha=0.3); ax.legend(fontsize=7)
        plt.suptitle(f"Paired difference (>0 ⇒ variant better) — {noise_type}, {mode}", y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"noise_paired_{mode}_{noise_type}.png"),
                    dpi=150, bbox_inches="tight")
        plt.show()
    return paired_all


def plot_degradation_summary(all_err, noise_levels, save_dir, noise_type):
    SUMMARY_HORIZONS = [1, 10, 20, 30]
    markers = {"Baseline": "o", "P3 semi": "s", "P3 weak": "d"}

    for mode in SEED_MODES:
        fig, axes = plt.subplots(1, len(SUMMARY_HORIZONS),
                                 figsize=(4.5 * len(SUMMARY_HORIZONS), 4.2), squeeze=False)
        for hi, h in enumerate(SUMMARY_HORIZONS):
            if h > SEQ_LEN:
                continue
            ax = axes[0][hi]
            for m in MODELS:
                label, color = m["label"], m["color"]
                med_vals, q25_vals, q75_vals = [], [], []
                for nl in noise_levels:
                    arr = all_err[nl][label][mode].mean(axis=2)[:, h - 1]
                    med_vals.append(np.median(arr))
                    q25_vals.append(np.percentile(arr, 25))
                    q75_vals.append(np.percentile(arr, 75))
                med_vals = np.array(med_vals)
                q25_vals = np.array(q25_vals)
                q75_vals = np.array(q75_vals)

                ax.plot(noise_levels, med_vals, color=color, lw=2,
                        marker=markers.get(label, "o"), markersize=6, label=label)
                ax.fill_between(noise_levels, q25_vals, q75_vals, color=color, alpha=0.12)
            if LOG_Y:
                ax.set_yscale("log")
            ax.set_title(f"h={h}")
            ax.set_xlabel(f"Noise level ({noise_type})")
            if hi == 0:
                ax.set_ylabel("State MSE (median)")
            ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
        plt.suptitle(f"Degradation under noise — {mode} ({noise_type})", y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"noise_degradation_{mode}_{noise_type}.png"),
                    dpi=150, bbox_inches="tight")
        plt.show()


def plot_perdim_per_level(all_err, noise_levels, save_dir, noise_type):
    pm = "encoded" if "encoded" in SEED_MODES else SEED_MODES[0]
    horizons = np.arange(1, SEQ_LEN + 1)
    for nl in noise_levels:
        fig, axes = plt.subplots(1, N_SUP, figsize=(4.2 * N_SUP, 4.0), squeeze=False)
        for d in range(N_SUP):
            ax = axes[0][d]
            for m in MODELS:
                label, color = m["label"], m["color"]
                med, q25, q75 = median_iqr(all_err[nl][label][pm][:, :, d])
                ax.plot(horizons, med, color=color, lw=2, label=label)
                ax.fill_between(horizons, q25, q75, color=color, alpha=0.15)
            if LOG_Y:
                ax.set_yscale("log")
            ax.set_title(f"{DIM_NAMES[d]} — σ={nl:.2f}")
            ax.set_xlabel("Horizon"); ax.set_xlim(1, SEQ_LEN); ax.grid(alpha=0.3, which="both")
            if d == 0:
                ax.set_ylabel("MSE (median, standardized)"); ax.legend()
        plt.suptitle(f"Per-dim MSE — {pm} | {noise_type} σ={nl:.2f}", y=1.02)
        plt.tight_layout()
        tag = f"{nl:.2f}".replace(".", "p")
        plt.savefig(os.path.join(save_dir, f"noise_perdim_{pm}_{noise_type}_{tag}.png"),
                    dpi=150, bbox_inches="tight")
        plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean_s, std_s = load_norm_stats(NORM_STATS)
    std8 = np.asarray(std_s[:N_SUP], dtype=np.float64)
    rng = np.random.default_rng(BOOT_SEED)
    horizons = np.arange(1, SEQ_LEN + 1)

    base = MODELS[0]["label"]
    variants = MODELS[1:]

    # Collect errors across all models and noise levels
    all_err = {}
    for nl in NOISE_LEVELS:
        print(f"\n{'='*60}")
        print(f"  NOISE LEVEL: {NOISE_TYPE} σ={nl:.2f}")
        print(f"{'='*60}")
        all_err[nl] = {}
        for m in MODELS:
            all_err[nl][m["label"]] = evaluate_model_noisy(
                m, device, mean_s, std_s, nl, NOISE_TYPE)

        # Align window sizes across models
        for mode in SEED_MODES:
            lens = [all_err[nl][m["label"]][mode].shape[0] for m in MODELS]
            n = min(lens)
            if max(lens) != min(lens):
                print(f"[WARN] mode={mode}: window counts differ {lens}; truncating to {n}")
            for m in MODELS:
                all_err[nl][m["label"]][mode] = all_err[nl][m["label"]][mode][:n]

    # ---- (1) Plot Median + IQR Curves ----
    for mode in SEED_MODES:
        plot_median_iqr_per_level(all_err, NOISE_LEVELS, mode, SAVE_DIR, NOISE_TYPE)

    # ---- (2) Plot Paired Differences ----
    paired_all = plot_paired_per_level(all_err, NOISE_LEVELS, SAVE_DIR, NOISE_TYPE, rng)

    # ---- (3) Plot Per-Dimension Curves ----
    plot_perdim_per_level(all_err, NOISE_LEVELS, SAVE_DIR, NOISE_TYPE)

    # ---- (4) Plot Degradation Summaries ----
    plot_degradation_summary(all_err, NOISE_LEVELS, SAVE_DIR, NOISE_TYPE)

    # ---- (5) Summary Table ----
    HS = [h for h in (1, 10, 20, 30) if h <= SEQ_LEN]
    print(f"\n{'='*80}")
    print(f"=== SUMMARY: median state-MSE (standardized) under {NOISE_TYPE} noise ===")
    print(f"{'='*80}")
    for nl in NOISE_LEVELS:
        print(f"\n  --- noise level σ={nl:.2f} ---")
        for mode in SEED_MODES:
            print(f"    [{mode}]")
            for m in MODELS:
                label = m["label"]
                med = np.median(all_err[nl][label][mode].mean(axis=2), axis=0)
                print(f"      {label:<12} " +
                      "  ".join(f"h{h}={med[h-1]:.5f}" for h in HS))
            for v in variants:
                vl = v["label"]
                if (mode, vl, nl) in paired_all:
                    med_d, lo_d, hi_d = paired_all[(mode, vl, nl)]
                    print(f"      paired Δ(>0⇒{vl})  " +
                          "  ".join(f"h{h}={med_d[h-1]:+.5f}"
                                    f"[{lo_d[h-1]:+.5f},{hi_d[h-1]:+.5f}]" for h in HS))

    # ---- Save Data ----
    save_dict = {"horizons": horizons, "std8": std8,
                 "noise_type": NOISE_TYPE,
                 "noise_levels": np.array(NOISE_LEVELS),
                 "seed_modes": np.array(SEED_MODES)}
    for nl in NOISE_LEVELS:
        tag = f"{nl:.2f}".replace(".", "p")
        for m in MODELS:
            label = m["label"]
            for md in SEED_MODES:
                save_dict[f"{label}_{md}_{tag}_err_median"] = np.median(
                    all_err[nl][label][md].mean(axis=2), axis=0)
        for md in SEED_MODES:
            for v in variants:
                vl = v["label"]
                if (md, vl, nl) in paired_all:
                    m_d, l, h = paired_all[(md, vl, nl)]
                    save_dict[f"paired_{vl}_{md}_{tag}_median"] = m_d
                    save_dict[f"paired_{vl}_{md}_{tag}_lo"] = l
                    save_dict[f"paired_{vl}_{md}_{tag}_hi"] = h
    np.savez(os.path.join(SAVE_DIR, "cmp_p3_noise_curves.npz"), **save_dict)
    print("\nsaved figures + cmp_p3_noise_curves.npz ->", SAVE_DIR)


if __name__ == "__main__":
    main()
