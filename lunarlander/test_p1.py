"""
test_p1.py — Αξιολόγηση Baseline vs Principle 1 με ΘΟΡΥΒΩΔΕΙΣ εικόνες (LunarLander).

Port του cart_pole/test_p1.py· state 4D -> 8D. Imports από τα canonical modules του lunarlander/.

ΕΣΤΙΑΣΜΕΝΗ ΕΚΔΟΣΗ (single-setting):
  * ΜΟΝΟ gaussian θόρυβος σ=0.1.
  * ΜΟΝΟ "encoded" seed mode (z_0 από VAE -> LSTM rollout).
  * Ο θόρυβος εφαρμόζεται ΑΠΟΚΛΕΙΣΤΙΚΑ στη φάση encoding (precompute), πριν τον encoder.

ΠΑΡΑΓΟΜΕΝΑ:
  (1) Overall median+IQR state-MSE ανά horizon (mean over dims)   [standardized]
  (2) Per-dim median+IQR state-MSE ανά horizon (2×4)              [standardized]
  (3) Paired Δ (baseline − p1) median + 95% bootstrap CI ανά horizon
  (4) ΦΥΣΙΚΑ ΜΕΓΕΘΗ ενός ΤΥΧΑΙΟΥ test window: GT vs baseline vs p1 (2×4)  [physical units]
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from vae import VAE
from vae_p1 import VAE_P1
from lstm import LatentPredictor
from loader import LatentSequenceDataset, load_norm_stats, list_npz

# ---------------------------------------------------------------------------
# CONFIG — placeholders <...> τα συμπληρώνει το bootstrap patcher (CONFIG_PATHS)
# ---------------------------------------------------------------------------
DATA_ROOT = "<lunarlander-dataset>"
NORM_STATS = os.path.join(DATA_ROOT, "norm_stats.npz")
SAVE_DIR = "/kaggle/working/lunarlander_p1_out"

SHIFT = 0
LATENT_SIZE, N_SUP, N_IMG = 64, 8, 56
N_ACTIONS, HIDDEN, LAYERS = 4, 64, 2
SEQ_LEN, TEST_STRIDE, BATCH = 30, 1, 128

DIM_NAMES = ["x", "y", "vx", "vy", "theta", "omega", "leg1", "leg2"]
DIM_LABELS = ["x", "y", r"$v_x$", r"$v_y$", r"$\theta$", r"$\omega$", "leg1", "leg2"]
DIM_UNITS = ["(pos)", "(pos)", "(vel)", "(vel)", "[rad]", "[rad/s]", "(contact)", "(contact)"]
N_BOOT = 1000
BOOT_SEED = 0
LOG_Y = True

# ---------------------------------------------------------------------------
# NOISE CONFIG — μοναδικό setting: gaussian σ=0.1
# ---------------------------------------------------------------------------
NOISE_TYPE = "gaussian"           # "gaussian" | "salt_pepper"
NOISE_SIGMA = 0.05                 # std (gaussian) πάνω σε [0,1] εικόνα
NOISE_SEED = 42

# Trajectory plot (4): τυχαίο test window
TRAJ_SEED = None                  # None -> ΓΝΗΣΙΑ τυχαίο· int -> reproducible
TRAJ_WINDOW = None                # None -> τυχαίο· ή ακέραιος index
N_TRAJ_WINDOWS = 1

# ---------------------------------------------------------------------------
# Model definitions — Baseline vs P1 (clean-trained VAE + encoded LSTM)
# ---------------------------------------------------------------------------
MODELS = [
    {"label": "Baseline", "color": "C0",
     "make_vae": lambda: VAE(latent_size=LATENT_SIZE),
     "vae_ckpt": "<lunarlander-baseline-vae>",
     "lstm_ckpt": "<lunarlander-baseline-lstm>",
     "latent_root": "/kaggle/working/lunarlander_p1_latents/baseline"},
    {"label": "Principle 1", "color": "C1",
     "make_vae": lambda: VAE_P1(n_sup=N_SUP, n_img=N_IMG),
     "vae_ckpt": "<lunarlander-p1-vae>",
     "lstm_ckpt": "<lunarlander-p1-lstm>",
     "latent_root": "/kaggle/working/lunarlander_p1_latents/p1"},
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
# NOISY precompute_latents — εφαρμόζει noise ΠΡΙΝ το encoding
# ---------------------------------------------------------------------------
@torch.no_grad()
def precompute_latents_noisy(encode_fn, root, out_root, noise_fn, shift=0, batch=256, device="cuda"):
    from os.path import join, basename
    from os import makedirs
    makedirs(out_root, exist_ok=True)
    for f in list_npz(root):
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


# ---------------------------------------------------------------------------
# Rollout (ENCODED) + collection of predicted physical dims & GT
# ---------------------------------------------------------------------------
@torch.no_grad()
def free_run(model, batch):
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
    return (np.median(arr, axis=0), np.percentile(arr, 25, axis=0), np.percentile(arr, 75, axis=0))


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
# Model evaluation at the single noise setting
# ---------------------------------------------------------------------------
def evaluate_model_noisy(m, device, mean_s, std_s):
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
    """(1) Overall median+IQR state-MSE (mean over dims) — Baseline vs P1."""
    horizons = np.arange(1, SEQ_LEN + 1)
    plt.figure(figsize=(6.8, 4.8))
    for m in MODELS:
        med, q25, q75 = median_iqr(err[m["label"]].mean(axis=2))
        plt.plot(horizons, med, color=m["color"], lw=2, label=m["label"])
        plt.fill_between(horizons, q25, q75, color=m["color"], alpha=0.18)
    if LOG_Y:
        plt.yscale("log")
    plt.title(f"median state-MSE (encoded) | {NOISE_TYPE} σ={NOISE_SIGMA:.2f}")
    plt.xlabel("Prediction Horizon"); plt.ylabel("State MSE (median, IQR band)")
    plt.xlim(1, SEQ_LEN); plt.grid(alpha=0.3, which="both"); plt.legend()
    plt.tight_layout()
    p = os.path.join(save_dir, "p1_median_iqr_encoded.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print("saved:", p)


def plot_perdim(err, save_dir):
    """(2) Per-dim median+IQR state-MSE (2×4) — Baseline vs P1."""
    horizons = np.arange(1, SEQ_LEN + 1)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), squeeze=False)
    for d in range(N_SUP):
        ax = axes[d // 4][d % 4]
        for m in MODELS:
            med, q25, q75 = median_iqr(err[m["label"]][:, :, d])
            ax.plot(horizons, med, color=m["color"], lw=2, label=m["label"])
            ax.fill_between(horizons, q25, q75, color=m["color"], alpha=0.18)
        if LOG_Y:
            ax.set_yscale("log")
        ax.set_title(f"{DIM_LABELS[d]} {DIM_UNITS[d]}")
        ax.set_xlabel("Horizon"); ax.set_xlim(1, SEQ_LEN); ax.grid(alpha=0.3, which="both")
        if d % 4 == 0:
            ax.set_ylabel("MSE (median, standardized)")
        if d == 0:
            ax.legend()
    plt.suptitle(f"Per-dim state-MSE (encoded) | {NOISE_TYPE} σ={NOISE_SIGMA:.2f}", y=1.01)
    plt.tight_layout()
    p = os.path.join(save_dir, "p1_perdim_encoded.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("saved:", p)


def plot_paired(err, save_dir, rng):
    """(3) Paired Δ (Baseline − P1) median + 95% bootstrap CI."""
    base, p1 = MODELS[0]["label"], MODELS[1]["label"]
    horizons = np.arange(1, SEQ_LEN + 1)
    diff = err[base].mean(axis=2) - err[p1].mean(axis=2)
    med, lo, hi = bootstrap_paired(diff, N_BOOT, rng)

    plt.figure(figsize=(6.8, 4.8))
    plt.axhline(0, color="k", lw=1)
    plt.plot(horizons, med, color="C2", lw=2, label=f"median({base} − {p1})")
    plt.fill_between(horizons, lo, hi, color="C2", alpha=0.25, label="95% bootstrap CI")
    plt.title(f"Paired difference (>0 ⇒ {p1} better) — encoded | {NOISE_TYPE} σ={NOISE_SIGMA:.2f}")
    plt.xlabel("Prediction Horizon"); plt.ylabel("Δ state-MSE")
    plt.xlim(1, SEQ_LEN); plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout()
    p = os.path.join(save_dir, "p1_paired_encoded.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print("saved:", p)
    return med, lo, hi


def plot_trajectory(data, mean_s, std_s, save_dir, rng):
    """(4) Physical trajectory ενός ΤΥΧΑΙΟΥ window: GT vs baseline vs p1 (2×4)."""
    base, p1 = MODELS[0]["label"], MODELS[1]["label"]
    mean8 = np.asarray(mean_s[:N_SUP], np.float64)
    std8 = np.asarray(std_s[:N_SUP], np.float64)
    gt_all = data[base]["gt"]
    N, L, _ = gt_all.shape
    horizons = np.arange(1, L + 1)

    for wi in range(N_TRAJ_WINDOWS):
        w = TRAJ_WINDOW if TRAJ_WINDOW is not None else int(rng.integers(0, N))
        if not np.allclose(data[base]["gt"][w], data[p1]["gt"][w], atol=1e-4):
            print(f"[warn] window {w}: GT differs between models (window alignment?).")

        gt_phys = gt_all[w] * std8 + mean8
        b_phys = data[base]["pred"][w] * std8 + mean8
        p_phys = data[p1]["pred"][w] * std8 + mean8

        fig, axes = plt.subplots(2, 4, figsize=(18, 8))
        for d in range(N_SUP):
            ax = axes[d // 4][d % 4]
            ax.plot(horizons, gt_phys[:, d], color="k", lw=2.0, label="GT")
            ax.plot(horizons, b_phys[:, d], color=MODELS[0]["color"], lw=1.6, ls="--", label=base)
            ax.plot(horizons, p_phys[:, d], color=MODELS[1]["color"], lw=1.6, ls="--", label=p1)
            ax.set_title(f"{DIM_LABELS[d]} {DIM_UNITS[d]}")
            ax.set_xlabel("Prediction Horizon"); ax.set_xlim(1, L); ax.grid(alpha=0.3)
            if d == 0:
                ax.legend(fontsize=9)
        plt.suptitle(f"Physical trajectory — test window #{w} | "
                     f"{NOISE_TYPE} σ={NOISE_SIGMA:.2f} (physical units)")
        plt.tight_layout()
        p = os.path.join(save_dir, f"p1_trajectory_window{w}.png")
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
    assert len(MODELS) == 2, "Paired analysis expects exactly 2 models (base, p1)."
    base, p1 = MODELS[0]["label"], MODELS[1]["label"]

    print(f"\n{'='*60}\n  NOISE: {NOISE_TYPE} σ={NOISE_SIGMA:.2f} | encoded mode\n{'='*60}")
    data = {m["label"]: evaluate_model_noisy(m, device, mean_s, std_s) for m in MODELS}

    n = min(data[base]["pred"].shape[0], data[p1]["pred"].shape[0])
    if data[base]["pred"].shape[0] != data[p1]["pred"].shape[0]:
        print(f"[WARN] #windows differ ({data[base]['pred'].shape[0]} vs "
              f"{data[p1]['pred'].shape[0]}); truncating to {n}.")
    for label in (base, p1):
        data[label]["pred"] = data[label]["pred"][:n]
        data[label]["gt"] = data[label]["gt"][:n]

    err = {label: (data[label]["pred"] - data[label]["gt"]) ** 2 for label in (base, p1)}

    plot_median_iqr(err, SAVE_DIR)
    plot_perdim(err, SAVE_DIR)
    med_d, lo_d, hi_d = plot_paired(err, SAVE_DIR, rng)
    plot_trajectory(data, mean_s, std_s, SAVE_DIR, traj_rng)

    HS = [h for h in (1, 10, 20, 30) if h <= SEQ_LEN]
    print(f"\n{'='*80}")
    print(f"=== SUMMARY: median state-MSE (standardized) | {NOISE_TYPE} σ={NOISE_SIGMA:.2f}, encoded ===")
    print(f"{'='*80}")
    for label in (base, p1):
        med = np.median(err[label].mean(axis=2), axis=0)
        print(f"  {label:<12} " + "  ".join(f"h{h}={med[h-1]:.5f}" for h in HS))
    print(f"  paired Δ(>0⇒{p1})  " +
          "  ".join(f"h{h}={med_d[h-1]:+.5f}[{lo_d[h-1]:+.5f},{hi_d[h-1]:+.5f}]" for h in HS))

    save_dict = {"horizons": np.arange(1, SEQ_LEN + 1),
                 "noise_type": NOISE_TYPE, "noise_sigma": NOISE_SIGMA}
    for label in (base, p1):
        save_dict[f"{label}_err_median"] = np.median(err[label].mean(axis=2), axis=0)
    save_dict["paired_median"], save_dict["paired_lo"], save_dict["paired_hi"] = med_d, lo_d, hi_d
    np.savez(os.path.join(SAVE_DIR, "cmp_p1_noise01_curves.npz"), **save_dict)
    print("\nsaved figures + cmp_p1_noise01_curves.npz ->", SAVE_DIR)


if __name__ == "__main__":
    main()
