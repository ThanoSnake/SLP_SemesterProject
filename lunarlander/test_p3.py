"""
test_p3.py — Αξιολόγηση Baseline vs P3-semi vs P3-weak με ΘΟΡΥΒΩΔΕΙΣ εικόνες (LunarLander).

Port του cart_pole/test_p3.py· state 4D -> 8D. Imports από τα canonical modules του lunarlander/.

ΕΣΤΙΑΣΜΕΝΗ ΕΚΔΟΣΗ (single-setting) — ίδια δομή/διαγράμματα με test_p1/test_p2, αλλά με ΤΡΙΑ
μοντέλα (Baseline, P3 semi, P3 weak· ίδια monolithic αρχιτεκτονική VAE, διαφέρει μόνο η ΕΠΟΠΤΕΙΑ):
  * ΜΟΝΟ μικρός gaussian θόρυβος σ=0.05.
  * ΜΟΝΟ "encoded" seed mode (z_0 από VAE -> LSTM rollout).
  * Ο θόρυβος εφαρμόζεται ΑΠΟΚΛΕΙΣΤΙΚΑ στη φάση encoding (precompute), πριν τον encoder.

ΠΑΡΑΓΟΜΕΝΑ:
  (1) Overall median+IQR state-MSE ανά horizon (mean over dims) — 3 καμπύλες   [standardized]
  (2) Per-dim median+IQR state-MSE ανά horizon (2×4) — 3 καμπύλες              [standardized]
  (3) Paired Δ (Baseline − semi) και (Baseline − weak) + 95% bootstrap CI
  (4) ΦΥΣΙΚΑ ΜΕΓΕΘΗ ενός ΤΥΧΑΙΟΥ test window: GT vs baseline vs semi vs weak (2×4)  [physical units]
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from vae import VAE
from vae_p3 import VAE_P3
from lstm import LatentPredictor
from loader import LatentSequenceDataset, load_norm_stats, list_npz

# ---------------------------------------------------------------------------
# CONFIG — placeholders <...> τα συμπληρώνει το bootstrap patcher (CONFIG_PATHS)
# ---------------------------------------------------------------------------
DATA_ROOT = "<lunarlander-dataset>"
NORM_STATS = os.path.join(DATA_ROOT, "norm_stats.npz")
SAVE_DIR = "/kaggle/working/lunarlander_p3_out"

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
# NOISE CONFIG — SWEEP πολλαπλών επιπέδων gaussian θορύβου
# ---------------------------------------------------------------------------
NOISE_TYPE = "gaussian"                      # "gaussian" | "salt_pepper"
NOISE_SIGMAS = [0.01, 0.02, 0.03, 0.05, 0.1]  # λίστα επιπέδων σ -> ένα run ανά επίπεδο
NOISE_SIGMA = NOISE_SIGMAS[0]                # ΤΡΕΧΟΝ επίπεδο (αλλάζει στο loop του main)
NOISE_SEED = 42

# Trajectory plot (4): τυχαίο test window
TRAJ_SEED = None
TRAJ_WINDOW = None
N_TRAJ_WINDOWS = 1

# ---------------------------------------------------------------------------
# Model definitions — Baseline vs P3 semi vs P3 weak
# (ΙΔΙΑ monolithic αρχιτεκτονική· διαφέρει μόνο η εποπτεία -> διαφορετικά checkpoints)
# ---------------------------------------------------------------------------
MODELS = [
    {"label": "Baseline", "color": "C0",
     "make_vae": lambda: VAE(latent_size=LATENT_SIZE),
     "vae_ckpt": "<lunarlander-baseline-vae>",
     "lstm_ckpt": "<lunarlander-baseline-lstm>",
     "latent_root": "/kaggle/working/lunarlander_p3_latents/baseline"},
    {"label": "P3 semi", "color": "C1",
     "make_vae": lambda: VAE_P3(latent_size=LATENT_SIZE),
     "vae_ckpt": "<lunarlander-p3-semi-vae>",
     "lstm_ckpt": "<lunarlander-p3-semi-lstm>",
     "latent_root": "/kaggle/working/lunarlander_p3_latents/p3_semi"},
    {"label": "P3 weak", "color": "C3",
     "make_vae": lambda: VAE_P3(latent_size=LATENT_SIZE),
     "vae_ckpt": "<lunarlander-p3-weak-vae>",
     "lstm_ckpt": "<lunarlander-p3-weak-lstm>",
     "latent_root": "/kaggle/working/lunarlander_p3_latents/p3_weak"},
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
    """(1) Overall median+IQR state-MSE (mean over dims) — όλα τα μοντέλα."""
    horizons = np.arange(1, SEQ_LEN + 1)
    plt.figure(figsize=(6.8, 4.8))
    for m in MODELS:
        med, q25, q75 = median_iqr(err[m["label"]].mean(axis=2))
        plt.plot(horizons, med, color=m["color"], lw=2, label=m["label"])
        plt.fill_between(horizons, q25, q75, color=m["color"], alpha=0.15)
    if LOG_Y:
        plt.yscale("log")
    plt.title(f"median state-MSE (encoded) | {NOISE_TYPE} σ={NOISE_SIGMA:.2f}")
    plt.xlabel("Prediction Horizon"); plt.ylabel("State MSE (median, IQR band)")
    plt.xlim(1, SEQ_LEN); plt.grid(alpha=0.3, which="both"); plt.legend()
    plt.tight_layout()
    tag = f"{NOISE_SIGMA:.2f}".replace(".", "p")
    p = os.path.join(save_dir, f"p3_median_iqr_encoded_s{tag}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print("saved:", p)


def plot_perdim(err, save_dir):
    """(2) Per-dim median+IQR state-MSE (2×4) — όλα τα μοντέλα."""
    horizons = np.arange(1, SEQ_LEN + 1)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), squeeze=False)
    for d in range(N_SUP):
        ax = axes[d // 4][d % 4]
        for m in MODELS:
            med, q25, q75 = median_iqr(err[m["label"]][:, :, d])
            ax.plot(horizons, med, color=m["color"], lw=2, label=m["label"])
            ax.fill_between(horizons, q25, q75, color=m["color"], alpha=0.15)
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
    tag = f"{NOISE_SIGMA:.2f}".replace(".", "p")
    p = os.path.join(save_dir, f"p3_perdim_encoded_s{tag}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("saved:", p)


def plot_paired(err, save_dir, rng):
    """(3) Paired Δ (Baseline − variant) median + 95% bootstrap CI, για κάθε variant."""
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
    tag = f"{NOISE_SIGMA:.2f}".replace(".", "p")
    p = os.path.join(save_dir, f"p3_paired_encoded_s{tag}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print("saved:", p)
    return paired


def plot_trajectory(data, mean_s, std_s, save_dir, rng):
    """(4) Physical trajectory ενός ΤΥΧΑΙΟΥ window: GT vs baseline vs semi vs weak (2×4)."""
    base = MODELS[0]["label"]
    mean8 = np.asarray(mean_s[:N_SUP], np.float64)
    std8 = np.asarray(std_s[:N_SUP], np.float64)
    gt_all = data[base]["gt"]
    N, L, _ = gt_all.shape
    horizons = np.arange(1, L + 1)

    for wi in range(N_TRAJ_WINDOWS):
        w = TRAJ_WINDOW if TRAJ_WINDOW is not None else int(rng.integers(0, N))
        for m in MODELS[1:]:
            if not np.allclose(data[base]["gt"][w], data[m["label"]]["gt"][w], atol=1e-4):
                print(f"[warn] window {w}: GT differs ({base} vs {m['label']}) — window alignment?")

        gt_phys = gt_all[w] * std8 + mean8
        preds_phys = {m["label"]: data[m["label"]]["pred"][w] * std8 + mean8 for m in MODELS}

        fig, axes = plt.subplots(2, 4, figsize=(18, 8))
        for d in range(N_SUP):
            ax = axes[d // 4][d % 4]
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
        tag = f"{NOISE_SIGMA:.2f}".replace(".", "p")
        p = os.path.join(save_dir, f"p3_trajectory_s{tag}_window{w}.png")
        plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
        print("saved:", p)


# ---------------------------------------------------------------------------
# (5) Degradation summary πάνω σε ΟΛΟ το σ-sweep
# ---------------------------------------------------------------------------
def plot_degradation(deg, save_dir):
    """median state-MSE (mean over dims) vs σ, σε σταθερούς ορίζοντες — μία γραμμή ανά μοντέλο.
    deg[σ][label] = median MSE curve (L,). Πιο flat καμπύλη = πιο robust στον θόρυβο."""
    SUMMARY_HORIZONS = [h for h in (1, 10, 20, 30) if h <= SEQ_LEN]
    sig = NOISE_SIGMAS
    fig, axes = plt.subplots(1, len(SUMMARY_HORIZONS),
                             figsize=(4.5 * len(SUMMARY_HORIZONS), 4.2), squeeze=False)
    for hi, h in enumerate(SUMMARY_HORIZONS):
        ax = axes[0][hi]
        for m in MODELS:
            vals = [deg[s][m["label"]][h - 1] for s in sig]
            ax.plot(sig, vals, color=m["color"], lw=2, marker="o", markersize=5, label=m["label"])
        if LOG_Y:
            ax.set_yscale("log")
        ax.set_title(f"h={h}"); ax.set_xlabel(f"noise σ ({NOISE_TYPE})")
        if hi == 0:
            ax.set_ylabel("median state-MSE")
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    plt.suptitle(f"Degradation under {NOISE_TYPE} noise — encoded (πιο flat = πιο robust)", y=1.02)
    plt.tight_layout()
    p = os.path.join(save_dir, "p3_degradation_encoded.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("saved:", p)


# ---------------------------------------------------------------------------
# Main — SWEEP πάνω στα NOISE_SIGMAS
# ---------------------------------------------------------------------------
def main():
    global NOISE_SIGMA
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean_s, std_s = load_norm_stats(NORM_STATS)
    rng = np.random.default_rng(BOOT_SEED)
    traj_rng = np.random.default_rng(TRAJ_SEED)
    assert len(MODELS) >= 2, "Paired analysis expects a baseline + ≥1 variant."
    base = MODELS[0]["label"]
    variants = MODELS[1:]

    HS = [h for h in (1, 10, 20, 30) if h <= SEQ_LEN]
    deg = {}                              # deg[σ][label] = median MSE curve (L,)
    save_dict = {"horizons": np.arange(1, SEQ_LEN + 1),
                 "noise_type": NOISE_TYPE, "noise_sigmas": np.array(NOISE_SIGMAS)}

    for sigma in NOISE_SIGMAS:
        NOISE_SIGMA = sigma               # οι plot/eval functions διαβάζουν αυτό το global
        tag = f"{sigma:.2f}".replace(".", "p")
        print(f"\n{'#'*64}\n#  NOISE LEVEL: {NOISE_TYPE} σ={sigma:.2f}\n{'#'*64}")
        data = {m["label"]: evaluate_model_noisy(m, device, mean_s, std_s) for m in MODELS}

        n = min(data[m["label"]]["pred"].shape[0] for m in MODELS)
        if any(data[m["label"]]["pred"].shape[0] != n for m in MODELS):
            counts = {m["label"]: data[m["label"]]["pred"].shape[0] for m in MODELS}
            print(f"[WARN] #windows differ {counts}; truncating to {n}.")
        for m in MODELS:
            data[m["label"]]["pred"] = data[m["label"]]["pred"][:n]
            data[m["label"]]["gt"] = data[m["label"]]["gt"][:n]

        err = {m["label"]: (data[m["label"]]["pred"] - data[m["label"]]["gt"]) ** 2 for m in MODELS}

        plot_median_iqr(err, SAVE_DIR)
        plot_perdim(err, SAVE_DIR)
        paired = plot_paired(err, SAVE_DIR, rng)
        plot_trajectory(data, mean_s, std_s, SAVE_DIR, traj_rng)

        deg[sigma] = {}
        print(f"\n  --- SUMMARY σ={sigma:.2f} (median state-MSE, standardized) ---")
        for m in MODELS:
            med = np.median(err[m["label"]].mean(axis=2), axis=0)
            deg[sigma][m["label"]] = med
            save_dict[f"{m['label']}_s{tag}_err_median"] = med
            print(f"  {m['label']:<12} " + "  ".join(f"h{h}={med[h-1]:.5f}" for h in HS))
        for v in variants:
            med_d, lo_d, hi_d = paired[v["label"]]
            save_dict[f"paired_{v['label']}_s{tag}_median"] = med_d
            save_dict[f"paired_{v['label']}_s{tag}_lo"] = lo_d
            save_dict[f"paired_{v['label']}_s{tag}_hi"] = hi_d
            print(f"  Δ(base−{v['label']}) " +
                  "  ".join(f"h{h}={med_d[h-1]:+.5f}[{lo_d[h-1]:+.5f},{hi_d[h-1]:+.5f}]" for h in HS))

    # ---- (5) degradation summary πάνω σε όλο το sweep ----
    plot_degradation(deg, SAVE_DIR)

    np.savez(os.path.join(SAVE_DIR, "cmp_p3_noise_sweep_curves.npz"), **save_dict)
    print("\nsaved figures + cmp_p3_noise_sweep_curves.npz ->", SAVE_DIR)


if __name__ == "__main__":
    main()
