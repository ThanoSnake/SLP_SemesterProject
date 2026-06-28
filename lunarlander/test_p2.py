"""
cmp_baseline_p2.py — Αξιολόγηση Baseline vs Principle 2 ΜΕ BRIGHTNESS/CONTRAST jitter (LunarLander).

Βασισμένο στο lunar_test_thanasis.py (robust median + IQR, paired bootstrap CI, degradation
sweep), αλλά αντί για Gaussian θόρυβο εφαρμόζει την ΙΔΙΑ photometric διαταραχή που το P2
εκπαιδεύτηκε να είναι ΑΜΕΤΑΒΛΗΤΟ (color invariance loss, Αρχή 2 του paper).

ΓΙΑΤΙ ΑΥΤΟ ΤΟ TRANSFORM:
  * Το P2 έμαθε INVARIANCE σε brightness/contrast -> το physical encoding του μένει σταθερό
    όταν αλλάζει η φωτεινότητα/αντίθεση. Ο Baseline δεν έχει τέτοια εκπαίδευση -> μετατοπίζεται.
  * Είναι INVARIANCE transform (όχι translation/rotation equivariance): αλλάζει την εικόνα
    ΧΩΡΙΣ να αλλάζει το πραγματικό physical state -> ταιριάζει με τη μεθοδολογία «αλλάζω input,
    μετράω vs το αρχικό clean GT». (Equivariance transforms θα άλλαζαν το GT -> ακατάλληλα εδώ.)
  * Gaussian noise ΔΕΝ θα ανέδειχνε το P2 (διαφορετική διαταραχή από το training του).

Σύγκριση ΜΟΝΟ Baseline vs P2, σε ENCODED mode (τα μοντέλα έχουν encoded-trained LSTM).
Σε notebook τρέξε ΠΡΩΤΑ τα cells: VAE, VAE_P2, LatentPredictor, loader.
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from vae import VAE, encode_fn as encode_fn_baseline
from vae_p2 import VAE_P2, encode_fn as encode_fn_p2
from lstm import LatentPredictor
from loader import precompute_latents, LatentSequenceDataset, load_norm_stats, list_npz

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATA_ROOT = "<lunarlander-dataset>"
NORM_STATS = os.path.join(DATA_ROOT, "norm_stats.npz")
SAVE_DIR = "/kaggle/working/compare_p2_bc_out"
SHIFT = 0

LATENT_SIZE, N_SUP, N_IMG = 64, 8, 56
N_ACTIONS, HIDDEN, LAYERS = 4, 64, 2
SEQ_LEN, TEST_STRIDE, BATCH = 30, 1, 128

SEED_MODES = ["encoded"]
DIM_NAMES = ["x", "y", "vx", "vy", "theta", "omega", "leg1", "leg2"]
N_BOOT = 1000
BOOT_SEED = 0
LOG_Y = True

# ---------------------------------------------------------------------------
# TRANSFORM CONFIG — photometric jitter (το invariance target του P2)
# ---------------------------------------------------------------------------
TRANSFORM_TYPE = "brightness_contrast"     # "brightness" | "contrast" | "brightness_contrast"
# Sweep πολλαπλών εντάσεων (factor = 1 ± level· level=0 -> clean).
TRANSFORM_LEVELS = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8]
TRANSFORM_SIGN = +1.0                       # +1 -> πιο φωτεινό/αντίθετο· -1 -> πιο σκούρο

# ---------------------------------------------------------------------------
# Model definitions — Baseline vs P2 (clean-trained VAE + encoded LSTM)
# ---------------------------------------------------------------------------
MODELS = [
    {"label": "Baseline", "color": "C0",
     "make_vae": lambda: VAE(latent_size=LATENT_SIZE),
     "vae_ckpt": "/kaggle/working/vae_out/vae_best.pth",
     "lstm_ckpt": {
         "encoded": "/kaggle/working/lstm_baseline_alt_out/lstm_baseline_alt_best.pth",
     },
     "latent_root": "/kaggle/working/cmp_latents_p2bc/baseline"},
    {"label": "Principle 2", "color": "C2",
     "make_vae": lambda: VAE_P2(latent_size=LATENT_SIZE),
     "vae_ckpt": "/kaggle/working/vae_p2_out/vae_p2_best.pth",
     "lstm_ckpt": {
         "encoded": "/kaggle/working/lstm_p2_alt_out/lstm_p2_alt_best.pth",
     },
     "latent_root": "/kaggle/working/cmp_latents_p2bc/p2"},
]


# ---------------------------------------------------------------------------
# Photometric transforms — εφαρμόζονται σε float [0,1] image tensors (T,3,H,W)
# ---------------------------------------------------------------------------
def apply_brightness(img, level, sign):
    """Multiplicative brightness: img * (1 ± level)."""
    return torch.clamp(img * (1.0 + sign * level), 0.0, 1.0)


def apply_contrast(img, level, sign):
    """Contrast γύρω από το per-frame mean: (img - m) * (1 ± level) + m."""
    m = img.mean(dim=(1, 2, 3), keepdim=True)
    return torch.clamp((img - m) * (1.0 + sign * level) + m, 0.0, 1.0)


def apply_brightness_contrast(img, level, sign):
    """Brightness + contrast μαζί (όπως το color_jitter του P2 training)."""
    out = img * (1.0 + sign * level)
    m = out.mean(dim=(1, 2, 3), keepdim=True)
    return torch.clamp((out - m) * (1.0 + sign * level) + m, 0.0, 1.0)


def make_transform_fn(transform_type, level, device):
    """Returns a deterministic function (img_tensor) -> transformed_img_tensor."""
    if level == 0.0:
        return lambda x: x  # no-op για clean baseline
    sign = TRANSFORM_SIGN
    if transform_type == "brightness":
        return lambda x: apply_brightness(x, level, sign)
    elif transform_type == "contrast":
        return lambda x: apply_contrast(x, level, sign)
    elif transform_type == "brightness_contrast":
        return lambda x: apply_brightness_contrast(x, level, sign)
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")


# ---------------------------------------------------------------------------
# precompute_latents — ίδια λογική με loader.precompute_latents,
# αλλά εφαρμόζει το transform ΠΡΙΝ το encoding.
# ---------------------------------------------------------------------------
@torch.no_grad()
def precompute_latents_transformed(encode_fn, root, out_root, transform_fn,
                                   shift=0, batch=256, device="cuda"):
    """Encode all episodes, applying transform_fn to each image BEFORE encoding.
    encode_fn(img_t, img_tp1) -> z.  transform_fn(img) -> transformed_img."""
    from os.path import join, basename
    from os import makedirs
    makedirs(out_root, exist_ok=True)
    for f in tqdm(list_npz(root), desc="encoding (transformed)"):
        with np.load(f) as d:
            imgs = torch.from_numpy(d["imgs"].astype(np.float32) / 255.0).permute(0, 3, 1, 2)
            acts = d["acts"].astype(np.float32)
            states = d["states"].astype(np.float32)
            x = (d[f"noisy_states_{shift}"] if shift in (2, 5, 10)
                 else d["states"]).astype(np.float32)

        # Εφαρμογή transform ΣΕ ΟΛΑ τα frames ΠΡΙΝ το encoding
        imgs = transform_fn(imgs.to(device))

        img_t, img_tp1 = imgs[:-1], imgs[1:]
        zs = []
        for b in range(0, img_t.shape[0], batch):
            zb = encode_fn(img_t[b:b + batch], img_tp1[b:b + batch])
            zs.append(zb.cpu().numpy())
        z = np.concatenate(zs, 0).astype(np.float32) if zs else np.empty((0, 0), np.float32)

        np.savez_compressed(join(out_root, basename(f)),
                            z=z, acts=acts[:-1], states=states[:-1], x=x[:-1])


# ---------------------------------------------------------------------------
# Rollout + per-window errors (ΙΔΙΟ με lunar_eval_baseline_vs_p1.py)
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
    """ -> (N, L, 8) STANDARDIZED squared error (preds[:, :8] vs GT standardized state). """
    model.eval()
    chunks = []
    for batch in loader:
        batch = [b.to(device, non_blocking=True) for b in batch]
        preds, state_tp1 = free_run(model, batch, encoded)
        err2 = (preds[..., :N_SUP] - state_tp1) ** 2
        chunks.append(err2.cpu().numpy())
    return np.concatenate(chunks, axis=0)


# ---------------------------------------------------------------------------
# Robust statistics (ΙΔΙΑ με lunar_eval_baseline_vs_p1.py)
# ---------------------------------------------------------------------------
def median_iqr(arr):
    """arr (N,L) -> median, q25, q75 per horizon."""
    return (np.median(arr, axis=0),
            np.percentile(arr, 25, axis=0),
            np.percentile(arr, 75, axis=0))


def bootstrap_paired(diff, n_boot, rng):
    """diff (N,L) = mse_base − mse_p2 per window/horizon (>0 => p2 better).
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
# Model evaluation at a given transform level
# ---------------------------------------------------------------------------
def evaluate_model_transformed(m, device, mean_s, std_s, level, transform_type):
    """Load VAE, encode test images WITH transform, evaluate LSTM -> (N,L,8) per mode."""
    transform_fn = make_transform_fn(transform_type, level, device)
    level_tag = f"{transform_type}_{level:.2f}".replace(".", "p")

    print(f"\n[{m['label']}] transform={transform_type} level={level:.2f}")
    print(f"  VAE ({m['vae_ckpt']}) -> precompute test latents (transformed)")
    vae = m["make_vae"]().to(device)
    vae.load_state_dict(torch.load(m["vae_ckpt"], map_location=device)); vae.eval()

    # Encode function — wraps the VAE's encode
    @torch.no_grad()
    def _encode(img_t, img_tp1):
        vae.eval()
        x = torch.cat([img_t, img_tp1], dim=1).to(device)
        mu, _ = vae.encode(x)
        return mu

    out_test = os.path.join(m["latent_root"], level_tag, "test")
    precompute_latents_transformed(_encode, os.path.join(DATA_ROOT, "test"),
                                   out_test, transform_fn=transform_fn, shift=SHIFT, device=device)
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
# Plotting helpers
# ---------------------------------------------------------------------------
def plot_median_iqr_per_level(all_err, levels, mode, save_dir, transform_type):
    """One figure per transform level: Baseline vs P2 median MSE curves."""
    base, p2 = MODELS[0]["label"], MODELS[1]["label"]
    cb, cp = MODELS[0]["color"], MODELS[1]["color"]
    horizons = np.arange(1, SEQ_LEN + 1)

    for nl in levels:
        plt.figure(figsize=(6.8, 4.8))
        for label, color in ((base, cb), (p2, cp)):
            arr = all_err[nl][label][mode].mean(axis=2)  # (N,L)
            med, q25, q75 = median_iqr(arr)
            plt.plot(horizons, med, color=color, lw=2, label=label)
            plt.fill_between(horizons, q25, q75, color=color, alpha=0.18)
        if LOG_Y:
            plt.yscale("log")
        plt.title(f"median state-MSE — {mode} | {transform_type} level={nl:.2f}")
        plt.xlabel("Prediction Horizon"); plt.ylabel("State MSE (median, IQR band)")
        plt.xlim(1, SEQ_LEN); plt.grid(alpha=0.3, which="both"); plt.legend()
        plt.tight_layout()
        tag = f"{nl:.2f}".replace(".", "p")
        plt.savefig(os.path.join(save_dir, f"bc_median_iqr_{mode}_{transform_type}_{tag}.png"), dpi=150)
        plt.show()


def plot_paired_per_level(all_err, levels, save_dir, transform_type, rng):
    """Paired Δ (base − p2) per transform level, all modes."""
    base, p2 = MODELS[0]["label"], MODELS[1]["label"]
    horizons = np.arange(1, SEQ_LEN + 1)
    paired_all = {}

    for mode in SEED_MODES:
        fig, axes = plt.subplots(1, len(levels),
                                 figsize=(5.5 * len(levels), 4.8), squeeze=False)
        for j, nl in enumerate(levels):
            diff = (all_err[nl][base][mode].mean(axis=2)
                    - all_err[nl][p2][mode].mean(axis=2))
            med, lo, hi = bootstrap_paired(diff, N_BOOT, rng)
            paired_all[(mode, nl)] = (med, lo, hi)
            ax = axes[0][j]
            ax.axhline(0, color="k", lw=1)
            ax.plot(horizons, med, color="C2", lw=2, label=f"median({base} − {p2})")
            ax.fill_between(horizons, lo, hi, color="C2", alpha=0.25, label="95% bootstrap CI")
            ax.set_title(f"paired Δ — {mode} | level={nl:.2f}")
            ax.set_xlabel("Horizon"); ax.set_ylabel("Δ state-MSE")
            ax.set_xlim(1, SEQ_LEN); ax.grid(alpha=0.3); ax.legend(fontsize=7)
        plt.suptitle(f"Paired difference (>0 ⇒ {p2} better) — {transform_type}, {mode}", y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"bc_paired_{mode}_{transform_type}.png"),
                    dpi=150, bbox_inches="tight")
        plt.show()
    return paired_all


def plot_degradation_summary(all_err, levels, save_dir, transform_type):
    """Summary: median MSE at horizon H vs transform level, for each model & mode.
    Δείχνει πόσο degrade-άρει κάθε μοντέλο σε αυξανόμενη διαταραχή (P2 = πιο flat = invariant)."""
    base, p2 = MODELS[0]["label"], MODELS[1]["label"]
    cb, cp = MODELS[0]["color"], MODELS[1]["color"]
    SUMMARY_HORIZONS = [1, 10, 20, 30]  # which horizons to show

    for mode in SEED_MODES:
        fig, axes = plt.subplots(1, len(SUMMARY_HORIZONS),
                                 figsize=(4.5 * len(SUMMARY_HORIZONS), 4.2), squeeze=False)
        for hi, h in enumerate(SUMMARY_HORIZONS):
            if h > SEQ_LEN:
                continue
            ax = axes[0][hi]
            for label, color, marker in ((base, cb, "o"), (p2, cp, "s")):
                med_vals = []
                q25_vals = []
                q75_vals = []
                for nl in levels:
                    arr = all_err[nl][label][mode].mean(axis=2)[:, h - 1]  # (N,)
                    med_vals.append(np.median(arr))
                    q25_vals.append(np.percentile(arr, 25))
                    q75_vals.append(np.percentile(arr, 75))
                med_vals = np.array(med_vals)
                q25_vals = np.array(q25_vals)
                q75_vals = np.array(q75_vals)
                ax.plot(levels, med_vals, color=color, lw=2, marker=marker,
                        markersize=6, label=label)
                ax.fill_between(levels, q25_vals, q75_vals, color=color, alpha=0.15)
            if LOG_Y:
                ax.set_yscale("log")
            ax.set_title(f"h={h}")
            ax.set_xlabel(f"Transform level ({transform_type})")
            if hi == 0:
                ax.set_ylabel("State MSE (median)")
            ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
        plt.suptitle(f"Degradation under {transform_type} — {mode} (πιο flat = πιο invariant)", y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"bc_degradation_{mode}_{transform_type}.png"),
                    dpi=150, bbox_inches="tight")
        plt.show()


def plot_perdim_per_level(all_err, levels, save_dir, transform_type):
    """Per-dimension curves at each transform level (encoded mode)."""
    base, p2 = MODELS[0]["label"], MODELS[1]["label"]
    cb, cp = MODELS[0]["color"], MODELS[1]["color"]
    horizons = np.arange(1, SEQ_LEN + 1)
    pm = "encoded" if "encoded" in SEED_MODES else SEED_MODES[0]

    for nl in levels:
        fig, axes = plt.subplots(1, N_SUP, figsize=(4.2 * N_SUP, 4.0), squeeze=False)
        for d in range(N_SUP):
            ax = axes[0][d]
            for label, color in ((base, cb), (p2, cp)):
                med, q25, q75 = median_iqr(all_err[nl][label][pm][:, :, d])
                ax.plot(horizons, med, color=color, lw=2, label=label)
                ax.fill_between(horizons, q25, q75, color=color, alpha=0.18)
            if LOG_Y:
                ax.set_yscale("log")
            ax.set_title(f"{DIM_NAMES[d]} — level={nl:.2f}")
            ax.set_xlabel("Horizon"); ax.set_xlim(1, SEQ_LEN); ax.grid(alpha=0.3, which="both")
            if d == 0:
                ax.set_ylabel("MSE (median, standardized)"); ax.legend()
        plt.suptitle(f"Per-dim MSE — {pm} | {transform_type} level={nl:.2f}", y=1.02)
        plt.tight_layout()
        tag = f"{nl:.2f}".replace(".", "p")
        plt.savefig(os.path.join(save_dir, f"bc_perdim_{pm}_{transform_type}_{tag}.png"),
                    dpi=150, bbox_inches="tight")
        plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean_s, std_s = load_norm_stats(NORM_STATS)
    std4 = np.asarray(std_s[:N_SUP], dtype=np.float64)
    rng = np.random.default_rng(BOOT_SEED)
    horizons = np.arange(1, SEQ_LEN + 1)
    assert len(MODELS) == 2, "Paired analysis expects exactly 2 models (base, p2)."

    base, p2 = MODELS[0]["label"], MODELS[1]["label"]

    # ---- Collect errors at each transform level ----
    # all_err[level][label][mode] = (N, L, 8)
    all_err = {}
    for nl in TRANSFORM_LEVELS:
        print(f"\n{'='*60}")
        print(f"  TRANSFORM LEVEL: {TRANSFORM_TYPE} level={nl:.2f}")
        print(f"{'='*60}")
        all_err[nl] = {}
        for m in MODELS:
            all_err[nl][m["label"]] = evaluate_model_transformed(
                m, device, mean_s, std_s, nl, TRANSFORM_TYPE)

        # Align window counts between models (same as original)
        for mode in SEED_MODES:
            nb = all_err[nl][base][mode].shape[0]
            np2 = all_err[nl][p2][mode].shape[0]
            n = min(nb, np2)
            if nb != np2:
                print(f"[WARN] mode={mode}: #windows differ ({nb} vs {np2}); "
                      f"truncating to {n} for pairing.")
            all_err[nl][base][mode] = all_err[nl][base][mode][:n]
            all_err[nl][p2][mode] = all_err[nl][p2][mode][:n]

    # ---- (1) Median + IQR curves per transform level ----
    for mode in SEED_MODES:
        plot_median_iqr_per_level(all_err, TRANSFORM_LEVELS, mode, SAVE_DIR, TRANSFORM_TYPE)

    # ---- (2) Paired Δ per transform level ----
    paired_all = plot_paired_per_level(all_err, TRANSFORM_LEVELS, SAVE_DIR, TRANSFORM_TYPE, rng)

    # ---- (3) Per-dim curves per transform level (encoded mode) ----
    plot_perdim_per_level(all_err, TRANSFORM_LEVELS, SAVE_DIR, TRANSFORM_TYPE)

    # ---- (4) DEGRADATION SUMMARY: MSE vs transform level at fixed horizons ----
    plot_degradation_summary(all_err, TRANSFORM_LEVELS, SAVE_DIR, TRANSFORM_TYPE)

    # ---- (5) Summary table ----
    HS = [h for h in (1, 10, 20, 30) if h <= SEQ_LEN]
    print(f"\n{'='*80}")
    print(f"=== SUMMARY: median state-MSE (standardized) under {TRANSFORM_TYPE} ===")
    print(f"{'='*80}")
    for nl in TRANSFORM_LEVELS:
        print(f"\n  --- level={nl:.2f} ---")
        for mode in SEED_MODES:
            print(f"    [{mode}]")
            for label in (base, p2):
                med = np.median(all_err[nl][label][mode].mean(axis=2), axis=0)
                print(f"      {label:<12} " +
                      "  ".join(f"h{h}={med[h-1]:.5f}" for h in HS))
            if (mode, nl) in paired_all:
                med_d, lo_d, hi_d = paired_all[(mode, nl)]
                print(f"      paired Δ(>0⇒{p2})  " +
                      "  ".join(f"h{h}={med_d[h-1]:+.5f}"
                                f"[{lo_d[h-1]:+.5f},{hi_d[h-1]:+.5f}]" for h in HS))

    # ---- save ----
    save_dict = {"horizons": horizons, "std4": std4,
                 "transform_type": TRANSFORM_TYPE,
                 "transform_levels": np.array(TRANSFORM_LEVELS),
                 "seed_modes": np.array(SEED_MODES)}
    for nl in TRANSFORM_LEVELS:
        tag = f"{nl:.2f}".replace(".", "p")
        for lab in (base, p2):
            for md in SEED_MODES:
                save_dict[f"{lab}_{md}_{tag}_err_median"] = np.median(
                    all_err[nl][lab][md].mean(axis=2), axis=0)
        for md in SEED_MODES:
            if (md, nl) in paired_all:
                m, l, h = paired_all[(md, nl)]
                save_dict[f"paired_{md}_{tag}_median"] = m
                save_dict[f"paired_{md}_{tag}_lo"] = l
                save_dict[f"paired_{md}_{tag}_hi"] = h
    np.savez(os.path.join(SAVE_DIR, "cmp_p2_bc_curves.npz"), **save_dict)
    print("\nsaved figures + cmp_p2_bc_curves.npz ->", SAVE_DIR)


if __name__ == "__main__":
    main()