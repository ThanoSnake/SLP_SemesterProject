# ========= Test Cartpole P2 vs Baseline =========
"""
test_p2.py — Evaluation of Baseline vs Principle 2 under BRIGHTNESS/CONTRAST jitter (CartPole).

FOCUSED VERSION (single setting) — same structure/plots as test_p1, but with the
PHOTOMETRIC transform of P2 (NOT gaussian noise):
  * ONLY brightness/contrast jitter at ONE level = 0.2 (the invariance target of Principle 2).
  * ONLY "encoded" seed mode (z_0 from the VAE -> LSTM rollout).
  * The transform is applied EXCLUSIVELY at the encoding stage (precompute_latents), before the
    encoder. It affects NEITHER the ground-truth states NOR the LSTM checkpoints.
  * The transform's IMPLEMENTATION LOGIC (apply_*, precompute_latents_transformed) is IDENTICAL
    to the original test_p2 — only the output structure changed (plots like test_p1).

OUTPUTS:
  (1) Overall median+IQR state-MSE per horizon (mean over dims)   [standardized]
  (2) Per-dim median+IQR state-MSE per horizon                    [standardized]
  (3) Paired Δ (baseline − p2) median + 95% bootstrap CI per horizon
  (4) PHYSICAL QUANTITIES of a RANDOM test window: GT vs predicted-baseline vs predicted-p2 [physical]
  (5) FRAME ENCODING CHECK: a random frame -> brightness/contrast transform -> visualization
      (original | transformed) -> encode with baseline & p2 -> each one's prediction & the gap from GT.
      (Shows P2's INVARIANCE: its physical encoding shifts less.)

WHY brightness/contrast (not gaussian noise): P2 was trained to be INVARIANT to
brightness/contrast -> this transform is what brings out its advantage (it changes the image
WITHOUT changing the real physical state -> matches "perturb the input, measure against clean GT").
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from paths import BASELINE_LSTM, BASELINE_VAE, DATA_ROOT, P2_LSTM, P2_VAE, outputs
from loader import LatentSequenceDataset, list_npz, load_norm_stats
from vae import VAE
from vae_p2 import VAE_P2
from lstm import LatentPredictor

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
NORM_STATS = os.path.join(DATA_ROOT, "norm_stats.npz")
SAVE_DIR = outputs("cartpole_p2_out")

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
# TRANSFORM CONFIG — a single setting: brightness/contrast level=0.2 (P2's invariance target)
# ---------------------------------------------------------------------------
TRANSFORM_TYPE = "brightness_contrast"   # "brightness" | "contrast" | "brightness_contrast"
TRANSFORM_LEVEL = 0.2                     # factor = 1 ± level
TRANSFORM_SIGN = +1.0                     # +1 -> brighter/higher contrast; -1 -> darker

# Trajectory plot (4): a random test window
TRAJ_SEED = None                  # None -> GENUINELY random (a different window every run); int -> reproducible
TRAJ_WINDOW = None                # None -> random; or an integer index for a specific window
N_TRAJ_WINDOWS = 1
# Frame-encoding check (5): a random frame
FRAME_SEED = None                 # None -> a GENUINELY random frame every run; int -> reproducible

# ---------------------------------------------------------------------------
# Model definitions — Baseline vs P2 (clean-trained VAE + encoded LSTM)
# ---------------------------------------------------------------------------
MODELS = [
    {"label": "Baseline", "color": "C0",
     "make_vae": lambda: VAE(latent_size=LATENT_SIZE),
     "vae_ckpt": BASELINE_VAE,
     "lstm_ckpt": BASELINE_LSTM,
     "latent_root": outputs("test_p2_latents/baseline")},
    {"label": "Principle 2", "color": "C2",
     "make_vae": lambda: VAE_P2(latent_size=LATENT_SIZE),
     "vae_ckpt": P2_VAE,
     "lstm_ckpt": P2_LSTM,
     "latent_root": outputs("test_p2_latents/p2")},
]


# ---------------------------------------------------------------------------
# Photometric transforms — SAME LOGIC as the original test_p2
# (float [0,1] image tensors (B/T,3,H,W))
# ---------------------------------------------------------------------------
def apply_brightness(img, level, sign):
    """Multiplicative brightness: img * (1 ± level)."""
    return torch.clamp(img * (1.0 + sign * level), 0.0, 1.0)


def apply_contrast(img, level, sign):
    """Contrast around the per-frame mean: (img - m) * (1 ± level) + m."""
    m = img.mean(dim=(1, 2, 3), keepdim=True)
    return torch.clamp((img - m) * (1.0 + sign * level) + m, 0.0, 1.0)


def apply_brightness_contrast(img, level, sign):
    """Brightness + contrast together (like P2's training color_jitter)."""
    out = img * (1.0 + sign * level)
    m = out.mean(dim=(1, 2, 3), keepdim=True)
    return torch.clamp((out - m) * (1.0 + sign * level) + m, 0.0, 1.0)


def make_transform_fn(transform_type, level):
    """Deterministic (img_tensor) -> transformed_img_tensor."""
    if level == 0.0:
        return lambda x: x
    sign = TRANSFORM_SIGN
    if transform_type == "brightness":
        return lambda x: apply_brightness(x, level, sign)
    elif transform_type == "contrast":
        return lambda x: apply_contrast(x, level, sign)
    elif transform_type == "brightness_contrast":
        return lambda x: apply_brightness_contrast(x, level, sign)
    raise ValueError(f"Unknown transform type: {transform_type}")


# ---------------------------------------------------------------------------
# precompute_latents_transformed — SAME LOGIC as the original test_p2
# (applies the transform BEFORE encoding)
# ---------------------------------------------------------------------------
@torch.no_grad()
def precompute_latents_transformed(encode_fn, root, out_root, transform_fn,
                                   shift=0, batch=256, device="cuda"):
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

        imgs = transform_fn(imgs.to(device))                   # transform on ALL frames before encoding
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
# Model evaluation at the single transform setting
# ---------------------------------------------------------------------------
def evaluate_model_transformed(m, device, mean_s, std_s):
    """Load VAE, encode test images WITH brightness/contrast level=TRANSFORM_LEVEL,
    rollout LSTM (encoded) -> {"pred": (N,L,4), "gt": (N,L,4)} standardized."""
    transform_fn = make_transform_fn(TRANSFORM_TYPE, TRANSFORM_LEVEL)
    tag = f"{TRANSFORM_TYPE}_{TRANSFORM_LEVEL:.2f}".replace(".", "p")

    print(f"\n[{m['label']}] transform={TRANSFORM_TYPE} level={TRANSFORM_LEVEL:.2f}")
    print(f"  VAE ({m['vae_ckpt']}) -> precompute test latents (transformed)")
    vae = m["make_vae"]().to(device)
    vae.load_state_dict(torch.load(m["vae_ckpt"], map_location=device)); vae.eval()

    @torch.no_grad()
    def _encode(img_t, img_tp1):
        vae.eval()
        x = torch.cat([img_t, img_tp1], dim=1).to(device)
        mu, _ = vae.encode(x)
        return mu

    out_test = os.path.join(m["latent_root"], tag, "test")
    precompute_latents_transformed(_encode, os.path.join(DATA_ROOT, "test"),
                                   out_test, transform_fn=transform_fn, shift=SHIFT, device=device)
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
# Plots (1)–(4): same as test_p1
# ---------------------------------------------------------------------------
def plot_median_iqr(err, save_dir):
    """(1) Overall median+IQR state-MSE (mean over dims) — Baseline vs P2."""
    horizons = np.arange(1, SEQ_LEN + 1)
    plt.figure(figsize=(6.8, 4.8))
    for m in MODELS:
        arr = err[m["label"]].mean(axis=2)
        med, q25, q75 = median_iqr(arr)
        plt.plot(horizons, med, color=m["color"], lw=2, label=m["label"])
        plt.fill_between(horizons, q25, q75, color=m["color"], alpha=0.18)
    if LOG_Y:
        plt.yscale("log")
    plt.title(f"median state-MSE (encoded) | {TRANSFORM_TYPE} level={TRANSFORM_LEVEL:.2f}")
    plt.xlabel("Prediction Horizon"); plt.ylabel("State MSE (median, IQR band)")
    plt.xlim(1, SEQ_LEN); plt.grid(alpha=0.3, which="both"); plt.legend()
    plt.tight_layout()
    p = os.path.join(save_dir, "p2_median_iqr_encoded.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print("saved:", p)


def plot_perdim(err, save_dir):
    """(2) Per-dim median+IQR state-MSE — Baseline vs P2."""
    horizons = np.arange(1, SEQ_LEN + 1)
    fig, axes = plt.subplots(1, N_SUP, figsize=(4.2 * N_SUP, 4.0), squeeze=False)
    for d in range(N_SUP):
        ax = axes[0][d]
        for m in MODELS:
            med, q25, q75 = median_iqr(err[m["label"]][:, :, d])
            ax.plot(horizons, med, color=m["color"], lw=2, label=m["label"])
            ax.fill_between(horizons, q25, q75, color=m["color"], alpha=0.18)
        if LOG_Y:
            ax.set_yscale("log")
        ax.set_title(f"{DIM_LABELS[d]} {DIM_UNITS[d]}")
        ax.set_xlabel("Horizon"); ax.set_xlim(1, SEQ_LEN); ax.grid(alpha=0.3, which="both")
        if d == 0:
            ax.set_ylabel("MSE (median, standardized)"); ax.legend()
    plt.suptitle(f"Per-dim state-MSE (encoded) | {TRANSFORM_TYPE} level={TRANSFORM_LEVEL:.2f}", y=1.02)
    plt.tight_layout()
    p = os.path.join(save_dir, "p2_perdim_encoded.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("saved:", p)


def plot_paired(err, save_dir, rng):
    """(3) Paired Δ (Baseline − P2) median + 95% bootstrap CI."""
    base, p2 = MODELS[0]["label"], MODELS[1]["label"]
    horizons = np.arange(1, SEQ_LEN + 1)
    diff = err[base].mean(axis=2) - err[p2].mean(axis=2)
    med, lo, hi = bootstrap_paired(diff, N_BOOT, rng)

    plt.figure(figsize=(6.8, 4.8))
    plt.axhline(0, color="k", lw=1)
    plt.plot(horizons, med, color="C2", lw=2, label=f"median({base} − {p2})")
    plt.fill_between(horizons, lo, hi, color="C2", alpha=0.25, label="95% bootstrap CI")
    plt.title(f"Paired difference (>0 ⇒ {p2} better) — encoded | {TRANSFORM_TYPE} level={TRANSFORM_LEVEL:.2f}")
    plt.xlabel("Prediction Horizon"); plt.ylabel("Δ state-MSE")
    plt.xlim(1, SEQ_LEN); plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout()
    p = os.path.join(save_dir, "p2_paired_encoded.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print("saved:", p)
    return med, lo, hi


def plot_trajectory(data, mean_s, std_s, save_dir, rng):
    """(4) Physical trajectory of a RANDOM test window: GT vs pred-baseline vs pred-p2."""
    base, p2 = MODELS[0]["label"], MODELS[1]["label"]
    mean4 = np.asarray(mean_s[:N_SUP], np.float64)
    std4 = np.asarray(std_s[:N_SUP], np.float64)
    gt_all = data[base]["gt"]                                  # (N,L,4) standardized (== p2 gt)
    N, L, _ = gt_all.shape
    horizons = np.arange(1, L + 1)

    for wi in range(N_TRAJ_WINDOWS):
        w = TRAJ_WINDOW if TRAJ_WINDOW is not None else int(rng.integers(0, N))
        if not np.allclose(data[base]["gt"][w], data[p2]["gt"][w], atol=1e-4):
            print(f"[warn] window {w}: GT differs between models (window alignment?).")

        gt_phys = gt_all[w] * std4 + mean4
        b_phys = data[base]["pred"][w] * std4 + mean4
        p_phys = data[p2]["pred"][w] * std4 + mean4

        fig, axes = plt.subplots(2, 2, figsize=(12, 6))
        for d in range(N_SUP):
            ax = axes[d // 2][d % 2]
            ax.plot(horizons, gt_phys[:, d], color="k", lw=2.0, label="GT")
            ax.plot(horizons, b_phys[:, d], color=MODELS[0]["color"], lw=1.6, ls="--", label=base)
            ax.plot(horizons, p_phys[:, d], color=MODELS[1]["color"], lw=1.6, ls="--", label=p2)
            ax.set_title(f"{DIM_LABELS[d]} {DIM_UNITS[d]}")
            ax.set_xlabel("Prediction Horizon"); ax.set_xlim(1, L); ax.grid(alpha=0.3)
            if d == 0:
                ax.legend(fontsize=9)
        plt.suptitle(f"Physical trajectory — test window #{w} | "
                     f"{TRANSFORM_TYPE} level={TRANSFORM_LEVEL:.2f} (physical units)")
        plt.tight_layout()
        p = os.path.join(save_dir, f"p2_trajectory_window{w}.png")
        plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
        print("saved:", p)


# ---------------------------------------------------------------------------
# (5) Frame-encoding invariance check
# ---------------------------------------------------------------------------
@torch.no_grad()
def plot_frame_encoding(mean_s, std_s, device, save_dir, rng):
    """Random frame -> brightness/contrast transform -> visualize & encode with baseline & p2.
    Shows each one's physical-state prediction (on the TRANSFORMED frame) and the gap from GT.
    P2's INVARIANCE shows up as a smaller clean->transformed shift."""
    base, p2 = MODELS[0]["label"], MODELS[1]["label"]
    mean4 = np.asarray(mean_s[:N_SUP], np.float64)
    std4 = np.asarray(std_s[:N_SUP], np.float64)
    transform_fn = make_transform_fn(TRANSFORM_TYPE, TRANSFORM_LEVEL)

    # --- random episode + frame (the encoder needs a pair t, t+1) ---
    files = list_npz(os.path.join(DATA_ROOT, "test"))
    ep = files[int(rng.integers(0, len(files)))]
    with np.load(ep) as d:
        imgs = d["imgs"]
        states = d["states"].astype(np.float32)
    T = imgs.shape[0]
    t = int(rng.integers(0, T - 1))
    gt = states[t]                                             # raw physical (4,)

    def _frame(i):
        return torch.from_numpy(imgs[i].astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    f_t, f_tp1 = _frame(t), _frame(t + 1)
    f_t_tf, f_tp1_tf = transform_fn(f_t), transform_fn(f_tp1)

    # --- encode clean & transformed with each VAE ---
    preds_clean, preds_tf = {}, {}
    for m in MODELS:
        vae = m["make_vae"]().to(device)
        vae.load_state_dict(torch.load(m["vae_ckpt"], map_location=device)); vae.eval()
        mu_c, _ = vae.encode(torch.cat([f_t, f_tp1], dim=1))
        mu_t, _ = vae.encode(torch.cat([f_t_tf, f_tp1_tf], dim=1))
        preds_clean[m["label"]] = mu_c[0, :N_SUP].cpu().numpy() * std4 + mean4
        preds_tf[m["label"]] = mu_t[0, :N_SUP].cpu().numpy() * std4 + mean4
        del vae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- figure: original | transformed  +  per-dim bars (GT vs baseline vs p2, on the transformed) ---
    orig_np = f_t[0].permute(1, 2, 0).cpu().numpy()
    tf_np = f_t_tf[0].permute(1, 2, 0).cpu().numpy()
    fig = plt.figure(figsize=(15, 7))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.3, 1.0], hspace=0.4, wspace=0.3)
    ax0 = fig.add_subplot(gs[0, 0:2]); ax0.imshow(orig_np); ax0.set_title("original frame_t"); ax0.axis("off")
    ax1 = fig.add_subplot(gs[0, 2:4]); ax1.imshow(tf_np)
    ax1.set_title(f"transformed ({TRANSFORM_TYPE} level={TRANSFORM_LEVEL:.2f})"); ax1.axis("off")
    for d in range(N_SUP):
        ax = fig.add_subplot(gs[1, d])
        vals = [gt[d], preds_tf[base][d], preds_tf[p2][d]]
        ax.bar([0, 1, 2], vals, color=["k", MODELS[0]["color"], MODELS[1]["color"]])
        ax.axhline(gt[d], color="k", ls=":", lw=1)            # GT reference
        ax.axhline(0, color="0.6", lw=0.6)
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["GT", base, p2], rotation=20, fontsize=8)
        eb = preds_tf[base][d] - gt[d]; ep2 = preds_tf[p2][d] - gt[d]
        ax.set_title(f"{DIM_LABELS[d]} {DIM_UNITS[d]}\nΔ {base[:4]}={eb:+.3f}  {p2[:4]}={ep2:+.3f}", fontsize=8)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"P2 invariance check — encoded physical state on a TRANSFORMED frame "
                 f"(ep={os.path.basename(ep)} t={t}, physical units)", y=1.0)
    p = os.path.join(save_dir, "p2_frame_encoding.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("saved:", p)

    # --- printed table: error vs GT (clean & transformed) + shift clean->transformed ---
    print(f"\n  FRAME ENCODING CHECK  (ep={os.path.basename(ep)} t={t})")
    print(f"  {'dim':<10}{'GT':>10}{'|err| base(tf)':>16}{'|err| p2(tf)':>16}"
          f"{'shift base':>14}{'shift p2':>12}")
    for d in range(N_SUP):
        eb = abs(preds_tf[base][d] - gt[d]); ep2 = abs(preds_tf[p2][d] - gt[d])
        sb = abs(preds_tf[base][d] - preds_clean[base][d])    # clean->transformed shift
        sp = abs(preds_tf[p2][d] - preds_clean[p2][d])
        print(f"  {DIM_NAMES[d]:<10}{gt[d]:>10.4f}{eb:>16.4f}{ep2:>16.4f}{sb:>14.4f}{sp:>12.4f}")
    print("  (shift = |transformed - clean| encoding; smaller shift = more invariant -> P2 advantage)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean_s, std_s = load_norm_stats(NORM_STATS)
    rng = np.random.default_rng(BOOT_SEED)
    traj_rng = np.random.default_rng(TRAJ_SEED)
    frame_rng = np.random.default_rng(FRAME_SEED)
    assert len(MODELS) == 2, "Paired analysis expects exactly 2 models (base, p2)."
    base, p2 = MODELS[0]["label"], MODELS[1]["label"]

    print(f"\n{'='*60}\n  TRANSFORM: {TRANSFORM_TYPE} level={TRANSFORM_LEVEL:.2f} | encoded mode\n{'='*60}")
    data = {m["label"]: evaluate_model_transformed(m, device, mean_s, std_s) for m in MODELS}

    # Align window counts (same windows -> same GT; truncate to the min for the paired analysis)
    n = min(data[base]["pred"].shape[0], data[p2]["pred"].shape[0])
    if data[base]["pred"].shape[0] != data[p2]["pred"].shape[0]:
        print(f"[WARN] #windows differ ({data[base]['pred'].shape[0]} vs "
              f"{data[p2]['pred'].shape[0]}); truncating to {n}.")
    for label in (base, p2):
        data[label]["pred"] = data[label]["pred"][:n]
        data[label]["gt"] = data[label]["gt"][:n]

    err = {label: (data[label]["pred"] - data[label]["gt"]) ** 2 for label in (base, p2)}

    # ---- plots ----
    plot_median_iqr(err, SAVE_DIR)
    plot_perdim(err, SAVE_DIR)
    med_d, lo_d, hi_d = plot_paired(err, SAVE_DIR, rng)
    plot_trajectory(data, mean_s, std_s, SAVE_DIR, traj_rng)
    plot_frame_encoding(mean_s, std_s, device, SAVE_DIR, frame_rng)

    # ---- summary table ----
    HS = [h for h in (1, 10, 20, 30) if h <= SEQ_LEN]
    print(f"\n{'='*80}")
    print(f"=== SUMMARY: median state-MSE (standardized) | {TRANSFORM_TYPE} level={TRANSFORM_LEVEL:.2f}, encoded ===")
    print(f"{'='*80}")
    for label in (base, p2):
        med = np.median(err[label].mean(axis=2), axis=0)
        print(f"  {label:<12} " + "  ".join(f"h{h}={med[h-1]:.5f}" for h in HS))
    print(f"  paired Δ(>0⇒{p2})  " +
          "  ".join(f"h{h}={med_d[h-1]:+.5f}[{lo_d[h-1]:+.5f},{hi_d[h-1]:+.5f}]" for h in HS))

    # ---- save ----
    save_dict = {"horizons": np.arange(1, SEQ_LEN + 1),
                 "transform_type": TRANSFORM_TYPE, "transform_level": TRANSFORM_LEVEL}
    for label in (base, p2):
        save_dict[f"{label}_err_median"] = np.median(err[label].mean(axis=2), axis=0)
    save_dict["paired_median"], save_dict["paired_lo"], save_dict["paired_hi"] = med_d, lo_d, hi_d
    np.savez(os.path.join(SAVE_DIR, "cmp_p2_bc02_curves.npz"), **save_dict)
    print("\nsaved figures + cmp_p2_bc02_curves.npz ->", SAVE_DIR)


if __name__ == "__main__":
    main()
