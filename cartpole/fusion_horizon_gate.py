"""
fusion_horizon_gate.py — (C) Horizon-gated blend SINDy↔LSTM (NO re-training).

ΙΔΕΑ: συνδυάζουμε τις δύο προβλέψεις με βάρος που εξαρτάται από τον ορίζοντα:
        pred(h) = w(h)·SINDy(h) + (1−w(h))·LSTM(h)
Περιμένουμε w(h) να γέρνει προς LSTM σε μικρό ορίζοντα (πιάνει λεπτομέρειες/μη-μοντελοποιημένα)
και προς SINDy σε μεγάλο (σταθερή φυσική, δεν συσσωρεύει error).

ΣΥΝΤΟΝΙΣΜΟΣ (ελάχιστο, closed-form fit στο val — καμία επανεκπαίδευση):
  Η blended-MSE είναι ΤΕΤΡΑΓΩΝΙΚΗ ως προς w. Με a=SINDy−LSTM, b=LSTM−GT (standardized):
        MSE(w) = E[(w·a + b)²]  ⇒  w* = −E[a·b]/E[a²],  clipped σε [0,1].
  Υπολογίζεται ανά horizon (per-dim aggregated) στο val· εφαρμόζεται αυτούσιο στο test.

ΠΑΡΑΓΟΜΕΝΑ ανά noise condition:
  (0) Το ίδιο το gate w(h) vs horizon (δείχνει το crossover LSTM→SINDy)
  (1) Overall median+IQR: LSTM vs SINDy vs Blend
  (2) Per-dim median+IQR· (3) Paired Δ (best-single − Blend, >0 ⇒ το blend κερδίζει)

Τοποθεσία: flat μέσα στο cartpole/. Τρέξε: !python3 cartpole/fusion_horizon_gate.py
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from sindy_core import *          # SINDy core (numpy) + path-bootstrap για vae/lstm/loader
from sindy_eval_utils import *    # VAE encode / LSTM rollout / noise / measurement helpers

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATA_ROOT = "<cartpole-dataset>"
NORM_STATS = os.path.join(DATA_ROOT, "norm_stats.npz")
VAE_CKPT = "<cartpole-baseline-vae>"
LSTM_CKPT = "<cartpole-baseline-lstm>"
LATENT_ROOT = "/kaggle/working/sindy_latents"
SAVE_DIR = "/kaggle/working/sindy_horizon_gate_out"

SEQ_LEN, TEST_STRIDE, N_SUP = 30, 1, 4

FEATURE_MODE = "physics"
THRESHOLD, RIDGE = 0.02, 1e-6
GATE_PERDIM = False            # False -> w(h) (per-horizon scalar)· True -> w(h,d) (ablation)

NOISE_CONDS = [("gaussian", 0.0), ("gaussian", 0.05), ("gaussian", 0.10)]
NOISE_SEED = 42
N_BOOT, BOOT_SEED = 1000, 0

DIM_NAMES = ["x", "x_dot", "theta", "theta_dot"]
DIM_LABELS = ["x", r"$\dot{x}$", r"$\theta$", r"$\dot{\theta}$"]
DIM_UNITS = ["(cart pos)", "(cart vel)", "[rad]", "[rad/s]"]
HS = [h for h in (1, 10, 20, 30) if h <= SEQ_LEN]
LOG_Y = True
C_LSTM, C_SINDY, C_BLEND = "C0", "C2", "C1"


# ---------------------------------------------------------------------------
# SINDy/LSTM standardized preds σε ένα latent dir (ίδια windows)
# ---------------------------------------------------------------------------
def get_preds_std(lstm, latent_dir, Xi, mean, std, device):
    """ -> (sindy_std, lstm_std, gt_std) σχήμα (N,L,4), standardized phys dims."""
    mean4 = np.asarray(mean[:N_SUP], np.float64)
    std4 = np.asarray(std[:N_SUP], np.float64)
    lstm_std, gt_std = lstm_free_run_dir(lstm, latent_dir, mean, std, device)
    seed_enc, U, gt_raw = assemble_windows(latent_dir, mean, std, seq_len=SEQ_LEN,
                                              stride=TEST_STRIDE, which_seed="encoded")
    sindy_raw = sindy_rollout(seed_enc, U, Xi, mode=FEATURE_MODE)
    n = min(lstm_std.shape[0], sindy_raw.shape[0])
    sindy_std = (sindy_raw[:n] - mean4) / std4
    return sindy_std, lstm_std[:n], gt_std[:n]


# ---------------------------------------------------------------------------
# Gate fit (closed-form) + apply
# ---------------------------------------------------------------------------
def fit_gate(sindy_std, lstm_std, gt_std, perdim=False):
    a = sindy_std - lstm_std
    b = lstm_std - gt_std
    axes = (0,) if perdim else (0, 2)
    num = (a * b).mean(axis=axes)
    den = (a * a).mean(axis=axes)
    w = -num / np.maximum(den, 1e-12)
    return np.clip(w, 0.0, 1.0)                # (L,4) αν perdim, αλλιώς (L,)


def apply_gate(sindy_std, lstm_std, w, perdim=False):
    wv = w[None, :, :] if perdim else w[None, :, None]
    return wv * sindy_std + (1.0 - wv) * lstm_std


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_gate(w, tag, label, save_dir):
    horizons = np.arange(1, SEQ_LEN + 1)
    plt.figure(figsize=(6.8, 4.2))
    if GATE_PERDIM:
        for d in range(N_SUP):
            plt.plot(horizons, w[:, d], lw=2, label=DIM_LABELS[d])
    else:
        plt.plot(horizons, w, color=C_BLEND, lw=2, label="w(h)")
    plt.ylim(-0.02, 1.02); plt.axhline(0.5, color="k", lw=0.6, ls=":")
    plt.title(f"Learned gate w(h)  (1=SINDy, 0=LSTM) | {label}")
    plt.xlabel("Prediction Horizon"); plt.ylabel("w (SINDy weight)")
    plt.xlim(1, SEQ_LEN); plt.grid(alpha=0.3); plt.legend(fontsize=8)
    plt.tight_layout()
    p = os.path.join(save_dir, f"gate_w_{tag}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print("saved:", p)


def plot_overall(err, tag, label, save_dir):
    horizons = np.arange(1, SEQ_LEN + 1)
    plt.figure(figsize=(6.8, 4.8))
    for name, c in [("LSTM", C_LSTM), ("SINDy", C_SINDY), ("Blend", C_BLEND)]:
        med, q25, q75 = median_iqr(err[name].mean(axis=2))
        plt.plot(horizons, med, color=c, lw=2, label=name)
        plt.fill_between(horizons, q25, q75, color=c, alpha=0.12)
    if LOG_Y:
        plt.yscale("log")
    plt.title(f"Horizon-gated blend (C) | {label}")
    plt.xlabel("Prediction Horizon"); plt.ylabel("State MSE (median, IQR) [standardized]")
    plt.xlim(1, SEQ_LEN); plt.grid(alpha=0.3, which="both"); plt.legend()
    plt.tight_layout()
    p = os.path.join(save_dir, f"gate_overall_{tag}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print("saved:", p)


def plot_perdim(err, tag, label, save_dir):
    horizons = np.arange(1, SEQ_LEN + 1)
    fig, axes = plt.subplots(1, N_SUP, figsize=(4.2 * N_SUP, 4.0), squeeze=False)
    for d in range(N_SUP):
        ax = axes[0][d]
        for name, c in [("LSTM", C_LSTM), ("SINDy", C_SINDY), ("Blend", C_BLEND)]:
            med, q25, q75 = median_iqr(err[name][:, :, d])
            ax.plot(horizons, med, color=c, lw=2, label=name)
            ax.fill_between(horizons, q25, q75, color=c, alpha=0.10)
        if LOG_Y:
            ax.set_yscale("log")
        ax.set_title(f"{DIM_LABELS[d]} {DIM_UNITS[d]}")
        ax.set_xlabel("Horizon"); ax.set_xlim(1, SEQ_LEN); ax.grid(alpha=0.3, which="both")
        if d == 0:
            ax.set_ylabel("MSE (median, standardized)"); ax.legend(fontsize=8)
    plt.suptitle(f"Horizon-gated blend (C) per-dim | {label}", y=1.02)
    plt.tight_layout()
    p = os.path.join(save_dir, f"gate_perdim_{tag}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("saved:", p)


def plot_paired(err, tag, label, save_dir, rng):
    """Paired Δ (best single model − Blend): >0 ⇒ το blend κερδίζει τον καλύτερο single."""
    lstm_m = np.median(err["LSTM"].mean(axis=2), axis=0)
    sindy_m = np.median(err["SINDy"].mean(axis=2), axis=0)
    best = "LSTM" if lstm_m.mean() < sindy_m.mean() else "SINDy"
    diff = err[best].mean(axis=2) - err["Blend"].mean(axis=2)
    med, lo, hi = bootstrap_paired(diff, N_BOOT, rng)
    horizons = np.arange(1, SEQ_LEN + 1)
    plt.figure(figsize=(7.5, 4.8))
    plt.axhline(0, color="k", lw=1)
    plt.plot(horizons, med, color=C_BLEND, lw=2, label=f"{best} − Blend")
    plt.fill_between(horizons, lo, hi, color=C_BLEND, alpha=0.18)
    plt.title(f"Paired Δ (>0 ⇒ Blend better than {best}) | {label}")
    plt.xlabel("Prediction Horizon"); plt.ylabel("Δ state-MSE")
    plt.xlim(1, SEQ_LEN); plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout()
    p = os.path.join(save_dir, f"gate_paired_{tag}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print("saved:", p)
    return best, med, lo, hi


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

    lstm = load_lstm(LSTM_CKPT, device)
    save = {"horizons": np.arange(1, SEQ_LEN + 1)}

    for ntype, level in NOISE_CONDS:
        tag = noise_tag(ntype, level)
        label = "clean" if level == 0.0 else f"{ntype} σ={level:.2f}"
        print(f"\n{'='*60}\n  CONDITION: {label}\n{'='*60}")

        nf = make_noise_fn(ntype, level, NOISE_SEED, device)
        val_dir = ensure_encoded(VAE_CKPT, DATA_ROOT, os.path.join(LATENT_ROOT, tag),
                                     device, noise_fn=nf, splits=("val",),
                                     force=(level != 0.0))["val"]
        test_dir = ensure_encoded(VAE_CKPT, DATA_ROOT, os.path.join(LATENT_ROOT, tag),
                                      device, noise_fn=nf, splits=("test",),
                                      force=(level != 0.0))["test"]

        # ---- fit gate στο val ----
        sv, lv, gv = get_preds_std(lstm, val_dir, Xi_e, mean, std, device)
        w = fit_gate(sv, lv, gv, perdim=GATE_PERDIM)

        # ---- apply στο test ----
        st, lt, gt = get_preds_std(lstm, test_dir, Xi_e, mean, std, device)
        blend = apply_gate(st, lt, w, perdim=GATE_PERDIM)
        err = {"LSTM": (lt - gt) ** 2, "SINDy": (st - gt) ** 2, "Blend": (blend - gt) ** 2}

        plot_gate(w, tag, label, SAVE_DIR)
        plot_overall(err, tag, label, SAVE_DIR)
        plot_perdim(err, tag, label, SAVE_DIR)
        best, med_d, lo_d, hi_d = plot_paired(err, tag, label, SAVE_DIR, rng)

        print(f"  gate w(h) (per-horizon{'·per-dim' if GATE_PERDIM else ''}): "
              + ("see fig" if GATE_PERDIM else "  ".join(f"h{h}={w[h-1]:.2f}" for h in HS)))
        for name in ("LSTM", "SINDy", "Blend"):
            m = np.median(err[name].mean(axis=2), axis=0)
            print(f"  {name:<8} " + "  ".join(f"h{h}={m[h-1]:.5f}" for h in HS))
        print(f"  Δ({best}−Blend)  " +
              "  ".join(f"h{h}={med_d[h-1]:+.5f}[{lo_d[h-1]:+.5f},{hi_d[h-1]:+.5f}]" for h in HS))

        save[f"{tag}__w"] = w
        for name in ("LSTM", "SINDy", "Blend"):
            save[f"{tag}__{name}_median"] = np.median(err[name].mean(axis=2), axis=0)
        save[f"{tag}__paired_best"] = best
        save[f"{tag}__paired_median"] = med_d

    np.savez(os.path.join(SAVE_DIR, "horizon_gate_curves.npz"), **save)
    print("\nsaved figures + horizon_gate_curves.npz ->", SAVE_DIR)


if __name__ == "__main__":
    main()
