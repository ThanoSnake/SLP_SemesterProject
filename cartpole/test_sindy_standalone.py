"""
test_sindy_standalone.py — Standalone αξιολόγηση του SINDy world-model (CartPole).

ΤΙ ΚΑΝΕΙ (ίδια μετρική/διαγράμματα με τα test_pX του baseline):
  1) Fit του αραιού Ξ (dyn_phys της Definition 2 του paper) στα TRAIN transitions (x_t,u_t)->x_{t+1},
     σε ΔΥΟ πηγές: ENCODED z[:4] (primary, ίδια «οπτική» με τον LSTM) και CLEAN GT (physics ref).
  2) Τυπώνει τις ΑΝΑΚΑΛΥΦΘΕΙΣΕΣ ΕΞΙΣΩΣΕΙΣ (interpretability — αυτό που ο LSTM δεν δίνει).
  3) Discrete rollout 30 βημάτων από τον ENCODED seed z_0[:4] (de-standardized) + actions,
     ολοκληρώνοντας τον χάρτη x_{t+1}=x_t+Θ(x_t,F_t)·Ξ.
  4) Μετρική ΙΔΙΑ με test_pX: standardized state-MSE, median+IQR ανά horizon (overall & per-dim).
  5) Τρέχει σε clean ΚΑΙ noisy-image conditions (θόρυβος μόνο στο test encoding, όπως test_p1/p3).

ΚΑΜΠΥΛΕΣ ανά noise condition:
  * SINDy (enc fit)        : Ξ από encoded train, ENCODED seed         [primary, fair vs LSTM]
  * SINDy (GT fit)         : Ξ από clean GT train, ENCODED seed        [πόσο καλή είναι η δυναμική]
  * SINDy (GT fit·GT seed) : Ξ από GT, GT seed                         [pure-physics ceiling]

Τοποθεσία: flat μέσα στο cartpole/ (δίπλα στα vae/lstm/loader). Τρέξε: !python3 cartpole/test_sindy_standalone.py
Συμπλήρωσε τα <...> paths (ίδιο convention με τα υπόλοιπα scripts).
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
LATENT_ROOT = "/kaggle/working/sindy_latents"            # cache encoded splits
SAVE_DIR = "/kaggle/working/sindy_standalone_out"

SEQ_LEN, TEST_STRIDE, N_SUP = 30, 1, 4

# SINDy
FEATURE_MODE = "physics"      # "physics" (cartpole-aware) | "poly2" (generic ablation)
THRESHOLD = 0.02
RIDGE = 1e-6

# Noise conditions (clean + δύο gaussian επίπεδα όπως test_p3/test_p1)
NOISE_CONDS = [
    ("gaussian", 0.0),
    ("gaussian", 0.05),
    ("gaussian", 0.10),
]
NOISE_SEED = 42

DIM_NAMES = ["x", "x_dot", "theta", "theta_dot"]
DIM_LABELS = ["x", r"$\dot{x}$", r"$\theta$", r"$\dot{\theta}$"]
DIM_UNITS = ["(cart pos)", "(cart vel)", "[rad]", "[rad/s]"]
HS = [h for h in (1, 10, 20, 30) if h <= SEQ_LEN]
LOG_Y = True


# ---------------------------------------------------------------------------
# Fit Ξ (encoded + GT) στο CLEAN train + τύπωσε εξισώσεις
# ---------------------------------------------------------------------------
def fit_models(train_dir, mean, std):
    Xe, Fe, Xne = assemble_fit_data(train_dir, "encoded", mean, std)
    Xi_e, names = fit_sindy(Xe, Fe, Xne, mode=FEATURE_MODE, threshold=THRESHOLD, ridge=RIDGE)
    Xg, Fg, Xng = assemble_fit_data(train_dir, "gt", mean, std)
    Xi_g, _ = fit_sindy(Xg, Fg, Xng, mode=FEATURE_MODE, threshold=THRESHOLD, ridge=RIDGE)

    print("\n" + "=" * 78)
    print("ΑΝΑΚΑΛΥΦΘΕΙΣΕΣ ΕΞΙΣΩΣΕΙΣ (discrete delta map)  —  dyn_phys της Definition 2")
    print("=" * 78)
    print(f"[ENCODED fit]  (#train transitions={Xe.shape[0]}, mode={FEATURE_MODE})")
    for line in format_equations(Xi_e, names):
        print("   " + line)
    print(f"\n[GT fit]       (#train transitions={Xg.shape[0]}, mode={FEATURE_MODE})")
    for line in format_equations(Xi_g, names):
        print("   " + line)
    print("=" * 78)
    return Xi_e, Xi_g, names


# ---------------------------------------------------------------------------
# Rollout + standardized sq-err ανά noise condition
# ---------------------------------------------------------------------------
def evaluate_condition(test_dir, Xi_e, Xi_g, mean, std):
    seed_enc, U, gt = assemble_windows(test_dir, mean, std, seq_len=SEQ_LEN,
                                          stride=TEST_STRIDE, which_seed="encoded")
    seed_gt, _, _ = assemble_windows(test_dir, mean, std, seq_len=SEQ_LEN,
                                        stride=TEST_STRIDE, which_seed="gt")
    curves = {
        "SINDy (enc fit)": sindy_rollout(seed_enc, U, Xi_e, mode=FEATURE_MODE),
        "SINDy (GT fit)": sindy_rollout(seed_enc, U, Xi_g, mode=FEATURE_MODE),
        "SINDy (GT fit·GT seed)": sindy_rollout(seed_gt, U, Xi_g, mode=FEATURE_MODE),
    }
    return {k: sq_err_standardized(v, gt, std) for k, v in curves.items()}, gt


COLORS = {"SINDy (enc fit)": "C2", "SINDy (GT fit)": "C4", "SINDy (GT fit·GT seed)": "C7"}
STYLES = {"SINDy (enc fit)": "-", "SINDy (GT fit)": "--", "SINDy (GT fit·GT seed)": ":"}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_overall(err, tag, title, save_dir):
    horizons = np.arange(1, SEQ_LEN + 1)
    plt.figure(figsize=(6.8, 4.8))
    for name, e in err.items():
        med, q25, q75 = median_iqr(e.mean(axis=2))
        plt.plot(horizons, med, color=COLORS[name], ls=STYLES[name], lw=2, label=name)
        plt.fill_between(horizons, q25, q75, color=COLORS[name], alpha=0.12)
    if LOG_Y:
        plt.yscale("log")
    plt.title(title)
    plt.xlabel("Prediction Horizon"); plt.ylabel("State MSE (median, IQR band) [standardized]")
    plt.xlim(1, SEQ_LEN); plt.grid(alpha=0.3, which="both"); plt.legend()
    plt.tight_layout()
    p = os.path.join(save_dir, f"sindy_standalone_overall_{tag}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print("saved:", p)


def plot_perdim(err, tag, title, save_dir):
    horizons = np.arange(1, SEQ_LEN + 1)
    fig, axes = plt.subplots(1, N_SUP, figsize=(4.2 * N_SUP, 4.0), squeeze=False)
    for d in range(N_SUP):
        ax = axes[0][d]
        for name, e in err.items():
            med, q25, q75 = median_iqr(e[:, :, d])
            ax.plot(horizons, med, color=COLORS[name], ls=STYLES[name], lw=2, label=name)
            ax.fill_between(horizons, q25, q75, color=COLORS[name], alpha=0.12)
        if LOG_Y:
            ax.set_yscale("log")
        ax.set_title(f"{DIM_LABELS[d]} {DIM_UNITS[d]}")
        ax.set_xlabel("Horizon"); ax.set_xlim(1, SEQ_LEN); ax.grid(alpha=0.3, which="both")
        if d == 0:
            ax.set_ylabel("MSE (median, standardized)"); ax.legend(fontsize=8)
    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    p = os.path.join(save_dir, f"sindy_standalone_perdim_{tag}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("saved:", p)


def print_summary(err, label):
    print(f"\n--- median standardized state-MSE | {label} ---")
    for name, e in err.items():
        med = np.median(e.mean(axis=2), axis=0)
        print(f"  {name:<24} " + "  ".join(f"h{h}={med[h-1]:.5f}" for h in HS))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    mean, std = load_norm_stats(NORM_STATS)

    # ---- encode CLEAN train (για το fit) ----
    train_dir = ensure_encoded(VAE_CKPT, DATA_ROOT,
                                   os.path.join(LATENT_ROOT, "clean"),
                                   device, noise_fn=None, splits=("train",))["train"]
    Xi_e, Xi_g, names = fit_models(train_dir, mean, std)

    save_curves = {"horizons": np.arange(1, SEQ_LEN + 1)}
    for ntype, level in NOISE_CONDS:
        tag = noise_tag(ntype, level)
        label = "clean" if level == 0.0 else f"{ntype} σ={level:.2f}"
        print(f"\n{'='*60}\n  CONDITION: {label}\n{'='*60}")

        nf = make_noise_fn(ntype, level, NOISE_SEED, device)
        test_dir = ensure_encoded(VAE_CKPT, DATA_ROOT,
                                      os.path.join(LATENT_ROOT, tag),
                                      device, noise_fn=nf, splits=("test",),
                                      force=(level != 0.0))["test"]
        err, _ = evaluate_condition(test_dir, Xi_e, Xi_g, mean, std)

        plot_overall(err, tag, f"SINDy standalone — {label}", SAVE_DIR)
        plot_perdim(err, tag, f"SINDy standalone (per-dim) — {label}", SAVE_DIR)
        print_summary(err, label)
        for name, e in err.items():
            save_curves[f"{tag}__{name}__median"] = np.median(e.mean(axis=2), axis=0)

    np.savez(os.path.join(SAVE_DIR, "sindy_standalone_curves.npz"), **save_curves)
    print("\nsaved figures + sindy_standalone_curves.npz ->", SAVE_DIR)


if __name__ == "__main__":
    main()
