"""
test_sindy_vs_lstm.py — Head-to-head: SINDy vs baseline LSTM vs GT (env-agnostic: CartPole/LunarLander).

ΙΔΕΑ: στα ΙΔΙΑ test windows, με ΙΔΙΟ seed/actions, συγκρίνουμε δύο εκδοχές του dyn_phys:
  * LSTM  = v(dyn(z))      (η implicit, μαθημένη δυναμική του baseline world model)
  * SINDy = dyn_phys(v(z)) (η ρητή, αραιή δυναμική)
Αυτό είναι ΑΚΡΙΒΩΣ το τεστ της συνθήκης (ii) της Definition 2 του paper: πόσο κοντά
«μετατίθενται» (commute) οι δύο δυναμικές προς το πραγματικό GT.

ΠΑΡΑΓΟΜΕΝΑ (ίδια μετρική/σύμβαση με τα test_pX):
  (1) Overall median+IQR state-MSE ανά horizon — LSTM vs SINDy (+ προαιρ. GT-fit SINDy ref)
  (2) Per-dim median+IQR state-MSE ανά horizon
  (3) Paired Δ (LSTM − SINDy) median + 95% bootstrap CI  (>0 ⇒ SINDy καλύτερο σε αυτόν τον ορίζοντα)
  (4) Trajectory ενός ΤΥΧΑΙΟΥ test window: GT vs LSTM vs SINDy (physical units)
  Όλα σε clean ΚΑΙ noisy-image conditions.

ΑΝΑΜΕΝΟΜΕΝΗ ΙΣΤΟΡΙΑ: το SINDy εκτείνεται καλύτερα σε μακρύ ορίζοντα (σταθερή φυσική, δεν
συσσωρεύει error όπως το NN)· το LSTM πιάνει λεπτομέρειες/μη-μοντελοποιημένα κομμάτια κοντά.

ENV-AGNOSTIC: ΙΔΙΟ αρχείο για cartpole & lunarlander· environment από sindy_core/sindy_eval_utils.
Τρέξε: !python3 <env-folder>/test_sindy_vs_lstm.py
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from sindy_core import *          # SINDy core (numpy) + path-bootstrap για vae/lstm/loader
from sindy_eval_utils import *    # VAE encode / LSTM rollout / noise / measurement helpers

# ---------------------------------------------------------------------------
# CONFIG — env-agnostic· το per-environment config έρχεται από sindy_eval_utils/sindy_core (import *).
# ---------------------------------------------------------------------------
SAVE_DIR = f"/kaggle/working/sindy_{ENV_TAG}_vs_lstm_out"
INCLUDE_GT_REF = True          # προσθέτει faint «SINDy (GT fit)» ref στο overall plot

# Trajectory plot
TRAJ_SEED = None               # None -> γνήσια τυχαίο window· int -> reproducible
TRAJ_WINDOW = None             # None -> τυχαίο· ή ακέραιος index
N_TRAJ_WINDOWS = 1

LOG_Y = True
C_LSTM, C_SINDY, C_GTREF = "C0", "C2", "C7"


# ---------------------------------------------------------------------------
# Evaluation σε ένα noise condition -> err dicts + raw arrays για trajectory
# ---------------------------------------------------------------------------
def evaluate_condition(lstm, test_dir, Xi_e, Xi_g, mean, std, device):
    # LSTM rollout (standardized phys)
    pred_lstm, gt_std = lstm_free_run_dir(lstm, test_dir, mean, std, device)
    # SINDy rollout (raw) στα ΙΔΙΑ windows (encoded seed)
    seed_enc, U, gt_raw = assemble_windows(test_dir, mean, std, seq_len=SEQ_LEN,
                                              stride=TEST_STRIDE, which_seed="encoded")
    pred_sindy = sindy_rollout(seed_enc, U, Xi_e, mode=FEATURE_MODE)

    n = min(pred_lstm.shape[0], pred_sindy.shape[0])
    if pred_lstm.shape[0] != pred_sindy.shape[0]:
        print(f"[WARN] window count mismatch LSTM={pred_lstm.shape[0]} "
              f"SINDy={pred_sindy.shape[0]}; truncating to {n}.")
    pred_lstm, gt_std = pred_lstm[:n], gt_std[:n]
    pred_sindy, gt_raw = pred_sindy[:n], gt_raw[:n]

    err = {
        "LSTM": (pred_lstm - gt_std) ** 2,
        "SINDy": sq_err_standardized(pred_sindy, gt_raw, std),
    }
    if INCLUDE_GT_REF:
        pred_sindy_g = sindy_rollout(seed_enc[:n], U[:n], Xi_g, mode=FEATURE_MODE)
        err["SINDy (GT fit)"] = sq_err_standardized(pred_sindy_g, gt_raw, std)

    raw = {"gt": gt_raw, "lstm": destandardize(pred_lstm, mean, std), "sindy": pred_sindy}
    return err, raw


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _color(name):
    return {"LSTM": C_LSTM, "SINDy": C_SINDY, "SINDy (GT fit)": C_GTREF}[name]


def _style(name):
    return {"LSTM": "-", "SINDy": "-", "SINDy (GT fit)": ":"}[name]


def plot_overall(err, tag, label, save_dir):
    horizons = np.arange(1, SEQ_LEN + 1)
    plt.figure(figsize=(6.8, 4.8))
    for name, e in err.items():
        med, q25, q75 = median_iqr(e.mean(axis=2))
        plt.plot(horizons, med, color=_color(name), ls=_style(name), lw=2, label=name)
        plt.fill_between(horizons, q25, q75, color=_color(name), alpha=0.12)
    if LOG_Y:
        plt.yscale("log")
    plt.title(f"LSTM vs SINDy — median state-MSE | {label}")
    plt.xlabel("Prediction Horizon"); plt.ylabel("State MSE (median, IQR band) [standardized]")
    plt.xlim(1, SEQ_LEN); plt.grid(alpha=0.3, which="both"); plt.legend()
    plt.tight_layout()
    p = os.path.join(save_dir, f"vs_overall_{tag}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print("saved:", p)


def plot_perdim(err, tag, label, save_dir):
    horizons = np.arange(1, SEQ_LEN + 1)
    fig, axes = plt.subplots(1, N_SUP, figsize=(4.2 * N_SUP, 4.0), squeeze=False)
    for d in range(N_SUP):
        ax = axes[0][d]
        for name, e in err.items():
            med, q25, q75 = median_iqr(e[:, :, d])
            ax.plot(horizons, med, color=_color(name), ls=_style(name), lw=2, label=name)
            ax.fill_between(horizons, q25, q75, color=_color(name), alpha=0.12)
        if LOG_Y:
            ax.set_yscale("log")
        ax.set_title(f"{DIM_LABELS[d]} {DIM_UNITS[d]}")
        ax.set_xlabel("Horizon"); ax.set_xlim(1, SEQ_LEN); ax.grid(alpha=0.3, which="both")
        if d == 0:
            ax.set_ylabel("MSE (median, standardized)"); ax.legend(fontsize=8)
    plt.suptitle(f"Per-dim state-MSE — LSTM vs SINDy | {label}", y=1.02)
    plt.tight_layout()
    p = os.path.join(save_dir, f"vs_perdim_{tag}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("saved:", p)


def plot_paired(err, tag, label, save_dir, rng):
    """Paired Δ (LSTM − SINDy): >0 ⇒ SINDy καλύτερο σε αυτόν τον ορίζοντα."""
    diff = err["LSTM"].mean(axis=2) - err["SINDy"].mean(axis=2)
    med, lo, hi = bootstrap_paired(diff, N_BOOT, rng)
    horizons = np.arange(1, SEQ_LEN + 1)
    plt.figure(figsize=(7.5, 4.8))
    plt.axhline(0, color="k", lw=1)
    plt.plot(horizons, med, color=C_SINDY, lw=2, label="LSTM − SINDy")
    plt.fill_between(horizons, lo, hi, color=C_SINDY, alpha=0.18)
    plt.title(f"Paired Δ (>0 ⇒ SINDy better) | {label}")
    plt.xlabel("Prediction Horizon"); plt.ylabel("Δ state-MSE (LSTM − SINDy)")
    plt.xlim(1, SEQ_LEN); plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout()
    p = os.path.join(save_dir, f"vs_paired_{tag}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print("saved:", p)
    return med, lo, hi


def plot_trajectory(raw, tag, label, save_dir, rng):
    gt = raw["gt"]; N, L, _ = gt.shape
    horizons = np.arange(1, L + 1)
    for _ in range(N_TRAJ_WINDOWS):
        w = TRAJ_WINDOW if TRAJ_WINDOW is not None else int(rng.integers(0, N))
        fig, axes = plt.subplots(2, 2, figsize=(12, 6))
        for d in range(N_SUP):
            ax = axes[d // 2][d % 2]
            ax.plot(horizons, gt[w, :, d], color="k", lw=2.0, label="GT")
            ax.plot(horizons, raw["lstm"][w, :, d], color=C_LSTM, lw=1.5, ls="--", label="LSTM")
            ax.plot(horizons, raw["sindy"][w, :, d], color=C_SINDY, lw=1.5, ls="--", label="SINDy")
            ax.set_title(f"{DIM_LABELS[d]} {DIM_UNITS[d]}")
            ax.set_xlabel("Prediction Horizon"); ax.set_xlim(1, L); ax.grid(alpha=0.3)
            if d == 0:
                ax.legend(fontsize=8)
        plt.suptitle(f"Trajectory — window #{w} | {label} (physical units)")
        plt.tight_layout()
        p = os.path.join(save_dir, f"vs_traj_{tag}_w{w}.png")
        plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
        print("saved:", p)


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
    traj_rng = np.random.default_rng(TRAJ_SEED)

    # ---- fit SINDy στο CLEAN encoded train (+ GT ref) ----
    train_dir = ensure_encoded(VAE_CKPT, DATA_ROOT,
                                   os.path.join(LATENT_ROOT, "clean"),
                                   device, noise_fn=None, splits=("train",))["train"]
    Xe, Fe, Xne = assemble_fit_data(train_dir, "encoded", mean, std)
    Xi_e, names = fit_sindy(Xe, Fe, Xne, mode=FEATURE_MODE, threshold=THRESHOLD, ridge=RIDGE)
    Xg, Fg, Xng = assemble_fit_data(train_dir, "gt", mean, std)
    Xi_g, _ = fit_sindy(Xg, Fg, Xng, mode=FEATURE_MODE, threshold=THRESHOLD, ridge=RIDGE)
    print("\n[SINDy enc-fit] discovered dynamics:")
    for line in format_equations(Xi_e, names):
        print("   " + line)

    lstm = load_lstm(LSTM_CKPT, device)

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
        err, raw = evaluate_condition(lstm, test_dir, Xi_e, Xi_g, mean, std, device)

        plot_overall(err, tag, label, SAVE_DIR)
        plot_perdim(err, tag, label, SAVE_DIR)
        med_d, lo_d, hi_d = plot_paired(err, tag, label, SAVE_DIR, rng)
        plot_trajectory(raw, tag, label, SAVE_DIR, traj_rng)

        print(f"\n--- median standardized state-MSE | {label} ---")
        for name, e in err.items():
            m = np.median(e.mean(axis=2), axis=0)
            print(f"  {name:<16} " + "  ".join(f"h{h}={m[h-1]:.5f}" for h in HS))
        print("  Δ(LSTM−SINDy)    " +
              "  ".join(f"h{h}={med_d[h-1]:+.5f}[{lo_d[h-1]:+.5f},{hi_d[h-1]:+.5f}]" for h in HS))

        for name, e in err.items():
            save_curves[f"{tag}__{name}__median"] = np.median(e.mean(axis=2), axis=0)
        save_curves[f"{tag}__paired_median"] = med_d
        save_curves[f"{tag}__paired_lo"] = lo_d
        save_curves[f"{tag}__paired_hi"] = hi_d

    np.savez(os.path.join(SAVE_DIR, "sindy_vs_lstm_curves.npz"), **save_curves)
    print("\nsaved figures + sindy_vs_lstm_curves.npz ->", SAVE_DIR)


if __name__ == "__main__":
    main()
