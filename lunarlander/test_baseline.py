"""
test_baseline_lunarlander.py — Αυτόνομη αξιολόγηση του Baseline world model (LunarLander).

Port του cart_pole/test_baseline_cartpole.py· state 4D -> 8D. ΧΡΗΣΙΜΟΠΟΙΕΙ IMPORTS από τα
canonical modules του φακέλου lunarlander/ (όπως το cart_pole importάρει vae/loader):
    from vae import VAE
    from loader import VaePairDataset, load_norm_stats, list_npz
-> καμία επανα-ορισμένη κλάση εδώ· τρέχει στο Kaggle αφού git-clone-άρεις το repo (cwd: lunarlander/).

MODULE 1 — «Physical encoding vs GT» (μόνο ο encoder, χωρίς LSTM):
  (1) Per-dim μετρικές: RMSE & MAE (φυσικές μονάδες), R², Pearson r  -> πίνακας + bar chart
  (2) Scatter pred-vs-GT ανά dim (+ γραμμή ταυτότητας, R²)
  (3) Per-dim ιστόγραμμα σφάλματος (φυσικές μονάδες) -> bias/spread ανά μέγεθος.
  (4) Time-series overlay GT vs encoded mu[:8] σε δείγμα επεισοδίων -> tracking + θόρυβος/υστέρηση.

ΠΩΣ ΔΙΑΒΑΖΕΤΑΙ:
  * Υψηλό R² (≈1) & χαμηλό RMSE σε x,y,theta,leg1,leg2 = καθαρή κωδικοποίηση ΣΤΑΤΙΚΗΣ κατάστασης.
  * Χαμηλότερο R² / μεγαλύτερο RMSE στις vx,vy,omega = αδυναμία κωδικοποίησης ΤΑΧΥΤΗΤΩΝ
    -> κίνητρο για τις αρχές (π.χ. P3 weak supervision).
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from vae import VAE
from loader import VaePairDataset, load_norm_stats, list_npz

# ---------------------------------------------------------------------------
# CONFIG — placeholders <...> τα συμπληρώνει το bootstrap patcher (CONFIG_PATHS)
# ---------------------------------------------------------------------------
DATA_ROOT = "<lunarlander-dataset>"
TEST_DIR = os.path.join(DATA_ROOT, "test")
NORM_STATS = os.path.join(DATA_ROOT, "norm_stats.npz")
VAE_CKPT = "<lunarlander-baseline-vae>"        # μόνο VAE — το LSTM ΔΕΝ χρειάζεται (encoder-only test)
SAVE_DIR = "/kaggle/working/lunar_baseline_test"

LATENT_SIZE = 64
N_SUP = 8
SHIFT = 0                 # 0=clean GT comparison (το σωστό για interpretability check)
BATCH = 128
NUM_WORKERS = 2
SEED = 0

DIM_NAMES = ["x", "y", "vx", "vy", "theta", "omega", "leg1", "leg2"]
DIM_LABELS = ["x", "y", r"$v_x$", r"$v_y$", r"$\theta$", r"$\omega$", "leg1", "leg2"]
DIM_UNITS = ["(pos)", "(pos)", "(vel)", "(vel)", "[rad]", "[rad/s]", "(contact)", "(contact)"]

N_EPISODE_PLOTS = 3       # πόσα επεισόδια για το time-series overlay
SCATTER_MAX_PTS = 8000    # subsample για ελαφρύ scatter
SEED_SCATTER = 0

# "standardized" (mean0/std1, κοινή κλίμακα) [ΠΡΟΕΠΙΛΟΓΗ] ή "physical" (de-standardized).
UNITS = "standardized"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_img(t, device):
    """uint8 ή float (B,3,H,W) -> float [0,1] στη συσκευή (robust)."""
    t = t.to(device, non_blocking=True)
    return t.float().div_(255.0) if t.dtype == torch.uint8 else t.float()


def _unit_suffix():
    return "(standardized)" if UNITS == "standardized" else "(physical units)"


def _to_units(mu_std, gt_std, mean, std):
    if UNITS == "standardized":
        return mu_std, gt_std
    mean8 = np.asarray(mean[:N_SUP], np.float64)
    std8 = np.asarray(std[:N_SUP], np.float64)
    return mu_std * std8 + mean8, gt_std * std8 + mean8


# ---------------------------------------------------------------------------
# (collect) Κωδικοποίηση όλου του test set: mu[:8] (std) vs GT state (std)
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_encoding(model, loader, device):
    model.eval()
    mus, gts = [], []
    for img_t, img_tp1, action, state_t, state_tp1 in loader:
        x = torch.cat([_to_img(img_t, device), _to_img(img_tp1, device)], dim=1)
        mu, _ = model.encode(x)
        mus.append(mu[:, :N_SUP].cpu().numpy())
        gts.append(state_t.numpy())                 # ήδη standardized από τον loader
    return np.concatenate(mus, 0), np.concatenate(gts, 0)


# ---------------------------------------------------------------------------
# (metrics) Per-dim: RMSE/MAE φυσικές μονάδες, R², Pearson r
# ---------------------------------------------------------------------------
def physical_metrics(mu_std, gt_std, mean, std):
    std8 = np.asarray(std[:N_SUP], np.float64)
    err = (mu_std - gt_std).astype(np.float64)
    rmse_std = np.sqrt((err ** 2).mean(0))
    mae_std = np.abs(err).mean(0)
    rmse_phys = rmse_std * std8
    mae_phys = mae_std * std8
    ss_res = (err ** 2).sum(0)
    ss_tot = ((gt_std - gt_std.mean(0)) ** 2).sum(0) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    pearson = np.array([np.corrcoef(mu_std[:, d], gt_std[:, d])[0, 1] for d in range(N_SUP)])
    return {"rmse_std": rmse_std, "mae_std": mae_std,
            "rmse_phys": rmse_phys, "mae_phys": mae_phys, "r2": r2, "pearson": pearson}


def print_metrics_table(m):
    print("\n" + "=" * 84)
    print("MODULE 1 — Physical encoding vs GT  (baseline VAE encoder, test set)")
    print("=" * 84)
    print(f"{'dim':<11}{'RMSE(std)':>12}{'MAE(std)':>12}{'RMSE(phys)':>13}{'R^2':>10}{'Pearson':>11}")
    print("-" * 84)
    for d in range(N_SUP):
        print(f"{DIM_NAMES[d]:<11}{m['rmse_std'][d]:>12.4f}{m['mae_std'][d]:>12.4f}"
              f"{m['rmse_phys'][d]:>13.5f}{m['r2'][d]:>10.4f}{m['pearson'][d]:>11.4f}")
    print("-" * 84)
    print(f"{'MEAN':<11}{m['rmse_std'].mean():>12.4f}{m['mae_std'].mean():>12.4f}"
          f"{m['rmse_phys'].mean():>13.5f}{m['r2'].mean():>10.4f}{m['pearson'].mean():>11.4f}")
    print("=" * 84)


# ---------------------------------------------------------------------------
# (plots)
# ---------------------------------------------------------------------------
def plot_bar(m, save_dir):
    rmse = m["rmse_std"] if UNITS == "standardized" else m["rmse_phys"]
    ylab = "RMSE (standardized)" if UNITS == "standardized" else "RMSE (physical units)"
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    xs = np.arange(N_SUP)
    axes[0].bar(xs, rmse, color="C0")
    axes[0].set_xticks(xs); axes[0].set_xticklabels(DIM_NAMES, rotation=30, ha="right")
    axes[0].set_ylabel(ylab); axes[0].set_title(f"Per-dim RMSE (↓)  {_unit_suffix()}")
    axes[0].grid(alpha=0.3, axis="y")
    axes[1].bar(xs, m["r2"], color="C1")
    axes[1].set_xticks(xs); axes[1].set_xticklabels(DIM_NAMES, rotation=30, ha="right")
    axes[1].set_ylim(min(0.0, float(m["r2"].min()) - 0.05), 1.0)
    axes[1].set_ylabel("R²"); axes[1].set_title("Per-dim R² (↑)")
    axes[1].axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.5)
    axes[1].grid(alpha=0.3, axis="y")
    plt.suptitle("Baseline encoder — physical fidelity per dimension")
    plt.tight_layout()
    p = os.path.join(save_dir, "enc_perdim_bars.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); print("saved:", p); plt.show()


def plot_scatter(mu_std, gt_std, m, mean, std, save_dir):
    mu_plot, gt_plot = _to_units(mu_std, gt_std, mean, std)
    rmse_key = "rmse_std" if UNITS == "standardized" else "rmse_phys"
    rng = np.random.default_rng(SEED_SCATTER)
    n = mu_plot.shape[0]
    idx = rng.choice(n, size=min(SCATTER_MAX_PTS, n), replace=False)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), squeeze=False)
    for d in range(N_SUP):
        ax = axes[d // 4][d % 4]
        gx, py = gt_plot[idx, d], mu_plot[idx, d]
        ax.scatter(gx, py, s=4, alpha=0.25, color="C0", edgecolors="none")
        lo = float(min(gx.min(), py.min())); hi = float(max(gx.max(), py.max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="ideal (y=x)")
        ax.set_title(f"{DIM_LABELS[d]} {DIM_UNITS[d]}\nR²={m['r2'][d]:.3f}  RMSE={m[rmse_key][d]:.3f}")
        ax.set_xlabel(f"GT {_unit_suffix()}"); ax.set_aspect("equal", adjustable="datalim")
        if d % 4 == 0:
            ax.set_ylabel("encoded mu[:8]")
        if d == 0:
            ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    plt.suptitle(f"Baseline encoder — predicted vs ground-truth physical state {_unit_suffix()}", y=1.01)
    plt.tight_layout()
    p = os.path.join(save_dir, "enc_scatter_vs_gt.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); print("saved:", p); plt.show()


def plot_error_hist(mu_std, gt_std, mean, std, save_dir):
    mu_plot, gt_plot = _to_units(mu_std, gt_std, mean, std)
    err = mu_plot - gt_plot
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), squeeze=False)
    for d in range(N_SUP):
        ax = axes[d // 4][d % 4]
        ax.hist(err[:, d], bins=60, color="C0", alpha=0.8)
        bias = float(err[:, d].mean()); sd = float(err[:, d].std())
        ax.axvline(0, color="k", lw=1)
        ax.axvline(bias, color="C3", lw=1.2, ls="--", label=f"bias={bias:+.4f}\nstd={sd:.4f}")
        ax.set_title(f"{DIM_LABELS[d]} {DIM_UNITS[d]}")
        ax.set_xlabel(f"error (encoded − GT) {_unit_suffix()}"); ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        if d % 4 == 0:
            ax.set_ylabel("count")
    plt.suptitle(f"Baseline encoder — error distribution per dimension {_unit_suffix()}", y=1.01)
    plt.tight_layout()
    p = os.path.join(save_dir, "enc_error_hist.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); print("saved:", p); plt.show()


# ---------------------------------------------------------------------------
# (episode) Κωδικοποίηση ΟΛΟΚΛΗΡΟΥ επεισοδίου -> time-series overlay
# ---------------------------------------------------------------------------
@torch.no_grad()
def encode_episode(model, npz_path, device, batch=256):
    model.eval()
    with np.load(npz_path) as d:
        imgs = d["imgs"].astype(np.float32) / 255.0
        states = d["states"].astype(np.float32)
    imgs_t = torch.from_numpy(imgs).permute(0, 3, 1, 2)
    img_t, img_tp1 = imgs_t[:-1], imgs_t[1:]
    mus = []
    for b in range(0, img_t.shape[0], batch):
        x = torch.cat([img_t[b:b + batch], img_tp1[b:b + batch]], dim=1).to(device)
        mu, _ = model.encode(x)
        mus.append(mu[:, :N_SUP].cpu().numpy())
    mu_std = np.concatenate(mus, 0) if mus else np.zeros((0, N_SUP), np.float32)
    return mu_std, states[:-1]


def plot_timeseries(model, mean, std, device, save_dir, n_eps=N_EPISODE_PLOTS):
    mean8 = np.asarray(mean[:N_SUP], np.float64)
    std8 = np.asarray(std[:N_SUP], np.float64)
    files = list_npz(TEST_DIR)
    if not files:
        print("[warn] no test episodes found for time-series.")
        return
    lengths = []
    for f in files:
        with np.load(f) as d:
            lengths.append(d["states"].shape[0])
    pick = [files[i] for i in np.argsort(lengths)[::-1][:n_eps]]

    for ei, ep in enumerate(pick):
        mu_std, gt_raw = encode_episode(model, ep, device)
        if mu_std.shape[0] == 0:
            continue
        if UNITS == "standardized":
            mu_plot = mu_std
            gt_plot = (gt_raw - mean8) / std8
        else:
            mu_plot = mu_std * std8 + mean8
            gt_plot = gt_raw
        t = np.arange(mu_std.shape[0])
        fig, axes = plt.subplots(2, 4, figsize=(18, 7))
        for d in range(N_SUP):
            ax = axes[d // 4][d % 4]
            ax.plot(t, gt_plot[:, d], color="k", lw=1.6, label="GT")
            ax.plot(t, mu_plot[:, d], color="C0", lw=1.4, ls="--", label="encoded")
            ax.set_title(f"{DIM_LABELS[d]} {DIM_UNITS[d]}")
            ax.set_xlabel("t (step)"); ax.grid(alpha=0.3)
            if d == 0:
                ax.legend(fontsize=9)
        plt.suptitle(f"Baseline encoder — episode {os.path.basename(ep)} "
                     f"(len={mu_std.shape[0]}) {_unit_suffix()}")
        plt.tight_layout()
        p = os.path.join(save_dir, f"enc_timeseries_ep{ei}.png")
        plt.savefig(p, dpi=150, bbox_inches="tight"); print("saved:", p); plt.show()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    np.random.seed(SEED); torch.manual_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = get_device()
    print("device:", device)

    mean, std = load_norm_stats(NORM_STATS)
    model = VAE(latent_size=LATENT_SIZE).to(device)
    model.load_state_dict(torch.load(VAE_CKPT, map_location=device))
    model.eval()

    pin = device.type == "cuda"
    nw = NUM_WORKERS if device.type == "cuda" else 0
    test_ds = VaePairDataset(TEST_DIR, shift=SHIFT, state_mean=mean, state_std=std)
    test_dl = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=nw, pin_memory=pin)
    print(f"test pairs: {len(test_ds)}")

    # ---- (collect) ----
    mu_std, gt_std = collect_encoding(model, test_dl, device)

    # ---- (1) metrics ----
    m = physical_metrics(mu_std, gt_std, mean, std)
    print_metrics_table(m)

    # ---- (2) plots: bars, scatter, error histograms ----
    plot_bar(m, SAVE_DIR)
    plot_scatter(mu_std, gt_std, m, mean, std, SAVE_DIR)
    plot_error_hist(mu_std, gt_std, mean, std, SAVE_DIR)

    # ---- (3) time-series overlay ----
    plot_timeseries(model, mean, std, device, SAVE_DIR)

    # ---- save metrics ----
    np.savez(os.path.join(SAVE_DIR, "baseline_encoding_metrics.npz"),
             dim_names=np.array(DIM_NAMES), **m)
    print("\nsaved metrics + figures ->", SAVE_DIR)


if __name__ == "__main__":
    main()
