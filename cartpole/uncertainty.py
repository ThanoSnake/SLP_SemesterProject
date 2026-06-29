"""
uncertainty.py — Interpretable uncertainty for the CartPole world model via MC Dropout.

Future Direction C (paper §5): "uncertainty estimation within physically meaningful latent
representations allows for more interpretable and actionable uncertainties." Because the
supervised latent dims (z[:, :N_SUP]) ARE the physical state [x, x_dot, theta, theta_dot],
an uncertainty estimate THERE is already in domain units (rad, rad/s, ...).

Two parts (each toggleable; neither touches the baseline vae.py / lstm.py):

  [LSTM]  Dynamics uncertainty.
    Trains/loads a SEPARATE dropout LSTM (same recipe as lstm.py + dropout). MC Dropout =
    T stochastic rollouts from the same seed/actions -> predictive mean +/- band over the
    future physical state. Optional aleatoric: sample seed z0 ~ N(mu, sigma) from the VAE.
    Compared against the deterministic baseline LSTM.

  [VAE]   Perception uncertainty.
    Trains/loads a SEPARATE dropout VAE (same recipe as vae.py + dropout). MC Dropout =
    T stochastic encodings of the SAME frame -> EPISTEMIC spread on the physical-state
    estimate. The VAE's own logvar gives ALEATORIC uncertainty (free). Their decomposition
    is the headline: epistemic spikes on OOD/noisy frames (encoder hasn't seen them) while
    aleatoric stays flat -> an actionable "don't trust this perception" signal. Compared
    against the baseline VAE (which has only aleatoric). Also renders a per-pixel
    reconstruction-uncertainty heatmap.

Kaggle: placeholders <...> are patched by kaggle-run.ipynb. Set TRAIN_DROPOUT_* to train+save
fresh dropout models, or False to load from <cartpole-dropout-lstm> / <cartpole-dropout-vae>.
Run:  !python3 cartpole/uncertainty.py
"""
import os
from os import makedirs
from os.path import join, basename

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from loader import list_npz, precompute_latents, LatentSequenceDataset, VaePairDataset, load_norm_stats
from vae import VAE, encode_fn
from lstm import LatentPredictor          # baseline (no dropout) — imported, NOT modified

# ---------------------------------------------------------------------------
# CONFIG  (placeholders <...> patched by kaggle-run.ipynb)
# ---------------------------------------------------------------------------
DATA_ROOT = "<cartpole-dataset>"
NORM_STATS = os.path.join(DATA_ROOT, "norm_stats.npz")
VAE_CKPT = "<cartpole-baseline-vae>"
BASELINE_LSTM_CKPT = "<cartpole-baseline-lstm>"
DROPOUT_LSTM_CKPT = "<dropout-lstm>"      # NEW: add to kaggle-run.ipynb CONFIG_PATHS
DROPOUT_VAE_CKPT = "<dropout-vae>"        # NEW: add to kaggle-run.ipynb CONFIG_PATHS

LATENT_ROOT = "/kaggle/working/cartpole_unc_latents"
SAVE_DIR = "/kaggle/working/cartpole_uncertainty"

LATENT_SIZE, N_SUP, N_IMG = 64, 4, 60
N_ACTIONS, HIDDEN, LAYERS = 2, 64, 2
SHIFT = 0
SEQ_LEN, TEST_STRIDE, BATCH = 30, 1, 128

# Which experiments to run
RUN_LSTM_UNCERTAINTY = True
RUN_VAE_UNCERTAINTY = True

# --- dropout LSTM training (mirrors lstm.py exactly, except P_DROP) ---
TRAIN_DROPOUT_LSTM = True   # True -> train a fresh dropout LSTM (+save); False -> load DROPOUT_LSTM_CKPT
P_DROP = 0.1                # dropout prob (training-time AND MC at test). 0 -> no uncertainty.
TRAIN_STRIDE, TRAIN_BATCH = 5, 64
EPOCHS, LR, CLIP, W_PHYS = 50, 1e-3, 1.0, 1.0
P_START, P_END, P_DECAY_EPOCHS = 1.0, 0.3, 40       # scheduled sampling
L_START, CURRICULUM_EPOCHS = 5, 15                  # horizon curriculum
EARLY_STOP_PATIENCE, SCHED_PATIENCE = 6, 3
NUM_WORKERS, SEED = 2, 0

# --- dropout VAE training (mirrors vae.py exactly, except P_DROP_VAE) ---
TRAIN_DROPOUT_VAE = True    # True -> train a fresh dropout VAE (+save); False -> load DROPOUT_VAE_CKPT
P_DROP_VAE = 0.1
VAE_EPOCHS, VAE_BATCH, VAE_LR = 40, 128, 1e-3
BETA_PHYS, BETA_STYLE_MAX, KL_ANNEAL_EPOCHS = 0.01, 1.0, 20
LAMBDA_SUP = 1.0
VAE_EARLY_STOP_PATIENCE = 5

# --- MC dropout / uncertainty ---
T_MC = 30                   # stochastic forward passes
USE_VAE_SAMPLING = True     # LSTM: also sample seed z0 ~ N(mu, sigma) from the VAE (aleatoric)

# --- visual-noise sweep on TEST images before encoding (mirrors test_p1.py) ---
NOISE_TYPE = "gaussian"                  # "gaussian" | "salt_pepper"
NOISE_LEVELS = [0.0, 0.10, 0.30]         # 0.0 = clean. MC eval is T x per level -> keep short.
NOISE_SEED = 42

# --- chosen-window / chosen-frame visualization ---
CHOSEN_WINDOW = None        # None -> random (with WINDOW_SEED); int -> fixed index
WINDOW_SEED = 0

DIM_LABELS = ["x", r"$\dot{x}$", r"$\theta$", r"$\dot{\theta}$"]
DIM_UNITS = ["(cart pos)", "(cart vel)", "[rad]", "[rad/s]"]
# Gaussian quantiles |N(0,1)| for nominal central intervals (no scipy dependency)
Z = {0.50: 0.6745, 0.80: 1.2816, 0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}
LOG_Y = True


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def noise_tag(ntype, level):
    return f"{ntype}_{level:.2f}".replace(".", "p")


def enable_dropout(model):
    """Keep ONLY dropout layers in train mode (everything else eval) -> MC dropout."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


# ---------------------------------------------------------------------------
# Shared noise injection (mirrors test_p1.py)
# ---------------------------------------------------------------------------
def add_gaussian_noise(img, std, gen):
    return torch.clamp(img + torch.randn(img.shape, generator=gen, device=img.device) * std, 0.0, 1.0)


def add_salt_pepper_noise(img, amount, gen):
    mask = torch.rand(img.shape, generator=gen, device=img.device)
    out = img.clone()
    out[mask < amount / 2] = 0.0
    out[mask > 1 - amount / 2] = 1.0
    return out


def make_noise_fn(ntype, level, seed, device):
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    if level == 0.0:
        return lambda x: x
    if ntype == "gaussian":
        return lambda x: add_gaussian_noise(x, level, gen)
    if ntype == "salt_pepper":
        return lambda x: add_salt_pepper_noise(x, level, gen)
    raise ValueError(f"Unknown noise type: {ntype}")


# ---------------------------------------------------------------------------
# Shared calibration helpers
# ---------------------------------------------------------------------------
def coverage(gt, mean, std, z, eps=1e-8):
    """Fraction of points with |gt - mean| <= z * std (overall scalar). Shape-agnostic."""
    return float((np.abs(gt - mean) <= z * (std + eps)).mean())


def recal_scalar(gt, mean, std, level=0.95, eps=1e-8):
    """Single multiplicative std-correction so the `level` interval is calibrated.
    s = empirical(level-quantile of |gt-mean|/std) / Z[level]. Fit on CLEAN, apply everywhere."""
    r = (np.abs(gt - mean) / (std + eps)).ravel()
    return float(np.percentile(r, 100 * level) / Z[level])


def plot_calibration(gt, mean, std, tag, save_dir, s_cal=1.0, prefix="unc"):
    levels = sorted(Z.keys())
    emp = [coverage(gt, mean, std, Z[q]) for q in levels]
    emp_cal = [coverage(gt, mean, s_cal * std, Z[q]) for q in levels]
    plt.figure(figsize=(5.6, 5.4))
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
    plt.plot(levels, emp, "o-", color="C3", label="MC dropout")
    plt.plot(levels, emp_cal, "s--", color="C2", label=f"recalibrated (s={s_cal:.2f})")
    plt.title(f"Calibration | {tag}")
    plt.xlabel("Nominal coverage")
    plt.ylabel("Empirical coverage")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    p = os.path.join(save_dir, f"{prefix}_calibration_{tag}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print("saved:", p)


# ===========================================================================
#  PART 1 — DYNAMICS UNCERTAINTY (dropout LSTM)
# ===========================================================================
class LatentPredictorMC(nn.Module):
    """Baseline architecture + ONE dropout layer before the residual head. MC variance comes
    from this nn.Dropout (re-enabled at test). NOTE: per-timestep dropout; proper variational
    recurrent dropout would fix one mask per rollout — a refinement, not done here."""
    def __init__(self, latent=64, action_dim=2, hidden=64, layers=2, p_drop=0.1):
        super().__init__()
        self.hidden, self.layers = hidden, layers
        self.lstm = nn.LSTM(latent + action_dim, hidden, layers, batch_first=True)
        self.drop = nn.Dropout(p_drop)
        self.fc = nn.Linear(hidden, latent)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def init_hidden(self, b, device):
        return (torch.zeros(self.layers, b, self.hidden, device=device),
                torch.zeros(self.layers, b, self.hidden, device=device))

    def step(self, z, a_onehot, hidden):
        inp = torch.cat([z, a_onehot], dim=-1).unsqueeze(1)
        out, hidden = self.lstm(inp, hidden)
        return z + self.fc(self.drop(out.squeeze(1))), hidden


def _train_rollout(model, batch, p_tf, free_running=False, max_len=None):
    z_t, action, z_tp1, state_t, state_tp1 = batch
    L = z_t.shape[1] if max_len is None else min(max_len, z_t.shape[1])
    B = z_t.shape[0]
    device = z_t.device
    z_gt = z_tp1[:, :L]
    hidden = model.init_hidden(B, device)
    z_in = z_t[:, 0]
    preds = []
    for k in range(L):
        a = F.one_hot(action[:, k].long(), N_ACTIONS).float()
        z_pred, hidden = model.step(z_in, a, hidden)
        preds.append(z_pred)
        if k < L - 1:
            if free_running:
                z_in = z_pred.detach()
            else:
                use_tf = (torch.rand(B, 1, device=device) < p_tf).float()
                z_in = use_tf * z_gt[:, k] + (1.0 - use_tf) * z_pred.detach()
    return torch.stack(preds, dim=1), z_gt, state_tp1


def _train_epoch_lstm(model, loader, optimizer, device, p_tf, cur_len):
    model.train()
    tot, n = 0.0, 0
    for batch in loader:
        batch = [b.to(device, non_blocking=True) for b in batch]
        preds, z_gt, _ = _train_rollout(model, batch, p_tf, free_running=False, max_len=cur_len)
        loss = (F.mse_loss(preds, z_gt, reduction="mean")
                + W_PHYS * F.mse_loss(preds[..., :N_SUP], z_gt[..., :N_SUP], reduction="mean"))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()
        bs = preds.size(0)
        tot += loss.item() * bs
        n += bs
    return tot / max(n, 1)


@torch.no_grad()
def _eval_epoch_lstm(model, loader, device, std4):
    model.eval()
    se, n = None, 0
    for batch in loader:
        batch = [b.to(device, non_blocking=True) for b in batch]
        preds, _, state_tp1 = _train_rollout(model, batch, 0.0, free_running=True, max_len=None)
        err = (preds[..., :N_SUP] - state_tp1) * std4
        s = (err ** 2).sum(dim=0)
        se = s if se is None else se + s
        n += preds.size(0)
    return (se / n).mean(dim=1).cpu().numpy()


def train_dropout_lstm(device, mean, std, std4):
    vae = VAE(latent_size=LATENT_SIZE).to(device)
    vae.load_state_dict(torch.load(VAE_CKPT, map_location=device))
    vae.eval()
    enc = encode_fn(vae, device)
    for split in ("train", "val"):
        src = os.path.join(DATA_ROOT, split)
        if os.path.isdir(src):
            print(f"[dropout-lstm] pre-encoding {split} (clean) ...")
            precompute_latents(enc, src, os.path.join(LATENT_ROOT, "clean", split), shift=SHIFT, device=device)
    del vae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    pw = NUM_WORKERS > 0
    train_ds = LatentSequenceDataset(os.path.join(LATENT_ROOT, "clean", "train"),
                                     seq_len=SEQ_LEN, stride=TRAIN_STRIDE, state_mean=mean, state_std=std)
    val_ds = LatentSequenceDataset(os.path.join(LATENT_ROOT, "clean", "val"),
                                   seq_len=SEQ_LEN, stride=TRAIN_STRIDE, state_mean=mean, state_std=std)
    train_dl = DataLoader(train_ds, batch_size=TRAIN_BATCH, shuffle=True, drop_last=True,
                          num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=pw)
    val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=pw)
    print(f"[dropout-lstm] train windows: {len(train_ds)} | val windows: {len(val_ds)} | p_drop={P_DROP}")

    model = LatentPredictorMC(LATENT_SIZE, N_ACTIONS, HIDDEN, LAYERS, p_drop=P_DROP).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=SCHED_PATIENCE)

    out_path = os.path.join(SAVE_DIR, "lstm_dropout_best.pth")
    best, bad = float("inf"), 0
    for epoch in range(1, EPOCHS + 1):
        p_tf = max(P_END, P_START - (P_START - P_END) * (epoch - 1) / max(P_DECAY_EPOCHS, 1))
        cur_len = int(round(min(SEQ_LEN, L_START + (SEQ_LEN - L_START) * (epoch - 1) / max(CURRICULUM_EPOCHS, 1))))
        tr = _train_epoch_lstm(model, train_dl, optimizer, device, p_tf, cur_len)
        mse_h = _eval_epoch_lstm(model, val_dl, device, std4)
        val_mean = float(mse_h.mean())
        scheduler.step(val_mean)
        print(f"E{epoch:03d} | p_tf={p_tf:.2f} H={cur_len} lr={optimizer.param_groups[0]['lr']:.1e} | "
              f"train={tr:.5f} | val phys-MSE={val_mean:.4f}")
        if val_mean < best - 1e-6:
            best, bad = val_mean, 0
            torch.save(model.state_dict(), out_path)
            print("  -> best dropout-lstm saved")
        else:
            bad += 1
            if bad >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch}.")
                break
    model.load_state_dict(torch.load(out_path, map_location=device))
    print(f"[dropout-lstm] best val phys-MSE: {best:.4f} | saved -> {out_path}")
    return model


@torch.no_grad()
def precompute_latents_mc(vae, root, out_root, noise_fn, device, batch=256):
    """Like loader.precompute_latents but ALSO caches logvar (for aleatoric) and applies
    image noise BEFORE encoding."""
    makedirs(out_root, exist_ok=True)
    vae.eval()
    for f in list_npz(root):
        with np.load(f) as d:
            imgs = torch.from_numpy(d["imgs"].astype(np.float32) / 255.0).permute(0, 3, 1, 2)
            acts = d["acts"].astype(np.float32)
            states = d["states"].astype(np.float32)
            x = (d[f"noisy_states_{SHIFT}"] if SHIFT in (2, 5, 10) else d["states"]).astype(np.float32)
        imgs = noise_fn(imgs.to(device)) if noise_fn is not None else imgs.to(device)
        img_t, img_tp1 = imgs[:-1], imgs[1:]
        mus, lvs = [], []
        for b in range(0, img_t.shape[0], batch):
            xb = torch.cat([img_t[b:b + batch], img_tp1[b:b + batch]], dim=1)
            mu, lv = vae.encode(xb)
            mus.append(mu.cpu().numpy())
            lvs.append(lv.cpu().numpy())
        z = np.concatenate(mus, 0).astype(np.float32) if mus else np.empty((0, 0), np.float32)
        zlv = np.concatenate(lvs, 0).astype(np.float32) if lvs else np.empty((0, 0), np.float32)
        np.savez_compressed(join(out_root, basename(f)),
                            z=z, zlogvar=zlv, acts=acts[:-1], states=states[:-1], x=x[:-1])


class UncLatentSeq(Dataset):
    """Like LatentSequenceDataset, but also returns the SEED logvar (for aleatoric sampling)."""
    def __init__(self, root, seq_len, stride, mean, std):
        self.seq_len = seq_len
        self.mean = np.asarray(mean, np.float32)
        self.std = np.asarray(std, np.float32)
        self.eps, self.index = [], []
        for fi, f in enumerate(list_npz(root)):
            with np.load(f) as d:
                ep = {k: d[k].astype(np.float32) for k in ("z", "zlogvar", "acts", "states", "x")}
            self.eps.append(ep)
            n = ep["z"].shape[0] - (seq_len + 1) + 1
            for s in range(0, max(n, 0), stride):
                self.index.append((fi, s))
        if not self.index:
            raise RuntimeError(f"No windows from {root} (seq_len={seq_len} too large?)")

    def _std(self, s):
        return ((s - self.mean) / self.std).astype(np.float32)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        fi, s = self.index[i]
        ep = self.eps[fi]
        L = self.seq_len
        z = ep["z"][s:s + L + 1]
        z_t = torch.from_numpy(z[:-1])
        z_tp1 = torch.from_numpy(z[1:])
        zlogvar0 = torch.from_numpy(ep["zlogvar"][s])
        action = torch.from_numpy(ep["acts"][s:s + L])
        state_t = torch.from_numpy(self._std(ep["x"][s:s + L]))
        state_tp1 = torch.from_numpy(self._std(ep["states"][s + 1:s + L + 1]))
        return z_t, action, z_tp1, state_t, state_tp1, zlogvar0


@torch.no_grad()
def _one_rollout(model, z_t, action, zlogvar0, sample_seed):
    B, L, _ = z_t.shape
    device = z_t.device
    z0 = z_t[:, 0]
    if sample_seed and zlogvar0 is not None:
        z0 = z0 + torch.randn_like(z0) * torch.exp(0.5 * zlogvar0)
    hidden = model.init_hidden(B, device)
    z_in = z0
    preds = []
    for k in range(L):
        a = F.one_hot(action[:, k].long(), N_ACTIONS).float()
        z_pred, hidden = model.step(z_in, a, hidden)
        preds.append(z_pred)
        z_in = z_pred
    return torch.stack(preds, dim=1)[..., :N_SUP]


@torch.no_grad()
def mc_collect(model, loader, device, T, sample_seed):
    model.eval()
    enable_dropout(model)
    means, stds, gts = [], [], []
    for batch in tqdm(loader, desc="MC rollout", leave=False):
        z_t, action, z_tp1, state_t, state_tp1, zlogvar0 = [b.to(device, non_blocking=True) for b in batch]
        passes = torch.stack([_one_rollout(model, z_t, action, zlogvar0, sample_seed) for _ in range(T)])
        means.append(passes.mean(0).cpu().numpy())
        stds.append(passes.std(0).cpu().numpy())
        gts.append(state_tp1[..., :N_SUP].cpu().numpy())
    return np.concatenate(means), np.concatenate(stds), np.concatenate(gts)


@torch.no_grad()
def baseline_collect(model, loader, device):
    model.eval()
    preds, gts = [], []
    for batch in loader:
        z_t, action, z_tp1, state_t, state_tp1, zlogvar0 = [b.to(device, non_blocking=True) for b in batch]
        p = _one_rollout(model, z_t, action, None, sample_seed=False)
        preds.append(p.cpu().numpy())
        gts.append(state_tp1[..., :N_SUP].cpu().numpy())
    return np.concatenate(preds), np.concatenate(gts)


def plot_band(gt, mc_mean, mc_std, base_pred, mean4, std4, w, tag, save_dir, s_cal=1.0):
    L = gt.shape[1]
    h = np.arange(1, L + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for d in range(N_SUP):
        ax = axes[d // 2][d % 2]
        g = gt[w, :, d] * std4[d] + mean4[d]
        m = mc_mean[w, :, d] * std4[d] + mean4[d]
        b = base_pred[w, :, d] * std4[d] + mean4[d]
        band = 1.96 * s_cal * mc_std[w, :, d] * std4[d]
        ax.plot(h, g, "k", lw=2.0, label="GT")
        ax.plot(h, b, "C0--", lw=1.5, label="Baseline (point)")
        ax.plot(h, m, "C3-", lw=1.6, label="MC mean")
        ax.fill_between(h, m - band, m + band, color="C3", alpha=0.20, label="MC 95% band")
        ax.set_title(f"{DIM_LABELS[d]} {DIM_UNITS[d]}")
        ax.set_xlabel("Prediction Horizon")
        ax.set_xlim(1, L)
        ax.grid(alpha=0.3)
        if d == 0:
            ax.legend(fontsize=8)
    plt.suptitle(f"Dynamics uncertainty — window #{w} | {tag} (physical units)")
    plt.tight_layout()
    p = os.path.join(save_dir, f"unc_band_window{w}_{tag}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved:", p)


def plot_std_vs_error(gt, mean, std, std4, tag, save_dir, s_cal=1.0):
    sd4 = std4[None, None, :]
    pred_unc = np.sqrt(((s_cal * std * sd4) ** 2).mean(axis=(0, 2)))
    act_err = np.sqrt((((gt - mean) * sd4) ** 2).mean(axis=(0, 2)))
    h = np.arange(1, gt.shape[1] + 1)
    plt.figure(figsize=(6.8, 4.6))
    plt.plot(h, act_err, "k-", lw=2, label="actual RMSE")
    plt.plot(h, pred_unc, "C3--", lw=2, label="predicted std (recal)")
    if LOG_Y:
        plt.yscale("log")
    plt.title(f"Predicted uncertainty vs actual error | {tag}")
    plt.xlabel("Prediction Horizon")
    plt.ylabel("physical units (RMS over dims)")
    plt.xlim(1, gt.shape[1])
    plt.grid(alpha=0.3, which="both")
    plt.legend()
    plt.tight_layout()
    p = os.path.join(save_dir, f"unc_std_vs_error_{tag}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print("saved:", p)


def plot_noise_sweep(levels, mean_std, cov95, save_dir):
    fig, ax1 = plt.subplots(figsize=(6.8, 4.6))
    ax1.plot(levels, mean_std, "C3-o", label="mean predictive std")
    ax1.set_xlabel("noise level σ")
    ax1.set_ylabel("mean predictive std (standardized)", color="C3")
    ax1.tick_params(axis="y", labelcolor="C3")
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(levels, cov95, "C2-s", label="95% coverage")
    ax2.axhline(0.95, color="C2", ls=":", lw=1)
    ax2.set_ylabel("empirical 95% coverage", color="C2")
    ax2.tick_params(axis="y", labelcolor="C2")
    ax2.set_ylim(0, 1)
    plt.title("Dynamics uncertainty under OOD noise")
    fig.tight_layout()
    p = os.path.join(save_dir, "unc_noise_sweep.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved:", p)


def run_lstm_uncertainty(device, mean, std, mean4, std4_np, std4):
    print(f"\n{'#'*64}\n#  DYNAMICS UNCERTAINTY (dropout LSTM)\n{'#'*64}")
    if TRAIN_DROPOUT_LSTM:
        mc_model = train_dropout_lstm(device, mean, std, std4)
    else:
        mc_model = LatentPredictorMC(LATENT_SIZE, N_ACTIONS, HIDDEN, LAYERS, p_drop=P_DROP).to(device)
        mc_model.load_state_dict(torch.load(DROPOUT_LSTM_CKPT, map_location=device))
        print(f"[dropout-lstm] loaded from {DROPOUT_LSTM_CKPT}")

    base_model = LatentPredictor(LATENT_SIZE, N_ACTIONS, HIDDEN, LAYERS).to(device)
    base_model.load_state_dict(torch.load(BASELINE_LSTM_CKPT, map_location=device))

    vae = VAE(latent_size=LATENT_SIZE).to(device)
    vae.load_state_dict(torch.load(VAE_CKPT, map_location=device))
    vae.eval()

    window_rng = np.random.default_rng(WINDOW_SEED)
    results, sweep = {}, {"levels": [], "mean_std": [], "cov95": []}
    for level in NOISE_LEVELS:
        tag = noise_tag(NOISE_TYPE, level)
        print(f"\n=== LSTM | NOISE {NOISE_TYPE} σ={level:.2f} ===")
        nf = make_noise_fn(NOISE_TYPE, level, NOISE_SEED, device)
        test_out = os.path.join(LATENT_ROOT, tag, "test")
        precompute_latents_mc(vae, os.path.join(DATA_ROOT, "test"), test_out, nf, device)
        ds = UncLatentSeq(test_out, SEQ_LEN, TEST_STRIDE, mean, std)
        dl = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        print(f"  test windows: {len(ds)}")
        mc_mean, mc_std, gt = mc_collect(mc_model, dl, device, T_MC, sample_seed=USE_VAE_SAMPLING)
        base_pred, _ = baseline_collect(base_model, dl, device)
        n = min(len(mc_mean), len(base_pred))
        mc_mean, mc_std, gt, base_pred = mc_mean[:n], mc_std[:n], gt[:n], base_pred[:n]
        results[tag] = dict(mc_mean=mc_mean, mc_std=mc_std, gt=gt, base_pred=base_pred, level=level)
        sweep["levels"].append(level)
        sweep["mean_std"].append(float(mc_std.mean()))
        sweep["cov95"].append(coverage(gt, mc_mean, mc_std, Z[0.95]))
        base_mse = float((((base_pred - gt) * std4_np) ** 2).mean())
        mc_mse = float((((mc_mean - gt) * std4_np) ** 2).mean())
        print(f"  phys-MSE baseline={base_mse:.4f} MC-mean={mc_mse:.4f} | "
              f"raw 95% cov={sweep['cov95'][-1]:.3f} mean-std={sweep['mean_std'][-1]:.4f}")

    clean_tag = noise_tag(NOISE_TYPE, 0.0) if 0.0 in NOISE_LEVELS else noise_tag(NOISE_TYPE, NOISE_LEVELS[0])
    R = results[clean_tag]
    s_cal = recal_scalar(R["gt"], R["mc_mean"], R["mc_std"], level=0.95)
    print(f"\n[recalibration] LSTM std-correction s={s_cal:.3f} (fit on {clean_tag})")

    w = CHOSEN_WINDOW if CHOSEN_WINDOW is not None else int(window_rng.integers(0, R["gt"].shape[0]))
    plot_band(R["gt"], R["mc_mean"], R["mc_std"], R["base_pred"], mean4, std4_np, w, clean_tag, SAVE_DIR, s_cal)
    for tag, R in results.items():
        plot_calibration(R["gt"], R["mc_mean"], R["mc_std"], tag, SAVE_DIR, s_cal, prefix="unc")
        plot_std_vs_error(R["gt"], R["mc_mean"], R["mc_std"], std4_np, tag, SAVE_DIR, s_cal)
    plot_noise_sweep(sweep["levels"], sweep["mean_std"], sweep["cov95"], SAVE_DIR)

    save = {"levels": np.array(sweep["levels"]), "mean_std": np.array(sweep["mean_std"]),
            "cov95": np.array(sweep["cov95"]), "s_cal": s_cal, "T_MC": T_MC, "p_drop": P_DROP}
    for tag, R in results.items():
        save[f"{tag}__pred_unc"] = np.sqrt(((s_cal * R["mc_std"] * std4_np[None, None, :]) ** 2).mean(axis=(0, 2)))
        save[f"{tag}__act_rmse"] = np.sqrt((((R["gt"] - R["mc_mean"]) * std4_np[None, None, :]) ** 2).mean(axis=(0, 2)))
    np.savez(os.path.join(SAVE_DIR, "unc_curves.npz"), **save)
    print("saved: unc_curves.npz")


# ===========================================================================
#  PART 2 — PERCEPTION UNCERTAINTY (dropout VAE)
# ===========================================================================
class VAE_MC(nn.Module):
    """Baseline VAE (vae.py) + dropout on the encoder feature vector and the decode features.
    encode() is stochastic under MC -> epistemic spread on mu; logvar gives aleatoric."""
    def __init__(self, latent_size=64, in_channels=6, out_channels=3, p_drop=0.1):
        super().__init__()
        self.latent_size = latent_size
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(inplace=True),
        )
        self.enc_drop = nn.Dropout(p_drop)
        self.fc_mu = nn.Linear(64 * 10 * 15, latent_size)
        self.fc_logvar = nn.Linear(64 * 10 * 15, latent_size)
        self.fc_decode = nn.Linear(latent_size, 64 * 10 * 15)
        self.dec_drop = nn.Dropout(p_drop)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, out_channels, 4, 2, 1), nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.enc_drop(self.encoder(x).flatten(1))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        h = self.dec_drop(self.fc_decode(z)).view(-1, 64, 10, 15)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_losses(recon, target, mu, logvar, state_t, n_sup):
    B, D = mu.size(0), mu.size(1)
    recon_l = F.mse_loss(recon, target, reduction="mean")
    sup = F.mse_loss(mu[:, :n_sup], state_t, reduction="mean")
    kl_per = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kld_phys = kl_per[:, :n_sup].sum() / B / n_sup
    kld_style = kl_per[:, n_sup:].sum() / B / (D - n_sup)
    return recon_l, kld_phys, kld_style, sup


def run_vae_epoch(model, loader, device, beta_style, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    tot = {"recon": 0.0, "sup": 0.0, "n": 0}
    for img_t, img_tp1, action, state_t, state_tp1 in loader:
        it = img_t.to(device, non_blocking=True).float() / 255.0
        itp = img_tp1.to(device, non_blocking=True).float() / 255.0
        x = torch.cat([it, itp], dim=1)
        st = state_t.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            recon, mu, logvar = model(x)
            r, kp, ks, s = vae_losses(recon, it, mu, logvar, st, N_SUP)
            loss = r + BETA_PHYS * kp + beta_style * ks + LAMBDA_SUP * s
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        bs = it.size(0)
        tot["recon"] += r.item() * bs
        tot["sup"] += s.item() * bs
        tot["n"] += bs
    n = tot["n"]
    return tot["recon"] / n, tot["sup"] / n


def train_dropout_vae(device, mean, std):
    pw = NUM_WORKERS > 0
    tr = VaePairDataset(os.path.join(DATA_ROOT, "train"), shift=SHIFT, state_mean=mean, state_std=std)
    va = VaePairDataset(os.path.join(DATA_ROOT, "val"), shift=SHIFT, state_mean=mean, state_std=std)
    tdl = DataLoader(tr, batch_size=VAE_BATCH, shuffle=True, drop_last=True,
                     num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=pw)
    vdl = DataLoader(va, batch_size=VAE_BATCH, shuffle=False,
                     num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=pw)
    print(f"[dropout-vae] train pairs: {len(tr)} | val pairs: {len(va)} | p_drop={P_DROP_VAE}")

    model = VAE_MC(LATENT_SIZE, p_drop=P_DROP_VAE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=VAE_LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=SCHED_PATIENCE)

    out_path = os.path.join(SAVE_DIR, "vae_dropout_best.pth")
    best, bad = float("inf"), 0
    for epoch in range(1, VAE_EPOCHS + 1):
        beta_style = BETA_STYLE_MAX * min(1.0, epoch / max(KL_ANNEAL_EPOCHS, 1))
        tr_recon, tr_sup = run_vae_epoch(model, tdl, device, beta_style, optimizer)
        va_recon, va_sup = run_vae_epoch(model, vdl, device, beta_style, optimizer=None)
        val_score = va_recon + LAMBDA_SUP * va_sup
        scheduler.step(val_score)
        print(f"E{epoch:03d} | beta_style={beta_style:.2f} | train recon={tr_recon:.5f} sup={tr_sup:.5f} "
              f"| val recon={va_recon:.5f} sup={va_sup:.5f} (select={val_score:.5f})")
        if val_score < best - 1e-6:
            best, bad = val_score, 0
            torch.save(model.state_dict(), out_path)
            print("  -> best dropout-vae saved")
        else:
            bad += 1
            if bad >= VAE_EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch}.")
                break
    model.load_state_dict(torch.load(out_path, map_location=device))
    print(f"[dropout-vae] best val score: {best:.5f} | saved -> {out_path}")
    return model


def _noisy_stack(img_t, img_tp1, device, noise_fn):
    it = img_t.to(device, non_blocking=True).float() / 255.0
    itp = img_tp1.to(device, non_blocking=True).float() / 255.0
    if noise_fn is not None:
        it, itp = noise_fn(it), noise_fn(itp)
    return torch.cat([it, itp], dim=1), it


@torch.no_grad()
def mc_encode_collect(model, loader, device, noise_fn, T):
    """T stochastic encodings per frame -> (mu_mean, epi_std, ale_std, gt), each (N, N_SUP)."""
    model.eval()
    enable_dropout(model)
    MU, EPI, ALE, GT = [], [], [], []
    for img_t, img_tp1, action, state_t, state_tp1 in loader:
        x, _ = _noisy_stack(img_t, img_tp1, device, noise_fn)
        mus, lvs = [], []
        for _ in range(T):
            mu, lv = model.encode(x)
            mus.append(mu[:, :N_SUP])
            lvs.append(lv[:, :N_SUP])
        M = torch.stack(mus)            # (T, B, 4)
        Lo = torch.stack(lvs)
        MU.append(M.mean(0).cpu().numpy())
        EPI.append(M.std(0).cpu().numpy())
        ALE.append(torch.exp(0.5 * Lo).mean(0).cpu().numpy())
        GT.append(state_t[:, :N_SUP].cpu().numpy())
    return np.concatenate(MU), np.concatenate(EPI), np.concatenate(ALE), np.concatenate(GT)


@torch.no_grad()
def baseline_encode_collect(model, loader, device, noise_fn):
    """Deterministic baseline VAE: point estimate + aleatoric (logvar) only. (mu, ale, gt)."""
    model.eval()
    MU, ALE, GT = [], [], []
    for img_t, img_tp1, action, state_t, state_tp1 in loader:
        x, _ = _noisy_stack(img_t, img_tp1, device, noise_fn)
        mu, lv = model.encode(x)
        MU.append(mu[:, :N_SUP].cpu().numpy())
        ALE.append(torch.exp(0.5 * lv[:, :N_SUP]).cpu().numpy())
        GT.append(state_t[:, :N_SUP].cpu().numpy())
    return np.concatenate(MU), np.concatenate(ALE), np.concatenate(GT)


@torch.no_grad()
def recon_uncertainty(model, x_one, T):
    """T stochastic decodes of one frame -> (recon_mean (3,H,W), per-pixel std (H,W))."""
    model.eval()
    enable_dropout(model)
    recs = []
    for _ in range(T):
        mu, _ = model.encode(x_one)
        recs.append(model.decode(mu))
    R = torch.stack(recs)               # (T, 1, 3, H, W)
    return R.mean(0)[0].cpu().numpy(), R.std(0)[0].mean(0).cpu().numpy()


def plot_state_estimate(mu_mean, epi, ale, gt, idx, std4, mean4, tag, save_dir, s_cal=1.0):
    dims = np.arange(N_SUP)
    m = mu_mean[idx] * std4 + mean4
    g = gt[idx] * std4 + mean4
    tot = np.sqrt(epi[idx] ** 2 + ale[idx] ** 2) * std4
    plt.figure(figsize=(6.4, 4.4))
    plt.errorbar(dims, m, yerr=1.96 * s_cal * tot, fmt="o", color="C3", capsize=5, label="estimate ±95%")
    plt.plot(dims, g, "kx", ms=10, label="GT")
    plt.xticks(dims, [f"{l}\n{u}" for l, u in zip(DIM_LABELS, DIM_UNITS)])
    plt.title(f"Perception: physical-state estimate ± uncertainty | {tag}")
    plt.ylabel("physical units")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    p = os.path.join(save_dir, f"vae_state_estimate_{tag}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print("saved:", p)


def plot_recon_uncertainty(inp_frame, recon_mean, std_map, tag, save_dir):
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    ax[0].imshow(np.transpose(inp_frame, (1, 2, 0)))
    ax[0].set_title("input frame_t")
    ax[1].imshow(np.transpose(recon_mean, (1, 2, 0)))
    ax[1].set_title("recon (MC mean)")
    im = ax[2].imshow(std_map, cmap="inferno")
    ax[2].set_title("per-pixel std (uncertainty)")
    fig.colorbar(im, ax=ax[2], fraction=0.046)
    for a in ax:
        a.axis("off")
    plt.suptitle(f"Reconstruction uncertainty | {tag}")
    plt.tight_layout()
    p = os.path.join(save_dir, f"vae_recon_uncertainty_{tag}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved:", p)


def plot_vae_noise_sweep(levels, epi, ale, base_ale, rmse_drop, rmse_base, save_dir):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    ax[0].plot(levels, epi, "C3-o", label="epistemic (MC dropout)")
    ax[0].plot(levels, ale, "C0-s", label="aleatoric (VAE logvar)")
    ax[0].plot(levels, base_ale, "C0--", alpha=0.6, label="baseline aleatoric")
    ax[0].set_title("Uncertainty vs noise (epistemic spikes OOD)")
    ax[0].set_xlabel("noise level σ")
    ax[0].set_ylabel("mean std (standardized)")
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    ax[1].plot(levels, rmse_drop, "C3-o", label="dropout VAE")
    ax[1].plot(levels, rmse_base, "C0-s", label="baseline VAE")
    ax[1].set_title("Perception RMSE vs noise")
    ax[1].set_xlabel("noise level σ")
    ax[1].set_ylabel("state RMSE (standardized)")
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=8)
    plt.tight_layout()
    p = os.path.join(save_dir, "vae_noise_sweep.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved:", p)


def run_vae_uncertainty(device, mean, std, mean4, std4_np):
    print(f"\n{'#'*64}\n#  PERCEPTION UNCERTAINTY (dropout VAE)\n{'#'*64}")
    if TRAIN_DROPOUT_VAE:
        mc_vae = train_dropout_vae(device, mean, std)
    else:
        mc_vae = VAE_MC(LATENT_SIZE, p_drop=P_DROP_VAE).to(device)
        mc_vae.load_state_dict(torch.load(DROPOUT_VAE_CKPT, map_location=device))
        print(f"[dropout-vae] loaded from {DROPOUT_VAE_CKPT}")

    base_vae = VAE(latent_size=LATENT_SIZE).to(device)
    base_vae.load_state_dict(torch.load(VAE_CKPT, map_location=device))
    base_vae.eval()

    test_ds = VaePairDataset(os.path.join(DATA_ROOT, "test"), shift=SHIFT, state_mean=mean, state_std=std)
    test_dl = DataLoader(test_ds, batch_size=VAE_BATCH, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    print(f"  test pairs: {len(test_ds)}")

    results = {}
    sweep = {"levels": [], "epi": [], "ale": [], "base_ale": [], "rmse_drop": [], "rmse_base": []}
    for level in NOISE_LEVELS:
        tag = noise_tag(NOISE_TYPE, level)
        print(f"\n=== VAE | NOISE {NOISE_TYPE} σ={level:.2f} ===")
        nf = make_noise_fn(NOISE_TYPE, level, NOISE_SEED, device)
        mu_mean, epi, ale, gt = mc_encode_collect(mc_vae, test_dl, device, nf, T_MC)
        b_mu, b_ale, _ = baseline_encode_collect(base_vae, test_dl, device, nf)
        total = np.sqrt(epi ** 2 + ale ** 2)
        results[tag] = dict(mu_mean=mu_mean, epi=epi, ale=ale, total=total, gt=gt, level=level)
        sweep["levels"].append(level)
        sweep["epi"].append(float(epi.mean()))
        sweep["ale"].append(float(ale.mean()))
        sweep["base_ale"].append(float(b_ale.mean()))
        sweep["rmse_drop"].append(float(np.sqrt(((mu_mean - gt) ** 2).mean())))
        sweep["rmse_base"].append(float(np.sqrt(((b_mu - gt) ** 2).mean())))
        print(f"  perception RMSE  dropout={sweep['rmse_drop'][-1]:.4f} baseline={sweep['rmse_base'][-1]:.4f} | "
              f"epistemic={sweep['epi'][-1]:.4f} aleatoric={sweep['ale'][-1]:.4f}")

    clean_tag = noise_tag(NOISE_TYPE, 0.0) if 0.0 in NOISE_LEVELS else noise_tag(NOISE_TYPE, NOISE_LEVELS[0])
    R = results[clean_tag]
    s_cal = recal_scalar(R["gt"], R["mu_mean"], R["total"], level=0.95)
    print(f"\n[recalibration] VAE std-correction s={s_cal:.3f} (fit on {clean_tag})")

    for tag, R in results.items():
        plot_calibration(R["gt"], R["mu_mean"], R["total"], tag, SAVE_DIR, s_cal, prefix="vae")
    plot_vae_noise_sweep(sweep["levels"], sweep["epi"], sweep["ale"], sweep["base_ale"],
                         sweep["rmse_drop"], sweep["rmse_base"], SAVE_DIR)

    # chosen-frame visualizations (clean + worst noise)
    rng = np.random.default_rng(WINDOW_SEED)
    idx = int(rng.integers(0, R["gt"].shape[0]))
    plot_state_estimate(R["mu_mean"], R["epi"], R["ale"], R["gt"], idx, std4_np, mean4, clean_tag, SAVE_DIR, s_cal)
    img_t, img_tp1, *_ = next(iter(test_dl))
    j = min(idx, img_t.shape[0] - 1)
    for level in (0.0, NOISE_LEVELS[-1]):
        nf = make_noise_fn(NOISE_TYPE, level, NOISE_SEED, device)
        x_one, it = _noisy_stack(img_t[j:j + 1], img_tp1[j:j + 1], device, nf)
        rmean, smap = recon_uncertainty(mc_vae, x_one, T_MC)
        plot_recon_uncertainty(it[0].cpu().numpy(), rmean, smap, noise_tag(NOISE_TYPE, level), SAVE_DIR)

    save = {"levels": np.array(sweep["levels"]), "epi": np.array(sweep["epi"]), "ale": np.array(sweep["ale"]),
            "base_ale": np.array(sweep["base_ale"]), "rmse_drop": np.array(sweep["rmse_drop"]),
            "rmse_base": np.array(sweep["rmse_base"]), "s_cal": s_cal, "T_MC": T_MC, "p_drop_vae": P_DROP_VAE}
    np.savez(os.path.join(SAVE_DIR, "vae_unc_curves.npz"), **save)
    print("saved: vae_unc_curves.npz")


# ===========================================================================
#  Main
# ===========================================================================
def main():
    set_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    mean, std = load_norm_stats(NORM_STATS)
    mean4 = np.asarray(mean[:N_SUP], np.float64)
    std4_np = np.asarray(std[:N_SUP], np.float64)
    std4 = torch.tensor(std4_np, device=device, dtype=torch.float32)

    if RUN_VAE_UNCERTAINTY:
        run_vae_uncertainty(device, mean, std, mean4, std4_np)
    if RUN_LSTM_UNCERTAINTY:
        run_lstm_uncertainty(device, mean, std, mean4, std4_np, std4)
    print("\nAll done ->", SAVE_DIR)


if __name__ == "__main__":
    main()
