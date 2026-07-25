"""
uncertainty_alt.py — Corrected/expanded interpretable uncertainty for the CartPole world model.

Based on uncertainty.py. The SAME two methods (nothing new — no ensembles):
    * MC Dropout  -> EPISTEMIC (model/OOD) uncertainty (dropout-VAE perception, dropout-LSTM dynamics)
    * VAE logvar  -> ALEATORIC (input/measurement) uncertainty
but fixed & properly calibrated/organized:

FIXES vs uncertainty.py
  (1) LOCKED (per-rollout) dropout in the LSTM. The old one resampled the mask at EVERY step -> uncorrelated,
      over-dispersed band (it needed s_cal~0.27). Here ONE mask per rollout (variational dropout
      a la Gal & Ghahramani), both in TRAINING and in MC -> consistent, correct epistemic.
  (2) PER-DIM recalibration (a vector s[dim]) instead of one global scalar. Position & velocity have
      different error scales; a single scalar cannot calibrate both.
  (3) PROPER metrics: Gaussian NLL + per-level coverage (reliability diagram) + sharpness, NOT
      coverage alone (a huge band covers trivially).
  (4) TOTAL uncertainty that REACTS to perception OOD: the dynamics seed z0 is perturbed with
      σ_perc = sqrt(aleatoric² + epistemic_perception²) (measured from the VAE). This way the dynamics
      band grows under noise (the old one did NOT). Decomposition: dynamics-only vs total.
  (5) Headline VISUALIZATIONS: the "umbrella" band on a CALM vs an OOD (near-failure) window, not a random one.

Epistemic vs Aleatoric (separately, where it makes sense):
  * Perception epistemic = spread of the T MC-dropout encodings of the SAME frame (how much the encoder "does not know").
  * Perception aleatoric = the VAE's logvar (input noise; here it turns out ~flat -> we say so honestly).
  * Dynamics epistemic   = spread of the T locked-mask MC-dropout rollouts (how much the dynamics "does not know").
  * Total                = perception ⊕ dynamics (seed-perturbation + dropout rollout).

Paths come from config.py (override with OUTPUT_DIR / CARTPOLE_* env vars). Run:  !python3 cartpole/uncertainty_alt.py
Set TRAIN_DROPOUT_* = False to load from the CARTPOLE_DROPOUT_LSTM/CARTPOLE_DROPOUT_VAE checkpoints.
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

from loader import list_npz, LatentSequenceDataset, VaePairDataset, load_norm_stats
from vae import VAE
from lstm import LatentPredictor          # baseline (no dropout) — imported, NOT modified

from paths import BASELINE_LSTM, BASELINE_VAE, DATA_ROOT, DROPOUT_LSTM, DROPOUT_VAE, outputs

# ---------------------------------------------------------------------------
# CONFIG  (paths from config.py via paths.py)
# ---------------------------------------------------------------------------
NORM_STATS = os.path.join(DATA_ROOT, "norm_stats.npz")
VAE_CKPT = BASELINE_VAE
BASELINE_LSTM_CKPT = BASELINE_LSTM
DROPOUT_LSTM_CKPT = DROPOUT_LSTM
DROPOUT_VAE_CKPT = DROPOUT_VAE

LATENT_ROOT = outputs("cartpole_unc_alt_latents")
SAVE_DIR = outputs("cartpole_uncertainty_alt")

LATENT_SIZE, N_SUP, N_IMG = 64, 4, 60
N_ACTIONS, HIDDEN, LAYERS = 2, 64, 2
SHIFT = 0
SEQ_LEN, TEST_STRIDE, BATCH = 30, 1, 128

RUN_PERCEPTION = True
RUN_DYNAMICS = True

# --- dropout training (mirror uncertainty.py recipe) ---
TRAIN_DROPOUT_LSTM = True
TRAIN_DROPOUT_VAE = False
P_DROP = 0.1                 # LSTM dropout (locked, per-rollout)
P_DROP_VAE = 0.1
TRAIN_STRIDE, TRAIN_BATCH = 5, 64
EPOCHS, LR, CLIP, W_PHYS = 50, 1e-3, 1.0, 1.0
P_START, P_END, P_DECAY_EPOCHS = 1.0, 0.3, 40
L_START, CURRICULUM_EPOCHS = 5, 15
EARLY_STOP_PATIENCE, SCHED_PATIENCE = 6, 3
VAE_EPOCHS, VAE_BATCH, VAE_LR = 40, 128, 1e-3
BETA_PHYS, BETA_STYLE_MAX, KL_ANNEAL_EPOCHS = 0.01, 1.0, 20
LAMBDA_SUP, VAE_EARLY_STOP_PATIENCE = 1.0, 5
NUM_WORKERS, SEED = 2, 0

# --- MC dropout / uncertainty ---
T_MC = 30                    # MC passes (epistemic)
T_SEED = 16                  # MC encodings/frame for perception-epistemic seed std (Part 3)
RECAL_LEVEL = 0.95           # level for the per-dim recalibration

# --- visual-noise sweep on TEST images before encoding ---
NOISE_TYPE = "gaussian"
NOISE_LEVELS = [0.0, 0.10, 0.30]         # 0.0 = clean; must contain 0.0 (fits recal/calibration)
NOISE_SEED = 42

WINDOW_SEED = 0
DIM_LABELS = ["x", r"$\dot{x}$", r"$\theta$", r"$\dot{\theta}$"]
DIM_UNITS = ["(cart pos)", "(cart vel)", "[rad]", "[rad/s]"]
# Gaussian central-interval multipliers |N(0,1)| (no scipy)
Z = {0.50: 0.6745, 0.80: 1.2816, 0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}
LOG_Y = True


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def noise_tag(ntype, level):
    return f"{ntype}_{level:.2f}".replace(".", "p")


def enable_dropout(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


# ---------------------------------------------------------------------------
# Noise injection (mirrors test_p1.py)
# ---------------------------------------------------------------------------
def add_gaussian_noise(img, std, gen):
    return torch.clamp(img + torch.randn(img.shape, generator=gen, device=img.device) * std, 0.0, 1.0)


def add_salt_pepper_noise(img, amount, gen):
    mask = torch.rand(img.shape, generator=gen, device=img.device)
    out = img.clone(); out[mask < amount / 2] = 0.0; out[mask > 1 - amount / 2] = 1.0
    return out


def make_noise_fn(ntype, level, seed, device):
    gen = torch.Generator(device=device); gen.manual_seed(seed)
    if level == 0.0:
        return lambda x: x
    if ntype == "gaussian":
        return lambda x: add_gaussian_noise(x, level, gen)
    if ntype == "salt_pepper":
        return lambda x: add_salt_pepper_noise(x, level, gen)
    raise ValueError(f"Unknown noise type: {ntype}")


# ---------------------------------------------------------------------------
# Metrics & calibration  (gt/mean/std in STANDARDIZED units; shapes (N,L,D) or (N,D))
# ---------------------------------------------------------------------------
def coverage(gt, mean, std, z, eps=1e-8):
    """Fraction of points with |gt-mean| <= z*std (scalar, overall)."""
    return float((np.abs(gt - mean) <= z * (std + eps)).mean())


def coverage_per_dim(gt, mean, std, z, eps=1e-8):
    ax = tuple(range(gt.ndim - 1))
    return (np.abs(gt - mean) <= z * (std + eps)).mean(axis=ax)        # (D,)


def gaussian_nll(gt, mean, std, eps=1e-8):
    """Mean negative log-likelihood of gt under N(mean, std²). LOWER=better.
    Penalizes BOTH the too-wide (over-dispersed) AND the too-narrow (over-confident) band."""
    s = std + eps
    return float((0.5 * np.log(2 * np.pi) + np.log(s) + 0.5 * ((gt - mean) / s) ** 2).mean())


def recal_per_dim(gt, mean, std, level=RECAL_LEVEL, eps=1e-8):
    """Per-dimension multiplier s[D] so that the `level` interval is calibrated per dim.
    s[d] = empirical(level-quantile of |gt-mean|/std in dim d) / Z[level]. Fit on CLEAN."""
    r = np.abs(gt - mean) / (std + eps)
    r2 = r.reshape(-1, r.shape[-1])
    return np.percentile(r2, 100 * level, axis=0) / Z[level]          # (D,)


def apply_recal(std, s_vec):
    return std * np.asarray(s_vec)[(None,) * (std.ndim - 1)]


def metrics_block(gt, mean, std):
    """ -> dict with NLL, sharpness (mean std), cov@95, RMSE (standardized)."""
    return {"nll": gaussian_nll(gt, mean, std),
            "sharp": float(std.mean()),
            "cov95": coverage(gt, mean, std, Z[0.95]),
            "rmse": float(np.sqrt(((gt - mean) ** 2).mean()))}


def plot_reliability(gt, mean, std, s_vec, tag, save_dir, prefix):
    """Reliability diagram (nominal vs empirical coverage) raw + per-dim-recalibrated, AND
    per-dim coverage@95 bars -> shows WHERE it was mis-calibrated."""
    levels = sorted(Z.keys())
    emp = [coverage(gt, mean, std, Z[q]) for q in levels]
    std_c = apply_recal(std, s_vec)
    emp_c = [coverage(gt, mean, std_c, Z[q]) for q in levels]
    cov95_dim_raw = coverage_per_dim(gt, mean, std, Z[0.95])
    cov95_dim_cal = coverage_per_dim(gt, mean, std_c, Z[0.95])

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    ax[0].plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
    ax[0].plot(levels, emp, "o-", color="C3", label="raw MC dropout")
    ax[0].plot(levels, emp_c, "s--", color="C2", label="per-dim recalibrated")
    ax[0].set_xlabel("Nominal coverage"); ax[0].set_ylabel("Empirical coverage")
    ax[0].set_title(f"Reliability | {tag}"); ax[0].set_xlim(0, 1); ax[0].set_ylim(0, 1)
    ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
    xs = np.arange(N_SUP)
    ax[1].bar(xs - 0.18, cov95_dim_raw, 0.36, color="C3", label="raw")
    ax[1].bar(xs + 0.18, cov95_dim_cal, 0.36, color="C2", label="recalibrated")
    ax[1].axhline(0.95, color="k", ls=":", lw=1)
    ax[1].set_xticks(xs); ax[1].set_xticklabels(DIM_LABELS)
    ax[1].set_ylim(0, 1.02); ax[1].set_ylabel("empirical 95% coverage")
    ax[1].set_title("Per-dim coverage@95 (ideal=0.95)"); ax[1].legend(fontsize=8)
    plt.tight_layout()
    p = os.path.join(save_dir, f"{prefix}_reliability_{tag}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("saved:", p)


# ===========================================================================
#  PERCEPTION (dropout VAE)  — epistemic (MC) vs aleatoric (logvar)
# ===========================================================================
class VAE_MC(nn.Module):
    """Baseline VAE + dropout on the encoder feature vector and on the decode features."""
    def __init__(self, latent_size=64, in_channels=6, out_channels=3, p_drop=0.1):
        super().__init__()
        self.latent_size = latent_size
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(inplace=True))
        self.enc_drop = nn.Dropout(p_drop)
        self.fc_mu = nn.Linear(64 * 10 * 15, latent_size)
        self.fc_logvar = nn.Linear(64 * 10 * 15, latent_size)
        self.fc_decode = nn.Linear(latent_size, 64 * 10 * 15)
        self.dec_drop = nn.Dropout(p_drop)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, out_channels, 4, 2, 1), nn.Sigmoid())

    def encode(self, x):
        h = self.enc_drop(self.encoder(x).flatten(1))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z):
        h = self.dec_drop(self.fc_decode(z)).view(-1, 64, 10, 15)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar


def vae_losses(recon, target, mu, logvar, state_t, n_sup):
    B, D = mu.size(0), mu.size(1)
    recon_l = F.mse_loss(recon, target, reduction="mean")
    sup = F.mse_loss(mu[:, :n_sup], state_t, reduction="mean")
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return recon_l, kl[:, :n_sup].sum() / B / n_sup, kl[:, n_sup:].sum() / B / (D - n_sup), sup


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
    opt = optim.Adam(model.parameters(), lr=VAE_LR)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=SCHED_PATIENCE)
    out_path = os.path.join(SAVE_DIR, "vae_dropout_best.pth")
    best, bad = float("inf"), 0
    for epoch in range(1, VAE_EPOCHS + 1):
        beta = BETA_STYLE_MAX * min(1.0, epoch / max(KL_ANNEAL_EPOCHS, 1))
        for phase, dl, train in (("tr", tdl, True), ("va", vdl, False)):
            model.train() if train else model.eval()
            agg = {"recon": 0.0, "sup": 0.0, "n": 0}
            for img_t, img_tp1, action, state_t, state_tp1 in dl:
                it = img_t.to(device).float() / 255.0
                itp = img_tp1.to(device).float() / 255.0
                x = torch.cat([it, itp], dim=1); st = state_t.to(device)
                with torch.set_grad_enabled(train):
                    recon, mu, logvar = model(x)
                    r, kp, ks, s = vae_losses(recon, it, mu, logvar, st, N_SUP)
                    loss = r + BETA_PHYS * kp + beta * ks + LAMBDA_SUP * s
                if train:
                    opt.zero_grad(); loss.backward(); opt.step()
                bs = it.size(0); agg["recon"] += r.item() * bs; agg["sup"] += s.item() * bs; agg["n"] += bs
            if phase == "va":
                val_score = agg["recon"] / agg["n"] + LAMBDA_SUP * agg["sup"] / agg["n"]
        sched.step(val_score)
        print(f"E{epoch:03d} | beta={beta:.2f} | val select={val_score:.5f}")
        if val_score < best - 1e-6:
            best, bad = val_score, 0; torch.save(model.state_dict(), out_path); print("  -> saved")
        else:
            bad += 1
            if bad >= VAE_EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch}."); break
    model.load_state_dict(torch.load(out_path, map_location=device))
    print(f"[dropout-vae] best val: {best:.5f} -> {out_path}")
    return model


def _noisy_stack(img_t, img_tp1, device, noise_fn):
    it = img_t.to(device).float() / 255.0
    itp = img_tp1.to(device).float() / 255.0
    if noise_fn is not None:
        it, itp = noise_fn(it), noise_fn(itp)
    return torch.cat([it, itp], dim=1), it


@torch.no_grad()
def mc_encode_collect(model, loader, device, noise_fn, T):
    """T stochastic encodings/frame -> (mu_mean, epi_std, ale_std, gt), each (N, N_SUP) standardized."""
    model.eval(); enable_dropout(model)
    MU, EPI, ALE, GT = [], [], [], []
    for img_t, img_tp1, action, state_t, state_tp1 in tqdm(loader, desc="VAE MC", leave=False):
        x, _ = _noisy_stack(img_t, img_tp1, device, noise_fn)
        mus, lvs = [], []
        for _ in range(T):
            mu, lv = model.encode(x)
            mus.append(mu[:, :N_SUP]); lvs.append(lv[:, :N_SUP])
        M, Lo = torch.stack(mus), torch.stack(lvs)
        MU.append(M.mean(0).cpu().numpy()); EPI.append(M.std(0).cpu().numpy())
        ALE.append(torch.exp(0.5 * Lo).mean(0).cpu().numpy()); GT.append(state_t[:, :N_SUP].cpu().numpy())
    return np.concatenate(MU), np.concatenate(EPI), np.concatenate(ALE), np.concatenate(GT)


@torch.no_grad()
def baseline_encode_collect(model, loader, device, noise_fn):
    model.eval()
    MU, ALE, GT = [], [], []
    for img_t, img_tp1, action, state_t, state_tp1 in loader:
        x, _ = _noisy_stack(img_t, img_tp1, device, noise_fn)
        mu, lv = model.encode(x)
        MU.append(mu[:, :N_SUP].cpu().numpy()); ALE.append(torch.exp(0.5 * lv[:, :N_SUP]).cpu().numpy())
        GT.append(state_t[:, :N_SUP].cpu().numpy())
    return np.concatenate(MU), np.concatenate(ALE), np.concatenate(GT)


@torch.no_grad()
def recon_uncertainty(model, x_one, T):
    model.eval(); enable_dropout(model)
    recs = [model.decode(model.encode(x_one)[0]) for _ in range(T)]
    R = torch.stack(recs)
    return R.mean(0)[0].cpu().numpy(), R.std(0)[0].mean(0).cpu().numpy()


def plot_perception_sweep(levels, epi, ale, base_ale, rmse_drop, rmse_base, save_dir):
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax[0].plot(levels, epi, "C3-o", label="epistemic (MC dropout)")
    ax[0].plot(levels, ale, "C0-s", label="aleatoric (VAE logvar)")
    ax[0].plot(levels, base_ale, "C0--", alpha=0.6, label="baseline aleatoric")
    ax[0].set_title("Uncertainty vs noise — epistemic spikes OOD, aleatoric ~flat")
    ax[0].set_xlabel("noise level σ"); ax[0].set_ylabel("mean std (standardized)")
    ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
    ax[1].plot(levels, rmse_drop, "C3-o", label="dropout VAE")
    ax[1].plot(levels, rmse_base, "C0-s", label="baseline VAE")
    ax[1].set_title("Perception RMSE vs noise"); ax[1].set_xlabel("noise level σ")
    ax[1].set_ylabel("state RMSE (standardized)"); ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)
    plt.suptitle("PERCEPTION uncertainty (dropout VAE)", y=1.02)
    plt.tight_layout()
    p = os.path.join(save_dir, "perc_noise_sweep.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); print("saved:", p)


def plot_state_estimate(mu_mean, epi, ale, gt, idx, std4, mean4, tag, save_dir, s_vec):
    dims = np.arange(N_SUP)
    m = mu_mean[idx] * std4 + mean4; g = gt[idx] * std4 + mean4
    tot = np.sqrt(epi[idx] ** 2 + ale[idx] ** 2) * s_vec * std4
    plt.figure(figsize=(6.6, 4.4))
    plt.errorbar(dims, m, yerr=1.96 * tot, fmt="o", color="C3", capsize=6, label="estimate ±95% (recal)")
    plt.plot(dims, g, "kx", ms=11, label="GT")
    plt.xticks(dims, [f"{l}\n{u}" for l, u in zip(DIM_LABELS, DIM_UNITS)])
    plt.title(f"PERCEPTION: physical-state estimate ± uncertainty | {tag}")
    plt.ylabel("physical units"); plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout()
    p = os.path.join(save_dir, f"perc_state_estimate_{tag}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close(); print("saved:", p)


def plot_recon_uncertainty(inp_frame, recon_mean, std_map, tag, save_dir):
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    ax[0].imshow(np.transpose(inp_frame, (1, 2, 0))); ax[0].set_title("input frame_t")
    ax[1].imshow(np.transpose(recon_mean, (1, 2, 0))); ax[1].set_title("recon (MC mean)")
    im = ax[2].imshow(std_map, cmap="inferno"); ax[2].set_title("per-pixel std (epistemic)")
    fig.colorbar(im, ax=ax[2], fraction=0.046)
    for a in ax:
        a.axis("off")
    plt.suptitle(f"Reconstruction uncertainty | {tag}"); plt.tight_layout()
    p = os.path.join(save_dir, f"perc_recon_uncertainty_{tag}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); print("saved:", p)


def run_perception(device, mean, std, mean4, std4_np, mc_vae):
    print(f"\n{'#'*64}\n#  PERCEPTION UNCERTAINTY (dropout VAE)\n{'#'*64}")
    base_vae = VAE(latent_size=LATENT_SIZE).to(device)
    base_vae.load_state_dict(torch.load(VAE_CKPT, map_location=device)); base_vae.eval()
    test_ds = VaePairDataset(os.path.join(DATA_ROOT, "test"), shift=SHIFT, state_mean=mean, state_std=std)
    test_dl = DataLoader(test_ds, batch_size=VAE_BATCH, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    results, sweep = {}, {k: [] for k in ("levels", "epi", "ale", "base_ale", "rmse_drop", "rmse_base")}
    for level in NOISE_LEVELS:
        tag = noise_tag(NOISE_TYPE, level)
        nf = make_noise_fn(NOISE_TYPE, level, NOISE_SEED, device)
        mu_mean, epi, ale, gt = mc_encode_collect(mc_vae, test_dl, device, nf, T_MC)
        b_mu, b_ale, _ = baseline_encode_collect(base_vae, test_dl, device, nf)
        total = np.sqrt(epi ** 2 + ale ** 2)
        results[tag] = dict(mu_mean=mu_mean, epi=epi, ale=ale, total=total, gt=gt)
        sweep["levels"].append(level); sweep["epi"].append(float(epi.mean())); sweep["ale"].append(float(ale.mean()))
        sweep["base_ale"].append(float(b_ale.mean()))
        sweep["rmse_drop"].append(float(np.sqrt(((mu_mean - gt) ** 2).mean())))
        sweep["rmse_base"].append(float(np.sqrt(((b_mu - gt) ** 2).mean())))
        print(f"  σ={level:.2f} | epi={sweep['epi'][-1]:.4f} ale={sweep['ale'][-1]:.4f} "
              f"RMSE drop={sweep['rmse_drop'][-1]:.4f} base={sweep['rmse_base'][-1]:.4f}")

    clean = noise_tag(NOISE_TYPE, 0.0)
    R0 = results[clean]
    s_vec = recal_per_dim(R0["gt"], R0["mu_mean"], R0["total"])
    print(f"[recal] perception per-dim s = {np.round(s_vec, 3)}")
    print(f"\n  {'cond':<10}{'NLL(raw)':>10}{'NLL(cal)':>10}{'cov95(raw)':>12}{'cov95(cal)':>12}{'sharp':>9}{'RMSE':>9}")
    for tag, R in results.items():
        m_raw = metrics_block(R["gt"], R["mu_mean"], R["total"])
        m_cal = metrics_block(R["gt"], R["mu_mean"], apply_recal(R["total"], s_vec))
        print(f"  {tag:<10}{m_raw['nll']:>10.3f}{m_cal['nll']:>10.3f}{m_raw['cov95']:>12.3f}"
              f"{m_cal['cov95']:>12.3f}{m_raw['sharp']:>9.3f}{m_raw['rmse']:>9.3f}")
        plot_reliability(R["gt"], R["mu_mean"], R["total"], s_vec, tag, SAVE_DIR, prefix="perc")

    plot_perception_sweep(sweep["levels"], sweep["epi"], sweep["ale"], sweep["base_ale"],
                          sweep["rmse_drop"], sweep["rmse_base"], SAVE_DIR)

    rng = np.random.default_rng(WINDOW_SEED)
    idx = int(rng.integers(0, R0["gt"].shape[0]))
    plot_state_estimate(R0["mu_mean"], R0["epi"], R0["ale"], R0["gt"], idx, std4_np, mean4, clean, SAVE_DIR, s_vec)
    img_t, img_tp1, *_ = next(iter(test_dl))
    j = min(idx, img_t.shape[0] - 1)
    for level in (0.0, NOISE_LEVELS[-1]):
        nf = make_noise_fn(NOISE_TYPE, level, NOISE_SEED, device)
        x_one, it = _noisy_stack(img_t[j:j + 1], img_tp1[j:j + 1], device, nf)
        rmean, smap = recon_uncertainty(mc_vae, x_one, T_MC)
        plot_recon_uncertainty(it[0].cpu().numpy(), rmean, smap, noise_tag(NOISE_TYPE, level), SAVE_DIR)

    np.savez(os.path.join(SAVE_DIR, "perc_curves.npz"), s_vec=s_vec, **{k: np.array(v) for k, v in sweep.items()})
    print("saved: perc_curves.npz")


# ===========================================================================
#  DYNAMICS (dropout LSTM, LOCKED masks)  +  TOTAL (perception ⊕ dynamics)
# ===========================================================================
class LatentPredictorVarMC(nn.Module):
    """Baseline arch + ONE locked dropout before the residual head. The mask is sampled ONCE
    per rollout (sample_mask) and stays fixed across all steps -> variational/recurrent-style MC."""
    def __init__(self, latent=64, action_dim=2, hidden=64, layers=2, p_drop=0.1):
        super().__init__()
        self.hidden, self.layers, self.p_drop = hidden, layers, p_drop
        self.lstm = nn.LSTM(latent + action_dim, hidden, layers, batch_first=True)
        self.fc = nn.Linear(hidden, latent)
        nn.init.zeros_(self.fc.weight); nn.init.zeros_(self.fc.bias)
        self._mask = None

    def init_hidden(self, b, device):
        return (torch.zeros(self.layers, b, self.hidden, device=device),
                torch.zeros(self.layers, b, self.hidden, device=device))

    def sample_mask(self, B, device):
        """ONE inverted-dropout Bernoulli mask (B,hidden) for the WHOLE rollout."""
        if self.p_drop <= 0:
            self._mask = None; return
        keep = 1.0 - self.p_drop
        self._mask = (torch.rand(B, self.hidden, device=device) < keep).float() / keep

    def clear_mask(self):
        self._mask = None                      # deterministic forward (full weights)

    def step(self, z, a_onehot, hidden):
        out, hidden = self.lstm(torch.cat([z, a_onehot], dim=-1).unsqueeze(1), hidden)
        h = out.squeeze(1)
        if self._mask is not None:
            h = h * self._mask
        return z + self.fc(h), hidden


def _rollout(model, z0, action, L):
    B = z0.shape[0]
    hidden = model.init_hidden(B, z0.device)
    z_in, preds = z0, []
    for k in range(L):
        a = F.one_hot(action[:, k].long(), N_ACTIONS).float()
        z_pred, hidden = model.step(z_in, a, hidden)
        preds.append(z_pred); z_in = z_pred
    return torch.stack(preds, dim=1)


def _train_epoch_lstm(model, loader, optimizer, device, p_tf, cur_len):
    model.train()
    tot, n = 0.0, 0
    for batch in loader:
        z_t, action, z_tp1, state_t, state_tp1 = [b.to(device, non_blocking=True) for b in batch]
        B = z_t.shape[0]; L = min(cur_len, z_t.shape[1])
        model.sample_mask(B, device)           # LOCKED mask per sequence (variational training)
        z_gt = z_tp1[:, :L]
        hidden = model.init_hidden(B, device); z_in = z_t[:, 0]; preds = []
        for k in range(L):
            a = F.one_hot(action[:, k].long(), N_ACTIONS).float()
            z_pred, hidden = model.step(z_in, a, hidden)
            preds.append(z_pred)
            if k < L - 1:
                use_tf = (torch.rand(B, 1, device=device) < p_tf).float()
                z_in = use_tf * z_gt[:, k] + (1.0 - use_tf) * z_pred.detach()
        preds = torch.stack(preds, dim=1)
        loss = (F.mse_loss(preds, z_gt) + W_PHYS * F.mse_loss(preds[..., :N_SUP], z_gt[..., :N_SUP]))
        optimizer.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CLIP); optimizer.step()
        tot += loss.item() * B; n += B
    return tot / max(n, 1)


@torch.no_grad()
def _eval_epoch_lstm(model, loader, device, std4):
    model.eval(); model.clear_mask()           # deterministic for model selection
    se, n = None, 0
    for batch in loader:
        z_t, action, z_tp1, state_t, state_tp1 = [b.to(device, non_blocking=True) for b in batch]
        preds = _rollout(model, z_t[:, 0], action, z_t.shape[1])[..., :N_SUP]
        s = (((preds - state_tp1) * std4) ** 2).sum(dim=0)
        se = s if se is None else se + s; n += preds.size(0)
    return (se / n).mean(dim=1).cpu().numpy()


def train_dropout_lstm(device, mean, std, std4):
    """Mirror the lstm.py recipe, with LOCKED dropout. Pre-encode CLEAN with the baseline VAE."""
    vae = VAE(latent_size=LATENT_SIZE).to(device)
    vae.load_state_dict(torch.load(VAE_CKPT, map_location=device)); vae.eval()
    from vae import encode_fn
    from loader import precompute_latents
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
    tr = LatentSequenceDataset(os.path.join(LATENT_ROOT, "clean", "train"), seq_len=SEQ_LEN,
                               stride=TRAIN_STRIDE, state_mean=mean, state_std=std)
    va = LatentSequenceDataset(os.path.join(LATENT_ROOT, "clean", "val"), seq_len=SEQ_LEN,
                               stride=TRAIN_STRIDE, state_mean=mean, state_std=std)
    tdl = DataLoader(tr, batch_size=TRAIN_BATCH, shuffle=True, drop_last=True,
                     num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=pw)
    vdl = DataLoader(va, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=pw)
    print(f"[dropout-lstm] train windows: {len(tr)} | val windows: {len(va)} | p_drop={P_DROP} (LOCKED)")
    model = LatentPredictorVarMC(LATENT_SIZE, N_ACTIONS, HIDDEN, LAYERS, p_drop=P_DROP).to(device)
    opt = optim.Adam(model.parameters(), lr=LR)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=SCHED_PATIENCE)
    out_path = os.path.join(SAVE_DIR, "lstm_dropout_best.pth")
    best, bad = float("inf"), 0
    for epoch in range(1, EPOCHS + 1):
        p_tf = max(P_END, P_START - (P_START - P_END) * (epoch - 1) / max(P_DECAY_EPOCHS, 1))
        cur_len = int(round(min(SEQ_LEN, L_START + (SEQ_LEN - L_START) * (epoch - 1) / max(CURRICULUM_EPOCHS, 1))))
        trl = _train_epoch_lstm(model, tdl, opt, device, p_tf, cur_len)
        val_mean = float(_eval_epoch_lstm(model, vdl, device, std4).mean())
        sched.step(val_mean)
        print(f"E{epoch:03d} | p_tf={p_tf:.2f} H={cur_len} | train={trl:.5f} | val phys-MSE={val_mean:.4f}")
        if val_mean < best - 1e-6:
            best, bad = val_mean, 0; torch.save(model.state_dict(), out_path); print("  -> saved")
        else:
            bad += 1
            if bad >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch}."); break
    model.load_state_dict(torch.load(out_path, map_location=device))
    print(f"[dropout-lstm] best val phys-MSE: {best:.4f} -> {out_path}")
    return model


@torch.no_grad()
def precompute_latents_unc(base_vae, drop_vae, root, out_root, noise_fn, device, batch=256):
    """Cache per frame: z (baseline μ), zlogvar (baseline -> aleatoric), zepi (dropout-VAE MC std
    -> perception epistemic). Image noise is applied BEFORE encoding."""
    makedirs(out_root, exist_ok=True)
    base_vae.eval(); drop_vae.eval(); enable_dropout(drop_vae)
    for f in tqdm(list_npz(root), desc="precompute unc", leave=False):
        with np.load(f) as d:
            imgs = torch.from_numpy(d["imgs"].astype(np.float32) / 255.0).permute(0, 3, 1, 2)
            acts = d["acts"].astype(np.float32); states = d["states"].astype(np.float32)
            x = (d[f"noisy_states_{SHIFT}"] if SHIFT in (2, 5, 10) else d["states"]).astype(np.float32)
        imgs = noise_fn(imgs.to(device)) if noise_fn is not None else imgs.to(device)
        img_t, img_tp1 = imgs[:-1], imgs[1:]
        zs, lvs, epis = [], [], []
        for b in range(0, img_t.shape[0], batch):
            xb = torch.cat([img_t[b:b + batch], img_tp1[b:b + batch]], dim=1)
            mu, lv = base_vae.encode(xb)
            zs.append(mu.cpu().numpy()); lvs.append(lv.cpu().numpy())
            passes = torch.stack([drop_vae.encode(xb)[0] for _ in range(T_SEED)])     # (T_SEED,B,64)
            epis.append(passes.std(0).cpu().numpy())
        cat = lambda a: np.concatenate(a, 0).astype(np.float32) if a else np.empty((0, 0), np.float32)
        np.savez_compressed(join(out_root, basename(f)), z=cat(zs), zlogvar=cat(lvs), zepi=cat(epis),
                            acts=acts[:-1], states=states[:-1], x=x[:-1])


class UncLatentSeq(Dataset):
    """Like LatentSequenceDataset + also returns the seed aleatoric (logvar0) & perception-epistemic (zepi0)."""
    def __init__(self, root, seq_len, stride, mean, std):
        self.seq_len = seq_len
        self.mean = np.asarray(mean, np.float32); self.std = np.asarray(std, np.float32)
        self.eps, self.index = [], []
        for fi, f in enumerate(list_npz(root)):
            with np.load(f) as d:
                ep = {k: d[k].astype(np.float32) for k in ("z", "zlogvar", "zepi", "acts", "states", "x")}
            self.eps.append(ep)
            n = ep["z"].shape[0] - (seq_len + 1) + 1
            for s in range(0, max(n, 0), stride):
                self.index.append((fi, s))
        if not self.index:
            raise RuntimeError(f"No windows from {root} (seq_len too large?)")

    def _std(self, s):
        return ((s - self.mean) / self.std).astype(np.float32)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        fi, s = self.index[i]; ep = self.eps[fi]; L = self.seq_len
        z = ep["z"][s:s + L + 1]
        return (torch.from_numpy(z[:-1]), torch.from_numpy(ep["acts"][s:s + L]),
                torch.from_numpy(ep["zlogvar"][s]), torch.from_numpy(ep["zepi"][s]),
                torch.from_numpy(self._std(ep["states"][s + 1:s + L + 1])))


@torch.no_grad()
def mc_collect_dynamics(model, loader, device, T, perturb_seed):
    """T locked-mask rollouts. perturb_seed=False -> DYNAMICS-only epistemic·
    True -> TOTAL (seed z0 ~ N(z0, ale²+epi²) from the VAE). -> (mean, std, gt) (N,L,N_SUP) standardized."""
    model.eval()
    MEAN, STD, GT = [], [], []
    for z_t, action, zlogvar0, zepi0, state_tp1 in tqdm(loader, desc=("TOTAL" if perturb_seed else "DYN"), leave=False):
        z_t, action = z_t.to(device), action.to(device)
        z0 = z_t[:, 0]; B = z0.shape[0]
        sig = None
        if perturb_seed:
            ale = torch.exp(0.5 * zlogvar0.to(device))            # (B,64) perception aleatoric
            epi = zepi0.to(device)                                # (B,64) perception epistemic
            sig = torch.sqrt(ale ** 2 + epi ** 2)
        passes = []
        for _ in range(T):
            seed = z0 + torch.randn_like(z0) * sig if sig is not None else z0
            model.sample_mask(B, device)
            passes.append(_rollout(model, seed, action, z_t.shape[1])[..., :N_SUP])
        P = torch.stack(passes)                                   # (T,B,L,4)
        MEAN.append(P.mean(0).cpu().numpy()); STD.append(P.std(0).cpu().numpy())
        GT.append(state_tp1[..., :N_SUP].cpu().numpy())
    return np.concatenate(MEAN), np.concatenate(STD), np.concatenate(GT)


@torch.no_grad()
def baseline_collect(model, loader, device):
    model.eval()
    P, G = [], []
    for z_t, action, zlogvar0, zepi0, state_tp1 in loader:
        z_t, action = z_t.to(device), action.to(device)
        P.append(_rollout(model, z_t[:, 0], action, z_t.shape[1])[..., :N_SUP].cpu().numpy())
        G.append(state_tp1[..., :N_SUP].numpy())
    return np.concatenate(P), np.concatenate(G)


def pick_windows(gt, mean4, std4):
    """ -> (calm_w, ood_w): the window with the SMALLEST / LARGEST max|θ| (physical). OOD ~ near-failure."""
    theta = gt[:, :, 2] * std4[2] + mean4[2]
    mx = np.abs(theta).max(axis=1)
    return int(np.argmin(mx)), int(np.argmax(mx))


def plot_band(gt, mc_mean, mc_std, base_pred, mean4, std4, w, tag, title, save_dir, s_vec):
    L = gt.shape[1]; h = np.arange(1, L + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for d in range(N_SUP):
        ax = axes[d // 2][d % 2]
        g = gt[w, :, d] * std4[d] + mean4[d]
        m = mc_mean[w, :, d] * std4[d] + mean4[d]
        b = base_pred[w, :, d] * std4[d] + mean4[d]
        band = 1.96 * s_vec[d] * mc_std[w, :, d] * std4[d]
        ax.plot(h, g, "k", lw=2.0, label="GT")
        ax.plot(h, b, "C0--", lw=1.4, label="baseline (point)")
        ax.plot(h, m, "C3-", lw=1.6, label="MC mean")
        ax.fill_between(h, m - band, m + band, color="C3", alpha=0.22, label="95% band (recal)")
        ax.set_title(f"{DIM_LABELS[d]} {DIM_UNITS[d]}"); ax.set_xlabel("Prediction Horizon")
        ax.set_xlim(1, L); ax.grid(alpha=0.3)
        if d == 0:
            ax.legend(fontsize=8)
    plt.suptitle(f"{title} — window #{w} | {tag} (physical units)"); plt.tight_layout()
    p = os.path.join(save_dir, f"dyn_band_{title.split()[0].lower()}_w{w}_{tag}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); print("saved:", p)


def plot_std_vs_error(gt, mean, std, std4, tag, save_dir, s_vec):
    sd = std4[None, None, :]
    pred = np.sqrt(((apply_recal(std, s_vec) * sd) ** 2).mean(axis=(0, 2)))
    err = np.sqrt((((gt - mean) * sd) ** 2).mean(axis=(0, 2)))
    h = np.arange(1, gt.shape[1] + 1)
    plt.figure(figsize=(6.8, 4.6))
    plt.plot(h, err, "k-", lw=2, label="actual RMSE")
    plt.plot(h, pred, "C3--", lw=2, label="predicted std (recal)")
    if LOG_Y:
        plt.yscale("log")
    plt.title(f"Predicted uncertainty vs actual error | {tag}")
    plt.xlabel("Prediction Horizon"); plt.ylabel("physical units (RMS over dims)")
    plt.xlim(1, gt.shape[1]); plt.grid(alpha=0.3, which="both"); plt.legend()
    plt.tight_layout()
    p = os.path.join(save_dir, f"dyn_std_vs_error_{tag}.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close(); print("saved:", p)


def plot_total_vs_dyn_sweep(levels, dyn_std, tot_std, rmse, save_dir):
    plt.figure(figsize=(7.2, 4.8))
    plt.plot(levels, rmse, "k-o", lw=2, label="actual RMSE")
    plt.plot(levels, tot_std, "C1-s", lw=2, label="TOTAL std (perception⊕dynamics)")
    plt.plot(levels, dyn_std, "C0-^", lw=2, label="dynamics-only std")
    plt.title("Does the band react to perception noise?  (TOTAL tracks error, dynamics-only doesn't)")
    plt.xlabel("noise level σ"); plt.ylabel("mean (standardized)")
    plt.grid(alpha=0.3); plt.legend(fontsize=9)
    plt.tight_layout()
    p = os.path.join(save_dir, "dyn_total_vs_dynonly_sweep.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close(); print("saved:", p)


def run_dynamics(device, mean, std, mean4, std4_np, std4, mc_vae):
    print(f"\n{'#'*64}\n#  DYNAMICS + TOTAL UNCERTAINTY (locked dropout LSTM)\n{'#'*64}")
    if TRAIN_DROPOUT_LSTM:
        mc_model = train_dropout_lstm(device, mean, std, std4)
    else:
        mc_model = LatentPredictorVarMC(LATENT_SIZE, N_ACTIONS, HIDDEN, LAYERS, p_drop=P_DROP).to(device)
        mc_model.load_state_dict(torch.load(DROPOUT_LSTM_CKPT, map_location=device))
        print(f"[dropout-lstm] loaded {DROPOUT_LSTM_CKPT}")
    base_model = LatentPredictor(LATENT_SIZE, N_ACTIONS, HIDDEN, LAYERS).to(device)
    base_model.load_state_dict(torch.load(BASELINE_LSTM_CKPT, map_location=device)); base_model.eval()
    base_vae = VAE(latent_size=LATENT_SIZE).to(device)
    base_vae.load_state_dict(torch.load(VAE_CKPT, map_location=device)); base_vae.eval()

    results, sweep = {}, {k: [] for k in ("levels", "dyn_std", "tot_std", "rmse")}
    for level in NOISE_LEVELS:
        tag = noise_tag(NOISE_TYPE, level)
        print(f"\n=== DYNAMICS | NOISE {NOISE_TYPE} σ={level:.2f} ===")
        nf = make_noise_fn(NOISE_TYPE, level, NOISE_SEED, device)
        test_out = os.path.join(LATENT_ROOT, tag, "test")
        precompute_latents_unc(base_vae, mc_vae, os.path.join(DATA_ROOT, "test"), test_out, nf, device)
        ds = UncLatentSeq(test_out, SEQ_LEN, TEST_STRIDE, mean, std)
        dl = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        print(f"  test windows: {len(ds)}")
        dyn_mean, dyn_std, gt = mc_collect_dynamics(mc_model, dl, device, T_MC, perturb_seed=False)
        tot_mean, tot_std, _ = mc_collect_dynamics(mc_model, dl, device, T_MC, perturb_seed=True)
        base_pred, _ = baseline_collect(base_model, dl, device)
        results[tag] = dict(gt=gt, dyn_mean=dyn_mean, dyn_std=dyn_std, tot_mean=tot_mean,
                            tot_std=tot_std, base_pred=base_pred)
        rmse = float(np.sqrt((((dyn_mean - gt) * std4_np) ** 2).mean()))
        sweep["levels"].append(level); sweep["dyn_std"].append(float(dyn_std.mean()))
        sweep["tot_std"].append(float(tot_std.mean())); sweep["rmse"].append(float(((dyn_mean - gt) ** 2).mean() ** 0.5))
        print(f"  phys-RMSE={rmse:.4f} | mean dyn-std={dyn_std.mean():.4f} tot-std={tot_std.mean():.4f}")

    clean = noise_tag(NOISE_TYPE, 0.0)
    R0 = results[clean]
    s_dyn = recal_per_dim(R0["gt"], R0["dyn_mean"], R0["dyn_std"])
    s_tot = recal_per_dim(R0["gt"], R0["tot_mean"], R0["tot_std"])
    print(f"[recal] dynamics per-dim s = {np.round(s_dyn, 3)} | total per-dim s = {np.round(s_tot, 3)}")

    print(f"\n  {'cond':<10}{'NLL(dyn)':>10}{'NLL(tot)':>10}{'cov95(dyn)':>12}{'cov95(tot)':>12}{'RMSE':>9}")
    for tag, R in results.items():
        md = metrics_block(R["gt"], R["dyn_mean"], apply_recal(R["dyn_std"], s_dyn))
        mt = metrics_block(R["gt"], R["tot_mean"], apply_recal(R["tot_std"], s_tot))
        print(f"  {tag:<10}{md['nll']:>10.3f}{mt['nll']:>10.3f}{md['cov95']:>12.3f}{mt['cov95']:>12.3f}{md['rmse']:>9.3f}")
        plot_reliability(R["gt"], R["dyn_mean"], R["dyn_std"], s_dyn, tag, SAVE_DIR, prefix="dyn")
        plot_std_vs_error(R["gt"], R["dyn_mean"], R["dyn_std"], std4_np, tag, SAVE_DIR, s_dyn)

    # headline: calm vs OOD window umbrellas (dynamics-only on clean, total on worst noise)
    calm_w, ood_w = pick_windows(R0["gt"], mean4, std4_np)
    plot_band(R0["gt"], R0["dyn_mean"], R0["dyn_std"], R0["base_pred"], mean4, std4_np, calm_w, clean,
              "DYNAMICS calm", SAVE_DIR, s_dyn)
    plot_band(R0["gt"], R0["dyn_mean"], R0["dyn_std"], R0["base_pred"], mean4, std4_np, ood_w, clean,
              "DYNAMICS OOD", SAVE_DIR, s_dyn)
    worst = noise_tag(NOISE_TYPE, NOISE_LEVELS[-1]); RW = results[worst]
    plot_band(RW["gt"], RW["tot_mean"], RW["tot_std"], RW["base_pred"], mean4, std4_np, ood_w, worst,
              "TOTAL OOD", SAVE_DIR, s_tot)
    plot_total_vs_dyn_sweep(sweep["levels"], sweep["dyn_std"], sweep["tot_std"], sweep["rmse"], SAVE_DIR)

    np.savez(os.path.join(SAVE_DIR, "dyn_curves.npz"), s_dyn=s_dyn, s_tot=s_tot,
             **{k: np.array(v) for k, v in sweep.items()})
    print("saved: dyn_curves.npz")


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

    # dropout VAE: needed by BOTH perception AND total -> build it once
    if TRAIN_DROPOUT_VAE:
        mc_vae = train_dropout_vae(device, mean, std)
    else:
        mc_vae = VAE_MC(LATENT_SIZE, p_drop=P_DROP_VAE).to(device)
        mc_vae.load_state_dict(torch.load(DROPOUT_VAE_CKPT, map_location=device))
        print(f"[dropout-vae] loaded {DROPOUT_VAE_CKPT}")

    if RUN_PERCEPTION:
        run_perception(device, mean, std, mean4, std4_np, mc_vae)
    if RUN_DYNAMICS:
        run_dynamics(device, mean, std, mean4, std4_np, std4, mc_vae)
    print("\nAll done ->", SAVE_DIR)


if __name__ == "__main__":
    main()
