"""
lstm_p1_control.py — ENCODED-mode LSTM (P1) trained on the FULL control+elite dataset (~12k).

Purpose: a STRONGER world model for the MPC (Extension 4). The control dataset has
random-action bursts / perturbed PID -> much better ACTION CONDITIONING than the
heuristic-only data (that was the problem that made the MPC fail).

Differences from lstm_p1.py:
  * REUSES the SAME (frozen) P1 VAE -> only precompute latents + train the LSTM.
  * MULTI-ROOT: unions control + elite (train/val/test) via loader_control (no copying).
  * WIND_FILTER: 'all' (default, uses all 12k) | 'clean' | 'wind'.
  * Imports instead of inline: VAE_P1 (vae_p1), LatentPredictor (lstm), loader (loader_control).

NOTE on norm_stats: the ORIGINAL norm_stats (the ones the VAE was trained with) — NOT combined — because
the VAE's mu[:8] lives in that standardized space (otherwise the physical-MSE eval comes out wrong).

Run: python lunarlander/lstm_p1_control.py   (cwd: lunarlander/ so the imports resolve).
"""
import os
from os.path import join, isdir
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from vae_p1 import VAE_P1
from lstm import LatentPredictor
from loader_control import load_norm_stats, precompute_latents, LatentSequenceDataset

from paths import NORM_STATS as DEFAULT_NORM_STATS, P1_VAE, outputs

# ===========================================================================
# CONFIG — local paths (M4 Max). Everything env-overridable.
# ===========================================================================
CONTROL_ROOT = os.environ.get("CONTROL_ROOT", os.path.expanduser("~/lunarlander_control_data"))
ELITE_ROOT = os.environ.get("ELITE_ROOT",
                            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "lunarlander_elite_recovery_4000"))
DATA_ROOTS = [CONTROL_ROOT, ELITE_ROOT]            # each with train/val/test

VAE_CKPT = os.environ.get("VAE_CKPT", P1_VAE)          # THE SAME trained P1 VAE
NORM_STATS = os.environ.get("NORM_STATS", DEFAULT_NORM_STATS)   # the ORIGINAL ones (the VAE's)
# output base: OUTPUT_DIR from config.py (override with OUT_ROOT/LATENT_ROOT/SAVE_DIR)
_OUT = os.environ.get("OUT_ROOT", outputs())
LATENT_ROOT = os.environ.get("LATENT_ROOT", join(_OUT, "lunarlander_p1_control_latents"))
SAVE_DIR = os.environ.get("SAVE_DIR", join(_OUT, "lunarlander_p1_control_lstm"))

WIND_FILTER = os.environ.get("WIND_FILTER", "all")  # 'all' | 'clean' | 'wind'

LATENT_SIZE, N_SUP, N_IMG = 64, 8, 56
N_ACTIONS = 4
SHIFT = 0

SEQ_LEN = 10                       # MPC-oriented: a short horizon (the MPC asks ~K steps ahead)
STRIDE = 3                         # shorter windows -> more samples per episode
BATCH = 256                        # small latents + SEQ_LEN=10 -> a large batch (throughput, especially on MPS)
HIDDEN = 64
LAYERS = 2

EPOCHS = 150
LR = 1e-3                          # fixed; ReduceLROnPlateau lowers it at a plateau
WEIGHT_DECAY = 1e-5                # mild AdamW regularization (more data/epochs)
CLIP = 1.0
W_PHYS = 1.0

# Scheduled sampling -> FULLY free-running at the end (P_END=0) for the MPC. Slopes ∝ EPOCHS:
P_START, P_END, P_DECAY_EPOCHS = 1.0, 0.0, 120    # p_tf 1.0->0.0 linearly over the first ~80% of epochs
L_START, CURRICULUM_EPOCHS = 3, 45                # horizon 3 -> SEQ_LEN(10) over the first ~30% of epochs

EARLY_STOP_PATIENCE = 10
SCHED_PATIENCE = 5                 # < EARLY_STOP -> allows 1-2 LR drops before stopping
MIN_LR = 1e-5
NUM_WORKERS = 0                    # locally on the Mac: spawn would re-pickle the eager dataset -> 0 (data is already in RAM)
SEED = 0
DO_PRECOMPUTE = True


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_vae(device):
    vae = VAE_P1(n_sup=N_SUP, n_img=N_IMG).to(device)
    vae.load_state_dict(torch.load(VAE_CKPT, map_location=device)); vae.eval()
    return vae


def encode_fn(model, device):
    @torch.no_grad()
    def _fn(img_t, img_tp1):
        model.eval()
        x = torch.cat([img_t, img_tp1], dim=1).to(device)
        mu, _ = model.encode(x)
        return mu
    return _fn


# ===========================================================================
# Rollout — ENCODED (seed/target = VAE latent; teacher forcing via scheduled sampling)
# ===========================================================================
def rollout(model, batch, p_tf, free_running=False, max_len=None):
    z_t, action, z_tp1, state_t, state_tp1 = batch
    L = z_t.shape[1] if max_len is None else min(max_len, z_t.shape[1])
    B = z_t.shape[0]; device = z_t.device
    z_in = z_t[:, 0]
    z_gt = z_tp1[:, :L]
    hidden = model.init_hidden(B, device)
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
    return torch.stack(preds, dim=1), z_gt, state_tp1[:, :L]


def train_epoch(model, loader, optimizer, device, p_tf, cur_len, desc=""):
    model.train()
    tot, n = 0.0, 0
    pbar = tqdm(loader, desc=desc, leave=False)
    for batch in pbar:
        batch = [b.to(device, non_blocking=True) for b in batch]
        preds, z_gt, _ = rollout(model, batch, p_tf, free_running=False, max_len=cur_len)
        loss = (F.mse_loss(preds, z_gt, reduction="mean")
                + W_PHYS * F.mse_loss(preds[..., :N_SUP], z_gt[..., :N_SUP], reduction="mean"))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()
        bs = preds.size(0)
        tot += loss.item() * bs; n += bs
        pbar.set_postfix(loss=f"{tot/n:.5f}", p_tf=f"{p_tf:.2f}", H=cur_len)
    return tot / n


@torch.no_grad()
def eval_epoch(model, loader, device, std_phys, desc=""):
    """ FREE-RUNNING at the FULL SEQ_LEN -> physical MSE per horizon vs the CLEAN state. """
    model.eval()
    se, n = None, 0
    for batch in tqdm(loader, desc=desc, leave=False):
        batch = [b.to(device, non_blocking=True) for b in batch]
        preds, _, state_tp1 = rollout(model, batch, 0.0, free_running=True, max_len=None)
        err = (preds[..., :N_SUP] - state_tp1) * std_phys
        s = (err ** 2).sum(dim=0)
        se = s if se is None else se + s
        n += preds.size(0)
    return (se / n).mean(dim=1).cpu().numpy()


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    set_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = get_device()
    print("device:", device, "| ENCODED P1-control LSTM | wind_filter:", WIND_FILTER)
    print("data roots:", DATA_ROOTS)

    mean, std = load_norm_stats(NORM_STATS)
    std_phys = torch.tensor(std[:N_SUP], device=device)

    if DO_PRECOMPUTE:
        vae = build_vae(device)
        enc = encode_fn(vae, device)
        for split in ("train", "val", "test"):
            roots_split = [join(r, split) for r in DATA_ROOTS if isdir(join(r, split))]
            if roots_split:
                print(f"pre-encoding '{split}' from {roots_split} ...")
                precompute_latents(enc, roots_split, join(LATENT_ROOT, split),
                                   shift=SHIFT, device=device, wind_filter=WIND_FILTER)
        del vae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    train_ds = LatentSequenceDataset(join(LATENT_ROOT, "train"),
                                     seq_len=SEQ_LEN, stride=STRIDE, state_mean=mean, state_std=std)
    val_ds = LatentSequenceDataset(join(LATENT_ROOT, "val"),
                                   seq_len=SEQ_LEN, stride=STRIDE, state_mean=mean, state_std=std)
    pin = device.type == "cuda"; pw = NUM_WORKERS > 0
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=True,
                          num_workers=NUM_WORKERS, pin_memory=pin, persistent_workers=pw)
    val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=pin, persistent_workers=pw)
    print(f"train windows: {len(train_ds)} | val windows: {len(val_ds)}")

    model = LatentPredictor(LATENT_SIZE, N_ACTIONS, HIDDEN, LAYERS).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5,
                                                     patience=SCHED_PATIENCE, min_lr=MIN_LR)

    best, bad = float("inf"), 0
    for epoch in range(1, EPOCHS + 1):
        p_tf = max(P_END, P_START - (P_START - P_END) * (epoch - 1) / max(P_DECAY_EPOCHS, 1))
        cur_len = int(round(min(SEQ_LEN, L_START + (SEQ_LEN - L_START) * (epoch - 1) / max(CURRICULUM_EPOCHS, 1))))

        tr = train_epoch(model, train_dl, optimizer, device, p_tf, cur_len, desc=f"E{epoch:03d} train")
        mse_h = eval_epoch(model, val_dl, device, std_phys, desc=f"E{epoch:03d} val")
        val_mean = float(mse_h.mean())
        scheduler.step(val_mean)
        lr_now = optimizer.param_groups[0]["lr"]

        h = {hh: mse_h[hh - 1] for hh in (1, 10, 20, SEQ_LEN) if hh <= SEQ_LEN}
        h_str = "  ".join(f"h{k}={v:.4f}" for k, v in h.items())
        print(f"E{epoch:03d} [p1-control] | p_tf={p_tf:.2f} H_train={cur_len} lr={lr_now:.1e} | "
              f"train={tr:.5f} | val phys-MSE mean={val_mean:.4f} | {h_str}")

        if val_mean < best - 1e-6:
            best, bad = val_mean, 0
            torch.save(model.state_dict(), join(SAVE_DIR, "lstm_p1_control_best.pth"))
            np.save(join(SAVE_DIR, "val_mse_per_horizon.npy"), mse_h)
            print("  -> best model saved")
        else:
            bad += 1
            if bad >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch}."); break

    torch.save(model.state_dict(), join(SAVE_DIR, "lstm_p1_control_last.pth"))
    print("Best val phys-MSE:", best)
