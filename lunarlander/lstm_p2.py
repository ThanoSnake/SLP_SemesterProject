"""
lunar_train_lstm_encoded_p2.py — ENCODED-mode LSTM training (Principle 2) για LunarLander, Kaggle/CUDA.

ΑΥΤΟΝΟΜΟ: VAE_P2, LatentPredictor, loader είναι ΜΕΣΑ στο αρχείο -> κανένα import. Φορτώνεις τον
ΠΑΓΩΜΕΝΟ P2 VAE (clean, χωρίς noisy labels), κωδικοποιείς τα frames σε 64-dim latents, και
εκπαιδεύεις το LSTM:
  * ΕΙΣΟΔΟΣ  : 64 latent dims (z_t) + one-hot action.
  * GROUND TRUTH (target) : το ΕΠΟΜΕΝΟ ENCODED latent z_{t+1} (όχι physical state injection).
  * TEACHER FORCING : scheduled sampling (p_tf: 1.0 -> 0.3). Για ΚΑΘΑΡΟ teacher forcing
    βάλε P_START = P_END = 1.0.

(Ο P2 VAE έχει ΙΔΙΑ αρχιτεκτονική encode με το baseline — ένας encoder· η Αρχή 2 ζει στο loss.)
"""
import os
from os.path import join, basename, isdir
from os import listdir, makedirs
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

# ===========================================================================
# CONFIG  — ΑΛΛΑΞΕ ΤΑ PATHS ΣΤΑ ΔΙΚΑ ΣΟΥ KAGGLE DATASETS
# ===========================================================================
MODEL = "p2"

DATA_ROOT = "/kaggle/input/<lunar-data>/lunarlander_data"     # train/val/test + norm_stats.npz
VAE_DIR   = "/kaggle/input/<vae-without-noise>"               # περιέχει lunar_vae_p2_best.pth
OUT_ROOT  = "/kaggle/working/lstm_img_noise"

VAE_CKPT_NAME = "lunar_vae_p2_best.pth"

LATENT_SIZE, N_SUP, N_IMG = 64, 8, 56
N_ACTIONS = 4
SHIFT = 0                          # clean labels (χωρίς θόρυβο)

SEQ_LEN = 30
STRIDE = 5
BATCH = 64
HIDDEN = 64
LAYERS = 2

EPOCHS = 50
LR = 1e-3
CLIP = 1.0
W_PHYS = 1.0

P_START, P_END, P_DECAY_EPOCHS = 1.0, 0.3, 40     # scheduled sampling (teacher forcing)
L_START, CURRICULUM_EPOCHS = 5, 15                # horizon curriculum

EARLY_STOP_PATIENCE = 5
SCHED_PATIENCE = 4
NUM_WORKERS = 2
SEED = 0
DO_PRECOMPUTE = True

VAE_CKPT = join(VAE_DIR, VAE_CKPT_NAME)
NORM_STATS = join(DATA_ROOT, "norm_stats.npz")
LATENT_ROOT = join(OUT_ROOT, f"latents_{MODEL}")
SAVE_DIR = join(OUT_ROOT, f"lstm_{MODEL}")


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


# ===========================================================================
# === VAE_P2 (inline· single-encoder αρχιτεκτονική, ΙΔΙΑ με baseline) ========
# ===========================================================================
class VAE_P2(nn.Module):
    """ encode(x): x = stack(frame_t, frame_t+1) (B,6,80,120) -> mu (64). """
    def __init__(self, latent_size=64, in_channels=6, out_channels=3):
        super().__init__()
        self.latent_size = latent_size
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(64 * 10 * 15, latent_size)
        self.fc_logvar = nn.Linear(64 * 10 * 15, latent_size)
        self.fc_decode = nn.Linear(latent_size, 64 * 10 * 15)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, out_channels, 4, 2, 1), nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


def build_vae(device):
    vae = VAE_P2(latent_size=LATENT_SIZE).to(device)
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
# === LSTM (residual latent predictor) ======================================
# ===========================================================================
class LatentPredictor(nn.Module):
    def __init__(self, latent=64, action_dim=4, hidden=64, layers=2):
        super().__init__()
        self.hidden, self.layers = hidden, layers
        self.lstm = nn.LSTM(latent + action_dim, hidden, layers, batch_first=True)
        self.fc = nn.Linear(hidden, latent)
        nn.init.zeros_(self.fc.weight); nn.init.zeros_(self.fc.bias)

    def init_hidden(self, b, device):
        return (torch.zeros(self.layers, b, self.hidden, device=device),
                torch.zeros(self.layers, b, self.hidden, device=device))

    def step(self, z, a_onehot, hidden):
        inp = torch.cat([z, a_onehot], dim=-1).unsqueeze(1)
        out, hidden = self.lstm(inp, hidden)
        return z + self.fc(out.squeeze(1)), hidden


# ===========================================================================
# === LOADER (inline) =======================================================
# ===========================================================================
def list_npz(root):
    files = []
    for sd in sorted(listdir(root)):
        p = join(root, sd)
        if isdir(p):
            files += [join(p, f) for f in sorted(listdir(p)) if f.endswith(".npz")]
        elif p.endswith(".npz"):
            files.append(p)
    return sorted(files)


def load_norm_stats(path):
    z = np.load(path)
    return z["mean"].astype(np.float32), z["std"].astype(np.float32)


def _standardize(s, mean, std):
    return s if mean is None else ((s - mean) / std).astype(np.float32)


@torch.no_grad()
def precompute_latents(enc, root, out_root, shift=0, batch=256, device="cuda"):
    makedirs(out_root, exist_ok=True)
    for f in tqdm(list_npz(root), desc="encoding"):
        with np.load(f) as d:
            imgs = torch.from_numpy(d["imgs"].astype(np.float32) / 255.0).permute(0, 3, 1, 2)
            acts = d["acts"].astype(np.float32)
            states = d["states"].astype(np.float32)
            x = (d[f"noisy_states_{shift}"] if shift in (2, 5, 10) else d["states"]).astype(np.float32)
        img_t, img_tp1 = imgs[:-1], imgs[1:]
        zs = []
        for b in range(0, img_t.shape[0], batch):
            zb = enc(img_t[b:b + batch].to(device), img_tp1[b:b + batch].to(device))
            zs.append(zb.cpu().numpy())
        z = np.concatenate(zs, 0).astype(np.float32) if zs else np.empty((0, 0), np.float32)
        np.savez_compressed(join(out_root, basename(f)),
                            z=z, acts=acts[:-1], states=states[:-1], x=x[:-1])


class LatentSequenceDataset(Dataset):
    def __init__(self, root, seq_len=30, stride=1, state_mean=None, state_std=None):
        self.seq_len = seq_len
        self.mean = None if state_mean is None else np.asarray(state_mean, np.float32)
        self.std = None if state_std is None else np.asarray(state_std, np.float32)
        self.eps, self.index = [], []
        for fi, f in enumerate(tqdm(list_npz(root), desc="loading latents -> RAM")):
            with np.load(f) as d:
                ep = {k: d[k].astype(np.float32) for k in ("z", "acts", "states", "x")}
            self.eps.append(ep)
            n = ep["z"].shape[0] - (seq_len + 1) + 1
            for s in range(0, max(n, 0), stride):
                self.index.append((fi, s))
        if not self.index:
            raise RuntimeError(f"Δεν βγήκαν παράθυρα από: {root}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        fi, s = self.index[i]
        ep = self.eps[fi]; L = self.seq_len
        z = ep["z"][s:s + L + 1]
        z_t = torch.from_numpy(z[:-1]); z_tp1 = torch.from_numpy(z[1:])
        action = torch.from_numpy(ep["acts"][s:s + L])
        state_t = torch.from_numpy(_standardize(ep["x"][s:s + L], self.mean, self.std))
        state_tp1 = torch.from_numpy(_standardize(ep["states"][s + 1:s + L + 1], self.mean, self.std))
        return z_t, action, z_tp1, state_t, state_tp1


# ===========================================================================
# Rollout — ENCODED (seed/target = VAE latent· teacher forcing μέσω scheduled sampling)
# ===========================================================================
def rollout(model, batch, p_tf, free_running=False, max_len=None):
    z_t, action, z_tp1, state_t, state_tp1 = batch
    L = z_t.shape[1] if max_len is None else min(max_len, z_t.shape[1])
    B = z_t.shape[0]
    device = z_t.device

    z_in = z_t[:, 0]                          # ENCODED seed
    z_gt = z_tp1[:, :L]                       # ground truth = ΕΠΟΜΕΝΑ encoded latents

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
                use_tf = (torch.rand(B, 1, device=device) < p_tf).float()   # teacher forcing
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
    """ FREE-RUNNING στο ΠΛΗΡΕΣ SEQ_LEN -> physical MSE ανά ορίζοντα vs CLEAN state. """
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


if __name__ == "__main__":
    set_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = get_device()
    print("device:", device, "| ENCODED-mode LSTM | MODEL:", MODEL, "| VAE:", VAE_CKPT)

    mean, std = load_norm_stats(NORM_STATS)
    std_phys = torch.tensor(std[:N_SUP], device=device)

    if DO_PRECOMPUTE:
        vae = build_vae(device)
        enc = encode_fn(vae, device)
        for split in ("train", "val", "test"):
            src = join(DATA_ROOT, split)
            if isdir(src):
                print(f"pre-encoding {split} ...")
                precompute_latents(enc, src, join(LATENT_ROOT, split), shift=SHIFT, device=device)
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
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=SCHED_PATIENCE)

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
        print(f"E{epoch:03d} [{MODEL}] | p_tf={p_tf:.2f} H_train={cur_len} lr={lr_now:.1e} | "
              f"train={tr:.5f} | val phys-MSE mean={val_mean:.4f} | {h_str}")

        if val_mean < best - 1e-6:
            best, bad = val_mean, 0
            torch.save(model.state_dict(), join(SAVE_DIR, f"lstm_{MODEL}_best.pth"))
            np.save(join(SAVE_DIR, "val_mse_per_horizon.npy"), mse_h)
            print("  -> best model saved")
        else:
            bad += 1
            if bad >= EARLY_STOP_PATIENCE:
                print(f"Early stopping στο epoch {epoch}."); break

    torch.save(model.state_dict(), join(SAVE_DIR, f"lstm_{MODEL}_last.pth"))
    print("Best val phys-MSE:", best)
