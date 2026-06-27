"""
LunarLstmBaseline_alt.py — ENCODED-mode LSTM για το WEAK-SUP baseline VAE (ΤΟΠΙΚΑ / MPS).

ENCODED (όχι hybrid): seed/target = VAE latent, ΚΑΜΙΑ GT injection -> η ποιότητα της
αναπαράστασης του VAE προπαγανδίζεται στον ορίζοντα -> φαίνεται το gap baseline vs P1/P2/P3
(paper Fig 3A, end-to-end). eval: free-run, preds[:,:8] vs CLEAN state (η μόνη χρήση clean).

Self-contained: VAE κλάση από το LunarVaeBaseline_alt· LatentPredictor inline· import μόνο
από LunarLoader. MPS auto-detect, num_workers=0, paths κάτω από ~/lunar_local_runs.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from LunarVaePrinciple3_weak_clean import VAE_P3, encode_fn
from LunarLoader import precompute_latents, LatentSequenceDataset, load_norm_stats

# ---------------------------------------------------------------------------
# CONFIG  (ΤΟΠΙΚΑ paths)
# ---------------------------------------------------------------------------
KEY = "p3_weak"
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lunarlander_data")
RUNS = os.path.expanduser("~/lunar_local_runs")
LATENT_ROOT = os.path.join(RUNS, f"latents_{KEY}_clean")
NORM_STATS = os.path.join(DATA_ROOT, "norm_stats.npz")
VAE_CKPT = os.path.join(RUNS, f"vae_{KEY}_clean", "vae_best.pth")
SAVE_DIR = os.path.join(RUNS, f"lstm_{KEY}_enc_tr")

LATENT_SIZE = 64
N_SUP = 8
N_ACTIONS = 4
SHIFT = 0                  # encoded mode ΔΕΝ χρησιμοποιεί το x (το latent είναι ήδη weak-sup)

SEQ_LEN = 30
STRIDE = 5
BATCH = 64
HIDDEN = 64
LAYERS = 2

EPOCHS = 50
LR = 1e-3
CLIP = 1.0
W_PHYS = 1.0               # επιπλέον βάρος στα N_SUP physical dims του latent target

P_START, P_END, P_DECAY_EPOCHS = 1.0, 0.3, 40     # scheduled sampling
L_START, CURRICULUM_EPOCHS = 5, 15                # horizon curriculum

EARLY_STOP_PATIENCE = 5
SCHED_PATIENCE = 4
NUM_WORKERS = 0            # Mac/MPS
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


# ---------------------------------------------------------------------------
# Model — residual latent predictor (zero-init -> ξεκινά identity)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Rollout — ENCODED (seed/target = VAE latent· ΚΑΜΙΑ GT injection)
# ---------------------------------------------------------------------------
def rollout(model, batch, p_tf, n_actions, free_running=False, max_len=None):
    z_t, action, z_tp1, state_t, state_tp1 = batch
    L = z_t.shape[1] if max_len is None else min(max_len, z_t.shape[1])
    B = z_t.shape[0]
    device = z_t.device

    z_in = z_t[:, 0]                          # ENCODED seed: latent του VAE (όχι GT)
    z_gt = z_tp1[:, :L]                       # target: latent trajectory

    hidden = model.init_hidden(B, device)
    preds = []
    for k in range(L):
        a = F.one_hot(action[:, k].long(), n_actions).float()
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
        preds, z_gt, _ = rollout(model, batch, p_tf, N_ACTIONS, free_running=False, max_len=cur_len)
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
    """ FREE-RUNNING στο ΠΛΗΡΕΣ SEQ_LEN -> physical MSE ανά ορίζοντα, vs CLEAN state. """
    model.eval()
    se, n = None, 0
    for batch in tqdm(loader, desc=desc, leave=False):
        batch = [b.to(device, non_blocking=True) for b in batch]
        preds, _, state_tp1 = rollout(model, batch, 0.0, N_ACTIONS, free_running=True, max_len=None)
        err = (preds[..., :N_SUP] - state_tp1) * std_phys
        s = (err ** 2).sum(dim=0)
        se = s if se is None else se + s
        n += preds.size(0)
    return (se / n).mean(dim=1).cpu().numpy()


def build_vae(device):
    """ Φορτώνει τον παγωμένο weak-sup VAE (override σε P1/P2/P3 variants). """
    vae = VAE_P3(latent_size=LATENT_SIZE).to(device)
    vae.load_state_dict(torch.load(VAE_CKPT, map_location=device)); vae.eval()
    return vae


if __name__ == "__main__":
    set_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = get_device()
    print("device:", device, f"| ENCODED-mode LSTM | KEY={KEY}")

    mean, std = load_norm_stats(NORM_STATS)
    std_phys = torch.tensor(std[:N_SUP], device=device)

    if DO_PRECOMPUTE:
        vae = build_vae(device)
        enc = encode_fn(vae, device)
        for split in ("train", "val", "test"):
            src = os.path.join(DATA_ROOT, split)
            if os.path.isdir(src):
                print(f"pre-encoding {split} ...")
                precompute_latents(enc, src, os.path.join(LATENT_ROOT, split), shift=SHIFT, device=device)
        del vae

    train_ds = LatentSequenceDataset(os.path.join(LATENT_ROOT, "train"),
                                     seq_len=SEQ_LEN, stride=STRIDE, state_mean=mean, state_std=std)
    val_ds = LatentSequenceDataset(os.path.join(LATENT_ROOT, "val"),
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
        print(f"E{epoch:03d} | p_tf={p_tf:.2f} H_train={cur_len} lr={lr_now:.1e} | "
              f"train={tr:.5f} | val phys-MSE mean={val_mean:.4f} | {h_str}")

        if val_mean < best - 1e-6:
            best, bad = val_mean, 0
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "lstm_best.pth"))
            np.save(os.path.join(SAVE_DIR, "val_mse_per_horizon.npy"), mse_h)
            print("  -> best model saved")
        else:
            bad += 1
            if bad >= EARLY_STOP_PATIENCE:
                print(f"Early stopping στο epoch {epoch}."); break

    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "lstm_last.pth"))
    print("Best val phys-MSE:", best)
