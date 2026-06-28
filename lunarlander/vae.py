"""
vae.py — Physically-interpretable VAE για LunarLander (baseline, notebook-ready).
Port του cart_pole/vae_baseline.py· state 4D -> 8D.

Χαρακτηριστικά:
  * ΕΙΣΟΔΟΣ 2 διαδοχικά frames (stack -> 6 κανάλια) -> κωδικοποίηση ταχυτήτων.
  * SPLIT-β KL: μικρό σταθερό BETA_PHYS στις 8 φυσικές dims, beta_style annealed 0->1
    στις υπόλοιπες 56.
  * Supervised alignment των 8 πρώτων dims με standardized φυσικά states
    ([x, y, vx, vy, theta, omega, leg1, leg2]).
  * Per-element mean losses· ξεχωριστά components στο log.
  * Validation με physical RMSE ανά μέγεθος + best-model saving + EARLY STOPPING.

  ΒΕΛΤΙΩΣΗ ΧΡΟΝΟΥ: ο loader επιστρέφει uint8 εικόνες· η μεταφορά στη GPU γίνεται σε
  uint8 (4× λιγότερα bytes στο PCIe) και το .float()/255 γίνεται ΣΤΗ GPU (όχι στη CPU)
  -> εξαφανίζει το CPU bottleneck της κανονικοποίησης.

Σε notebook: τρέξε ΠΡΩΤΑ το cell του LunarLoader.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from loader import VaePairDataset, load_norm_stats

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATA_ROOT = "<lunarlander-dataset>"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR = os.path.join(DATA_ROOT, "val")
NORM_STATS = os.path.join(DATA_ROOT, "norm_stats.npz")
SAVE_DIR = "/kaggle/working/lunarlander_baseline_vae"

STATE_NAMES = ("x", "y", "vx", "vy", "theta", "omega", "leg1", "leg2")

LATENT_SIZE = 64
N_SUP = 8                  # [x, y, vx, vy, theta, omega, leg1, leg2]
SHIFT = 0                  # 0=clean· 2/5/10 -> noisy (weak supervision)

BATCH = 128
EPOCHS = 40
LR = 1e-3

# --- SPLIT-β KL ---
BETA_PHYS = 0.01           # σταθερό, ΜΙΚΡΟ: ελάχιστη πίεση στις φυσικές dims
BETA_STYLE_MAX = 1.0       # τελικό βάρος KL στις style dims
KL_ANNEAL_EPOCHS = 20      # beta_style: 0 -> BETA_STYLE_MAX

LAMBDA_SUP = 1.0           # per-element mean -> O(1) knob

EARLY_STOP_PATIENCE = 5
SCHED_PATIENCE = 3         # < EARLY_STOP -> προλαβαίνει να πέσει το LR πρώτα

NUM_WORKERS = 2
SEED = 0


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def _to_img(t, device):
    """ uint8 (B,3,H,W) -> float [0,1] στη GPU (η μεταφορά γίνεται σε uint8). """
    return t.to(device, non_blocking=True).float().div_(255.0)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class VAE(nn.Module):
    """ Είσοδος (B,6,80,120)=stack(frame_t,frame_t+1)· ανακατασκευή (B,3,80,120)=frame_t. """

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

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        h = self.fc_decode(z).view(-1, 64, 10, 15)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def encode_fn(model, device):
    """ Callable για LunarLoader.precompute_latents — δέχεται float [0,1] εικόνες. """
    @torch.no_grad()
    def _fn(img_t, img_tp1):
        model.eval()
        x = torch.cat([img_t, img_tp1], dim=1).to(device)
        mu, _ = model.encode(x)
        return mu
    return _fn


# ---------------------------------------------------------------------------
# Loss — per-element means· SPLIT-β KL (phys vs style)
# ---------------------------------------------------------------------------
def vae_losses(recon, target, mu, logvar, state_t, n_sup):
    B, D = mu.size(0), mu.size(1)
    recon_l = F.mse_loss(recon, target, reduction="mean")               # ανά pixel
    sup = F.mse_loss(mu[:, :n_sup], state_t, reduction="mean")          # ανά supervised dim

    kl_per = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())             # (B, D) ανά διάσταση
    kld_phys = kl_per[:, :n_sup].sum() / B / n_sup
    kld_style = kl_per[:, n_sup:].sum() / B / (D - n_sup)
    return recon_l, kld_phys, kld_style, sup


# ---------------------------------------------------------------------------
# Train / Eval
# ---------------------------------------------------------------------------
def run_epoch(model, loader, device, beta_style, optimizer=None, desc=""):
    train = optimizer is not None
    model.train() if train else model.eval()
    tot = {"recon": 0.0, "kld_phys": 0.0, "kld_style": 0.0, "sup": 0.0, "n": 0}

    pbar = tqdm(loader, desc=desc, leave=False)
    for img_t, img_tp1, action, state_t, state_tp1 in pbar:
        img_t = _to_img(img_t, device)                       # uint8 -> float/255 ΣΤΗ GPU
        img_tp1 = _to_img(img_tp1, device)
        x = torch.cat([img_t, img_tp1], dim=1)               # (B,6,H,W)
        target = img_t                                       # ανακατασκευή του frame_t
        st = state_t.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            recon, mu, logvar = model(x)
            r, kp, ks, s = vae_losses(recon, target, mu, logvar, st, N_SUP)
            loss = r + BETA_PHYS * kp + beta_style * ks + LAMBDA_SUP * s

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        bs = target.size(0)
        tot["recon"] += r.item() * bs
        tot["kld_phys"] += kp.item() * bs
        tot["kld_style"] += ks.item() * bs
        tot["sup"] += s.item() * bs
        tot["n"] += bs

        n = tot["n"]
        cur = (tot["recon"] + BETA_PHYS * tot["kld_phys"]
               + beta_style * tot["kld_style"] + LAMBDA_SUP * tot["sup"]) / n
        pbar.set_postfix(total=f"{cur:.4f}", recon=f"{tot['recon']/n:.4f}", sup=f"{tot['sup']/n:.4f}")

    return {k: tot[k] / tot["n"] for k in ("recon", "kld_phys", "kld_style", "sup")}


@torch.no_grad()
def physical_rmse(model, loader, device, std_phys):
    model.eval()
    se = torch.zeros(N_SUP, device=device)
    n = 0
    for img_t, img_tp1, action, state_t, state_tp1 in loader:
        x = torch.cat([_to_img(img_t, device), _to_img(img_tp1, device)], dim=1)
        st = state_t.to(device)
        mu, _ = model.encode(x)
        se += (((mu[:, :N_SUP] - st) * std_phys) ** 2).sum(0)
        n += st.size(0)
    return torch.sqrt(se / n).cpu().numpy()

'''
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    set_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, "  (αν 'cpu' -> ενεργοποίησε GPU στην Kaggle!)")

    mean, std = load_norm_stats(NORM_STATS)
    std_phys = torch.tensor(std[:N_SUP], device=device)

    train_ds = VaePairDataset(TRAIN_DIR, shift=SHIFT, state_mean=mean, state_std=std)
    val_ds = VaePairDataset(VAL_DIR, shift=SHIFT, state_mean=mean, state_std=std)
    pw = NUM_WORKERS > 0
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=True,
                          num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=pw)
    val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=pw)
    print(f"train pairs: {len(train_ds)} | val pairs: {len(val_ds)}")

    model = VAE(latent_size=LATENT_SIZE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=SCHED_PATIENCE)

    best_val, bad_epochs = float("inf"), 0
    for epoch in range(1, EPOCHS + 1):
        beta_style = BETA_STYLE_MAX * min(1.0, epoch / max(KL_ANNEAL_EPOCHS, 1))

        tr = run_epoch(model, train_dl, device, beta_style, optimizer, desc=f"E{epoch:03d} train")
        va = run_epoch(model, val_dl, device, beta_style, optimizer=None, desc=f"E{epoch:03d} val")
        rmse = physical_rmse(model, val_dl, device, std_phys)

        def total(d):
            return d["recon"] + BETA_PHYS * d["kld_phys"] + beta_style * d["kld_style"] + LAMBDA_SUP * d["sup"]
        tr_total, va_total = total(tr), total(va)
        val_score = va["recon"] + LAMBDA_SUP * va["sup"]
        scheduler.step(val_score)
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"E{epoch:03d} | beta_style={beta_style:.2f} beta_phys={BETA_PHYS} lr={lr_now:.1e}")
        print(f"  TRAIN total={tr_total:.5f} | recon={tr['recon']:.5f}  "
              f"kld_phys={tr['kld_phys']:.4f}  kld_style={tr['kld_style']:.4f}  sup={tr['sup']:.5f}")
        print(f"  VAL   total={va_total:.5f} | recon={va['recon']:.5f}  "
              f"kld_phys={va['kld_phys']:.4f}  kld_style={va['kld_style']:.4f}  sup={va['sup']:.5f}  "
              f"(select={val_score:.5f})")
        rmse_str = "  ".join(f"{nm}={rmse[i]:.4f}" for i, nm in enumerate(STATE_NAMES))
        print(f"  VAL   phys RMSE | {rmse_str}")

        if val_score < best_val - 1e-6:
            best_val, bad_epochs = val_score, 0
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "vae_best.pth"))
            print("  -> best model saved")
        else:
            bad_epochs += 1
            print(f"  (no improvement: {bad_epochs}/{EARLY_STOP_PATIENCE})")
            if bad_epochs >= EARLY_STOP_PATIENCE:
                print(f"Early stopping στο epoch {epoch}.")
                break

    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "vae_last.pth"))
    print("Best val score:", best_val)
    '''
