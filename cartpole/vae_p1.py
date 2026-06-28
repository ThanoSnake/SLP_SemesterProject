"""
vae_p1.py — Principle 1 (Structured latent) VAE για CartPole (notebook-ready, Kaggle).

Διαφορά από το baseline vae.py:
  * ΔΥΟ ΑΝΕΞΑΡΤΗΤΟΙ encoders (διαφορετικά βάρη) με ΔΙΑΦΟΡΕΤΙΚΗ είσοδο:
        - state_encoder : 6 καν. = stack(frame_t, frame_t+1) -> N_SUP   (x, x_dot, theta, theta_dot)
                          (χρειάζεται 2 frames για να βγάλει ταχύτητα)
        - image_encoder : 3 καν. = ΜΟΝΟ frame_t            -> N_IMG   (low-level visual features)
                          (1 frame -> δεν μπορεί να κωδικοποιήσει ταχύτητα: τα image dims
                           αποσυζεύγνυνται από τη δυναμική/ταχύτητα)
    Τα δύο κομμάτια συνενώνονται -> latent (N_SUP + N_IMG = LATENT_SIZE).
    (Paper, Principle 1: "split the encoder into the image part ... and the state part".)
  * ΚΟΙΝΟΣ decoder: παίρνει ΟΛΟΚΛΗΡΟ το latent (64) -> reconstruct frame_t.
  * Losses (ίδια φιλοσοφία με baseline):
        - reconstruction (MSE) μετά τον κοινό decoder
        - physical supervision (MSE) στα πρώτα N_SUP dims (mu[:, :N_SUP] ~ state_t)
        - SPLIT-β KL (ίδιο με baseline): σταθερό μικρό BETA_PHYS στα 4 phys dims,
          annealed beta_style 0->1 στα 60 image dims (ώστε το KL να μη σπρώχνει τα
          φυσικά μεγέθη προς N(0,1)· τα image dims καταρρέουν -> καθαρά latents για LSTM).
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

#
#  Config
#
DATA_ROOT = "<cartpole-dataset>"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR = os.path.join(DATA_ROOT, "val")
NORM_STATS = os.path.join(DATA_ROOT, "norm_stats.npz")
SAVE_DIR = "/kaggle/working/cartpole_p1_vae"

LATENT_SIZE = 64
N_SUP = 4                    # [x, x_dot, theta, theta_dot] -> state_encoder
N_IMG = LATENT_SIZE - N_SUP  # 60 -> image_encoder
SHIFT = 0                    # 0=clean, 2/5/10=noisy (weak supervision)

BATCH = 128
EPOCHS = 40
LR = 1e-3

# Split-beta KL (identical to baseline vae): tiny fixed pressure on physical dims,
# annealed pressure on image dims
BETA_PHYS = 0.01
BETA_STYLE_MAX = 1.0
KL_ANNEAL_EPOCHS = 20

LAMBDA_SUP = 1.0   # per-element mean keeps this an O(1) knob

EARLY_STOP_PATIENCE = 5
SCHED_PATIENCE = 3  # below early-stop so the LR drops before we give up

# Dataset is fully eager-loaded in __init__. num_workers>0 only parallelizes the
# /255+permute; data is shared via fork copy-on-write (no RAM duplication).
NUM_WORKERS = 2
SEED = 0


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


#
#  Model — two independent encoders + a shared decoder
#
class VAE_P1(nn.Module):
    """encode(x) input: x = stack(frame_t, frame_t+1) (B,6,80,120). Reconstructs frame_t (B,3,80,120).

    latent = [ z_state (N_SUP) | z_image (N_IMG) ], from TWO separate encoders:
      * state_encoder : sees BOTH frames (6 channels) -> can extract velocity.
      * image_encoder : sees ONLY frame_t (3 channels) -> architecturally cannot encode
        dynamics/velocity in the image dims (decouples static from dynamic).
    """

    @staticmethod
    def _conv_stack(in_channels):
        """Same architecture as the baseline encoder: 80x120 -> (64,10,15)."""
        return nn.Sequential(
            nn.Conv2d(in_channels, 16, 4, 2, 1), nn.ReLU(inplace=True),   # 80x120 -> 40x60
            nn.Conv2d(16, 32, 4, 2, 1), nn.ReLU(inplace=True),           # -> 20x30
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(inplace=True),           # -> 10x15
        )

    def __init__(self, n_sup=4, n_img=60, frame_channels=3, out_channels=3):
        super().__init__()
        self.n_sup = n_sup
        self.n_img = n_img
        self.frame_channels = frame_channels   # channels of ONE frame (RGB=3)
        self.latent_size = n_sup + n_img
        feat = 64 * 10 * 15

        # Two independent conv stacks (different weights, different input channel counts)
        self.state_encoder = self._conv_stack(2 * frame_channels)   # 6: stack(frame_t, frame_t+1)
        self.image_encoder = self._conv_stack(frame_channels)       # 3: only frame_t

        # Separate mu/logvar heads per branch
        self.fc_mu_s = nn.Linear(feat, n_sup)
        self.fc_logvar_s = nn.Linear(feat, n_sup)
        self.fc_mu_i = nn.Linear(feat, n_img)
        self.fc_logvar_i = nn.Linear(feat, n_img)

        # Shared decoder: takes the FULL latent -> image
        self.fc_decode = nn.Linear(self.latent_size, feat)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(inplace=True),  # 10x15 -> 20x30
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(inplace=True),  # -> 40x60
            nn.ConvTranspose2d(16, out_channels, 4, 2, 1), nn.Sigmoid(), # -> 80x120, [0,1]
        )

    def encode(self, x):
        """x = stack(frame_t, frame_t+1) (B,6,H,W). First N_SUP dims come from the state_encoder
        (both frames); the N_IMG dims from the image_encoder on frame_t only (first 3 channels)."""
        hs = self.state_encoder(x).flatten(1)                          # 6 channels: frame_t ++ frame_t+1
        hi = self.image_encoder(x[:, :self.frame_channels]).flatten(1) # 3 channels: only frame_t
        mu = torch.cat([self.fc_mu_s(hs), self.fc_mu_i(hi)], dim=1)
        logvar = torch.cat([self.fc_logvar_s(hs), self.fc_logvar_i(hi)], dim=1)
        return mu, logvar

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
    """Callable for loader.precompute_latents: (img_t, img_tp1) -> mu (deterministic)."""
    @torch.no_grad()
    def _fn(img_t, img_tp1):
        model.eval()
        x = torch.cat([img_t, img_tp1], dim=1).to(device)
        mu, _ = model.encode(x)
        return mu
    return _fn


#
#  Loss — per-element means (~O(1)). KL split into physical vs image dims.
#
def vae_losses(recon, target, mu, logvar, state_t, n_sup):
    recon_l = F.mse_loss(recon, target, reduction="mean")       # per pixel
    sup = F.mse_loss(mu[:, :n_sup], state_t, reduction="mean")  # per supervised dim

    kld_elem = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())   # (B, latent)
    b = mu.size(0)
    n_img = mu.size(1) - n_sup
    kld_phys = kld_elem[:, :n_sup].sum() / b / n_sup            # per physical dim
    kld_img = kld_elem[:, n_sup:].sum() / b / max(n_img, 1)     # per image dim
    return recon_l, kld_phys, kld_img, sup


#
#  Train / Eval (with progress bar)
#
def run_epoch(model, loader, device, beta_style, optimizer=None, desc=""):
    train = optimizer is not None
    model.train() if train else model.eval()
    tot = {"recon": 0.0, "kld_phys": 0.0, "kld_img": 0.0, "sup": 0.0, "n": 0}

    pbar = tqdm(loader, desc=desc, leave=False)
    for img_t, img_tp1, action, state_t, state_tp1 in pbar:
        # Transfer AS uint8 (small payload), then .float()/255 on the GPU.
        img_t = img_t.to(device, non_blocking=True).float() / 255.0
        img_tp1 = img_tp1.to(device, non_blocking=True).float() / 255.0
        x = torch.cat([img_t, img_tp1], dim=1)
        target = img_t
        st = state_t.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            recon, mu, logvar = model(x)
            r, kp, ki, s = vae_losses(recon, target, mu, logvar, st, N_SUP)
            # Split-beta: fixed small BETA_PHYS on phys dims, annealed beta_style on image dims
            loss = r + BETA_PHYS * kp + beta_style * ki + LAMBDA_SUP * s

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        bs = target.size(0)
        tot["recon"] += r.item() * bs
        tot["kld_phys"] += kp.item() * bs
        tot["kld_img"] += ki.item() * bs
        tot["sup"] += s.item() * bs
        tot["n"] += bs

        n = tot["n"]
        cur_total = (tot["recon"] + BETA_PHYS * tot["kld_phys"]
                     + beta_style * tot["kld_img"] + LAMBDA_SUP * tot["sup"]) / n
        pbar.set_postfix(total=f"{cur_total:.4f}",
                         recon=f"{tot['recon'] / n:.4f}",
                         sup=f"{tot['sup'] / n:.4f}")

    return {key: tot[key] / tot["n"] for key in ("recon", "kld_phys", "kld_img", "sup")}


@torch.no_grad()
def physical_rmse(model, loader, device, std4):
    """RMSE of the supervised dims in PHYSICAL units (undoes the standardization)."""
    model.eval()
    se = torch.zeros(N_SUP, device=device)
    n = 0
    for img_t, img_tp1, action, state_t, state_tp1 in loader:
        # Same uint8 -> float/255 conversion as run_epoch (keep the input consistent!)
        img_t = img_t.to(device).float() / 255.0
        img_tp1 = img_tp1.to(device).float() / 255.0
        x = torch.cat([img_t, img_tp1], dim=1)
        st = state_t.to(device)
        mu, _ = model.encode(x)
        se += (((mu[:, :N_SUP] - st) * std4) ** 2).sum(0)
        n += st.size(0)
    return torch.sqrt(se / n).cpu().numpy()

"""
#
#  Main
#
if __name__ == "__main__":
    set_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, "  (if 'cpu' -> enable GPU on Kaggle!)")

    mean, std = load_norm_stats(NORM_STATS)
    std4 = torch.tensor(std[:N_SUP], device=device)

    # Datasets load EVERYTHING into RAM in __init__ (with their own progress bar).
    train_ds = VaePairDataset(TRAIN_DIR, shift=SHIFT, state_mean=mean, state_std=std)
    val_ds = VaePairDataset(VAL_DIR, shift=SHIFT, state_mean=mean, state_std=std)

    pw = NUM_WORKERS > 0
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=True, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=pw)
    val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=pw)
    print(f"train pairs: {len(train_ds)} | val pairs: {len(val_ds)}")

    model = VAE_P1(n_sup=N_SUP, n_img=N_IMG).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=SCHED_PATIENCE)

    best_val = float("inf")
    epochs_no_improve = 0
    for epoch in range(1, EPOCHS + 1):
        beta_style = BETA_STYLE_MAX * min(1.0, epoch / max(KL_ANNEAL_EPOCHS, 1))

        tr = run_epoch(model, train_dl, device, beta_style, optimizer, desc=f"E{epoch:03d} train")
        va = run_epoch(model, val_dl, device, beta_style, optimizer=None, desc=f"E{epoch:03d} val")
        rmse = physical_rmse(model, val_dl, device, std4)

        def total(m):  # weighted loss (the one we backprop)
            return (m["recon"] + BETA_PHYS * m["kld_phys"] + beta_style * m["kld_img"] + LAMBDA_SUP * m["sup"])

        tr_total = total(tr)
        va_total = total(va)
        val_score = va["recon"] + LAMBDA_SUP * va["sup"]   # selection (beta-independent, stable)
        scheduler.step(val_score)
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"E{epoch:03d} | beta_style={beta_style:.2f} beta_phys={BETA_PHYS} lr={lr_now:.1e}")
        print(f"  TRAIN total={tr_total:.5f} | recon={tr['recon']:.5f}  "
              f"kld_img={tr['kld_img']:.5f}  kld_phys={tr['kld_phys']:.5f}  sup={tr['sup']:.5f}")
        print(f"  VAL   total={va_total:.5f} | recon={va['recon']:.5f}  "
              f"kld_img={va['kld_img']:.5f}  kld_phys={va['kld_phys']:.5f}  sup={va['sup']:.5f}  "
              f"(select={val_score:.5f})")
        print(f"  VAL   phys RMSE | x={rmse[0]:.4f}  x_dot={rmse[1]:.4f}  "
              f"theta={rmse[2]:.4f}  theta_dot={rmse[3]:.4f}")

        if val_score < best_val - 1e-6:
            best_val = val_score
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "vae_p1_best.pth"))
            print("  -> best model saved")
        else:
            epochs_no_improve += 1
            print(f"  (no improvement: {epochs_no_improve}/{EARLY_STOP_PATIENCE})")
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch}.")
                break

    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "vae_p1_last.pth"))
    print("Best val score:", best_val)
 """
