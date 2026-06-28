"""
vae_p3.py — Principle 3 (Multi-level / multi-strength supervision) VAE για CartPole.

ΣΧΕΔΙΑΣΤΙΚΗ ΑΠΟΦΑΣΗ — "P3 ΜΟΝΟ ΤΟΥ" (απομονωμένο, πάνω στο baseline):
  Όπως το paper (Fig. 3D) συγκρίνει ΚΑΘΕ αρχή ΞΕΧΩΡΙΣΤΑ ως ablation πάνω στο baseline,
  κρατάμε ΑΚΡΙΒΩΣ την αρχιτεκτονική του baseline (ΕΝΑΣ monolithic encoder, 6-κάναλη
  είσοδος stack(frame_t,frame_t+1), 64 latent, SPLIT-β KL) και αλλάζουμε ΜΟΝΟ τον ΣΤΟΧΟ
  ΕΠΟΠΤΕΙΑΣ -> κάθε διαφορά οφείλεται καθαρά στην Αρχή 3. ΔΕΝ δανειζόμαστε τίποτα από
  P1 (split encoder) ή P2 (in/equivariance losses).

ΑΡΧΗ 3 (paper, §3.3): ενσωμάτωσε ΠΟΛΛΑΠΛΕΣ μορφές & ισχείς εποπτείας. Για CartPole, δύο
ρυθμίσεις πάνω στο [x, ẋ, θ, θ̇] (static=[0,2], velocity=[1,3]):

  (1) SEMI : εποπτεύεται ΜΟΝΟ το ΣΤΑΤΙΚΟ (θέση x, γωνία θ)· οι ΤΑΧΥΤΗΤΕΣ άγνωστες
             (dims 1,3 ΑΝΕΠΟΠΤΕΥΤΕΣ).
  (2) WEAK : + εποπτεία ταχύτητας με ΕΚΤΙΜΗΣΗ από φυσική γνώση (finite diff):
             ẋ_est=(x_{t+1}-x_t)/dt , θ̇_est=(θ_{t+1}-θ_t)/dt   (dt=tau=0.02).

ΑΝΑΦΟΡΑ "FULL" (εποπτεία και των 4 dims με ΑΛΗΘΙΝΑ states) = ΤΑΥΤΟΣΗΜΟ με το baseline
(vae_baseline.py) -> χρησιμοποίησε εκείνη την καμπύλη ως reference· εδώ μόνο semi/weak.

ΕΚΤΙΜΗΣΗ ΤΑΧΥΤΗΤΑΣ & STANDARDIZATION: τα states είναι standardized. Το finite diff
υπολογίζεται σε RAW μονάδες και ΞΑΝΑ-standardize-άρεται στις μονάδες της velocity-dim,
ώστε ο στόχος weak να βρίσκεται στον ΙΔΙΟ χώρο με την αληθινή standardized ταχύτητα.

ΣΗΜ. (2-frame): επειδή ο encoder βλέπει 2 frames, η ταχύτητα είναι ΠΑΡΑΤΗΡΗΣΙΜΗ -> η
διαφορά semi vs weak θα είναι ΗΠΙΟΤΕΡΗ απ' ό,τι στο paper (single-frame). Συνειδητή
επιλογή για ΚΟΙΝΟ backbone με baseline/P1/P2.
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
#  Config  (same paths/hyper as baseline -> fair comparison)
#
DATA_ROOT = "<cartpole-dataset>"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR = os.path.join(DATA_ROOT, "val")
NORM_STATS = os.path.join(DATA_ROOT, "norm_stats.npz")

LATENT_SIZE = 64
N_SUP = 4    # [x, x_dot, theta, theta_dot]
SHIFT = 0    # 0=clean, 2/5/10=noisy (weak supervision on the positions)

# Principle 3: supervision setting
SUPERVISION = "weak"       # "semi" (static only) | "weak" (+ estimated velocity)
SAVE_DIR = f"/kaggle/working/cartpole_p3_{SUPERVISION}_vae"
STATIC_DIMS = (0, 2)       # x, theta
VEL_DIMS = (1, 3)          # x_dot, theta_dot
DT = 0.02                  # gym CartPole tau -> used for the velocity estimate

BATCH = 128
EPOCHS = 40
LR = 1e-3

# Split-beta KL (identical to baseline)
BETA_PHYS = 0.01
BETA_STYLE_MAX = 1.0
KL_ANNEAL_EPOCHS = 20

LAMBDA_SUP = 1.0   # per-element mean keeps this an O(1) knob

EARLY_STOP_PATIENCE = 5
SCHED_PATIENCE = 3  # below early-stop so the LR drops before we give up

NUM_WORKERS = 2
SEED = 0


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def _to_img(t, device):
    # Move as uint8 (cheaper transfer), then convert to float [0,1] on device
    return t.to(device, non_blocking=True).float().div_(255.0)


#
#  Model — exactly the baseline VAE (single monolithic encoder); only the sup target changes.
#
class VAE(nn.Module):
    """Input (B,6,80,120)=stack(frame_t,frame_t+1). Reconstructs frame_t (B,3,80,120)."""

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
    """Callable for loader.precompute_latents. Expects float [0,1] images."""
    @torch.no_grad()
    def _fn(img_t, img_tp1):
        model.eval()
        x = torch.cat([img_t, img_tp1], dim=1).to(device)
        mu, _ = model.encode(x)
        return mu
    return _fn


#
#  Principle 3 — supervision target & mask depending on SUPERVISION
#
def _sup_mask(device):
    """Which state dims are supervised: semi -> static; weak -> static + velocity."""
    m = torch.zeros(N_SUP, device=device)
    for d in STATIC_DIMS:
        m[d] = 1.0
    if SUPERVISION == "weak":
        for d in VEL_DIMS:
            m[d] = 1.0
    return m


def _sup_target(state_t, state_tp1, mean_t, std_t):
    """Target (standardized) for the physical dims:
        static = TRUE state_t; velocity (weak only) = finite-diff ESTIMATE.
    The finite diff is computed in RAW units and re-standardized to the velocity-dim:
        Δx_raw = (x_{t+1}-x_t)_std * std[x]      (the mean cancels in the difference)
        ẋ_est_raw = Δx_raw / dt
        ẋ_est_std = (ẋ_est_raw - mean[ẋ]) / std[ẋ]"""
    target = state_t.clone()
    if SUPERVISION == "weak":
        dx_raw = (state_tp1[:, 0] - state_t[:, 0]) * std_t[0]
        dth_raw = (state_tp1[:, 2] - state_t[:, 2]) * std_t[2]
        target[:, 1] = ((dx_raw / DT) - mean_t[1]) / std_t[1]      # ẋ_est_std
        target[:, 3] = ((dth_raw / DT) - mean_t[3]) / std_t[3]     # θ̇_est_std
    return target


#
#  Loss — per-element means; split-beta KL (phys vs style); masked supervision (Principle 3)
#
def vae_losses(recon, img_target, mu, logvar, sup_target, sup_mask, n_sup):
    B, D = mu.size(0), mu.size(1)
    recon_l = F.mse_loss(recon, img_target, reduction="mean")           # per pixel

    # Masked supervision: mean over the supervised dims ONLY (full -> /4 == baseline)
    diff = (mu[:, :n_sup] - sup_target) * sup_mask
    sup = (diff ** 2).sum() / (B * sup_mask.sum())

    kl_per = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())             # (B, D) per dimension
    kld_phys = kl_per[:, :n_sup].sum() / B / n_sup
    kld_style = kl_per[:, n_sup:].sum() / B / (D - n_sup)
    return recon_l, kld_phys, kld_style, sup


#
#  Train / Eval
#
def run_epoch(model, loader, device, beta_style, mean_t, std_t, optimizer=None, desc=""):
    train = optimizer is not None
    model.train() if train else model.eval()
    mask = _sup_mask(device)
    tot = {"recon": 0.0, "kld_phys": 0.0, "kld_style": 0.0, "sup": 0.0, "n": 0}

    pbar = tqdm(loader, desc=desc, leave=False)
    for img_t, img_tp1, action, state_t, state_tp1 in pbar:
        img_t = _to_img(img_t, device)        # uint8 -> float/255 on device
        img_tp1 = _to_img(img_tp1, device)
        x = torch.cat([img_t, img_tp1], dim=1)  # (B,6,H,W)
        img_target = img_t                      # reconstruct frame_t
        st = state_t.to(device, non_blocking=True)
        stp = state_tp1.to(device, non_blocking=True)
        sup_target = _sup_target(st, stp, mean_t, std_t)

        with torch.set_grad_enabled(train):
            recon, mu, logvar = model(x)
            r, kp, ks, s = vae_losses(recon, img_target, mu, logvar, sup_target, mask, N_SUP)
            loss = r + BETA_PHYS * kp + beta_style * ks + LAMBDA_SUP * s

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        bs = img_target.size(0)
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
def physical_rmse(model, loader, device, std4):
    """RMSE of all 4 dims in PHYSICAL units vs TRUE state (diagnostic): shows the
    velocity error even under semi/weak -> why the estimated supervision helps."""
    model.eval()
    se = torch.zeros(N_SUP, device=device)
    n = 0
    for img_t, img_tp1, action, state_t, state_tp1 in loader:
        x = torch.cat([_to_img(img_t, device), _to_img(img_tp1, device)], dim=1)
        st = state_t.to(device)
        mu, _ = model.encode(x)
        se += (((mu[:, :N_SUP] - st) * std4) ** 2).sum(0)
        n += st.size(0)
    return torch.sqrt(se / n).cpu().numpy()


#
#  Main
#
if __name__ == "__main__":
    assert SUPERVISION in ("semi", "weak"), "SUPERVISION in {'semi','weak'} (full == baseline)"
    set_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, " | supervision:", SUPERVISION,
          "  (if 'cpu' -> enable GPU on Kaggle!)")

    mean, std = load_norm_stats(NORM_STATS)
    std4 = torch.tensor(std[:N_SUP], device=device)
    mean_t = torch.tensor(mean, device=device)
    std_t = torch.tensor(std, device=device)

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

        tr = run_epoch(model, train_dl, device, beta_style, mean_t, std_t, optimizer, desc=f"E{epoch:03d} train")
        va = run_epoch(model, val_dl, device, beta_style, mean_t, std_t, optimizer=None, desc=f"E{epoch:03d} val")
        rmse = physical_rmse(model, val_dl, device, std4)

        def total(d):
            return d["recon"] + BETA_PHYS * d["kld_phys"] + beta_style * d["kld_style"] + LAMBDA_SUP * d["sup"]
        tr_total, va_total = total(tr), total(va)
        val_score = va["recon"] + LAMBDA_SUP * va["sup"]    # selection (beta-independent)
        scheduler.step(val_score)
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"E{epoch:03d} [{SUPERVISION}] | beta_style={beta_style:.2f} beta_phys={BETA_PHYS} lr={lr_now:.1e}")
        print(f"  TRAIN total={tr_total:.5f} | recon={tr['recon']:.5f}  "
              f"kld_phys={tr['kld_phys']:.4f}  kld_style={tr['kld_style']:.4f}  sup={tr['sup']:.5f}")
        print(f"  VAL   total={va_total:.5f} | recon={va['recon']:.5f}  "
              f"kld_phys={va['kld_phys']:.4f}  kld_style={va['kld_style']:.4f}  sup={va['sup']:.5f}  "
              f"(select={val_score:.5f})")
        print(f"  VAL   phys RMSE | x={rmse[0]:.4f}  x_dot={rmse[1]:.4f}  "
              f"theta={rmse[2]:.4f}  theta_dot={rmse[3]:.4f}")

        if val_score < best_val - 1e-6:
            best_val, bad_epochs = val_score, 0
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "vae_best.pth"))
            print("  -> best model saved")
        else:
            bad_epochs += 1
            print(f"  (no improvement: {bad_epochs}/{EARLY_STOP_PATIENCE})")
            if bad_epochs >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch}.")
                break

    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "vae_last.pth"))
    print("Best val score:", best_val)
