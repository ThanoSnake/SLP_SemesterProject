"""
LunarVaePrinciple3.py — Principle 3 (Multi-level / multi-strength supervision) VAE
για LunarLander.  Port του cart_pole/vae_principle3.py· state 4D -> 8D· τρέχει σε MPS
(Apple Silicon) ώστε να εκπαιδεύεται ΤΟΠΙΚΑ στο Mac.

ΣΧΕΔΙΑΣΤΙΚΗ ΑΠΟΦΑΣΗ — "P3 ΜΟΝΟ ΤΟΥ" (απομονωμένο, πάνω στο baseline):
  Κρατάμε ΑΚΡΙΒΩΣ την αρχιτεκτονική του baseline (ΕΝΑΣ monolithic encoder, 6-κάναλη
  είσοδος stack(frame_t,frame_t+1), 64 latent, SPLIT-β KL) και αλλάζουμε ΜΟΝΟ τον ΣΤΟΧΟ
  ΕΠΟΠΤΕΙΑΣ -> κάθε διαφορά οφείλεται καθαρά στην Αρχή 3.

obs = [x, y, vx, vy, theta, omega, leg1, leg2]
  STATIC (άμεσα μετρήσιμα)   : x(0), y(1), theta(4), leg1(6), leg2(7)
  VELOCITY (παράγωγα)        : vx(2), vy(3), omega(5)

ΑΡΧΗ 3 (paper, §3.3): πολλαπλές ισχείς εποπτείας:
  (1) SEMI : εποπτεύονται ΜΟΝΟ τα STATIC (θέσεις/γωνία/πόδια)· οι ΤΑΧΥΤΗΤΕΣ άγνωστες.
  (2) WEAK : + εποπτεία ταχύτητας με ΕΚΤΙΜΗΣΗ finite-diff (vx<-Δx, vy<-Δy, omega<-Δtheta).
ΑΝΑΦΟΡΑ "FULL" (όλα τα 8 dims με ΑΛΗΘΙΝΑ states) == baseline -> εκείνη η καμπύλη ως reference.

=====================================================================================
*** ΚΡΙΣΙΜΗ ΔΙΟΡΘΩΣΗ vs CartPole (το πιθανό "λάθος" στο port) ***
  Στο CartPole το gym κάνει explicit Euler -> (x_{t+1}-x_t)/dt = vx_t ΑΚΡΙΒΩΣ, άρα ένα
  σταθερό /DT (=tau=0.02) αρκούσε. ΣΤΟ LUNARLANDER ΟΧΙ: οι ταχύτητες του obs έχουν ΔΙΚΗ
  τους κανονικοποίηση (vel*k/FPS) που ΔΙΑΦΕΡΕΙ ανά διάσταση. Εμπειρικά (από τα ΔΕΔΟΜΕΝΑ):
        vx ≈ 98.9 * Δx ,  vy ≈ 44.4 * Δy ,  omega ≈ 18.0 * Δtheta   (r=0.95–0.998)
  Ένα σκέτο /DT θα έδινε ΤΕΛΕΙΩΣ λάθος κλίμακα. Γι' αυτό ΒΑΘΜΟΝΟΜΟΥΜΕ τις σταθερές
  vel_scale[d] ΑΠΟ ΤΑ ΔΕΔΟΜΕΝΑ στην αρχή (calibrate_vel_scales) -> robust & version-proof.

ΑΛΛΕΣ ΒΕΛΤΙΩΣΕΙΣ ΣΤΟ PORT:
  * physical_rmse: regime-aware report -> ξεχωρίζει supervised (★) από unsupervised dims,
    ώστε να ΜΗΝ μπερδεύει το (νοηματικά κενό) velocity-RMSE του semi.
  * MPS device + num_workers=0/pin_memory=False στο Mac (αποφυγή pickling του eager dataset).

ΣΗΜ. (2-frame): ο encoder βλέπει 2 frames -> η ταχύτητα είναι ΠΑΡΑΤΗΡΗΣΙΜΗ· η διαφορά
semi vs weak θα είναι ΗΠΙΟΤΕΡΗ (συνειδητή επιλογή για ΚΟΙΝΟ backbone με baseline/P1/P2).
Τρέξε ΞΕΧΩΡΙΣΤΑ για SUPERVISION="semi" και ="weak".
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from loader import VaePairDataset, load_norm_stats, list_npz

# ---------------------------------------------------------------------------
# CONFIG  (ΤΟΠΙΚΑ paths -> τρέχει στο Mac· για Kaggle άλλαξε σε /kaggle/working/...)
# ---------------------------------------------------------------------------
SUPERVISION = "<supervision>"

DATA_ROOT = "<lunarlander-dataset>"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR = os.path.join(DATA_ROOT, "val")
NORM_STATS = os.path.join(DATA_ROOT, "norm_stats.npz")
SAVE_DIR = "/kaggle/working/lunarlander_p3_vae"

STATE_NAMES = ("x", "y", "vx", "vy", "theta", "omega", "leg1", "leg2")

LATENT_SIZE = 64
N_SUP = 8                  # [x, y, vx, vy, theta, omega, leg1, leg2]
SHIFT = 0                  # CLEAN ground truth (no label noise)

# --- ΑΡΧΗ 3: ρύθμιση εποπτείας ---
STATIC_DIMS = (0, 1, 4, 6, 7)    # x, y, theta, leg1, leg2 -> ΠΑΝΤΑ εποπτευόμενα
VEL_DIMS = (2, 3, 5)             # vx, vy, omega
VEL_SRC = {2: 0, 3: 1, 5: 4}     # vel-dim <- pos-dim για το finite-diff (vx<-x, vy<-y, omega<-theta)

BATCH = 128
EPOCHS = 40
LR = 1e-3

# --- SPLIT-β KL (ΤΑΥΤΟΣΗΜΟ με baseline) ---
BETA_PHYS = 0.01
BETA_STYLE_MAX = 1.0
KL_ANNEAL_EPOCHS = 20

LAMBDA_SUP = 1.0

EARLY_STOP_PATIENCE = 5
SCHED_PATIENCE = 3

NUM_WORKERS = 2
SEED = 0


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def get_device():
    """ CUDA (Kaggle) -> MPS (Apple Silicon, τοπικά) -> CPU. """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_img(t, device):
    """ uint8 (B,3,H,W) -> float [0,1] στη συσκευή. """
    return t.to(device, non_blocking=True).float().div_(255.0)


# ---------------------------------------------------------------------------
# Model — ΑΚΡΙΒΩΣ το baseline VAE (single monolithic encoder)
# ---------------------------------------------------------------------------
class VAE_P3(nn.Module):
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
# Βαθμονόμηση finite-diff -> ταχύτητα ΑΠΟ ΤΑ ΔΕΔΟΜΕΝΑ (όχι hardcoded physics)
# ---------------------------------------------------------------------------
def calibrate_vel_scales(train_dir):
    """ Για κάθε vel-dim d (πηγή pos pd): least-squares κλίμακα C ώστε vel_t ≈ C * Δpos_t
    σε RAW obs μονάδες (Σ(v·Δp)/Σ(Δp²)). Επιστρέφει {d: C}. Διαβάζει ΜΟΝΟ τα "states"
    (φθηνό· np.load lazy). """
    num = {d: 0.0 for d in VEL_DIMS}
    den = {d: 0.0 for d in VEL_DIMS}
    for f in tqdm(list_npz(train_dir), desc="calibrating vel-scales"):
        with np.load(f) as z:
            s = z["states"].astype(np.float64)            # (T,8) raw
        if s.shape[0] < 2:
            continue
        for d in VEL_DIMS:
            pd = VEL_SRC[d]
            dp = s[1:, pd] - s[:-1, pd]                    # Δpos_t
            v = s[:-1, d]                                  # vel_t
            num[d] += float((v * dp).sum())
            den[d] += float((dp * dp).sum())
    C = {d: (num[d] / den[d] if den[d] > 1e-12 else 0.0) for d in VEL_DIMS}
    print("vel-scales (vel ≈ C·Δpos, RAW units):  " +
          "  ".join(f"{STATE_NAMES[d]}<-Δ{STATE_NAMES[VEL_SRC[d]]}: C={C[d]:.2f}" for d in VEL_DIMS))
    return C


# ---------------------------------------------------------------------------
# ΑΡΧΗ 3 — στόχος & μάσκα εποπτείας ανάλογα με SUPERVISION
# ---------------------------------------------------------------------------
def _sup_mask(device):
    """ Ποιες state dims εποπτεύονται: semi -> static· weak -> static + velocity. """
    m = torch.zeros(N_SUP, device=device)
    for d in STATIC_DIMS:
        m[d] = 1.0
    if SUPERVISION == "weak":
        for d in VEL_DIMS:
            m[d] = 1.0
    return m


def _sup_target(state_t, state_tp1, mean_t, std_t, vel_scale):
    """ Στόχος (standardized) για τις φυσικές dims:
        static  = ΑΛΗΘΙΝΟ state_t.
        velocity (μόνο weak) = ΕΚΤΙΜΗΣΗ finite-diff, ΒΑΘΜΟΝΟΜΗΜΕΝΗ ανά dim:
            Δpos_raw = (pos_{t+1}-pos_t)_std * std[pos]      (mean ακυρώνεται στη διαφορά)
            vel_est_raw = C[d] * Δpos_raw
            vel_est_std = (vel_est_raw - mean[d]) / std[d]   (ίδιος χώρος με αληθινή ταχύτητα) """
    target = state_t.clone()
    if SUPERVISION == "weak":
        for d in VEL_DIMS:
            pd = VEL_SRC[d]
            dpos_raw = (state_tp1[:, pd] - state_t[:, pd]) * std_t[pd]
            vel_est_raw = vel_scale[d] * dpos_raw
            target[:, d] = (vel_est_raw - mean_t[d]) / std_t[d]
    return target


# ---------------------------------------------------------------------------
# Loss — per-element means· SPLIT-β KL (phys vs style)· masked supervision (Αρχή 3)
# ---------------------------------------------------------------------------
def vae_losses(recon, img_target, mu, logvar, sup_target, sup_mask, n_sup):
    B, D = mu.size(0), mu.size(1)
    recon_l = F.mse_loss(recon, img_target, reduction="mean")

    diff = (mu[:, :n_sup] - sup_target) * sup_mask
    sup = (diff ** 2).sum() / (B * sup_mask.sum())                      # mean ΜΟΝΟ σε supervised dims

    kl_per = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kld_phys = kl_per[:, :n_sup].sum() / B / n_sup
    kld_style = kl_per[:, n_sup:].sum() / B / (D - n_sup)
    return recon_l, kld_phys, kld_style, sup


# ---------------------------------------------------------------------------
# Train / Eval
# ---------------------------------------------------------------------------
def run_epoch(model, loader, device, beta_style, mean_t, std_t, vel_scale, optimizer=None, desc=""):
    train = optimizer is not None
    model.train() if train else model.eval()
    mask = _sup_mask(device)
    tot = {"recon": 0.0, "kld_phys": 0.0, "kld_style": 0.0, "sup": 0.0, "n": 0}

    pbar = tqdm(loader, desc=desc, leave=False)
    for img_t, img_tp1, action, state_t, state_tp1 in pbar:
        img_t = _to_img(img_t, device)
        img_tp1 = _to_img(img_tp1, device)
        x = torch.cat([img_t, img_tp1], dim=1)
        img_target = img_t
        st = state_t.to(device, non_blocking=True)
        stp = state_tp1.to(device, non_blocking=True)
        sup_target = _sup_target(st, stp, mean_t, std_t, vel_scale)

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
def physical_rmse(model, loader, device, std_phys):
    """ RMSE των 8 dims σε ΦΥΣΙΚΕΣ μονάδες vs ΑΛΗΘΙΝΟ state (διαγνωστικό).
    ΣΗΜ.: για semi, οι velocity dims είναι ΑΝΕΠΟΠΤΕΥΤΕΣ -> το RMSE τους ΔΕΝ είναι νοηματικό
    (ελεύθερα latents)· γι' αυτό τα ξεχωρίζουμε με ★ (supervised) στο print. """
    model.eval()
    se = torch.zeros(N_SUP, device=device); n = 0
    for img_t, img_tp1, action, state_t, state_tp1 in loader:
        x = torch.cat([_to_img(img_t, device), _to_img(img_tp1, device)], dim=1)
        st = state_t.to(device)
        mu, _ = model.encode(x)
        se += (((mu[:, :N_SUP] - st) * std_phys) ** 2).sum(0)
        n += st.size(0)
    return torch.sqrt(se / n).cpu().numpy()


def _rmse_str(rmse, sup_dims):
    parts = []
    for i, nm in enumerate(STATE_NAMES):
        star = "★" if i in sup_dims else " "
        parts.append(f"{star}{nm}={rmse[i]:.4f}")
    return "  ".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    assert SUPERVISION in ("semi", "weak"), "SUPERVISION ∈ {'semi','weak'} (full == baseline)"
    set_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = get_device()
    print("device:", device, " | supervision:", SUPERVISION)

    mean, std = load_norm_stats(NORM_STATS)
    std_phys = torch.tensor(std[:N_SUP], device=device)
    mean_t = torch.tensor(mean, device=device)
    std_t = torch.tensor(std, device=device)

    # *** βαθμονόμηση finite-diff -> ταχύτητα από τα δεδομένα (ΜΟΝΟ για weak, αλλά φθηνό) ***
    vel_scale = calibrate_vel_scales(TRAIN_DIR)

    sup_dims = set(STATIC_DIMS) | (set(VEL_DIMS) if SUPERVISION == "weak" else set())

    train_ds = VaePairDataset(TRAIN_DIR, shift=SHIFT, state_mean=mean, state_std=std)
    val_ds = VaePairDataset(VAL_DIR, shift=SHIFT, state_mean=mean, state_std=std)
    pin = device.type == "cuda"
    pw = NUM_WORKERS > 0
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=True,
                          num_workers=NUM_WORKERS, pin_memory=pin, persistent_workers=pw)
    val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=pin, persistent_workers=pw)
    print(f"train pairs: {len(train_ds)} | val pairs: {len(val_ds)}")

    model = VAE_P3(latent_size=LATENT_SIZE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=SCHED_PATIENCE)

    best_val, bad_epochs = float("inf"), 0
    for epoch in range(1, EPOCHS + 1):
        beta_style = BETA_STYLE_MAX * min(1.0, epoch / max(KL_ANNEAL_EPOCHS, 1))

        tr = run_epoch(model, train_dl, device, beta_style, mean_t, std_t, vel_scale, optimizer, desc=f"E{epoch:03d} train")
        va = run_epoch(model, val_dl, device, beta_style, mean_t, std_t, vel_scale, optimizer=None, desc=f"E{epoch:03d} val")
        rmse = physical_rmse(model, val_dl, device, std_phys)

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
        print(f"  VAL   phys RMSE (★=supervised) | {_rmse_str(rmse, sup_dims)}")

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
