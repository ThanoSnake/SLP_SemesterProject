"""
LunarVaePrinciple2.py — Principle 2 (Aligned in/equivariance) VAE για LunarLander.
Port του cart_pole/vae_principle2-2.py· state 4D -> 8D, ένας movable (το lander).

ΣΧΕΔΙΑΣΤΙΚΗ ΑΠΟΦΑΣΗ — "P2 ΜΟΝΟ ΤΟΥ" (απομονωμένο, πάνω στο baseline):
  Το paper (Fig. 3C/D) συγκρίνει ΚΑΘΕ αρχή ΞΕΧΩΡΙΣΤΑ ως ablation πάνω στο baseline.
  Άρα κρατάμε ΑΚΡΙΒΩΣ την αρχιτεκτονική του baseline (ΕΝΑΣ encoder, 6-κάναλη είσοδος
  stack(frame_t,frame_t+1), 64 latent, supervised τα 8 πρώτα dims) και προσθέτουμε ΜΟΝΟ
  το in/equivariance loss -> κάθε διαφορά οφείλεται καθαρά στην Αρχή 2.

ΑΡΧΗ 2 (paper, Def. 3): enc equivariant αν enc(g_Θ(x)) =d h_Φ(enc(x)),
με invariance την ειδική περίπτωση h=identity.  L ∝ E[ ||enc(g(x)) − h(enc(x))||² ].

obs = [x, y, vx, vy, theta, omega, leg1, leg2]  -> index: x=0, y=1, theta=4.

SEGMENTATION (επιβεβαιωμένο από τα ΠΡΑΓΜΑΤΙΚΑ frames):
  ουρανός=μαύρο (0,0,0) | έδαφος=λευκό/γκρι (~240) | σημαίες pad=κίτρινο (204,204,0)
  *** LANDER = μωβ (128,102,230) *** -> διακριτικό color-mask. ΕΝΑ mask, κοινό για
  translation ΚΑΙ rotation (το lander είναι το μοναδικό κινούμενο σώμα· έδαφος/σημαίες/
  ουρανός = world frame, σταθερά). Background όπου σβήνουμε το lander -> ΜΑΥΡΟ (ουρανός).

ΕΝΕΡΓΟΙ ΜΕΤΑΣΧΗΜΑΤΙΣΜΟΙ (πάνω στα ερμηνεύσιμα φυσικά dims mu[:, :8]):

  (A) EQUIVARIANCE ΘΕΣΗΣ — 2D object-only translation       -> x (idx 0) & y (idx 1)
      g = μετατόπιση ΜΟΝΟ του lander (μωβ mask) κατά (dpx, dpy) px, ΙΔΙΑ στα 2 frames·
          έδαφος/σημαίες/ουρανός ΣΤΑΘΕΡΑ (world frame). Η τρύπα που αφήνει -> ΜΑΥΡΟ.
      h = x += dpx*scale_x, y += dpy*scale_y. (vx,vy,theta,omega invariant: ίδιο shift
          στα 2 frames -> διαφορά=ταχύτητα αμετάβλητη· legs invariant: αέρας.)
      ΚΛΙΜΑΚΑ: το obs x,y είναι ΚΑΝΟΝΙΚΟΠΟΙΗΜΕΝΟ ώστε ΟΛΟ το πλάτος/ύψος = εύρος 2.0
          (x∈[-1,1] σε W, y παρόμοια σε H). render flips κάθετα (world-up=image-up):
          shift ΚΑΤΩ (dpy>0) -> ΜΙΚΡΟΤΕΡΟ y -> Y_SIGN=-1.
          PX_TO_X = 2/IMG_W, PX_TO_Y = 2/IMG_H. scale = (±PX_TO)/std.

  (B) EQUIVARIANCE ΓΩΝΙΑΣ — rotation γύρω από το ΚΕΝΤΡΟΕΙΔΕΣ του lander  -> theta (idx 4)
      g = περιστροφή ΜΟΝΟ του masked lander κατά Δθ γύρω από το κεντροειδές του, ΙΔΙΑ στα
          2 frames. (Στο LunarLander obs[4]=self.lander.angle = ΑΚΑΤΕΡΓΑΣΤΑ rad.)
      h = theta += Δθ_std = Δθ_rad / std[theta]. (x,y invariant: κεντροειδές αμετάβλητο·
          ίδια Δθ στα 2 frames -> omega αμετάβλητο.)
      ΣΗΜ.: VIS_SIGN αντιστοιχεί ΦΥΣΙΚΟ +θ -> ΟΠΤΙΚΗ φορά· επιβεβαίωσε με SAVE_VIZ.

  (C) INVARIANCE ΦΩΤΕΙΝΟΤΗΤΑΣ (το παράδειγμα Αρχής 2 του paper)  -> όλα τα φυσικά dims
      g = brightness/contrast jitter (ίδιο στα 2 frames). h = identity (δ=0).

  (D) REAL-PAIR difference-consistency (repo-style)  -> OFF (μολύνει το "P2-only" ablation·
      ανάγεται σε reweighted supervision/Αρχή 3· βάλε True ΜΟΝΟ σκόπιμα).

ΣΗΜ.: σε notebook τρέξε ΠΡΩΤΑ το cell του LunarLoader. Ο encode_fn επιστρέφει mu
-> ΙΔΙΟ LSTM pipeline (precompute_latents) με baseline/P1.
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
# CONFIG  (ΙΔΙΑ paths/hyper με baseline & P1 -> τίμια σύγκριση)
# ---------------------------------------------------------------------------
DATA_ROOT = "<lunarlander-dataset>"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR = os.path.join(DATA_ROOT, "val")
NORM_STATS = os.path.join(DATA_ROOT, "norm_stats.npz")
SAVE_DIR = "/kaggle/working/lunarlander_p2_vae"

STATE_NAMES = ("x", "y", "vx", "vy", "theta", "omega", "leg1", "leg2")
X_DIM, Y_DIM, THETA_DIM = 0, 1, 4          # index στα supervised dims

LATENT_SIZE = 64
N_SUP = 8                  # [x, y, vx, vy, theta, omega, leg1, leg2]
SHIFT = 0                  # 0=clean· 2/5/10 -> noisy (weak supervision)

BATCH = 128
EPOCHS = 40
LR = 1e-3

# --- SPLIT-β KL (ΤΑΥΤΟΣΗΜΟ με baseline/P1) ---
BETA_PHYS = 0.01
BETA_STYLE_MAX = 1.0
KL_ANNEAL_EPOCHS = 20

LAMBDA_SUP = 1.0           # P1-style supervision (per-element mean -> O(1) knob)

# --- ΑΡΧΗ 2 (per-element mean losses -> O(1) knobs, ίδια κλίμακα με sup) ---
LAMBDA_EQUIV = 1.0         # (A) equivariance θέσης (2D translation)
LAMBDA_ROT = 1.0           # (B) equivariance γωνίας (rotation γύρω από κεντροειδές)
LAMBDA_COLOR = 1.0         # (C) invariance φωτεινότητας/αντίθεσης
LAMBDA_PAIR = 1.0          # (D) real-pair difference-consistency (repo-style)
USE_EQUIV = True
USE_ROT = True
USE_COLOR = True
USE_PAIR = False           # ΟΧΙ Αρχή 2 -> OFF για καθαρό ablation (δες docstring (D))

# --- (A) 2D translation (object-only, via μωβ mask) ---
IMG_H, IMG_W = 80, 120
X_RANGE = 2.0                          # obs x ∈ [-1,1] -> ΟΛΟ το πλάτος = εύρος 2.0
Y_RANGE = 2.0                          # obs y παρόμοια κανονικοποίηση -> εύρος ~2.0 σε H
PX_TO_X = X_RANGE / IMG_W              # raw x-units ανά pixel (οριζόντια)
PX_TO_Y = Y_RANGE / IMG_H             # raw y-units ανά pixel (κάθετα)
Y_SIGN = -1.0                          # shift ΚΑΤΩ (dpy>0) -> ΜΙΚΡΟΤΕΡΟ y (render flipped)
MAX_SHIFT_PX = 8                       # per-sample dpx,dpy ∈ [-8, 8]

# --- lander color-segmentation (επιβεβαιωμένο από τα data: μωβ (128,102,230)) ---
LANDER_RGB = (128 / 255.0, 102 / 255.0, 230 / 255.0)
LANDER_TOL = 0.30                      # color-distance κατώφλι (πιάνει και σκιασμένα μωβ)
LANDER_MIN_PX = 5                      # ελάχιστα pixels για "ανιχνεύτηκε lander" (small obj)

# --- (B) rotation γύρω από το κεντροειδές ---
MAX_ROT_RAD = 0.15                     # per-sample Δθ ∈ [-0.15, 0.15] rad (~8.6°)
VIS_SIGN = 1.0                         # ΦΥΣΙΚΟ +θ -> ΟΠΤΙΚΗ φορά (επιβεβαίωσε με SAVE_VIZ)

# --- (C) color jitter (ίδιο στα 2 frames) ---
COLOR_BRIGHTNESS = 0.3
COLOR_CONTRAST = 0.3

EARLY_STOP_PATIENCE = 5
SCHED_PATIENCE = 3

NUM_WORKERS = 2
SEED = 0
SAVE_VIZ = True            # γράφει λίγα before/after PNGs πριν το train (validation transforms)


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def _to_img(t, device):
    """ (B,3,H,W) -> float [0,1] στη GPU. Robust σε uint8 (loader) ή ήδη-float. """
    t = t.to(device, non_blocking=True)
    return t.float().div_(255.0) if t.dtype == torch.uint8 else t.float()


# ---------------------------------------------------------------------------
# Model — ΑΚΡΙΒΩΣ το baseline VAE (single encoder)· αλλάζει ΜΟΝΟ το loss.
# ---------------------------------------------------------------------------
class VAE_P2(nn.Module):
    """ encode(x): x = stack(frame_t, frame_t+1) (B,6,80,120). Decode (B,3,80,120)=frame_t.
    latent[:, :N_SUP] = φυσικά (supervised). """

    def __init__(self, latent_size=64, in_channels=6, out_channels=3):
        super().__init__()
        self.latent_size = latent_size
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 4, 2, 1), nn.ReLU(inplace=True),   # 80x120 -> 40x60
            nn.Conv2d(16, 32, 4, 2, 1), nn.ReLU(inplace=True),           # -> 20x30
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(inplace=True),           # -> 10x15
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
    """ Callable για LunarLoader.precompute_latents: (img_t,img_tp1) -> mu (ντετερμινιστικό). """
    @torch.no_grad()
    def _fn(img_t, img_tp1):
        model.eval()
        x = torch.cat([img_t, img_tp1], dim=1).to(device)
        mu, _ = model.encode(x)
        return mu
    return _fn


# ---------------------------------------------------------------------------
# Lander segmentation (μωβ color-mask) — κοινό για translation & rotation
# ---------------------------------------------------------------------------
def lander_mask(frame):
    """ frame:(B,3,H,W) σε [0,1] -> soft mask (B,1,H,W): color-distance στο μωβ lander. """
    c = torch.tensor(LANDER_RGB, device=frame.device).view(1, 3, 1, 1)
    dist = torch.sqrt(((frame - c) ** 2).sum(1, keepdim=True) + 1e-8)
    return (1.0 - dist / LANDER_TOL).clamp(0.0, 1.0)


def _has_lander(mask):
    return (mask.sum(dim=(2, 3)) > LANDER_MIN_PX).squeeze(1).float()   # (B,)


def _blend(bg, frame_t, mask_t, frame_orig, has):
    """ out = bg όπου ΟΧΙ-lander, μετασχηματισμένο-lander όπου mask_t.
    Samples χωρίς ανιχνευμένο lander (has=0) -> no-op (κρατούν το original). """
    out = bg * (1 - mask_t) + frame_t * mask_t
    keep = has.view(-1, 1, 1, 1)
    return out * keep + frame_orig * (1 - keep)


# ---------------------------------------------------------------------------
# (A) 2D TRANSLATION (object-only) — shift ΜΟΝΟ το lander· έδαφος/σημαίες/ουρανός σταθερά
# ---------------------------------------------------------------------------
def shift_image_h(x, dpx):
    """ Per-sample οριζόντια μετατόπιση (border-replicate). x:(B,C,H,W), dpx:(B,) long.
    dpx>0 -> περιεχόμενο ΔΕΞΙΑ. out[...,j]=x[..., clamp(j-dpx,0,W-1)]. """
    B, C, H, W = x.shape
    cols = (torch.arange(W, device=x.device).unsqueeze(0) - dpx.unsqueeze(1)).clamp(0, W - 1)
    idx = cols.view(B, 1, 1, W).expand(B, C, H, W)
    return torch.gather(x, 3, idx)


def shift_image_v(x, dpy):
    """ Per-sample κάθετη μετατόπιση (border-replicate). dpy>0 -> περιεχόμενο ΚΑΤΩ.
    out[...,i,:]=x[..., clamp(i-dpy,0,H-1), :]. """
    B, C, H, W = x.shape
    rows = (torch.arange(H, device=x.device).unsqueeze(0) - dpy.unsqueeze(1)).clamp(0, H - 1)
    idx = rows.view(B, 1, H, 1).expand(B, C, H, W)
    return torch.gather(x, 2, idx)


def _shift2d(x, dpx, dpy):
    return shift_image_v(shift_image_h(x, dpx), dpy)


def translate_object_frame(frame, dpx, dpy):
    """ Μετατοπίζει ΜΟΝΟ το lander κατά (dpx,dpy)· η τρύπα -> μαύρο (ουρανός).
    frame:(B,3,H,W)· dpx,dpy:(B,) long. Επιστρέφει (νέο frame, has(B,)). """
    mask = lander_mask(frame)
    has = _has_lander(mask)
    bg = frame * (1 - mask)                              # σβήσε το lander -> ΜΑΥΡΟ
    frame_sh = _shift2d(frame, dpx, dpy)
    mask_sh = _shift2d(mask, dpx, dpy)
    return _blend(bg, frame_sh, mask_sh, frame, has), has


def equivariance_translation_loss(model, x, mu, scale_x, scale_y):
    B, dev = x.size(0), x.device
    dpx = torch.randint(-MAX_SHIFT_PX, MAX_SHIFT_PX + 1, (B,), device=dev)
    dpy = torch.randint(-MAX_SHIFT_PX, MAX_SHIFT_PX + 1, (B,), device=dev)
    f0, h0 = translate_object_frame(x[:, :3], dpx, dpy)
    f1, h1 = translate_object_frame(x[:, 3:6], dpx, dpy)
    mu_sh, _ = model.encode(torch.cat([f0, f1], dim=1))
    delta = torch.zeros(B, N_SUP, device=dev)
    delta[:, X_DIM] = dpx.float() * scale_x             # Δx_std = dpx * PX_TO_X / std[x]
    delta[:, Y_DIM] = dpy.float() * scale_y             # Δy_std = dpy * Y_SIGN*PX_TO_Y / std[y]
    w = (h0 * h1).view(B, 1)                            # μόνο όπου ανιχνεύτηκε lander στα 2 frames
    diff = (mu_sh[:, :N_SUP] - (mu[:, :N_SUP] + delta)) * w
    return (diff ** 2).sum() / (w.sum() * N_SUP + 1e-6)


# ---------------------------------------------------------------------------
# (B) ROTATION γύρω από το κεντροειδές του lander — g(x) + γνωστό h
# ---------------------------------------------------------------------------
def _estimate_centroid(mask, thr=0.5):
    """ pivot = κεντροειδές του lander (μέσο x,y των masked pixels).
    Επιστρέφει Xc,(B,) Yc,(B,) has,(B,). """
    B, _, H, W = mask.shape
    dev = mask.device
    present = (mask > thr).float()                                   # (B,1,H,W)
    ys = torch.arange(H, device=dev).float()
    xs = torch.arange(W, device=dev).float()
    denom = present.sum(dim=(2, 3)) + 1e-6                           # (B,1)
    Yc = (present * ys.view(1, 1, H, 1)).sum(dim=(2, 3)) / denom     # (B,1)
    Xc = (present * xs.view(1, 1, 1, W)).sum(dim=(2, 3)) / denom     # (B,1)
    has = (present.sum(dim=(2, 3)) > LANDER_MIN_PX).float()          # (B,1)
    return Xc.squeeze(1), Yc.squeeze(1), has.squeeze(1)


def _rotate(img, Xp, Yp, vis_ang):
    """ Περιστρέφει το ΠΕΡΙΕΧΟΜΕΝΟ κατά +vis_ang (rad) γύρω από pivot (Xp,Yp) σε pixel-space
    (σωστό για H≠W). img:(B,C,H,W), Xp/Yp/vis_ang:(B,). """
    B, C, H, W = img.shape
    dev = img.device
    ys = torch.arange(H, device=dev).float().view(1, H, 1).expand(B, H, W)
    xs = torch.arange(W, device=dev).float().view(1, 1, W).expand(B, H, W)
    phi = (-vis_ang).view(B, 1, 1)                                   # sample-from γωνία
    c, s = torch.cos(phi), torch.sin(phi)
    Xc, Yc = xs - Xp.view(B, 1, 1), ys - Yp.view(B, 1, 1)
    Xi = c * Xc - s * Yc + Xp.view(B, 1, 1)
    Yi = s * Xc + c * Yc + Yp.view(B, 1, 1)
    grid = torch.stack([2.0 * Xi / (W - 1) - 1.0, 2.0 * Yi / (H - 1) - 1.0], dim=-1)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)


def rotate_lander_frame(frame, vis_ang):
    """ Περιστρέφει ΜΟΝΟ το masked lander κατά vis_ang γύρω από το κεντροειδές του.
    frame:(B,3,H,W) σε [0,1]· vis_ang:(B,). Επιστρέφει (νέο frame, has(B,)). """
    mask = lander_mask(frame)
    Xc, Yc, has = _estimate_centroid(mask)
    bg = frame * (1 - mask)                                  # σβήσε το παλιό lander -> ΜΑΥΡΟ
    frame_rot = _rotate(frame, Xc, Yc, vis_ang)
    mask_rot = _rotate(mask, Xc, Yc, vis_ang)
    return _blend(bg, frame_rot, mask_rot, frame, has), has


def equivariance_rotation_loss(model, x, mu, scale_theta):
    B, dev = x.size(0), x.device
    dtheta = (torch.rand(B, device=dev) * 2 - 1) * MAX_ROT_RAD       # ΦΥΣΙΚΟ Δθ (rad)
    f0r, h0 = rotate_lander_frame(x[:, :3], VIS_SIGN * dtheta)
    f1r, h1 = rotate_lander_frame(x[:, 3:6], VIS_SIGN * dtheta)
    mu_rot, _ = model.encode(torch.cat([f0r, f1r], dim=1))
    delta = torch.zeros(B, N_SUP, device=dev)
    delta[:, THETA_DIM] = dtheta * scale_theta                      # Δθ_std στο theta-dim
    w = (h0 * h1).view(B, 1)                                        # μόνο όπου ανιχνεύτηκε lander
    diff = (mu_rot[:, :N_SUP] - (mu[:, :N_SUP] + delta)) * w
    return (diff ** 2).sum() / (w.sum() * N_SUP + 1e-6)


# ---------------------------------------------------------------------------
# (C) COLOR INVARIANCE  +  (D) REAL-PAIR difference-consistency
# ---------------------------------------------------------------------------
def color_jitter(x):
    """ Per-sample brightness+contrast (ΙΔΙΑ σε όλα τα κανάλια -> συνεπές στα 2 frames). """
    B, dev = x.size(0), x.device
    bright = 1.0 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * COLOR_BRIGHTNESS
    contrast = 1.0 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * COLOR_CONTRAST
    out = x * bright
    m = out.mean(dim=(1, 2, 3), keepdim=True)
    return ((out - m) * contrast + m).clamp(0, 1)


def color_invariance_loss(model, x, mu):
    mu_c, _ = model.encode(color_jitter(x))
    return F.mse_loss(mu_c[:, :N_SUP], mu[:, :N_SUP], reduction="mean")   # h=identity (δ=0)


def pair_equivariance_loss(mu, state_t):
    """ 1ο vs 2ο μισό batch (τυχαία ζεύγη λόγω shuffle): latent_diff ≈ true_state_diff. """
    h = mu.size(0) // 2
    if h == 0:
        return mu.new_zeros(())
    md = mu[h:2 * h, :N_SUP] - mu[:h, :N_SUP]
    sd = state_t[h:2 * h] - state_t[:h]
    return F.mse_loss(md, sd, reduction="mean")


# ---------------------------------------------------------------------------
# Loss — per-element means· SPLIT-β KL (phys vs style)· + Αρχή 2 (A,B,C,D)
# ---------------------------------------------------------------------------
def vae_losses(model, x, target, state_t, scale_x, scale_y, scale_theta):
    recon, mu, logvar = model(x)
    recon_l = F.mse_loss(recon, target, reduction="mean")
    sup = F.mse_loss(mu[:, :N_SUP], state_t, reduction="mean")

    kl_per = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    B, D = mu.size(0), mu.size(1)
    kld_phys = kl_per[:, :N_SUP].sum() / B / N_SUP
    kld_style = kl_per[:, N_SUP:].sum() / B / (D - N_SUP)

    z = mu.new_zeros(())
    equiv = equivariance_translation_loss(model, x, mu, scale_x, scale_y) if USE_EQUIV else z
    rot = equivariance_rotation_loss(model, x, mu, scale_theta) if USE_ROT else z
    color = color_invariance_loss(model, x, mu) if USE_COLOR else z
    pair = pair_equivariance_loss(mu, state_t) if USE_PAIR else z
    return {"recon": recon_l, "kld_phys": kld_phys, "kld_style": kld_style, "sup": sup,
            "equiv": equiv, "rot": rot, "color": color, "pair": pair}


def weighted_total(L, beta_style):
    return (L["recon"] + BETA_PHYS * L["kld_phys"] + beta_style * L["kld_style"]
            + LAMBDA_SUP * L["sup"] + LAMBDA_EQUIV * L["equiv"] + LAMBDA_ROT * L["rot"]
            + LAMBDA_COLOR * L["color"] + LAMBDA_PAIR * L["pair"])


def p2_score(L):
    """ Selection score: ανεξάρτητο του beta, αλλά ΜΕ τους όρους της Αρχής 2. """
    return (L["recon"] + LAMBDA_SUP * L["sup"] + LAMBDA_EQUIV * L["equiv"]
            + LAMBDA_ROT * L["rot"] + LAMBDA_COLOR * L["color"] + LAMBDA_PAIR * L["pair"])


_KEYS = ("recon", "kld_phys", "kld_style", "sup", "equiv", "rot", "color", "pair")


# ---------------------------------------------------------------------------
# Train / Eval
# ---------------------------------------------------------------------------
def run_epoch(model, loader, device, beta_style, scale_x, scale_y, scale_theta,
              optimizer=None, desc=""):
    train = optimizer is not None
    model.train() if train else model.eval()
    tot = {k: 0.0 for k in _KEYS}; tot["n"] = 0

    pbar = tqdm(loader, desc=desc, leave=False)
    for img_t, img_tp1, action, state_t, state_tp1 in pbar:
        img_t = _to_img(img_t, device)
        img_tp1 = _to_img(img_tp1, device)
        x = torch.cat([img_t, img_tp1], dim=1)
        target = img_t
        st = state_t.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            L = vae_losses(model, x, target, st, scale_x, scale_y, scale_theta)
            loss = weighted_total(L, beta_style)

        if train:
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        bs = target.size(0)
        for k in _KEYS:
            tot[k] += float(L[k]) * bs
        tot["n"] += bs
        n = tot["n"]
        pbar.set_postfix(total=f"{weighted_total({k: tot[k]/n for k in _KEYS}, beta_style):.4f}",
                         sup=f"{tot['sup']/n:.4f}", equiv=f"{tot['equiv']/n:.4f}",
                         rot=f"{tot['rot']/n:.4f}")

    return {k: tot[k] / tot["n"] for k in _KEYS}


@torch.no_grad()
def physical_rmse(model, loader, device, std_phys):
    """ RMSE των supervised dims σε ΦΥΣΙΚΕΣ μονάδες (αναιρεί το standardization). """
    model.eval()
    se = torch.zeros(N_SUP, device=device); n = 0
    for img_t, img_tp1, action, state_t, state_tp1 in loader:
        x = torch.cat([_to_img(img_t, device), _to_img(img_tp1, device)], dim=1)
        st = state_t.to(device)
        mu, _ = model.encode(x)
        se += (((mu[:, :N_SUP] - st) * std_phys) ** 2).sum(0)
        n += st.size(0)
    return torch.sqrt(se / n).cpu().numpy()


@torch.no_grad()
def save_transform_samples(loader, device, out_dir, n=6):
    """ Γράφει [original | translation | rotation] PNGs -> οπτική επιβεβαίωση των g(x)
    (ποιότητα μωβ-mask, ΦΟΡΑ περιστροφής/VIS_SIGN, σωστό κάθετο πρόσημο shift). """
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    img_t, *_ = next(iter(loader))
    f0 = _to_img(img_t[:n], device)
    B = f0.size(0)
    dpx = torch.full((B,), MAX_SHIFT_PX, device=device, dtype=torch.long)
    dpy = torch.full((B,), MAX_SHIFT_PX, device=device, dtype=torch.long)
    f_sh, _ = translate_object_frame(f0, dpx, dpy)
    f_rot, _ = rotate_lander_frame(f0, torch.full((B,), VIS_SIGN * MAX_ROT_RAD, device=device))

    def to_np(t):
        return (t.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
    a, b, c = to_np(f0), to_np(f_sh), to_np(f_rot)
    for i in range(a.shape[0]):
        row = np.concatenate([a[i], b[i], c[i]], axis=1)   # orig | +shift(↘) | +rot
        Image.fromarray(row).save(os.path.join(out_dir, f"transform_sample_{i}.png"))
    print(f"[viz] saved {a.shape[0]} samples (orig|shift|rot) -> {out_dir}")


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
    scale_x = float(PX_TO_X / std[X_DIM])              # raw px -> standardized x-shift / pixel
    scale_y = float(Y_SIGN * PX_TO_Y / std[Y_DIM])     # raw px -> standardized y-shift / pixel
    scale_theta = float(1.0 / std[THETA_DIM])          # rad -> standardized theta-shift / rad
    print(f"scale_x={scale_x:.5f} scale_y={scale_y:.5f} scale_theta={scale_theta:.4f} | "
          f"std[x]={std[X_DIM]:.4f} std[y]={std[Y_DIM]:.4f} std[theta]={std[THETA_DIM]:.4f}")

    train_ds = VaePairDataset(TRAIN_DIR, shift=SHIFT, state_mean=mean, state_std=std)
    val_ds = VaePairDataset(VAL_DIR, shift=SHIFT, state_mean=mean, state_std=std)
    pw = NUM_WORKERS > 0
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=True,
                          num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=pw)
    val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=pw)
    print(f"train pairs: {len(train_ds)} | val pairs: {len(val_ds)}")

    if SAVE_VIZ:
        save_transform_samples(val_dl, device, os.path.join(SAVE_DIR, "viz"))

    model = VAE_P2(latent_size=LATENT_SIZE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=SCHED_PATIENCE)

    best_val, bad_epochs = float("inf"), 0
    for epoch in range(1, EPOCHS + 1):
        beta_style = BETA_STYLE_MAX * min(1.0, epoch / max(KL_ANNEAL_EPOCHS, 1))

        tr = run_epoch(model, train_dl, device, beta_style, scale_x, scale_y, scale_theta,
                       optimizer, desc=f"E{epoch:03d} train")
        va = run_epoch(model, val_dl, device, beta_style, scale_x, scale_y, scale_theta,
                       optimizer=None, desc=f"E{epoch:03d} val")
        rmse = physical_rmse(model, val_dl, device, std_phys)

        val_score = p2_score(va)
        scheduler.step(val_score)
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"E{epoch:03d} | beta_style={beta_style:.2f} lr={lr_now:.1e}")
        print(f"  TRAIN total={weighted_total(tr, beta_style):.5f} | recon={tr['recon']:.5f}  "
              f"sup={tr['sup']:.5f}  equiv={tr['equiv']:.5f}  rot={tr['rot']:.5f}  "
              f"color={tr['color']:.5f}  pair={tr['pair']:.5f}")
        print(f"  VAL   total={weighted_total(va, beta_style):.5f} | recon={va['recon']:.5f}  "
              f"sup={va['sup']:.5f}  equiv={va['equiv']:.5f}  rot={va['rot']:.5f}  "
              f"color={va['color']:.5f}  pair={va['pair']:.5f}  (select={val_score:.5f})")
        rmse_str = "  ".join(f"{nm}={rmse[i]:.4f}" for i, nm in enumerate(STATE_NAMES))
        print(f"  VAL   phys RMSE | {rmse_str}")

        if val_score < best_val - 1e-6:
            best_val, bad_epochs = val_score, 0
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "vae_p2_best.pth"))
            print("  -> best model saved")
        else:
            bad_epochs += 1
            print(f"  (no improvement: {bad_epochs}/{EARLY_STOP_PATIENCE})")
            if bad_epochs >= EARLY_STOP_PATIENCE:
                print(f"Early stopping στο epoch {epoch}.")
                break

    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "vae_p2_last.pth"))
    print("Best val score:", best_val)
