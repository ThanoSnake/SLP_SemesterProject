"""
vae_principle2.py — Principle 2 (Aligned in/equivariance) VAE για CartPole.

ΣΧΕΔΙΑΣΤΙΚΗ ΑΠΟΦΑΣΗ — "P2 ΜΟΝΟ ΤΟΥ" (απομονωμένο, πάνω στο baseline):
  Το paper (Fig. 3C/D) συγκρίνει ΚΑΘΕ αρχή ΞΕΧΩΡΙΣΤΑ ως ablation πάνω στο baseline
  ("Baseline", "Enhancement by Principle 1", "Enhancement by Principle 2"). Άρα εδώ
  κρατάμε ΑΚΡΙΒΩΣ την αρχιτεκτονική του baseline (ένας encoder, 6-κάναλη είσοδος
  stack(frame_t,frame_t+1), 64 latent, supervised τα 4 πρώτα dims) και προσθέτουμε
  ΜΟΝΟ το in/equivariance loss -> κάθε διαφορά οφείλεται καθαρά στην Αρχή 2.

ΑΡΧΗ 2 (paper, Def. 3): enc είναι equivariant αν enc(g_Θ(x)) =d h_Φ(enc(x)),
με invariance την ειδική περίπτωση h=identity. Loss:
        L_wm(x) ∝ E_{Θ,Φ}[ || enc(g_Θ(x)) − h_Φ(enc(x)) ||² ].

ΕΝΕΡΓΟΙ ΜΕΤΑΣΧΗΜΑΤΙΣΜΟΙ (πάνω στα ερμηνεύσιμα φυσικά dims mu[:, :4]):

  (A) EQUIVARIANCE ΘΕΣΗΣ (object-only translation via segmentation)  -> dim x (index 0)
      g = οριζόντια μετατόπιση ΜΟΝΟ του ΚΙΝΟΥΜΕΝΟΥ αντικειμένου (κάρο+κοντάρι) κατά dpx
          pixels· η ΡΑΓΑ/φόντο μένουν ΣΤΑΘΕΡΑ (φυσικά σωστό: η ράγα είναι το world frame).
          Ίδιο pipeline με το rotation: segment -> shift μόνο του αντικειμένου -> recompose.
          Κάρο & ράγα είναι ΚΑΙ ΤΑ ΔΥΟ μαύρα -> τα ξεχωρίζουμε γεωμετρικά: η ράγα είναι
          full-width οριζόντια γραμμή, το κάρο τοπικό blob. Το bg ανακατασκευάζεται ως
          λευκό + συνεχής ράγα.
      h = μετατόπιση ΜΟΝΟ του x-dim κατά Δx_std. (x_dot/theta/theta_dot invariant:
          ίδιο dpx και στα 2 frames -> η διαφορά=ταχύτητα δεν αλλάζει.)
      ΚΛΙΜΑΚΑ: render 600px, world_width=4.8 -> 125 px/μονάδα· resize IMG_W=120 (×0.2)
      -> 25 px/μονάδα -> px_to_x = 4.8/120 = 0.04 raw/px. Επειδή ο loader επιβλέπει
      STANDARDIZED states: Δx_std = dpx * px_to_x / std[x].

  (B) EQUIVARIANCE ΓΩΝΙΑΣ (rotation μέσω color-segmentation)  -> dim theta (index 2)
      Το κοντάρι έχει ΣΤΑΘΕΡΟ χρώμα tan (202,152,101) -> color-mask απομονώνει ΜΟΝΟ
      το κοντάρι (κάρο/ράγα μαύρα, φόντο λευκό). Περιστρέφουμε ΜΟΝΟ το masked κοντάρι
      γύρω από τον άξονα (pivot = βάση του κονταριού) κατά Δθ, ΙΔΙΑ και στα 2 frames.
      h = μετατόπιση ΜΟΝΟ του theta-dim κατά Δθ_std = Δθ_rad / std[theta]. (x/ταχύτητες
      invariant: pivot αμετάβλητο -> x ίδιο· ίδια Δθ στα 2 frames -> theta_dot ίδιο.)
      ΣΗΜ.: VIS_SIGN αντιστοιχεί τη ΦΥΣΙΚΗ φορά του theta στην ΟΠΤΙΚΗ φορά περιστροφής·
      επιβεβαίωσέ την με SAVE_VIZ=True (γράφει before/after PNGs) και γύρισέ την αν χρειαστεί.

  (C) INVARIANCE ΦΩΤΕΙΝΟΤΗΤΑΣ (το παράδειγμα Αρχής 2 του paper)  -> όλα τα φυσικά dims
      g = brightness/contrast jitter (ίδιο στα 2 frames). h = identity (δ=0).

  (D) REAL-PAIR difference-consistency (η προσέγγιση του original repo)  -> όλα τα dims
      Για ζεύγη ΠΡΑΓΜΑΤΙΚΩΝ frames: latent_diff ≈ true_state_diff. Δίνει equivariance
      σήμα ΚΑΙ στη γωνία/ταχύτητες μέσω φυσικών μεταβάσεων (όχι synthetic transform).

ΣΗΜ.: σε notebook τρέξε ΠΡΩΤΑ το cell του loader (VaePairDataset, load_norm_stats).
Ο encode_fn επιστρέφει mu -> ΙΔΙΟ LSTM pipeline (precompute_latents) με baseline/p1.
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
# CONFIG  (ΙΔΙΑ paths/hyper με baseline & p1 -> τίμια σύγκριση)
# ---------------------------------------------------------------------------
DATA_ROOT = "<cartpole-dataset>"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR = os.path.join(DATA_ROOT, "val")
NORM_STATS = os.path.join(DATA_ROOT, "norm_stats.npz")
SAVE_DIR = "/kaggle/working/cartpole_p2_vae"

LATENT_SIZE = 64
N_SUP = 4                  # [x, x_dot, theta, theta_dot] -> index 0=x, 2=theta
SHIFT = 0                  # 0=clean· 2/5/10 -> noisy (weak supervision)

BATCH = 128
EPOCHS = 40
LR = 1e-3

# --- SPLIT-β KL (ΤΑΥΤΟΣΗΜΟ με baseline/p1) ---
BETA_PHYS = 0.01
BETA_STYLE_MAX = 1.0
KL_ANNEAL_EPOCHS = 20

LAMBDA_SUP = 1.0           # P1-style supervision (per-element mean -> O(1) knob)

# --- ΑΡΧΗ 2 (per-element mean losses -> O(1) knobs, ίδια κλίμακα με sup) ---
LAMBDA_EQUIV = 1.0         # (A) equivariance θέσης (translation)
LAMBDA_ROT = 1.0           # (B) equivariance γωνίας (rotation via segmentation)
LAMBDA_COLOR = 1.0         # (C) invariance φωτεινότητας/αντίθεσης
LAMBDA_PAIR = 1.0          # (D) real-pair difference-consistency (repo-style)
USE_EQUIV = True
USE_ROT = True
USE_COLOR = True
# (D) ΔΕΝ είναι Αρχή 2: συγκρίνει latents ΔΥΟ ΠΡΑΓΜΑΤΙΚΩΝ (ήδη supervised) frames, χωρίς
# input-transform g -> ανάγεται σε L_pair = 2*tr(Cov(mu-state)), δηλ. reweighted supervision
# (Αρχή 3), πλεονάζον με το sup. Μολύνει το "P2-only" ablation. -> OFF για καθαρή σύγκριση.
# Βάλε True ΜΟΝΟ αν θες σκόπιμα το repo-style supervised-consistency extra (όχι paper-P2).
USE_PAIR = False

# --- (A) translation (object-only, via segmentation) ---
MAX_SHIFT_PX = 8                       # per-sample dpx ∈ [-8, 8]
X_THRESHOLD = 2.4                      # world_width = 2*x_threshold = 4.8
IMG_W = 120
PX_TO_X = (2 * X_THRESHOLD) / IMG_W    # 0.04 raw μονάδες x ανά pixel
NONWHITE_TOL = 0.30                    # color-distance από λευκό -> foreground (κάρο+κοντάρι+ράγα)
DARK_THR = 0.30                        # max-channel < DARK_THR -> "μαύρο" (κάρο ή ράγα)
TRACK_ROW_FRAC = 0.40                  # γραμμή με dark-fraction > τιμή -> ΡΑΓΑ (full-width)
CART_COL_MIN = 3                       # στήλη με > τόσα dark pixels -> ΚΑΡΟ (όχι μόνο ράγα)

# --- (B) rotation via segmentation ---
POLE_RGB = (202 / 255.0, 152 / 255.0, 101 / 255.0)   # gym pole tan χρώμα
POLE_TOL = 0.25                        # κατώφλι color-distance για το mask
MAX_ROT_RAD = 0.10                     # per-sample Δθ ∈ [-0.10, 0.10] rad (~5.7°)
VIS_SIGN = 1.0                         # ΦΥΣΙΚΟ +θ -> ΟΠΤΙΚΗ φορά (επιβεβαίωσε με SAVE_VIZ)

# --- (C) color jitter (ίδιο στα 2 frames) ---
COLOR_BRIGHTNESS = 0.3
COLOR_CONTRAST = 0.3

EARLY_STOP_PATIENCE = 5
SCHED_PATIENCE = 3

NUM_WORKERS = 2
SEED = 0
SAVE_VIZ = True            # γράφει λίγα before/after PNGs πριν το train (validation των transforms)


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def _to_img(t, device):
    """ (B,3,H,W) -> float [0,1] στη GPU. Robust σε uint8 (loader επιστρέφει uint8)
    ή ήδη-float [0,1] (κάποιες εκδόσεις loader κάνουν /255 στο __getitem__). """
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
    """ Callable για loader.precompute_latents: (img_t,img_tp1) -> mu (ντετερμινιστικό). """
    @torch.no_grad()
    def _fn(img_t, img_tp1):
        model.eval()
        x = torch.cat([img_t, img_tp1], dim=1).to(device)
        mu, _ = model.encode(x)
        return mu
    return _fn


# ---------------------------------------------------------------------------
# ΚΟΙΝΟ pipeline μετασχηματισμού αντικειμένου:
#   segment(object) -> transform(object + mask) -> recompose πάνω σε σταθερό bg.
# Χρησιμοποιείται ΚΑΙ από το translation (A) ΚΑΙ από το rotation (B).
# ---------------------------------------------------------------------------
def _blend(bg, frame_t, mask_t, frame_orig, has):
    """ out = bg όπου ΟΧΙ-αντικείμενο, μετασχηματισμένο-αντικείμενο όπου mask_t.
    Samples χωρίς ανιχνευμένο αντικείμενο (has=0) -> no-op (κρατούν το original). """
    out = bg * (1 - mask_t) + frame_t * mask_t
    keep = has.view(-1, 1, 1, 1)
    return out * keep + frame_orig * (1 - keep)


# ---------------------------------------------------------------------------
# (A) TRANSLATION (object-only) — segment κάρο+κοντάρι, shift ΜΟΝΟ αυτό, ράγα σταθερή
# ---------------------------------------------------------------------------
def shift_image_h(x, dpx):
    """ Per-sample οριζόντια μετατόπιση με border-replicate. x:(B,C,H,W), dpx:(B,) long.
    dpx>0 -> περιεχόμενο ΔΕΞΙΑ (x αυξάνεται). out[...,j]=x[..., clamp(j-dpx,0,W-1)]. """
    B, C, H, W = x.shape
    cols = (torch.arange(W, device=x.device).unsqueeze(0) - dpx.unsqueeze(1)).clamp(0, W - 1)
    idx = cols.view(B, 1, 1, W).expand(B, C, H, W)
    return torch.gather(x, 3, idx)


def movable_mask_and_bg(frame):
    """ Διαχωρίζει το ΚΙΝΟΥΜΕΝΟ αντικείμενο (κάρο+κοντάρι) από τη ΣΤΑΘΕΡΗ ράγα/φόντο.
    frame:(B,3,H,W) σε [0,1]. Επιστρέφει:
      movable (B,1,H,W) soft: κάρο+κοντάρι (ΟΧΙ η full-width ράγα),
      bg (B,3,H,W): λευκό + ανακατασκευασμένη συνεχής ράγα,
      has (B,): αρκετά object pixels.
    Κάρο & ράγα είναι ΚΑΙ ΤΑ ΔΥΟ μαύρα -> γεωμετρικός διαχωρισμός: η ράγα = γραμμή με
    υψηλό dark-fraction σε όλο το πλάτος· το κάρο = στήλες με πολλά dark pixels. """
    B, _, H, W = frame.shape
    nonwhite = (torch.sqrt(((frame - 1.0) ** 2).sum(1, keepdim=True) + 1e-8) / NONWHITE_TOL).clamp(0, 1)
    dark = (frame.amax(dim=1, keepdim=True) < DARK_THR).float()      # (B,1,H,W) κάρο+ράγα
    is_track_row = (dark.mean(dim=3, keepdim=True) > TRACK_ROW_FRAC).float()   # (B,1,H,1)
    is_cart_col = (dark.sum(dim=2, keepdim=True) > CART_COL_MIN).float()       # (B,1,1,W)
    track_only = dark * is_track_row * (1 - is_cart_col)             # ράγα ΕΚΤΟΣ στηλών κάρου
    movable = (nonwhite * (1 - track_only)).clamp(0, 1)
    track_fw = is_track_row.expand(B, 1, H, W)                       # full-width ράγα
    bg = torch.ones_like(frame) * (1 - track_fw)                     # λευκό· μαύρη συνεχής ράγα
    has = (movable.sum(dim=(2, 3)) > 8).squeeze(1)                   # (B,)
    return movable, bg, has.float()


def translate_object_frame(frame, dpx):
    """ Μετατοπίζει ΜΟΝΟ το κινούμενο αντικείμενο κατά dpx· ράγα/φόντο σταθερά.
    frame:(B,3,H,W)· dpx:(B,) long. Επιστρέφει (νέο frame, has(B,)). """
    movable, bg, has = movable_mask_and_bg(frame)
    frame_sh = shift_image_h(frame, dpx)
    mask_sh = shift_image_h(movable, dpx)
    return _blend(bg, frame_sh, mask_sh, frame, has), has


def equivariance_translation_loss(model, x, mu, scale_x):
    B, dev = x.size(0), x.device
    dpx = torch.randint(-MAX_SHIFT_PX, MAX_SHIFT_PX + 1, (B,), device=dev)
    f0, h0 = translate_object_frame(x[:, :3], dpx)
    f1, h1 = translate_object_frame(x[:, 3:6], dpx)
    mu_sh, _ = model.encode(torch.cat([f0, f1], dim=1))
    delta = torch.zeros(B, N_SUP, device=dev)
    delta[:, 0] = dpx.float() * scale_x                    # Δx_std = dpx * px_to_x / std[x]
    w = (h0 * h1).view(B, 1)                               # μόνο όπου ανιχνεύτηκε κάρο στα 2 frames
    diff = (mu_sh[:, :N_SUP] - (mu[:, :N_SUP] + delta)) * w
    return (diff ** 2).sum() / (w.sum() * N_SUP + 1e-6)


# ---------------------------------------------------------------------------
# (B) ROTATION via color-segmentation — g(x) (περιστροφή ΜΟΝΟ του κονταριού) + γνωστό h
# ---------------------------------------------------------------------------
def pole_mask(frame):
    """ frame:(B,3,H,W) σε [0,1] -> soft mask (B,1,H,W) με color-distance στο tan. """
    tan = torch.tensor(POLE_RGB, device=frame.device).view(1, 3, 1, 1)
    dist = torch.sqrt(((frame - tan) ** 2).sum(1, keepdim=True) + 1e-8)
    return (1.0 - dist / POLE_TOL).clamp(0.0, 1.0)


def _estimate_pivot(mask, thr=0.5):
    """ pivot = βάση κονταριού: y=κατώτερη γραμμή με κοντάρι, x=μέσο x κοντά στη βάση.
    Επιστρέφει Xp,(B,) Yp,(B,) has_pole,(B,). """
    B, _, H, W = mask.shape
    dev = mask.device
    present = (mask > thr).float()                                   # (B,1,H,W)
    ys = torch.arange(H, device=dev).float()
    xs = torch.arange(W, device=dev).float()
    row_has = (present.sum(dim=3) > 0).float()                       # (B,1,H)
    base_y = (row_has * ys.view(1, 1, H)).amax(dim=2)                # (B,1) κατώτερο y
    near = present * (ys.view(1, 1, H, 1) >= (base_y.view(B, 1, 1, 1) - 2.0)).float()
    denom = near.sum(dim=(2, 3)) + 1e-6                              # (B,1)
    base_x = (near * xs.view(1, 1, 1, W)).sum(dim=(2, 3)) / denom    # (B,1)
    has = (present.sum(dim=(2, 3)) > 4).float()                      # (B,1) αρκετά pixels
    return base_x.squeeze(1), base_y.squeeze(1), has.squeeze(1)


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


def rotate_pole_frame(frame, vis_ang):
    """ Περιστρέφει ΜΟΝΟ το masked κοντάρι κατά vis_ang γύρω από τη βάση του.
    frame:(B,3,H,W) σε [0,1]· vis_ang:(B,). Επιστρέφει (νέο frame, has_pole(B,)). """
    mask = pole_mask(frame)                                  # (B,1,H,W)
    Xp, Yp, has = _estimate_pivot(mask)
    bg = frame * (1 - mask) + 1.0 * mask                     # σβήσε το παλιό κοντάρι (-> λευκό)
    frame_rot = _rotate(frame, Xp, Yp, vis_ang)
    mask_rot = _rotate(mask, Xp, Yp, vis_ang)
    return _blend(bg, frame_rot, mask_rot, frame, has), has  # ίδιο recompose με το translation


def equivariance_rotation_loss(model, x, mu, scale_theta):
    B, dev = x.size(0), x.device
    dtheta = (torch.rand(B, device=dev) * 2 - 1) * MAX_ROT_RAD       # ΦΥΣΙΚΟ Δθ (rad)
    f0r, h0 = rotate_pole_frame(x[:, :3], VIS_SIGN * dtheta)
    f1r, h1 = rotate_pole_frame(x[:, 3:6], VIS_SIGN * dtheta)
    mu_rot, _ = model.encode(torch.cat([f0r, f1r], dim=1))
    delta = torch.zeros(B, N_SUP, device=dev)
    delta[:, 2] = dtheta * scale_theta                              # Δθ_std στο theta-dim
    w = (h0 * h1).view(B, 1)                                        # μόνο όπου ανιχνεύτηκε κοντάρι
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
def vae_losses(model, x, target, state_t, scale_x, scale_theta):
    recon, mu, logvar = model(x)
    recon_l = F.mse_loss(recon, target, reduction="mean")
    sup = F.mse_loss(mu[:, :N_SUP], state_t, reduction="mean")

    kl_per = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    B, D = mu.size(0), mu.size(1)
    kld_phys = kl_per[:, :N_SUP].sum() / B / N_SUP
    kld_style = kl_per[:, N_SUP:].sum() / B / (D - N_SUP)

    z = mu.new_zeros(())
    equiv = equivariance_translation_loss(model, x, mu, scale_x) if USE_EQUIV else z
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
def run_epoch(model, loader, device, beta_style, scale_x, scale_theta, optimizer=None, desc=""):
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
            L = vae_losses(model, x, target, st, scale_x, scale_theta)
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
def physical_rmse(model, loader, device, std4):
    """ RMSE των supervised dims σε ΦΥΣΙΚΕΣ μονάδες (αναιρεί το standardization). """
    model.eval()
    se = torch.zeros(N_SUP, device=device); n = 0
    for img_t, img_tp1, action, state_t, state_tp1 in loader:
        x = torch.cat([_to_img(img_t, device), _to_img(img_tp1, device)], dim=1)
        st = state_t.to(device)
        mu, _ = model.encode(x)
        se += (((mu[:, :N_SUP] - st) * std4) ** 2).sum(0)
        n += st.size(0)
    return torch.sqrt(se / n).cpu().numpy()


@torch.no_grad()
def save_transform_samples(loader, device, out_dir, n=4):
    """ Γράφει [original | translation | rotation] PNGs -> οπτική επιβεβαίωση των g(x)
    (ειδικά της ΦΟΡΑΣ περιστροφής / VIS_SIGN και της ποιότητας του pole-mask). """
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    img_t, *_ = next(iter(loader))
    f0 = _to_img(img_t[:n], device)
    f_sh, _ = translate_object_frame(f0, torch.full((f0.size(0),), MAX_SHIFT_PX, device=device, dtype=torch.long))
    f_rot, _ = rotate_pole_frame(f0, torch.full((f0.size(0),), VIS_SIGN * MAX_ROT_RAD, device=device))

    def to_np(t):
        return (t.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
    a, b, c = to_np(f0), to_np(f_sh), to_np(f_rot)
    for i in range(a.shape[0]):
        row = np.concatenate([a[i], b[i], c[i]], axis=1)   # orig | +shift | +rot
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
    std4 = torch.tensor(std[:N_SUP], device=device)
    scale_x = float(PX_TO_X / std[0])          # raw px -> standardized x-shift / pixel
    scale_theta = float(1.0 / std[2])          # rad -> standardized theta-shift / rad
    print(f"scale_x={scale_x:.5f} std/px | scale_theta={scale_theta:.4f} std/rad "
          f"| std[x]={std[0]:.4f} std[theta]={std[2]:.4f}")

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

        tr = run_epoch(model, train_dl, device, beta_style, scale_x, scale_theta, optimizer, desc=f"E{epoch:03d} train")
        va = run_epoch(model, val_dl, device, beta_style, scale_x, scale_theta, optimizer=None, desc=f"E{epoch:03d} val")
        rmse = physical_rmse(model, val_dl, device, std4)

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
        print(f"  VAL   phys RMSE | x={rmse[0]:.4f}  x_dot={rmse[1]:.4f}  "
              f"theta={rmse[2]:.4f}  theta_dot={rmse[3]:.4f}")

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
