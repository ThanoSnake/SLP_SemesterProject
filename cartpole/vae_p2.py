"""
vae_p2.py — Principle 2 (aligned in/equivariance) VAE for CartPole.

DESIGN DECISION — "P2 ON ITS OWN" (isolated, on top of the baseline):
  The paper (Fig. 3C/D) compares EACH principle SEPARATELY as an ablation over the baseline
  ("Baseline", "Enhancement by Principle 1", "Enhancement by Principle 2"). So here we
  keep the baseline architecture EXACTLY (one encoder, 6-channel input
  stack(frame_t,frame_t+1), 64 latent, first 4 dims supervised) and add
  ONLY the in/equivariance loss -> any difference is cleanly attributable to Principle 2.

ACTIVE TRANSFORMS (on the interpretable physical dims mu[:, :4]):

  (A) POSITION EQUIVARIANCE (object-only translation via segmentation)  -> dim x (index 0)
      g = horizontal shift of ONLY the MOVING object (cart+pole) by dpx
          pixels; the TRACK/background stay FIXED (physically right: the track is the world frame).
          Same pipeline as the rotation: segment -> shift only the object -> recompose.
          Cart & track are BOTH black -> we separate them geometrically: the track is a
          full-width horizontal line, the cart a local blob. The bg is rebuilt as
          white + a continuous track.
      h = shift ONLY the x-dim by Δx_std. (x_dot/theta/theta_dot invariant:
          the same dpx in both frames -> the difference (= velocity) does not change.)
      SCALE: render 600px, world_width=4.8 -> 125 px/unit; resize IMG_W=120 (x0.2)
      -> 25 px/unit -> px_to_x = 4.8/120 = 0.04 raw/px. Because the loader supervises
      STANDARDIZED states: Δx_std = dpx * px_to_x / std[x].

  (B) ANGLE EQUIVARIANCE (rotation via color segmentation)  -> dim theta (index 2)
      The pole has a CONSTANT tan color (202,152,101) -> a color mask isolates ONLY
      the pole (cart/track black, background white). We rotate ONLY the masked pole
      about the axle (pivot = base of the pole) by Δθ, IDENTICALLY in both frames.
      h = shift ONLY the theta-dim by Δθ_std = Δθ_rad / std[theta]. (x/velocities
      invariant: the pivot is unchanged -> same x; same Δθ in both frames -> same theta_dot.)
      NOTE: VIS_SIGN maps the PHYSICAL sign of theta to the VISUAL direction of rotation;
      confirm it with SAVE_VIZ=True (writes before/after PNGs) and flip it if needed.

  (C) BRIGHTNESS INVARIANCE (the paper's own Principle 2 example)  -> all physical dims
      g = brightness/contrast jitter (identical in both frames). h = identity (δ=0).

  (D) REAL-PAIR difference-consistency (the original repo's approach)  -> all dims
      For pairs of REAL frames: latent_diff ~ true_state_diff. Gives an equivariance
      signal for the angle/velocities too, through physical transitions (not a synthetic transform).
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

from paths import DATA_ROOT, outputs

#
#  Config  (same paths/hyper as baseline & p1 -> fair comparison)
#
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR = os.path.join(DATA_ROOT, "val")
NORM_STATS = os.path.join(DATA_ROOT, "norm_stats.npz")
SAVE_DIR = outputs("cartpole_p2_vae")

LATENT_SIZE = 64
N_SUP = 4    # [x, x_dot, theta, theta_dot] -> index 0=x, 2=theta
SHIFT = 0    # 0=clean, 2/5/10=noisy (weak supervision)

BATCH = 128
EPOCHS = 40
LR = 1e-3

# Split-beta KL (identical to baseline/p1)
BETA_PHYS = 0.01
BETA_STYLE_MAX = 1.0
KL_ANNEAL_EPOCHS = 20

LAMBDA_SUP = 1.0   # P1-style supervision (per-element mean -> O(1) knob)

# Principle 2 (per-element mean losses -> O(1) knobs, same scale as sup)
LAMBDA_EQUIV = 1.0   # (A) position equivariance (translation)
LAMBDA_ROT = 1.0     # (B) angle equivariance (rotation via segmentation)
LAMBDA_COLOR = 1.0   # (C) brightness/contrast invariance
LAMBDA_PAIR = 1.0    # (D) real-pair difference-consistency (repo-style)
USE_EQUIV = True
USE_ROT = True
USE_COLOR = True
# (D) is NOT Principle 2: it compares latents of TWO REAL (already supervised) frames, with
# no input-transform g -> reduces to L_pair = 2*tr(Cov(mu-state)), i.e. reweighted supervision
# (Principle 3), redundant with sup. It contaminates the "P2-only" ablation. -> OFF for a clean
# comparison. Set True only if you deliberately want the repo-style supervised-consistency extra.
USE_PAIR = False

# (A) translation (object-only, via segmentation)
MAX_SHIFT_PX = 8                       # per-sample dpx in [-8, 8]
X_THRESHOLD = 2.4                      # world_width = 2*x_threshold = 4.8
IMG_W = 120
PX_TO_X = (2 * X_THRESHOLD) / IMG_W    # 0.04 raw x units per pixel
NONWHITE_TOL = 0.30                    # color-distance from white -> foreground (cart+pole+track)
DARK_THR = 0.30                        # max-channel < DARK_THR -> "black" (cart or track)
TRACK_ROW_FRAC = 0.40                  # row with dark-fraction > value -> TRACK (full-width)
CART_COL_MIN = 3                       # column with > this many dark pixels -> CART (not just track)

# (B) rotation via segmentation
POLE_RGB = (202 / 255.0, 152 / 255.0, 101 / 255.0)   # gym pole tan color
POLE_TOL = 0.25                        # color-distance threshold for the mask
MAX_ROT_RAD = 0.10                     # per-sample Δθ in [-0.10, 0.10] rad (~5.7°)
VIS_SIGN = 1.0                         # maps physical +θ to visual rotation (confirm with SAVE_VIZ)

# (C) color jitter (same on both frames)
COLOR_BRIGHTNESS = 0.3
COLOR_CONTRAST = 0.3

EARLY_STOP_PATIENCE = 5
SCHED_PATIENCE = 3

NUM_WORKERS = 2
SEED = 0
SAVE_VIZ = True    # writes a few before/after PNGs before training (validates the transforms)


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def _to_img(t, device):
    """(B,3,H,W) -> float [0,1] on the GPU. Robust to uint8 (loader returns uint8)
    or already-float [0,1] (some loader versions do /255 in __getitem__)."""
    t = t.to(device, non_blocking=True)
    return t.float().div_(255.0) if t.dtype == torch.uint8 else t.float()


#
#  Model — exactly the baseline VAE (single encoder); only the loss changes.
#
class VAE_P2(nn.Module):
    """encode(x): x = stack(frame_t, frame_t+1) (B,6,80,120). Decode (B,3,80,120)=frame_t.
    latent[:, :N_SUP] = physical (supervised)."""

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
    """Callable for loader.precompute_latents: (img_t, img_tp1) -> mu (deterministic)."""
    @torch.no_grad()
    def _fn(img_t, img_tp1):
        model.eval()
        x = torch.cat([img_t, img_tp1], dim=1).to(device)
        mu, _ = model.encode(x)
        return mu
    return _fn


#
#  Shared object-transform pipeline:
#    segment(object) -> transform(object + mask) -> recompose onto a fixed bg.
#  Used by both translation (A) and rotation (B).
#
def _blend(bg, frame_t, mask_t, frame_orig, has):
    """out = bg where NOT-object, transformed-object where mask_t.
    Samples with no detected object (has=0) -> no-op (keep the original)."""
    out = bg * (1 - mask_t) + frame_t * mask_t
    keep = has.view(-1, 1, 1, 1)
    return out * keep + frame_orig * (1 - keep)


#
#  (A) TRANSLATION (object-only) — segment cart+pole, shift ONLY it, track fixed
#
def shift_image_h(x, dpx):
    """Per-sample horizontal shift with border-replicate. x:(B,C,H,W), dpx:(B,) long.
    dpx>0 -> content moves RIGHT (x increases). out[...,j]=x[..., clamp(j-dpx,0,W-1)]."""
    B, C, H, W = x.shape
    cols = (torch.arange(W, device=x.device).unsqueeze(0) - dpx.unsqueeze(1)).clamp(0, W - 1)
    idx = cols.view(B, 1, 1, W).expand(B, C, H, W)
    return torch.gather(x, 3, idx)


def movable_mask_and_bg(frame):
    """Separates the MOVING object (cart+pole) from the FIXED track/background.
    frame:(B,3,H,W) in [0,1]. Returns:
      movable (B,1,H,W) soft: cart+pole (NOT the full-width track),
      bg (B,3,H,W): white + reconstructed continuous track,
      has (B,): enough object pixels.
    Cart & track are BOTH black -> geometric separation: track = row with high
    dark-fraction across the full width; cart = columns with many dark pixels."""
    B, _, H, W = frame.shape
    nonwhite = (torch.sqrt(((frame - 1.0) ** 2).sum(1, keepdim=True) + 1e-8) / NONWHITE_TOL).clamp(0, 1)
    dark = (frame.amax(dim=1, keepdim=True) < DARK_THR).float()      # (B,1,H,W) cart+track
    is_track_row = (dark.mean(dim=3, keepdim=True) > TRACK_ROW_FRAC).float()   # (B,1,H,1)
    is_cart_col = (dark.sum(dim=2, keepdim=True) > CART_COL_MIN).float()       # (B,1,1,W)
    track_only = dark * is_track_row * (1 - is_cart_col)             # track OUTSIDE cart columns
    movable = (nonwhite * (1 - track_only)).clamp(0, 1)
    track_fw = is_track_row.expand(B, 1, H, W)                       # full-width track
    bg = torch.ones_like(frame) * (1 - track_fw)                    # white; black continuous track
    has = (movable.sum(dim=(2, 3)) > 8).squeeze(1)                   # (B,)
    return movable, bg, has.float()


def translate_object_frame(frame, dpx):
    """Shifts ONLY the moving object by dpx; track/background fixed.
    frame:(B,3,H,W); dpx:(B,) long. Returns (new frame, has(B,))."""
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
    w = (h0 * h1).view(B, 1)                               # only where the cart was detected in both frames
    diff = (mu_sh[:, :N_SUP] - (mu[:, :N_SUP] + delta)) * w
    return (diff ** 2).sum() / (w.sum() * N_SUP + 1e-6)


#
#  (B) ROTATION via color-segmentation — g(x) (rotate ONLY the pole) + known h
#
def pole_mask(frame):
    """frame:(B,3,H,W) in [0,1] -> soft mask (B,1,H,W) from color-distance to tan."""
    tan = torch.tensor(POLE_RGB, device=frame.device).view(1, 3, 1, 1)
    dist = torch.sqrt(((frame - tan) ** 2).sum(1, keepdim=True) + 1e-8)
    return (1.0 - dist / POLE_TOL).clamp(0.0, 1.0)


def _estimate_pivot(mask, thr=0.5):
    """pivot = pole base: y=lowest row with pole, x=mean x near the base.
    Returns Xp,(B,) Yp,(B,) has_pole,(B,)."""
    B, _, H, W = mask.shape
    dev = mask.device
    present = (mask > thr).float()                                   # (B,1,H,W)
    ys = torch.arange(H, device=dev).float()
    xs = torch.arange(W, device=dev).float()
    row_has = (present.sum(dim=3) > 0).float()                       # (B,1,H)
    base_y = (row_has * ys.view(1, 1, H)).amax(dim=2)                # (B,1) lowest y
    near = present * (ys.view(1, 1, H, 1) >= (base_y.view(B, 1, 1, 1) - 2.0)).float()
    denom = near.sum(dim=(2, 3)) + 1e-6                              # (B,1)
    base_x = (near * xs.view(1, 1, 1, W)).sum(dim=(2, 3)) / denom    # (B,1)
    has = (present.sum(dim=(2, 3)) > 4).float()                      # (B,1) enough pixels
    return base_x.squeeze(1), base_y.squeeze(1), has.squeeze(1)


def _rotate(img, Xp, Yp, vis_ang):
    """Rotates the CONTENT by +vis_ang (rad) about pivot (Xp,Yp) in pixel-space
    (correct for H≠W). img:(B,C,H,W), Xp/Yp/vis_ang:(B,)."""
    B, C, H, W = img.shape
    dev = img.device
    ys = torch.arange(H, device=dev).float().view(1, H, 1).expand(B, H, W)
    xs = torch.arange(W, device=dev).float().view(1, 1, W).expand(B, H, W)
    phi = (-vis_ang).view(B, 1, 1)                                   # sample-from angle
    c, s = torch.cos(phi), torch.sin(phi)
    Xc, Yc = xs - Xp.view(B, 1, 1), ys - Yp.view(B, 1, 1)
    Xi = c * Xc - s * Yc + Xp.view(B, 1, 1)
    Yi = s * Xc + c * Yc + Yp.view(B, 1, 1)
    grid = torch.stack([2.0 * Xi / (W - 1) - 1.0, 2.0 * Yi / (H - 1) - 1.0], dim=-1)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)


def rotate_pole_frame(frame, vis_ang):
    """Rotates ONLY the masked pole by vis_ang about its base.
    frame:(B,3,H,W) in [0,1]; vis_ang:(B,). Returns (new frame, has_pole(B,))."""
    mask = pole_mask(frame)                                  # (B,1,H,W)
    Xp, Yp, has = _estimate_pivot(mask)
    bg = frame * (1 - mask) + 1.0 * mask                     # erase the old pole (-> white)
    frame_rot = _rotate(frame, Xp, Yp, vis_ang)
    mask_rot = _rotate(mask, Xp, Yp, vis_ang)
    return _blend(bg, frame_rot, mask_rot, frame, has), has  # same recompose as translation


def equivariance_rotation_loss(model, x, mu, scale_theta):
    B, dev = x.size(0), x.device
    dtheta = (torch.rand(B, device=dev) * 2 - 1) * MAX_ROT_RAD       # physical Δθ (rad)
    f0r, h0 = rotate_pole_frame(x[:, :3], VIS_SIGN * dtheta)
    f1r, h1 = rotate_pole_frame(x[:, 3:6], VIS_SIGN * dtheta)
    mu_rot, _ = model.encode(torch.cat([f0r, f1r], dim=1))
    delta = torch.zeros(B, N_SUP, device=dev)
    delta[:, 2] = dtheta * scale_theta                              # Δθ_std on the theta-dim
    w = (h0 * h1).view(B, 1)                                        # only where the pole was detected
    diff = (mu_rot[:, :N_SUP] - (mu[:, :N_SUP] + delta)) * w
    return (diff ** 2).sum() / (w.sum() * N_SUP + 1e-6)


#
#  (C) COLOR INVARIANCE  +  (D) REAL-PAIR difference-consistency
#
def color_jitter(x):
    """Per-sample brightness+contrast (SAME across channels -> consistent on both frames)."""
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
    """1st vs 2nd half of the batch (random pairs thanks to shuffle): latent_diff ≈ true_state_diff."""
    h = mu.size(0) // 2
    if h == 0:
        return mu.new_zeros(())
    md = mu[h:2 * h, :N_SUP] - mu[:h, :N_SUP]
    sd = state_t[h:2 * h] - state_t[:h]
    return F.mse_loss(md, sd, reduction="mean")


#
#  Loss — per-element means; split-beta KL (phys vs style); + Principle 2 (A,B,C,D)
#
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
    """Selection score: beta-independent, but WITH the Principle 2 terms."""
    return (L["recon"] + LAMBDA_SUP * L["sup"] + LAMBDA_EQUIV * L["equiv"]
            + LAMBDA_ROT * L["rot"] + LAMBDA_COLOR * L["color"] + LAMBDA_PAIR * L["pair"])


_KEYS = ("recon", "kld_phys", "kld_style", "sup", "equiv", "rot", "color", "pair")


#
#  Train / Eval
#
def run_epoch(model, loader, device, beta_style, scale_x, scale_theta, optimizer=None, desc=""):
    train = optimizer is not None
    model.train() if train else model.eval()
    tot = {k: 0.0 for k in _KEYS}
    tot["n"] = 0

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
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
    """RMSE of the supervised dims in PHYSICAL units (undoes the standardization)."""
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


@torch.no_grad()
def save_transform_samples(loader, device, out_dir, n=4):
    """Writes [original | translation | rotation] PNGs -> visual check of the g(x)
    (especially the rotation DIRECTION / VIS_SIGN and the pole-mask quality)."""
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
                print(f"Early stopping at epoch {epoch}.")
                break

    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "vae_p2_last.pth"))
    print("Best val score:", best_val)
