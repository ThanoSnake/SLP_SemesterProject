"""
vae_principle4.py — Principle 4 (Compositional / object-centric DECODING) VAE για LunarLander.
Loaders/helpers από το LunarLoader.py (τρέξε εκείνο το cell πρώτα). Ανάλογο του cart_pole/principles4.

ΣΚΗΝΗ (χρώματα): μωβ διαστημόπλοιο, κίτρινες σημαίες, γκρι κοντάρια, μαύρος ουρανός, άσπρο έδαφος.
3 DECODERS (όπως στο paper) -> 3 χρωματικές ομάδες (color segmentation):
    dec_lander ← physical dims [0:8]   : το ΚΙΝΟΥΜΕΝΟ σκάφος (μωβ)            -> RGB + alpha
    dec_flags  ← λίγα style dims        : στατικές σημαίες+κοντάρια (κίτρινο+γκρι) -> RGB + alpha
    dec_bg     ← style dims             : terrain (μαύρος ουρανός + άσπρο έδαφος)   -> RGB (opaque)

ΓΙΑΤΙ alpha (διαφορά από CartPole): στο CartPole το φόντο ήταν ΕΝΙΑΙΑ λευκό -> δούλευε το «sum−2»
χωρίς alpha. Εδώ το φόντο έχει ΔΥΟ χρώματα (μαύρο/άσπρο) και τα foreground αντικείμενα ΚΡΥΒΟΥΝ
το άσπρο έδαφος, άρα χρειάζεται κανονικό LAYERED ALPHA COMPOSITING (bg πίσω -> flags -> lander).

LOSS = σταθμισμένο άθροισμα (ίδια δομή με CartPole):
    W_OBJ  * mean( MSE(obj_lander, img·mask_lander) + MSE(obj_flags, img·mask_flags)
                   + MSE(rgb_bg·mask_bg, img·mask_bg) )      # ανακατασκευή ΚΑΘΕ αντικειμένου (πάνω σε μαύρο)
  + W_FULL * MSE(composite, img)                              # ανακατασκευή ΟΛΗΣ της εικόνας
  + LAMBDA_SUP * sup + split-β KL                             # ίδιος baseline backbone (P4-only ablation)
  όπου obj_lander = alpha_lander · rgb_lander  (το alpha εκπαιδεύεται ΕΜΜΕΣΑ, χωρίς ξεχωριστό mask-loss).

SEGMENTATION: on-the-fly (color_segment_lunar) — nearest-reference-color σε 5 χρώματα -> 3 ομάδες.
ΑΞΙΟΛΟΓΗΣΗ (paper): full-image reconstruction MSE + SSIM + ΜΕΓΕΘΟΣ μοντέλου (eval_principle4.py).
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from loader import VaePairDataset, load_norm_stats   # ΟΛΟΙ οι loaders ζουν εδώ (VAE+LSTM)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATA_ROOT = "<lunarlander-dataset>"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR = os.path.join(DATA_ROOT, "val")
NORM_STATS = os.path.join(DATA_ROOT, "norm_stats.npz")
SAVE_DIR = "/kaggle/working/lunarlander_p4_vae"

LATENT_SIZE = 64
N_SUP = 8                  # [x, y, vx, vy, theta, omega, leg1, leg2]
SHIFT = 0

# --- ποια dims παίρνει κάθε decoder ---
LANDER_DIMS = list(range(N_SUP))                 # [0..7] full physical (pose + legs)
FLAGS_DIMS = list(range(N_SUP, N_SUP + 8))       # [8..15] λίγα style dims (σημαίες ~στατικές)
# bg_dims = list(range(N_SUP, LATENT_SIZE))      # [8..63] όλα τα style (terrain) — αυτόματα

DEC_BASE_LANDER = 16
DEC_BASE_FLAGS = 8
DEC_BASE_BG = 16           # ↑ αν το terrain βγαίνει θολό· ↓ για μεγαλύτερη μείωση μεγέθους

W_OBJ = 1.0
W_FULL = 1.0

BATCH = 128
EPOCHS = 40
LR = 1e-3
BETA_PHYS = 0.01
BETA_STYLE_MAX = 1.0
KL_ANNEAL_EPOCHS = 20
LAMBDA_SUP = 1.0
EARLY_STOP_PATIENCE = 5
SCHED_PATIENCE = 3
NUM_WORKERS = 2
SEED = 0

# --- Color-segmentation references (σε [0,1]). ΠΡΟΣΑΡΜΟΣΕ αν το render σου διαφέρει — δες visualize. ---
LUNAR_REF_COLORS = torch.tensor([
    [0.60, 0.20, 0.80],   # 0 lander  (μωβ)
    [0.90, 0.90, 0.10],   # 1 flag    (κίτρινο)
    [0.55, 0.55, 0.55],   # 2 pole    (γκρι)
    [0.00, 0.00, 0.00],   # 3 sky     (μαύρο)
    [1.00, 1.00, 1.00],   # 4 ground  (άσπρο)
], dtype=torch.float32)
LUNAR_GROUPS = torch.tensor([0, 1, 1, 2, 2], dtype=torch.long)   # lander / flags(yellow,gray) / bg(sky,ground)


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def _to_img(t, device):
    return t.to(device, non_blocking=True).float().div_(255.0)


# ---------------------------------------------------------------------------
# COLOR SEGMENTATION (on-the-fly, vectorized) — nearest reference color -> 3 ομάδες
# ---------------------------------------------------------------------------
def color_segment_lunar(img, refs=LUNAR_REF_COLORS, groups=LUNAR_GROUPS):
    """ img: (B,3,H,W) σε [0,1].  Επιστρέφει (mask_lander, mask_flags, mask_bg), καθένα (B,1,H,W) float {0,1},
    ΑΥΣΤΗΡΟ PARTITION: κάθε pixel ανατίθεται στο ΠΛΗΣΙΕΣΤΕΡΟ reference χρώμα και μετά στην ομάδα του.
    Robust σε anti-aliasing (το nearest κερδίζει). """
    B, C, H, W = img.shape
    refs = refs.to(img.device); groups = groups.to(img.device)
    px = img.permute(0, 2, 3, 1).reshape(-1, 3)                       # (N,3)
    d2 = ((px.unsqueeze(1) - refs.unsqueeze(0)) ** 2).sum(-1)         # (N,R)
    grp = groups[d2.argmin(1)].view(B, 1, H, W)                       # (B,1,H,W)
    return (grp == 0).float(), (grp == 1).float(), (grp == 2).float()


@torch.no_grad()
def visualize_segmentation(npz_path, idx=0, save_path=None):
    """ Sanity-check ΠΡΙΝ την εκπαίδευση: frame + 3 μάσκες + 3 component GT (αντικείμενο σε ΜΑΥΡΟ). """
    import matplotlib.pyplot as plt
    with np.load(npz_path) as d:
        frame = d["imgs"][idx]
    img = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    ml, mf, mb = color_segment_lunar(img)
    comps = [(img * m)[0].permute(1, 2, 0).numpy() for m in (ml, mf, mb)]   # object σε μαύρο
    masks = [m[0, 0].numpy() for m in (ml, mf, mb)]
    titles = ["frame", "mask lander", "mask flags", "mask bg", "obj lander", "obj flags", "obj bg"]
    ims = [frame, masks[0], masks[1], masks[2], comps[0], comps[1], comps[2]]
    fig, ax = plt.subplots(1, 7, figsize=(16, 2.6))
    for a, t, im in zip(ax, titles, ims):
        a.imshow(im, cmap=("gray" if im.ndim == 2 else None), vmin=0, vmax=1)
        a.set_title(t, fontsize=9); a.axis("off")
    cover = float((ml + mf + mb).max())
    fig.suptitle(f"partition check: max overlap = {cover:.2f} (πρέπει=1.0)", fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120); print("saved:", save_path)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# DATASET / HELPERS: από LunarLoader.py (VaePairDataset 5-item, load_norm_stats, precompute_latents,
# LatentSequenceDataset για LSTM). Οι μάσκες ΔΕΝ αποθηκεύονται — παράγονται on-the-fly.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Model — baseline encoder + 3 object decoders -> layered alpha composite
# ---------------------------------------------------------------------------
class _BaseEncoder(nn.Module):
    def __init__(self, latent_size=64, in_channels=6):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(64 * 10 * 15, latent_size)
        self.fc_logvar = nn.Linear(64 * 10 * 15, latent_size)

    def encode(self, x):
        h = self.encoder(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class TinyDecoder(nn.Module):
    """ z -> (out_ch, 80, 120). 10x15 -> 80x120 (×8). out_ch=4 (RGB+alpha) ή 3 (RGB). """
    def __init__(self, in_dim, base_channels, out_ch):
        super().__init__()
        self.base = base_channels
        self.fc = nn.Linear(in_dim, base_channels * 10 * 15)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels, base_channels, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels, max(4, base_channels // 2), 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(max(4, base_channels // 2), out_ch, 4, 2, 1),
        )

    def forward(self, z):
        h = self.fc(z).view(-1, self.base, 10, 15)
        return self.decoder(h)


class VAE_P4(nn.Module):
    def __init__(self, latent_size=64, n_sup=8):
        super().__init__()
        self.latent_size = latent_size
        self.n_sup = n_sup
        self.lander_dims = list(LANDER_DIMS)
        self.flags_dims = list(FLAGS_DIMS)
        self.bg_dims = list(range(n_sup, latent_size))
        self.enc = _BaseEncoder(latent_size)
        self.dec_lander = TinyDecoder(len(self.lander_dims), DEC_BASE_LANDER, out_ch=4)  # RGB+alpha
        self.dec_flags = TinyDecoder(len(self.flags_dims), DEC_BASE_FLAGS, out_ch=4)      # RGB+alpha
        self.dec_bg = TinyDecoder(len(self.bg_dims), DEC_BASE_BG, out_ch=3)               # RGB opaque

    def encode(self, x):
        return self.enc.encode(x)

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def _compose(self, z):
        ol = self.dec_lander(z[:, self.lander_dims])
        of = self.dec_flags(z[:, self.flags_dims])
        rgb_b = torch.sigmoid(self.dec_bg(z[:, self.bg_dims]))         # opaque backdrop
        rgb_l, a_l = torch.sigmoid(ol[:, :3]), torch.sigmoid(ol[:, 3:4])
        rgb_f, a_f = torch.sigmoid(of[:, :3]), torch.sigmoid(of[:, 3:4])
        # layered alpha compositing: bg (πίσω) -> flags -> lander (μπροστά)
        out = rgb_b
        out = a_f * rgb_f + (1.0 - a_f) * out
        out = a_l * rgb_l + (1.0 - a_l) * out
        # «αντικείμενο πάνω σε μαύρο» (για το per-object recon)
        obj_l, obj_f = a_l * rgb_l, a_f * rgb_f
        return out, (obj_l, obj_f, rgb_b)

    def decode(self, z):
        """ render predicted latent -> (composite, (obj_lander, obj_flags, bg)). """
        return self._compose(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        composite, comps = self._compose(z)
        return composite, mu, logvar, comps


def encode_fn(model, device):
    @torch.no_grad()
    def _fn(img_t, img_tp1):
        model.eval()
        x = torch.cat([img_t, img_tp1], dim=1).to(device)
        mu, _ = model.encode(x)
        return mu
    return _fn


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def vae_losses(composite, img, comps, masks, mu, logvar, state_t, n_sup):
    B, D = mu.size(0), mu.size(1)
    obj_l, obj_f, rgb_b = comps
    m_l, m_f, m_b = masks
    recon_full = F.mse_loss(composite, img, reduction="mean")
    recon_obj = (F.mse_loss(obj_l, img * m_l, reduction="mean")
                 + F.mse_loss(obj_f, img * m_f, reduction="mean")
                 + F.mse_loss(rgb_b * m_b, img * m_b, reduction="mean")) / 3.0
    sup = F.mse_loss(mu[:, :n_sup], state_t, reduction="mean")
    kl_per = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kld_phys = kl_per[:, :n_sup].sum() / B / n_sup
    kld_style = kl_per[:, n_sup:].sum() / B / max(D - n_sup, 1)
    return recon_full, recon_obj, sup, kld_phys, kld_style


_KEYS = ("recon_full", "recon_obj", "sup", "kld_phys", "kld_style")


def run_epoch(model, loader, device, beta_style, optimizer=None, desc=""):
    train = optimizer is not None
    model.train() if train else model.eval()
    tot = {k: 0.0 for k in _KEYS}; tot["n"] = 0

    pbar = tqdm(loader, desc=desc, leave=False)
    for img_t, img_tp1, action, state_t, state_tp1 in pbar:
        img_t = _to_img(img_t, device)
        img_tp1 = _to_img(img_tp1, device)
        x = torch.cat([img_t, img_tp1], dim=1)
        st = state_t.to(device, non_blocking=True)
        masks = color_segment_lunar(img_t)                # on-the-fly στη GPU

        with torch.set_grad_enabled(train):
            composite, mu, logvar, comps = model(x)
            rf, ro, s, kp, ks = vae_losses(composite, img_t, comps, masks, mu, logvar, st, N_SUP)
            loss = W_FULL * rf + W_OBJ * ro + LAMBDA_SUP * s + BETA_PHYS * kp + beta_style * ks

        if train:
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        bs = img_t.size(0)
        for k, v in zip(_KEYS, (rf, ro, s, kp, ks)):
            tot[k] += float(v) * bs
        tot["n"] += bs
        n = tot["n"]
        pbar.set_postfix(full=f"{tot['recon_full']/n:.4f}", obj=f"{tot['recon_obj']/n:.4f}", sup=f"{tot['sup']/n:.4f}")

    return {k: tot[k] / tot["n"] for k in _KEYS}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    set_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean, std = load_norm_stats(NORM_STATS)
    train_ds = VaePairDataset(TRAIN_DIR, shift=SHIFT, state_mean=mean, state_std=std)
    val_ds = VaePairDataset(VAL_DIR, shift=SHIFT, state_mean=mean, state_std=std)
    pw = NUM_WORKERS > 0
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=True,
                          num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=pw)
    val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=pw)

    model = VAE_P4(latent_size=LATENT_SIZE, n_sup=N_SUP).to(device)
    dec_params = sum(p.numel() for m in (model.dec_lander, model.dec_flags, model.dec_bg) for p in m.parameters())
    print(f"device: {device} | P4 decoder params: {dec_params:,} (lander+flags+bg)")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=SCHED_PATIENCE)

    best_val, bad_epochs = float("inf"), 0
    for epoch in range(1, EPOCHS + 1):
        beta_style = BETA_STYLE_MAX * min(1.0, epoch / max(KL_ANNEAL_EPOCHS, 1))
        tr = run_epoch(model, train_dl, device, beta_style, optimizer, desc=f"E{epoch:03d} train")
        va = run_epoch(model, val_dl, device, beta_style, optimizer=None, desc=f"E{epoch:03d} val")

        val_score = va["recon_full"] + W_OBJ * va["recon_obj"] + LAMBDA_SUP * va["sup"]
        scheduler.step(val_score)
        print(f"E{epoch:03d} | TRAIN full={tr['recon_full']:.5f} obj={tr['recon_obj']:.5f} sup={tr['sup']:.5f} "
              f"| VAL full={va['recon_full']:.5f} obj={va['recon_obj']:.5f}  (select={val_score:.5f})")

        if val_score < best_val - 1e-6:
            best_val, bad_epochs = val_score, 0
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "vae_p4_best.pth"))
            print("  -> best model saved")
        else:
            bad_epochs += 1
            if bad_epochs >= EARLY_STOP_PATIENCE:
                print(f"Early stopping στο epoch {epoch}."); break

    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "vae_p4_last.pth"))
    print("Best val score:", best_val)