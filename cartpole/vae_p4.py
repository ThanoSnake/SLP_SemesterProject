"""
vae_principle4.py — Principle 4 (Compositional / object-centric DECODING) VAE για CartPole.
Loaders/helpers έρχονται από το loader_final.py (τρέξε εκείνο το cell πρώτα). Στόχος (= L2P των
ερευνητών): ΑΝΤΙ για ΕΝΑΝ μεγάλο decoder, ΤΡΕΙΣ
ΜΙΚΡΟΥΣ object decoders (cart, pole, background)· καθένας ΑΝΑΚΑΤΑΣΚΕΥΑΖΕΙ ΤΗΝ ΕΙΚΟΝΑ ΤΟΥ
ΑΝΤΙΚΕΙΜΕΝΟΥ ΤΟΥ (segmented component), και τα 3 συντίθενται (overlay) στο πλήρες frame.

LOSS = σταθμισμένο άθροισμα:
    W_OBJ  * mean_i MSE(dec_i, comp_gt_i)      # ανακατασκευή ΚΑΘΕ αντικειμένου ξεχωριστά
  + W_FULL * MSE(composite, full_frame)        # ανακατασκευή ΟΛΗΣ της εικόνας
  + LAMBDA_SUP * sup + split-β KL              # ίδιος baseline backbone (P4-only ablation)

SEGMENTATION: ΔΕΝ χρειάζονται προ-αποθηκευμένα masks. Υπολογίζονται ON-THE-FLY ανά batch στη GPU
με απλό COLOR-segmentation (color_segment_cartpole): bg=λευκό, cart=σκούρα pixels (+track), pole=υπόλοιπα.
Τα GT components = «αντικείμενο πάνω σε λευκό»:  comp_i = img * mask_i + (1 - mask_i)  (λευκό = 1).
Η segmentation είναι PARTITION (κάθε pixel ΑΚΡΙΒΩΣ ένα από cart/pole/bg) -> το overlay ανακατασκευάζει το frame.

COMPOSITE (λευκό φόντο, σκούρα αντικείμενα): composite = clamp(comp_cart + comp_pole + comp_bg − 2, 0, 1).

ΑΞΙΟΛΟΓΗΣΗ (όπως το paper): full-image reconstruction MSE + SSIM + ΜΕΓΕΘΟΣ μοντέλου (#params).
Δείχνεις ότι ο P4 (3 μικροί decoders) πετυχαίνει ~ίδιο MSE/SSIM με τον baseline (1 μεγάλος) με
ΠΟΛΥ λιγότερες παραμέτρους.

ΔΕΔΟΜΕΝΑ: μόνο standard keys (imgs, acts, states). ΔΕΝ απαιτούνται mask keys.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from loader import VaePairDataset, load_norm_stats


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATA_ROOT = "<cartpole-dataset>"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR = os.path.join(DATA_ROOT, "val")
NORM_STATS = os.path.join(DATA_ROOT, "norm_stats.npz")
SAVE_DIR = "/kaggle/working/cartpole_p4_vae"

LATENT_SIZE = 64
N_SUP = 4
SHIFT = 0

# --- Color-segmentation thresholds (σε [0,1]· gym CartPole: bg=λευκό, cart/track=μαύρο, pole=tan) ---
SEG_WHITE_THR = 0.78       # pixel είναι background αν ΟΛΑ τα κανάλια > αυτό
SEG_DARK_THR = 0.59        # pixel είναι cart αν ΟΛΑ τα κανάλια < αυτό (το track λαμβάνεται ως cart)

# --- P4: ποια dims παίρνει κάθε object decoder (τα bg παίρνει ΟΛΑ τα style dims) ---
CART_DIMS = [0, 1]         # x, x_dot
POLE_DIMS = [0, 2]         # x (βάση) + theta — το pole ΧΡΕΙΑΖΕΤΑΙ το x· για αυστηρό split βάλε [2,3]
# bg_dims = list(range(N_SUP, LATENT_SIZE))  (αυτόματα)
DEC_BASE = 8               # base channels των ΜΙΚΡΩΝ object decoders (μικρό = λιγότερες παράμετροι)
DEC_BASE_BG = 16           # bg λίγο μεγαλύτερο (περισσότερα style dims)

W_OBJ = 1.0                # βάρος object-recon
W_FULL = 1.0               # βάρος full-recon

BATCH = 128
EPOCHS = 40
LR = 1e-3
BETA_PHYS = 0.01
BETA_STYLE_MAX = 1.0
KL_ANNEAL_EPOCHS = 20
LAMBDA_SUP = 1.0
EARLY_STOP_PATIENCE = 5
SCHED_PATIENCE = 3
NUM_WORKERS = 4
SEED = 0


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def _to_img(t, device):
    return t.to(device, non_blocking=True).float().div_(255.0)


# ---------------------------------------------------------------------------
# COLOR SEGMENTATION (on-the-fly, vectorized) — ο "απλός αλγόριθμος" της P4
# ---------------------------------------------------------------------------
def color_segment_cartpole(img, white_thr=SEG_WHITE_THR, dark_thr=SEG_DARK_THR):
    """ img: (B,3,H,W) σε [0,1].  Επιστρέφει (mask_cart, mask_pole, mask_bg), καθένα (B,1,H,W) float {0,1},
    ΑΥΣΤΗΡΟ PARTITION (κάθε pixel ακριβώς σε μία κλάση).
        bg   = ~λευκό  (όλα τα κανάλια > white_thr)
        cart = ~μαύρο  (όλα τα κανάλια < dark_thr)  -> περιλαμβάνει cart + τη μαύρη γραμμή του track
        pole = ό,τι απομένει (το tan κοντάρι + ο μπλε άξονας + edges)
    Καθαρά color-based· δεν χρειάζεται γεωμετρία ή προ-αποθηκευμένα masks. """
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    is_bg = (r > white_thr) & (g > white_thr) & (b > white_thr)
    is_cart = (r < dark_thr) & (g < dark_thr) & (b < dark_thr) & (~is_bg)
    is_pole = ~(is_bg | is_cart)
    return is_cart.float(), is_pole.float(), is_bg.float()


@torch.no_grad()
def visualize_segmentation(npz_path, idx=0, save_path=None):
    """ Sanity-check: δείχνει ένα frame + τις 3 μάσκες + τα 3 component GT. Τρέξε ΠΡΙΝ την εκπαίδευση. """
    import matplotlib.pyplot as plt
    with np.load(npz_path) as d:
        frame = d["imgs"][idx]
    img = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0   # (1,3,H,W)
    mc, mp, mb = color_segment_cartpole(img)
    comps = [(img * m + (1.0 - m))[0].permute(1, 2, 0).numpy() for m in (mc, mp, mb)]
    masks = [m[0, 0].numpy() for m in (mc, mp, mb)]
    titles = ["frame", "mask cart", "mask pole", "mask bg", "comp cart", "comp pole", "comp bg"]
    ims = [frame, masks[0], masks[1], masks[2], comps[0], comps[1], comps[2]]
    fig, ax = plt.subplots(1, 7, figsize=(16, 2.6))
    for a, t, im in zip(ax, titles, ims):
        a.imshow(im, cmap=("gray" if im.ndim == 2 else None), vmin=0, vmax=1)
        a.set_title(t, fontsize=9); a.axis("off")
    cover = float((mc + mp + mb).max())   # πρέπει να είναι 1.0 (partition)
    fig.suptitle(f"partition check: max overlap = {cover:.2f} (πρέπει=1.0)", fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120); print("saved:", save_path)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# DATASET / HELPERS: ΔΕΝ ξαναδηλώνονται εδώ. Έρχονται από το loader_final.py (τρέξε το cell πρώτα):
#   VaePairDataset (5-item: img_t, img_tp1, action, state_t, state_tp1), load_norm_stats, list_npz,
#   precompute_latents, LatentSequenceDataset (για το LSTM). Η P4 χρησιμοποιεί ΑΥΤΟΥΣΙΟ τον loader —
#   οι μάσκες παράγονται on-the-fly με color_segment_cartpole (πιο πάνω), όχι από το dataset.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Model — baseline encoder + 3 ΜΙΚΡΟΙ object decoders (RGB) -> overlay composite
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


class ObjDecoder(nn.Module):
    """ z -> RGB εικόνα (3,80,120) σε [0,1]: το αντικείμενο πάνω σε λευκό φόντο. 10x15 -> 80x120. """
    def __init__(self, in_dim, base_channels):
        super().__init__()
        self.base = base_channels
        self.fc = nn.Linear(in_dim, base_channels * 10 * 15)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels, base_channels, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels, max(4, base_channels // 2), 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(max(4, base_channels // 2), 3, 4, 2, 1),
        )

    def forward(self, z):
        h = self.fc(z).view(-1, self.base, 10, 15)
        return torch.sigmoid(self.decoder(h))


class VAE_P4(nn.Module):
    def __init__(self, latent_size=64, n_sup=4):
        super().__init__()
        self.latent_size = latent_size
        self.n_sup = n_sup
        self.cart_dims = list(CART_DIMS)
        self.pole_dims = list(POLE_DIMS)
        self.bg_dims = list(range(n_sup, latent_size))
        self.enc = _BaseEncoder(latent_size)
        self.dec_cart = ObjDecoder(len(self.cart_dims), DEC_BASE)
        self.dec_pole = ObjDecoder(len(self.pole_dims), DEC_BASE)
        self.dec_bg = ObjDecoder(len(self.bg_dims), DEC_BASE_BG)

    def encode(self, x):
        return self.enc.encode(x)

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def _compose(self, z):
        comp_c = self.dec_cart(z[:, self.cart_dims])
        comp_p = self.dec_pole(z[:, self.pole_dims])
        comp_b = self.dec_bg(z[:, self.bg_dims])
        composite = torch.clamp(comp_c + comp_p + comp_b - 2.0, 0.0, 1.0)   # overlay σε λευκό
        return composite, (comp_c, comp_p, comp_b)

    def decode(self, z):
        """ render ενός latent (π.χ. LSTM-predicted) -> (composite, (comp_cart, comp_pole, comp_bg)). """
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
def vae_losses(composite, full, comps, comp_gt, mu, logvar, state_t, n_sup):
    B, D = mu.size(0), mu.size(1)
    recon_full = F.mse_loss(composite, full, reduction="mean")
    recon_obj = (F.mse_loss(comps[0], comp_gt[0], reduction="mean")
                 + F.mse_loss(comps[1], comp_gt[1], reduction="mean")
                 + F.mse_loss(comps[2], comp_gt[2], reduction="mean")) / 3.0
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
        # ON-THE-FLY color segmentation στη GPU -> 3 μάσκες (B,1,H,W) -> component GT = αντικείμενο σε λευκό
        masks = color_segment_cartpole(img_t)
        comp_gt = tuple(img_t * mk + (1.0 - mk) for mk in masks)

        with torch.set_grad_enabled(train):
            composite, mu, logvar, comps = model(x)
            rf, ro, s, kp, ks = vae_losses(composite, img_t, comps, comp_gt, mu, logvar, st, N_SUP)
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
    dec_params = sum(p.numel() for m in (model.dec_cart, model.dec_pole, model.dec_bg) for p in m.parameters())
    print(f"device: {device} | P4 decoder params: {dec_params:,} (3 object decoders)")

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