"""
eval_principle4.py — Αξιολόγηση Principle 4 (compositional decoding) όπως στο paper.

Μετράει & συγκρίνει baseline VAE (1 μονολιθικός decoder)  vs  VAE_P4 (3 μικροί object decoders):
    1) full-image reconstruction MSE   (όσο χαμηλότερο τόσο καλύτερα)
    2) SSIM                            (όσο υψηλότερο τόσο καλύτερα)
    3) ΜΕΓΕΘΟΣ μοντέλου = #params      (decoder-only ΚΑΙ total)
Στόχος (claim του paper): ο P4 χάνει ΕΛΑΧΙΣΤΑ σε MSE/SSIM για ΜΕΓΑΛΗ μείωση παραμέτρων decoder.

ΠΡΟΫΠΟΘΕΣΕΙΣ (notebook convention — ΔΕΝ ξαναδηλώνουμε models/loaders εδώ):
  Από προηγούμενα cells πρέπει να υπάρχουν στο scope:
    - VAE                (baseline, με .decode/.fc_decode/.decoder)
    - VAE_P4             (από το vae_principle4.py, με .decode -> (composite, comps))
    - VaePairDataset, load_norm_stats   (από το vae_principle4.py — 5-item loader, χωρίς masks)
  Αλλιώς κάνε import/run τα αντίστοιχα cells πρώτα.

ΕΚΤΕΛΕΣΗ: ρύθμισε τα paths στο CFG και τρέξε. Παράγει πίνακα σύγκρισης + (προαιρετικά) εικόνες.
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from loader import VaePairDataset, load_norm_stats
from vae import VAE
from vae_p4 import VAE_P4


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATA_ROOT = "/kaggle/working/cartpole_data"
EVAL_DIR = os.path.join(DATA_ROOT, "val")          # ή ένα ξεχωριστό test split
NORM_STATS = os.path.join(DATA_ROOT, "norm_stats.npz")
BASELINE_CKPT = "/kaggle/working/vae_baseline_out/vae_best.pth"
P4_CKPT = "/kaggle/working/vae_p4_out/vae_p4_best.pth"
LATENT_SIZE = 64
N_SUP = 4
BATCH = 128
NUM_WORKERS = 4
SAVE_FIG = "/kaggle/working/vae_p4_out/p4_recon_compare.png"   # None για παράλειψη


# ---------------------------------------------------------------------------
# SSIM (self-contained, gaussian-windowed, channel-averaged) — δεν χρειάζεται skimage
# ---------------------------------------------------------------------------
def _gaussian_window(ch, ksize=11, sigma=1.5, device="cpu"):
    coords = torch.arange(ksize, dtype=torch.float32, device=device) - (ksize - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    w2d = (g.t() @ g).unsqueeze(0).unsqueeze(0)            # (1,1,k,k)
    return w2d.expand(ch, 1, ksize, ksize).contiguous()


@torch.no_grad()
def ssim_batch(a, b, window=None):
    """ a,b: (B,3,H,W) σε [0,1]. Επιστρέφει mean SSIM (scalar). """
    ch = a.size(1)
    if window is None:
        window = _gaussian_window(ch, device=a.device)
    pad = window.size(-1) // 2
    mu_a = F.conv2d(a, window, padding=pad, groups=ch)
    mu_b = F.conv2d(b, window, padding=pad, groups=ch)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sa = F.conv2d(a * a, window, padding=pad, groups=ch) - mu_a2
    sb = F.conv2d(b * b, window, padding=pad, groups=ch) - mu_b2
    sab = F.conv2d(a * b, window, padding=pad, groups=ch) - mu_ab
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    smap = ((2 * mu_ab + C1) * (2 * sab + C2)) / ((mu_a2 + mu_b2 + C1) * (sa + sb + C2))
    return smap.mean().item()


def count_params(*modules):
    return sum(p.numel() for m in modules for p in m.parameters())


# ---------------------------------------------------------------------------
# Εξαγωγή του reconstruction από ΟΠΟΙΟΔΗΠΟΤΕ μοντέλο (baseline ή P4)
# ---------------------------------------------------------------------------
@torch.no_grad()
def model_recon(model, x):
    out = model(x)
    return out[0]            # baseline -> recon ;  P4 -> composite (πρώτο στοιχείο)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    win = None
    se, ssim_sum, n = 0.0, 0.0, 0
    for img_t, img_tp1, action, state_t, state_tp1 in tqdm(loader, desc="eval", leave=False):
        img_t = img_t.to(device).float().div_(255.0)
        img_tp1 = img_tp1.to(device).float().div_(255.0)
        x = torch.cat([img_t, img_tp1], dim=1)
        recon = model_recon(model, x).clamp(0, 1)
        if win is None:
            win = _gaussian_window(recon.size(1), device=device)
        bs = img_t.size(0)
        se += F.mse_loss(recon, img_t, reduction="mean").item() * bs
        ssim_sum += ssim_batch(recon, img_t, win) * bs
        n += bs
    return se / n, ssim_sum / n


def decoder_param_count(model):
    """ Μετράει ΜΟΝΟ τις παραμέτρους του decoder (το μέρος που αλλάζει το P4). """
    if hasattr(model, "dec_cart"):           # VAE_P4
        return count_params(model.dec_cart, model.dec_pole, model.dec_bg)
    parts = []
    if hasattr(model, "fc_decode"): parts.append(model.fc_decode)
    if hasattr(model, "decoder"):   parts.append(model.decoder)
    return count_params(*parts)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = load_norm_stats(NORM_STATS)                       # noqa: F821 (prior cell)
    ds = VaePairDataset(EVAL_DIR, shift=0, state_mean=mean, state_std=std)  # noqa: F821
    dl = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    base = VAE(latent_size=LATENT_SIZE).to(device)               # noqa: F821 (prior cell)
    base.load_state_dict(torch.load(BASELINE_CKPT, map_location=device))
    p4 = VAE_P4(latent_size=LATENT_SIZE, n_sup=N_SUP).to(device)  # noqa: F821 (vae_principle4.py)
    p4.load_state_dict(torch.load(P4_CKPT, map_location=device))

    b_mse, b_ssim = evaluate(base, dl, device)
    p_mse, p_ssim = evaluate(p4, dl, device)
    b_dec, p_dec = decoder_param_count(base), decoder_param_count(p4)
    b_tot = count_params(base); p_tot = count_params(p4)

    print("\n" + "=" * 64)
    print(f"{'metric':<22}{'baseline':>14}{'P4 (3 dec)':>14}{'Δ':>14}")
    print("-" * 64)
    print(f"{'recon MSE  (↓)':<22}{b_mse:>14.6f}{p_mse:>14.6f}{(p_mse-b_mse):>+14.6f}")
    print(f"{'SSIM       (↑)':<22}{b_ssim:>14.4f}{p_ssim:>14.4f}{(p_ssim-b_ssim):>+14.4f}")
    print(f"{'decoder params (↓)':<22}{b_dec:>14,}{p_dec:>14,}{f'{100*(1-p_dec/b_dec):.1f}% less':>14}")
    print(f"{'total params':<22}{b_tot:>14,}{p_tot:>14,}{f'{100*(1-p_tot/b_tot):.1f}% less':>14}")
    print("=" * 64)
    print(f"\n>> P4: {100*(1-p_dec/b_dec):.1f}% λιγότερες decoder-params, ΔSSIM={p_ssim-b_ssim:+.4f}, "
          f"ΔMSE={p_mse-b_mse:+.6f}")

    # --- προαιρετικά: οπτική σύγκριση (full + 3 components του P4) ---
    if SAVE_FIG:
        import matplotlib.pyplot as plt
        img_t, img_tp1, *_ = next(iter(dl))
        img_t = img_t.to(device).float().div_(255.0)
        x = torch.cat([img_t, img_tp1.to(device).float().div_(255.0)], dim=1)
        with torch.no_grad():
            comp, (cc, cp, cb) = p4.decode(p4.encode(x)[0])
            base_r = model_recon(base, x).clamp(0, 1)
        k = min(4, img_t.size(0))
        cols = [("GT", img_t), ("baseline", base_r), ("P4 full", comp),
                ("P4 cart", cc), ("P4 pole", cp), ("P4 bg", cb)]
        fig, ax = plt.subplots(k, len(cols), figsize=(2.2 * len(cols), 2.2 * k))
        for r in range(k):
            for c, (title, im) in enumerate(cols):
                a = ax[r, c] if k > 1 else ax[c]
                a.imshow(im[r].permute(1, 2, 0).cpu().numpy()); a.axis("off")
                if r == 0: a.set_title(title, fontsize=9)
        plt.tight_layout(); plt.savefig(SAVE_FIG, dpi=120); print("saved:", SAVE_FIG)