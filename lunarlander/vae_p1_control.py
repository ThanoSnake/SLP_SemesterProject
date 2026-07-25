"""
vae_p1_control.py — Retrain of the VAE on a WIDE-COVERAGE control dataset (8k), with a LAZY low-RAM loader.

WHY: the diagnostic showed that the encoder is near-perfect in-distribution (theta corr 0.98, omega 0.83)
but collapses under RL/MPC exploration (theta 0.11-0.70) — a CLEAR distribution shift, NOT a lack of
capacity. Fix: retrain on the control dataset (random bursts/perturbed PID -> wide coverage of
theta/omega/vx) so that the encoder generalizes to the states RL & MPC actually visit.

EVERYTHING is import-form:
  * model + losses + train/eval loops -> from vae_p1 (or vae for the baseline)  [NO re-implementation]
  * lazy dataset + chunked sampler                                             -> from lazy_vae_loader
Run:  python3 lunarlander/vae_p1_control.py        (VAE_MODEL=baseline for the baseline VAE)

NOTE on norm_stats: this is a NEW retrain -> use the CONTROL dataset's norm_stats (mu[:8] will live
in that standardized space; the downstream scripts must load THE SAME norm_stats).
"""
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from lazy_vae_loader import VaePairDatasetLazy, ChunkedEpisodeSampler
from loader_control import load_norm_stats

from paths import CONTROL_DATA, outputs

# --- model choice: P1 (default) or baseline; both expose the same API/globals ---
VAE_MODEL = os.environ.get("VAE_MODEL", "p1")          # "p1" | "baseline"
if VAE_MODEL == "p1":
    import vae_p1 as M
    make_net = lambda: M.VAE_P1(n_sup=M.N_SUP, n_img=M.N_IMG)
    BEST_NAME, LAST_NAME = "vae_p1_control_best.pth", "vae_p1_control_last.pth"
else:
    import vae as M
    make_net = lambda: M.VAE(latent_size=M.LATENT_SIZE)
    BEST_NAME, LAST_NAME = "vae_baseline_control_best.pth", "vae_baseline_control_last.pth"

# ---------------------------------------------------------------------------
# CONFIG  (paths from config.py via paths.py)
# ---------------------------------------------------------------------------
DATA_ROOT = os.environ.get("LL_DATA_ROOT") or CONTROL_DATA   # local: LL_DATA_ROOT=...
TRAIN_DIRS = [os.path.join(DATA_ROOT, "train")]        # multi-root ready (e.g. +elite if you want)
VAL_DIRS = [os.path.join(DATA_ROOT, "val")]
NORM_STATS = os.path.join(DATA_ROOT, "norm_stats.npz")  # THE CONTROL DATASET'S NORM_STATS
SAVE_DIR = outputs("lunarlander_vae_control")

SHIFT = int(os.environ.get("SHIFT", "0"))              # 0=clean· 2/5/10 -> weak supervision
WIND_FILTER = os.environ.get("WIND_FILTER", "all")     # "all"|"clean"|"wind"
BATCH = int(os.environ.get("BATCH", "128"))
EPOCHS = int(os.environ.get("EPOCHS", str(M.EPOCHS)))
LR = float(os.environ.get("LR", str(M.LR)))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))  # Kaggle Linux/fork; the lazy ds pickles cheaply
CHUNK = int(os.environ.get("CHUNK", "64"))             # episodes per chunk (sampler)
CACHE = int(os.environ.get("CACHE", str(CHUNK + 8)))   # cache >= chunk (guarantees hits)
SEED = int(os.environ.get("SEED", "0"))


def main():
    M.set_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else
                          ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"device: {device} | model: {VAE_MODEL} | save -> {SAVE_DIR}")
    if device.type == "cpu":
        print("  WARNING device=cpu -> enable the GPU (Kaggle).")

    mean, std = load_norm_stats(NORM_STATS)
    std_phys = torch.tensor(std[:M.N_SUP], device=device)

    train_ds = VaePairDatasetLazy(TRAIN_DIRS, shift=SHIFT, state_mean=mean, state_std=std,
                                  cache_size=CACHE, wind_filter=WIND_FILTER)
    val_ds = VaePairDatasetLazy(VAL_DIRS, shift=SHIFT, state_mean=mean, state_std=std,
                                cache_size=CACHE, wind_filter=WIND_FILTER)
    print(f"train pairs: {len(train_ds):,} | val pairs: {len(val_ds):,}")

    sampler = ChunkedEpisodeSampler(train_ds.ep_ranges, chunk_size=CHUNK, seed=SEED)
    pw = NUM_WORKERS > 0
    train_dl = DataLoader(train_ds, batch_size=BATCH, sampler=sampler, drop_last=True,
                          num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=pw)
    val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False,    # sequential -> already episode-local
                        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=pw)

    model = make_net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=M.SCHED_PATIENCE)

    best_val, no_improve = float("inf"), 0
    for epoch in range(1, EPOCHS + 1):
        sampler.set_epoch(epoch)
        beta_style = M.BETA_STYLE_MAX * min(1.0, epoch / max(M.KL_ANNEAL_EPOCHS, 1))

        tr = M.run_epoch(model, train_dl, device, beta_style, optimizer, desc=f"E{epoch:03d} train")
        va = M.run_epoch(model, val_dl, device, beta_style, optimizer=None, desc=f"E{epoch:03d} val")
        rmse = M.physical_rmse(model, val_dl, device, std_phys)

        def total(m):
            return (m["recon"] + M.BETA_PHYS * m["kld_phys"]
                    + beta_style * m["kld_img"] + M.LAMBDA_SUP * m["sup"])

        val_score = va["recon"] + M.LAMBDA_SUP * va["sup"]      # beta-independent selection
        scheduler.step(val_score)
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"E{epoch:03d} | beta_style={beta_style:.2f} lr={lr_now:.1e}")
        print(f"  TRAIN total={total(tr):.5f} recon={tr['recon']:.5f} sup={tr['sup']:.5f} "
              f"kld_img={tr['kld_img']:.5f} kld_phys={tr['kld_phys']:.5f}")
        print(f"  VAL   total={total(va):.5f} recon={va['recon']:.5f} sup={va['sup']:.5f} "
              f"(select={val_score:.5f})")
        print("  VAL phys RMSE | " + "  ".join(f"{nm}={rmse[i]:.4f}" for i, nm in enumerate(M.STATE_NAMES)))

        if val_score < best_val - 1e-6:
            best_val, no_improve = val_score, 0
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, BEST_NAME))
            print(f"  -> best saved ({BEST_NAME})")
        else:
            no_improve += 1
            print(f"  (no improvement {no_improve}/{M.EARLY_STOP_PATIENCE})")
            if no_improve >= M.EARLY_STOP_PATIENCE:
                print(f"Early stopping @ epoch {epoch}.")
                break

    torch.save(model.state_dict(), os.path.join(SAVE_DIR, LAST_NAME))
    print("Best val score:", best_val)


if __name__ == "__main__":
    main()
