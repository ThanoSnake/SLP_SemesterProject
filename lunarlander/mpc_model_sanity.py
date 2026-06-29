"""
mpc_model_sanity.py — Γιατί αποτυγχάνει ο MPC; Μετράει ΑΜΕΣΑ την ποιότητα του world-model
με το DEPLOYED ζευγάρι (VAE encoder + LSTM), σε ΠΡΑΓΜΑΤΙΚΑ test επεισόδια.

Τρεις δοκιμές:
  (1) 1-STEP ENCODER-CONSISTENCY: z_{t+1}^pred (LSTM από z_t, a_t) vs z_{t+1}^enc (encoder).
      Αν είναι ΜΕΓΑΛΟ -> VAE↔LSTM mismatch (λάθος ζευγάρι) ή σπασμένο μοντέλο.
  (2) ROLLOUT MSE vs TRUE state, free-running H βήματα, με (a) HEURISTIC (πραγματικές) ενέργειες
      [in-distribution] vs (b) RANDOM ενέργειες [off-distribution]. Αν (a) μικρό αλλά (b) ΜΕΓΑΛΟ
      -> το μοντέλο είναι αξιόπιστο μόνο σε heuristic actions -> ο MPC (free actions) ονειρεύεται σκουπίδια.
  (3) ACTION-SENSITIVITY: σταθερό action K βήματα -> Δ ανά dim (αποκρίνεται στις ενέργειες;).

Imports από τα canonical modules· τρέξε: python lunarlander/mpc_model_sanity.py (αφού patch-αριστούν τα paths).
"""
import os
import numpy as np
import torch
import torch.nn.functional as F

from vae_p1 import VAE_P1
from lstm import LatentPredictor
from loader import list_npz, load_norm_stats

# --- CONFIG (ίδια placeholders με το extension· ο patcher τα γεμίζει) ---
DATA_ROOT = "<lunarlander-dataset>"
NORM_STATS = os.path.join(DATA_ROOT, "norm_stats.npz")
VAE_CKPT = "<lunarlander-p1-vae>"
LSTM_CKPT = "<lunarlander-p1-lstm>"

LATENT_SIZE, N_SUP, N_IMG = 64, 8, 56
N_ACTIONS, HIDDEN, LAYERS = 4, 64, 2
N_EP = 8                 # πόσα test επεισόδια
H = 10                   # rollout horizon (όσο το MPC effective horizon)
WINDOWS_PER_EP = 8
SEED = 0
DIM = ["x", "y", "vx", "vy", "theta", "omega", "leg1", "leg2"]


def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def encode_episode(vae, f, device):
    """ -> z (T-1,64) encoder latents, acts (T-1,), states_next (T-1,8) RAW (το state στο t+1). """
    with np.load(f) as d:
        imgs = torch.from_numpy(d["imgs"].astype(np.float32) / 255.0).permute(0, 3, 1, 2)
        acts = d["acts"].astype(np.float32)
        states = d["states"].astype(np.float32)
    img_t, img_tp1 = imgs[:-1], imgs[1:]
    zs = []
    for b in range(0, img_t.shape[0], 256):
        x = torch.cat([img_t[b:b+256], img_tp1[b:b+256]], dim=1).to(device)
        mu, _ = vae.encode(x)
        zs.append(mu.cpu())
    z = torch.cat(zs, 0) if zs else torch.empty(0, LATENT_SIZE)
    return z, acts[:-1], states[1:]                    # z[t] ~ (frame_t,frame_{t+1}); next state = states[t+1]


@torch.no_grad()
def main():
    np.random.seed(SEED); torch.manual_seed(SEED)
    device = get_device()
    print("device:", device, "| VAE:", VAE_CKPT, "| LSTM:", LSTM_CKPT)
    mean, std = load_norm_stats(NORM_STATS)
    mean_t = torch.tensor(mean, device=device); std_t = torch.tensor(std, device=device)
    std8 = std_t[:N_SUP]; mean8 = mean_t[:N_SUP]

    vae = VAE_P1(n_sup=N_SUP, n_img=N_IMG).to(device)
    vae.load_state_dict(torch.load(VAE_CKPT, map_location=device)); vae.eval()
    lstm = LatentPredictor(LATENT_SIZE, N_ACTIONS, HIDDEN, LAYERS).to(device)
    lstm.load_state_dict(torch.load(LSTM_CKPT, map_location=device)); lstm.eval()

    files = list_npz(os.path.join(DATA_ROOT, "test"))[:N_EP]
    rng = np.random.default_rng(SEED)

    resid1, roll_h, roll_r = [], np.zeros(H), np.zeros(H)
    nb_h = nb_r = 0
    for f in files:
        z, acts, snext = encode_episode(vae, f, device)
        z = z.to(device)
        T = z.shape[0]
        if T < H + 2:
            continue
        # (1) 1-step consistency: z_pred(z_t,a_t)[:8] vs encoder z_{t+1}[:8]  (φυσικές μονάδες)
        a = F.one_hot(torch.from_numpy(acts).long().to(device), N_ACTIONS).float()
        zp, _ = lstm.step(z[:-1], a[:-1], lstm.init_hidden(T - 1, device))
        d1 = torch.norm((zp[:, :N_SUP] - z[1:, :N_SUP]) * std8, dim=1)        # phys L2
        resid1.append(d1.mean().item())

        # (2) H-step rollout vs TRUE state, (a) heuristic actions (b) random actions
        starts = rng.integers(0, T - H - 1, size=min(WINDOWS_PER_EP, T - H - 1))
        sn = torch.from_numpy(snext).to(device)
        for s in starts:
            for mode in ("heur", "rand"):
                zi = z[s:s+1].clone(); hid = lstm.init_hidden(1, device)
                acc = np.zeros(H)
                for k in range(H):
                    if mode == "heur":
                        ak = int(acts[s + k])
                    else:
                        ak = int(rng.integers(0, N_ACTIONS))
                    aoh = F.one_hot(torch.tensor([ak], device=device), N_ACTIONS).float()
                    zi, hid = lstm.step(zi, aoh, hid)
                    pred_phys = zi[0, :N_SUP] * std8 + mean8
                    true_phys = sn[s + k]                              # RAW true state at t+1+k
                    acc[k] = ((pred_phys - true_phys) ** 2).mean().item()
                if mode == "heur":
                    roll_h += acc; nb_h += 1
                else:
                    roll_r += acc; nb_r += 1

    print("\n" + "=" * 60)
    print(f"(1) 1-STEP encoder-consistency (phys L2): {np.mean(resid1):.4f}")
    print("    (μικρό ~<0.3 -> VAE+LSTM ταιριάζουν· μεγάλο -> mismatch/σπασμένο)")
    print("\n(2) ROLLOUT MSE vs TRUE state (physical, mean over dims) ανά horizon:")
    print(f"    {'h':>3}{'heuristic(in-dist)':>22}{'random(off-dist)':>20}{'ratio':>9}")
    for k in range(H):
        rh, rr = roll_h[k] / max(nb_h, 1), roll_r[k] / max(nb_r, 1)
        print(f"    {k+1:>3}{rh:>22.4f}{rr:>20.4f}{rr/max(rh,1e-9):>9.1f}x")
    print("    -> αν random >> heuristic: το μοντέλο είναι αναξιόπιστο σε free actions -> MPC αποτυγχάνει.")

    # (3) action-sensitivity από ένα airborne z
    z0 = None
    for f in files:
        zz, _, _ = encode_episode(vae, f, device)
        if zz.shape[0] > 25:
            z0 = zz[20:21].to(device); break
    if z0 is not None:
        p0 = (z0[0, :N_SUP] * std8 + mean8).cpu().numpy()
        print(f"\n(3) ACTION-SENSITIVITY (σταθερό action {H} βήματα)  start y={p0[1]:+.3f} vy={p0[3]:+.3f}")
        print(f"    {'action':<7}{'Δy':>9}{'Δvy':>9}{'Δvx':>9}{'Δtheta':>9}")
        for act, nm in ((0, "noop"), (2, "main"), (1, "left"), (3, "right")):
            zi = z0.clone(); hid = lstm.init_hidden(1, device)
            aoh = F.one_hot(torch.tensor([act], device=device), N_ACTIONS).float()
            for _ in range(H):
                zi, hid = lstm.step(zi, aoh, hid)
            d = (zi[0, :N_SUP] * std8 + mean8).cpu().numpy() - p0
            print(f"    {nm:<7}{d[1]:>+9.3f}{d[3]:>+9.3f}{d[2]:>+9.3f}{d[4]:>+9.3f}")
        print("    -> 'main' πρέπει Δy/Δvy αισθητά > 'noop'· left/right να αλλάζουν vx/theta.")


if __name__ == "__main__":
    main()
