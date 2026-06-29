"""
sindy_eval_utils.py — Κοινά helpers για όλα τα SINDy eval/fusion scripts (LunarLander).

ΤΙ ΠΡΟΣΦΕΡΕΙ (ώστε να ΜΗΝ επαναλαμβάνεται κώδικας στα test_sindy_* / fusion_*):
  * make_noise_fn            : gaussian/salt-pepper θόρυβος σε [0,1] εικόνες (ίδιο με test_p1/p3)
  * encode_split / ensure_encoded : φορτώνουν το (frozen) baseline VAE και κωδικοποιούν splits σε
                               latent .npz dir (z, acts, states, x) — ΙΔΙΟ format/indexing με το
                               loader.precompute_latents -> τα windows ευθυγραμμίζονται 1-1 με το
                               sindy_core.assemble_windows ΚΑΙ το LatentSequenceDataset.
  * lstm_free_run_dir        : ENCODED free-running rollout του baseline LSTM (+ optional seed override)
  * encoded_measurements / seed_context : per-frame encoded z[:8] ανά window — «μέτρηση» για το
                               Kalman filtering + το seed-denoise.
  * sq_err_standardized / destandardize : κοινή σύμβαση μετρικής (standardized) όπως τα test_pX.

ΣΥΜΒΑΣΗ ΘΟΡΥΒΟΥ (συνεπής με test_p1/p3): ο θόρυβος εφαρμόζεται ΜΟΝΟ στο test encoding (πριν τον
encoder)· το SINDy/LSTM fit μένει ΠΑΝΤΑ σε CLEAN train encodings.

ΣΗΜ. (run convention): module που γίνεται import από τα standalone SINDy scripts
(`!python3 lunarlander/extra/<file>.py`). Τα αρχεία ζουν στο lunarlander/extra/ (subfolder)· το
`from sindy_core import ...` κάνει path-bootstrap που βάζει τον ΓΟΝΙΚΟ φάκελο (lunarlander/) στο
sys.path -> γίνονται importable τα vae/lstm/loader. Τα torch/vae/lstm/loader imports μένουν LAZY
(μέσα στις συναρτήσεις) ώστε οι καθαρά-numpy helpers να δουλεύουν ΧΩΡΙΣ torch.

ΑΥΤΟ ΕΙΝΑΙ ENV-SPECIFIC αρχείο (constants + config). Τα consumer scripts (test_sindy_*, fusion_*)
είναι env-AGNOSTIC και ΙΔΙΑ με του cartpole — τα τραβούν από εδώ μέσω `from sindy_eval_utils import *`.
"""
import os

import numpy as np

from sindy_core import list_npz, N_SUP, DIM_NAMES  # + path-bootstrap (vae/lstm/loader importable)


# ---------------------------------------------------------------------------
# Σταθερές αρχιτεκτονικής (LunarLander baseline)
# ---------------------------------------------------------------------------
LATENT_SIZE = 64
N_ACTIONS = 4                      # {0:noop, 1:left, 2:main, 3:right}
HIDDEN = 64
LAYERS = 2
SEQ_LEN = 30
TEST_STRIDE = 1
BATCH = 128

# ---------------------------------------------------------------------------
# ENV-SPECIFIC CONFIG — ΕΔΩ ζει όλο το per-environment config (placeholders + labels + params).
# Τα consumer scripts είναι env-AGNOSTIC και τα τραβούν από εδώ μέσω `from sindy_eval_utils import *`.
# ΜΟΝΟ αυτό το αρχείο + το sindy_core αλλάζουν ανά environment.
# Ο patcher του kaggle-run αντικαθιστά τα <...> placeholders εδώ.
# ---------------------------------------------------------------------------
ENV_TAG = "lunarlander"
DATA_ROOT = "<lunarlander-dataset>"
NORM_STATS = os.path.join(DATA_ROOT, "norm_stats.npz")
VAE_CKPT = "<lunarlander-baseline-vae>"
LSTM_CKPT = "<lunarlander-baseline-lstm>"
LATENT_ROOT = f"/kaggle/working/sindy_{ENV_TAG}_latents"          # κοινό cache encoded splits

FEATURE_MODE = "physics"          # "physics" (LunarLander-aware library) | "poly2" (generic ablation)
THRESHOLD, RIDGE = 0.02, 1e-6     # STLSQ (threshold σε normalized columns· tunable)

NOISE_CONDS = [("gaussian", 0.0), ("gaussian", 0.05), ("gaussian", 0.10)]  # θόρυβος μόνο στο test
NOISE_SEED = 42
N_BOOT, BOOT_SEED = 1000, 0

DIM_LABELS = ["x", "y", r"$v_x$", r"$v_y$", r"$\theta$", r"$\omega$", "leg1", "leg2"]   # plot labels
DIM_UNITS = ["(pos x)", "(pos y)", "(vel x)", "(vel y)", "[rad]", "[rad/s]", "(contact)", "(contact)"]
HS = [h for h in (1, 10, 20, 30) if h <= SEQ_LEN]                          # horizons στο summary


# ---------------------------------------------------------------------------
# Noise injection — float [0,1] image tensors (αντιγραφή από test_p1/p3)
# ---------------------------------------------------------------------------
def add_gaussian_noise(img_tensor, std, rng_gen):
    import torch
    noise = torch.randn(img_tensor.shape, generator=rng_gen, device=img_tensor.device) * std
    return torch.clamp(img_tensor + noise, 0.0, 1.0)


def add_salt_pepper_noise(img_tensor, amount, rng_gen):
    import torch
    mask = torch.rand(img_tensor.shape, generator=rng_gen, device=img_tensor.device)
    out = img_tensor.clone()
    out[mask < amount / 2] = 0.0
    out[mask > 1 - amount / 2] = 1.0
    return out


def make_noise_fn(noise_type, level, seed, device):
    """ -> callable(img[0,1]) -> noisy img. level==0 -> identity (clean)."""
    if level == 0.0:
        return lambda x: x
    import torch
    rng_gen = torch.Generator(device=device)
    rng_gen.manual_seed(seed)
    if noise_type == "gaussian":
        return lambda x: add_gaussian_noise(x, level, rng_gen)
    if noise_type == "salt_pepper":
        return lambda x: add_salt_pepper_noise(x, level, rng_gen)
    raise ValueError(f"Unknown noise type: {noise_type}")


def noise_tag(noise_type, level):
    return "clean" if level == 0.0 else f"{noise_type}_{level:.2f}".replace(".", "p")


# ---------------------------------------------------------------------------
# VAE / LSTM load
# ---------------------------------------------------------------------------
def load_vae(ckpt, device):
    import torch
    from vae import VAE
    vae = VAE(latent_size=LATENT_SIZE).to(device)
    vae.load_state_dict(torch.load(ckpt, map_location=device))
    vae.eval()
    return vae


def load_lstm(ckpt, device):
    import torch
    from lstm import LatentPredictor
    lstm = LatentPredictor(LATENT_SIZE, N_ACTIONS, HIDDEN, LAYERS).to(device)
    lstm.load_state_dict(torch.load(ckpt, map_location=device))
    lstm.eval()
    return lstm


# ---------------------------------------------------------------------------
# Encode ένα split -> latent .npz dir (clean ή noisy)
# ---------------------------------------------------------------------------
def encode_split(vae, src_dir, out_dir, device, noise_fn=None, shift=0, batch=256):
    """Κωδικοποιεί κάθε επεισόδιο του src_dir με τον (frozen) vae σε .npz ΙΔΙΟΥ format με
    loader.precompute_latents: z (T-1,64), acts (T-1), states (T-1,8 RAW), x (T-1,8 RAW).
    noise_fn: αν δοθεί, εφαρμόζεται σε ΟΛΑ τα frames ΠΡΙΝ το encoding (test-time noise)."""
    import torch
    os.makedirs(out_dir, exist_ok=True)
    vae.eval()
    with torch.no_grad():
        for f in list_npz(src_dir):
            with np.load(f) as d:
                imgs = torch.from_numpy(d["imgs"].astype(np.float32) / 255.0).permute(0, 3, 1, 2)
                acts = d["acts"].astype(np.float32)
                states = d["states"].astype(np.float32)
                x = (d[f"noisy_states_{shift}"] if shift in (2, 5, 10)
                     else d["states"]).astype(np.float32)

            imgs = imgs.to(device)
            if noise_fn is not None:
                imgs = noise_fn(imgs)
            img_t, img_tp1 = imgs[:-1], imgs[1:]

            zs = []
            for b in range(0, img_t.shape[0], batch):
                x_in = torch.cat([img_t[b:b + batch], img_tp1[b:b + batch]], dim=1)
                mu, _ = vae.encode(x_in)
                zs.append(mu.cpu().numpy())
            z = np.concatenate(zs, 0).astype(np.float32) if zs else np.empty((0, 0), np.float32)
            np.savez_compressed(os.path.join(out_dir, os.path.basename(f)),
                                z=z, acts=acts[:-1], states=states[:-1], x=x[:-1])


def ensure_encoded(vae_ckpt, data_root, out_root, device, noise_fn=None,
                   splits=("train", "test"), shift=0, force=False):
    """Κωδικοποιεί τα ζητούμενα splits σε <out_root>/<split>. Επιστρέφει dict split->dir.
       Παρακάμπτει split αν υπάρχει ήδη (εκτός force). Ο θόρυβος εφαρμόζεται ΜΟΝΟ στο test."""
    import torch
    dirs, vae = {}, None
    for sp in splits:
        src = os.path.join(data_root, sp)
        out = os.path.join(out_root, sp)
        dirs[sp] = out
        if not os.path.isdir(src):
            print(f"[warn] missing split dir: {src}")
            continue
        if (not force) and os.path.isdir(out) and list_npz(out):
            print(f"  [skip] already encoded: {out}")
            continue
        if vae is None:
            vae = load_vae(vae_ckpt, device)
        nf = noise_fn if sp == "test" else None      # noise μόνο στο test· train/val μένουν clean
        print(f"  encoding '{sp}' -> {out}" + (" (noisy)" if nf else " (clean)"))
        encode_split(vae, src, out, device, noise_fn=nf, shift=shift)
    if vae is not None:
        del vae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return dirs


# ---------------------------------------------------------------------------
# LSTM ENCODED free-running rollout (ίδιο με test_p1/p3 free_run)
# ---------------------------------------------------------------------------
def _free_run(model, batch, seed_phys=None):
    """seed_phys: (B,8) standardized override για τις dims[:8] του seed (None -> ως έχει)."""
    import torch
    import torch.nn.functional as F
    z_t, action, z_tp1, state_t, state_tp1 = batch
    B, L, _ = z_t.shape
    z_in = z_t[:, 0].clone()
    if seed_phys is not None:
        z_in[:, :N_SUP] = seed_phys
    hidden = model.init_hidden(B, z_t.device)
    preds = []
    for k in range(L):
        a = F.one_hot(action[:, k].long(), N_ACTIONS).float()
        z_pred, hidden = model.step(z_in, a, hidden)
        preds.append(z_pred)
        z_in = z_pred
    return torch.stack(preds, dim=1), state_tp1


def lstm_free_run_dir(lstm, latent_dir, mean, std, device,
                      seq_len=SEQ_LEN, stride=TEST_STRIDE, batch=BATCH,
                      full_latent=False, seed_phys_std=None):
    """ENCODED free-running rollout πάνω σε latent_dir (ΙΔΙΑ windows με assemble_windows).
       -> (pred (N,L,8) standardized phys, gt (N,L,8) standardized phys).
       full_latent=True   -> επιστρέφει ΕΠΙΣΗΣ pred_full (N,L,64) για παραγωγικά fusion.
       seed_phys_std (N,8) -> standardized override των seed dims[:8] (για το seed-denoise).
       ΣΗΜ.: shuffle=False ώστε η σειρά των windows να ταυτίζεται με assemble_windows."""
    import torch
    from torch.utils.data import DataLoader
    from loader import LatentSequenceDataset
    ds = LatentSequenceDataset(latent_dir, seq_len=seq_len, stride=stride,
                               state_mean=mean, state_std=std)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0, pin_memory=False)
    lstm.eval()
    P, G, PF = [], [], []
    off = 0
    with torch.no_grad():
        for b in dl:
            b = [t.to(device) for t in b]
            bs = b[0].shape[0]
            sp = None
            if seed_phys_std is not None:
                sp = torch.from_numpy(np.asarray(seed_phys_std[off:off + bs], np.float32)).to(device)
            preds, state_tp1 = _free_run(lstm, b, seed_phys=sp)
            P.append(preds[..., :N_SUP].cpu().numpy())
            G.append(state_tp1.cpu().numpy())
            if full_latent:
                PF.append(preds.cpu().numpy())
            off += bs
    pred = np.concatenate(P, 0)
    gt = np.concatenate(G, 0)
    if full_latent:
        return pred, gt, np.concatenate(PF, 0)
    return pred, gt


# ---------------------------------------------------------------------------
# Per-frame encoded z[:8] ανά window — numpy-only (μέτρηση για Kalman / seed-denoise)
# (ίδιο indexing με assemble_windows: window s -> frames s+1 .. s+L)
# ---------------------------------------------------------------------------
def seed_context(latent_dir, mean, std, seq_len=SEQ_LEN, stride=TEST_STRIDE):
    """Per-window context του seed για το SINDy seed-denoising (F). Αλιγναρισμένο ΘΕΣΗ-ΘΕΣΗ με
       sindy_core.assemble_windows. -> dict:
         enc_seed_raw (N,8): de-standardized encoded z[s,:8]
         prev_raw     (N,8): de-standardized encoded z[s-1,:8] (=enc_seed αν s==0)
         prev_act     (N,)  : action u[s-1] (αδιάφορο όπου has_prev=False)
         has_prev     (N,)  : bool — αν υπάρχει προηγούμενο frame (s>0)
       physics-predicted seed = sindy_step(prev_raw, prev_act) -> ΜΟΝΟ παρελθούσα πληροφορία."""
    mean8 = np.asarray(mean[:N_SUP], np.float64)
    std8 = np.asarray(std[:N_SUP], np.float64)
    enc, prev, pact, hp = [], [], [], []
    for f in list_npz(latent_dir):
        d = np.load(f)
        z, acts = d["z"], d["acts"]
        n = z.shape[0] - (seq_len + 1) + 1
        zphys = z[:, :N_SUP].astype(np.float64) * std8 + mean8
        for s in range(0, max(n, 0), stride):
            enc.append(zphys[s])
            if s > 0:
                prev.append(zphys[s - 1]); pact.append(acts[s - 1]); hp.append(True)
            else:
                prev.append(zphys[s]); pact.append(acts[s]); hp.append(False)
    if not enc:
        raise RuntimeError(f"No windows from {latent_dir} (seq_len={seq_len} too large?)")
    return {"enc_seed_raw": np.asarray(enc), "prev_raw": np.asarray(prev),
            "prev_act": np.asarray(pact), "has_prev": np.asarray(hp)}


def encoded_measurements(latent_dir, mean, std, seq_len=SEQ_LEN, stride=TEST_STRIDE):
    """ -> (meas_raw (N,L,8), seed_raw (N,8)) σε RAW physical units.
       meas_raw[w,k] = de-standardized encoded z[:8] στο frame s+k+1  (k=0..L-1)
       seed_raw[w]   = de-standardized encoded z[:8] στο seed frame s.
       Ταυτίζεται θέση-θέση με τα windows του sindy_core.assemble_windows."""
    mean8 = np.asarray(mean[:N_SUP], np.float64)
    std8 = np.asarray(std[:N_SUP], np.float64)
    seeds, meas = [], []
    for f in list_npz(latent_dir):
        d = np.load(f)
        z = d["z"]
        n = z.shape[0] - (seq_len + 1) + 1
        zphys = z[:, :N_SUP].astype(np.float64) * std8 + mean8
        for s in range(0, max(n, 0), stride):
            seeds.append(zphys[s])
            meas.append(zphys[s + 1:s + seq_len + 1])
    if not seeds:
        raise RuntimeError(f"No windows from {latent_dir} (seq_len={seq_len} too large?)")
    return np.asarray(meas), np.asarray(seeds)


# ---------------------------------------------------------------------------
# Μετρική: standardized squared error (N,L,8) — κοινή σύμβαση για ΟΛΕΣ τις μεθόδους
# ---------------------------------------------------------------------------
def sq_err_standardized(pred_raw, gt_raw, std):
    """RAW preds/gt -> (N,L,8) standardized squared error (όπως test_pX)."""
    std8 = np.asarray(std[:N_SUP], np.float64)
    return (((np.asarray(pred_raw, np.float64) - np.asarray(gt_raw, np.float64)) / std8) ** 2)


def destandardize(arr_std, mean, std):
    mean8 = np.asarray(mean[:N_SUP], np.float64)
    std8 = np.asarray(std[:N_SUP], np.float64)
    return np.asarray(arr_std, np.float64) * std8 + mean8
