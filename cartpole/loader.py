""" loader.py — EAGER all-in-RAM loading (καμία cache/LRU, μηδέν IO στο train).

Φιλοσοφία:
  * Όλο το dataset φορτώνεται ΜΙΑ φορά στο __init__ (με progress bar). Μετά κάθε
    __getitem__ είναι καθαρή RAM-πρόσβαση -> το γρηγορότερο δυνατό train.
  * Οι εικόνες μένουν uint8 στη RAM· το /255 γίνεται per-sample (φθηνό). Έτσι ΔΕΝ
    τετραπλασιάζουμε τη μνήμη (float θα ήταν 4×).
  * Global flat index (αρχείο, θέση t) -> DataLoader(shuffle=True) δίνει αληθινό
    ανακάτεμα ΑΝΑΜΕΣΑ σε επεισόδια.
  * shift in {2,5,10} -> noisy_states_{shift}. Προαιρετικό standardize states.
  * Με num_workers>0 σε Linux, το self.eps μοιράζεται μέσω fork copy-on-write
    (δεν διπλασιάζεται η RAM). Χρησιμοποίησε persistent_workers=True.

Ροή: VaePairDataset -> (train VAE) -> precompute_latents -> LatentSequenceDataset.
"""
from os import listdir, makedirs
from os.path import join, isdir, basename

import numpy as np
import torch
import torch.utils.data
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def list_npz(root):
    files = []
    for sd in sorted(listdir(root)):
        p = join(root, sd)
        if isdir(p):
            files += [join(p, f) for f in sorted(listdir(p)) if f.endswith(".npz")]
        elif p.endswith(".npz"):
            files.append(p)
    return sorted(files)


def load_norm_stats(path):
    z = np.load(path)
    return z["mean"].astype(np.float32), z["std"].astype(np.float32)


def _standardize(s, mean, std):
    return s if mean is None else ((s - mean) / std).astype(np.float32)


# ---------------------------------------------------------------------------
# Κοινή βάση: EAGER load όλων των επεισοδίων στη RAM + flat index
# ---------------------------------------------------------------------------
class _BaseRollout(torch.utils.data.Dataset):
    def __init__(self, root, window, stride=1, shift=0,
                 state_mean=None, state_std=None, cache_size=None):
        # cache_size: αγνοείται (κρατιέται για συμβατότητα με παλιές κλήσεις)
        self.window = window
        self.shift = shift
        self.mean = None if state_mean is None else np.asarray(state_mean, np.float32)
        self.std = None if state_std is None else np.asarray(state_std, np.float32)

        files = list_npz(root)
        if not files:
            raise RuntimeError(f"Κανένα .npz στο: {root}")

        self.eps, self.index, angles = [], [], []
        tag = basename(root.rstrip("/")) or root
        for fi, f in enumerate(tqdm(files, desc=f"loading '{tag}' -> RAM")):
            with np.load(f) as d:
                ep = {
                    "imgs": d["imgs"],                                  # uint8 (T,H,W,3) — μένει uint8
                    "acts": d["acts"].astype(np.float32),               # (T,)
                    "states": d["states"].astype(np.float32),           # (T,4) clean
                    "x": (d[f"noisy_states_{shift}"] if shift in (2, 5, 10)
                          else d["states"]).astype(np.float32),         # (T,4) input
                }
            self.eps.append(ep)
            T = ep["states"].shape[0]
            for s in range(0, max(T - window + 1, 0), stride):
                self.index.append((fi, s))
                angles.append(ep["states"][s, 2])
        self.angles = np.asarray(angles, np.float32)

    def __len__(self):
        return len(self.index)


# ---------------------------------------------------------------------------
# VAE: 1 δείγμα = ζεύγος (t, t+1)
# ---------------------------------------------------------------------------
class VaePairDataset(_BaseRollout):
    """ Επιστρέφει: img_t, img_tp1 (3,H,W), action_t, state_t (4), state_tp1 (4). """
    def __init__(self, root, **kw):
        super().__init__(root, window=2, **kw)

    def __getitem__(self, i):
        fi, t = self.index[i]
        ep = self.eps[fi]
        # Επιστρέφουμε uint8 (C,H,W)· το .float()/255 γίνεται στη GPU (run_epoch),
        # ώστε να μη φορτώνεται η CPU και να μεταφέρονται 4× λιγότερα bytes στο PCIe.
        img_t = torch.from_numpy(ep["imgs"][t]).permute(2, 0, 1)        # uint8
        img_tp1 = torch.from_numpy(ep["imgs"][t + 1]).permute(2, 0, 1)  # uint8
        action = torch.tensor(ep["acts"][t])
        state_t = torch.from_numpy(_standardize(ep["x"][t], self.mean, self.std))
        state_tp1 = torch.from_numpy(_standardize(ep["states"][t + 1], self.mean, self.std))
        return img_t, img_tp1, action, state_t, state_tp1


# ---------------------------------------------------------------------------
# LSTM (image-based εναλλακτική): παράθυρο seq_len ζευγών
# ---------------------------------------------------------------------------
class SequenceDataset(_BaseRollout):
    """ seq_len διαδοχικά ζεύγη (t,t+1). Για 30 βήματα -> seq_len=31.
    Επιστρέφει: img_t (L,3,H,W), img_tp1 (L,3,H,W), actions (L,), states (L,4),
                states_clean (L,4).  Ανακάτεψε τα ΠΑΡΑΘΥΡΑ (shuffle=True), όχι μέσα τους. """
    def __init__(self, root, seq_len=31, stride=1, **kw):
        self.seq_len = seq_len
        super().__init__(root, window=seq_len + 1, stride=stride, **kw)

    def __getitem__(self, i):
        fi, s = self.index[i]
        ep = self.eps[fi]
        L = self.seq_len
        frames = ep["imgs"][s:s + L + 1].astype(np.float32) / 255.0      # (L+1,H,W,3)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)            # (L+1,3,H,W)
        img_t = frames[:L].contiguous()
        img_tp1 = frames[1:].contiguous()
        actions = torch.from_numpy(ep["acts"][s:s + L])
        states = torch.from_numpy(_standardize(ep["x"][s:s + L], self.mean, self.std))
        states_clean = torch.from_numpy(_standardize(ep["states"][s:s + L], self.mean, self.std))
        return img_t, img_tp1, actions, states, states_clean


# ---------------------------------------------------------------------------
# Pre-encoding: τρέξε τον (παγωμένο) VAE μία φορά -> z-ακολουθίες
# ---------------------------------------------------------------------------
@torch.no_grad()
def precompute_latents(encode_fn, root, out_root, shift=0, batch=256, device="cuda"):
    """ encode_fn(img_t, img_tp1) -> z. Ανά επεισόδιο σώζει .npz με z,acts,states,x (N=T-1). """
    makedirs(out_root, exist_ok=True)
    for f in tqdm(list_npz(root), desc="encoding"):
        with np.load(f) as d:
            imgs = torch.from_numpy(d["imgs"].astype(np.float32) / 255.0).permute(0, 3, 1, 2)
            acts = d["acts"].astype(np.float32)
            states = d["states"].astype(np.float32)
            x = (d[f"noisy_states_{shift}"] if shift in (2, 5, 10) else d["states"]).astype(np.float32)

        img_t, img_tp1 = imgs[:-1], imgs[1:]
        zs = []
        for b in range(0, img_t.shape[0], batch):
            zb = encode_fn(img_t[b:b + batch].to(device), img_tp1[b:b + batch].to(device))
            zs.append(zb.cpu().numpy())
        z = np.concatenate(zs, 0).astype(np.float32) if zs else np.empty((0, 0), np.float32)

        np.savez_compressed(join(out_root, basename(f)),
                            z=z, acts=acts[:-1], states=states[:-1], x=x[:-1])


# ---------------------------------------------------------------------------
# LSTM (συνιστώμενο): παράθυρα latents (ήδη all-in-RAM)
# ---------------------------------------------------------------------------
class LatentSequenceDataset(torch.utils.data.Dataset):
    """ seq_len ΜΕΤΑΒΑΣΕΙΣ latents. Για 30 βήματα -> seq_len=30.
    Επιστρέφει: z_t (L,latent), action (L,), z_tp1 (L,latent), state_t (L,4), state_tp1 (L,4).
      - state_t   : input state στο t  (x -> noisy αν shift), standardized
      - state_tp1 : CLEAN state στο t+1, standardized (στόχος για physical MSE / hybrid gt) """
    def __init__(self, root, seq_len=30, stride=1, state_mean=None, state_std=None):
        self.seq_len = seq_len
        self.mean = None if state_mean is None else np.asarray(state_mean, np.float32)
        self.std = None if state_std is None else np.asarray(state_std, np.float32)

        self.eps, self.index = [], []
        for fi, f in enumerate(tqdm(list_npz(root), desc="loading latents -> RAM")):
            with np.load(f) as d:
                ep = {k: d[k].astype(np.float32) for k in ("z", "acts", "states", "x")}
            self.eps.append(ep)
            n = ep["z"].shape[0] - (seq_len + 1) + 1
            for s in range(0, max(n, 0), stride):
                self.index.append((fi, s))
        if not self.index:
            raise RuntimeError(f"Δεν βγήκαν παράθυρα από: {root} (seq_len πολύ μεγάλο;)")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        fi, s = self.index[i]
        ep = self.eps[fi]
        L = self.seq_len
        z = ep["z"][s:s + L + 1]
        z_t = torch.from_numpy(z[:-1])
        z_tp1 = torch.from_numpy(z[1:])
        action = torch.from_numpy(ep["acts"][s:s + L])
        state_t = torch.from_numpy(_standardize(ep["x"][s:s + L], self.mean, self.std))
        state_tp1 = torch.from_numpy(_standardize(ep["states"][s + 1:s + L + 1], self.mean, self.std))
        return z_t, action, z_tp1, state_t, state_tp1


# ---------------------------------------------------------------------------
# Προαιρετικό: ισορρόπηση ουρών (σπάνιες γωνίες)
# ---------------------------------------------------------------------------
def angle_balanced_weights(dataset, bins=40):
    ang = dataset.angles
    hist, edges = np.histogram(ang, bins=bins)
    idx = np.clip(np.digitize(ang, edges[:-1]) - 1, 0, len(hist) - 1)
    return torch.as_tensor(1.0 / (hist[idx] + 1.0), dtype=torch.double)
