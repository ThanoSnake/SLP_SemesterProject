"""
loader_control.py — MULTI-ROOT loader για το ΣΥΝΔΥΑΣΜΕΝΟ control + elite dataset (~12k).

Αντίστοιχο του loader.py, αλλά:
  * MULTI-ROOT: κάθε function/dataset δέχεται str Ή λίστα από split-dirs
    (π.χ. [control/train, elite/train]) -> ένωση χωρίς αντιγραφή αρχείων.
  * precompute_latents: STREAMING (ένα επεισόδιο τη φορά -> χαμηλή RAM), unique ονόματα
    εξόδου (ep000000.npz, ...) ώστε να μη συγκρούονται μεταξύ datasets, + WIND_FILTER.
  * WIND_FILTER ("all"|"clean"|"wind"): φιλτράρει βάσει του key 'wind_enabled' των νέων .npz.
  * LatentSequenceDataset: multi-root, eager (latents ~MB -> ΟΚ ακόμα και για 12k).
  * TransitionDataset: states/acts/next_states/rewards/dones (ΧΩΡΙΣ εικόνες) για reward selection.

ΣΗΜ. norm_stats: επειδή ΞΑΝΑΧΡΗΣΙΜΟΠΟΙΟΥΜΕ το ίδιο VAE, χρησιμοποίησε τα ΑΡΧΙΚΑ norm_stats
(αυτά που εκπαιδεύτηκε το VAE) -> το mu[:8] είναι σε εκείνο το standardized space.
"""
import os
from os import listdir, makedirs
from os.path import join, isdir

import numpy as np
import torch
import torch.utils.data
from tqdm.auto import tqdm


STATE_DIM = 8
ANGLE_DIM = 4
_NOISY = (2, 5, 10)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _as_roots(roots):
    if isinstance(roots, (str, bytes, os.PathLike)):
        return [os.fspath(roots)]
    return [os.fspath(r) for r in roots]


def list_npz(roots):
    """ΟΛΑ τα .npz κάτω από ένα ή περισσότερα roots (str ή λίστα)."""
    files = []
    for root in _as_roots(roots):
        if not isdir(root):
            continue
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


def _is_wind(d):
    return bool(np.asarray(d["wind_enabled"]).item()) if "wind_enabled" in d else False


def _wind_skip(d, wind_filter):
    if wind_filter == "all":
        return False
    w = _is_wind(d)
    return (wind_filter == "clean" and w) or (wind_filter == "wind" and not w)


def compute_norm_stats(roots, out_path):
    """mean/std των states στο ΣΥΝΔΥΑΣΜΕΝΟ train (φθηνό· διαβάζει μόνο 'states').
    ΣΗΜ.: για reuse του ίδιου VAE προτίμησε τα ΑΡΧΙΚΑ norm_stats αντί γι' αυτά."""
    files = list_npz(roots)
    if not files:
        raise RuntimeError(f"No .npz under {roots}")
    n, ssum, ssq = 0, np.zeros(STATE_DIM, np.float64), np.zeros(STATE_DIM, np.float64)
    for f in tqdm(files, desc="norm-stats"):
        with np.load(f) as d:
            s = d["states"].astype(np.float64)
        n += s.shape[0]; ssum += s.sum(0); ssq += (s * s).sum(0)
    mean = (ssum / n).astype(np.float32)
    std = (np.sqrt(np.maximum(ssq / n - (ssum / n) ** 2, 0.0)) + 1e-8).astype(np.float32)
    np.savez(out_path, mean=mean, std=std)
    print("State mean:", mean, "\nState std :", std)
    return mean, std


# ---------------------------------------------------------------------------
# Pre-encoding (STREAMING) -> z-ακολουθίες. Multi-root + WIND_FILTER + unique names.
# ---------------------------------------------------------------------------
@torch.no_grad()
def precompute_latents(encode_fn, roots, out_root, shift=0, batch=256, device="cuda", wind_filter="all"):
    """ encode_fn(img_t, img_tp1) -> z. Ένα επεισόδιο τη φορά (χαμηλή RAM).
    wind_filter: 'all' (όλα) | 'clean' (μόνο χωρίς άνεμο) | 'wind' (μόνο με άνεμο). """
    makedirs(out_root, exist_ok=True)
    gi, skipped = 0, 0
    for f in tqdm(list_npz(roots), desc=f"encoding ({wind_filter})"):
        with np.load(f) as d:
            if _wind_skip(d, wind_filter):
                skipped += 1
                continue
            imgs = torch.from_numpy(d["imgs"].astype(np.float32) / 255.0).permute(0, 3, 1, 2)
            acts = d["acts"].astype(np.float32)
            states = d["states"].astype(np.float32)
            x = (d[f"noisy_states_{shift}"] if shift in _NOISY else states).astype(np.float32)
            img_t, img_tp1 = imgs[:-1], imgs[1:]
            zs = []
            for b in range(0, img_t.shape[0], batch):
                zb = encode_fn(img_t[b:b + batch].to(device), img_tp1[b:b + batch].to(device))
                zs.append(zb.cpu().numpy())
            z = np.concatenate(zs, 0).astype(np.float32) if zs else np.empty((0, 0), np.float32)
        np.savez_compressed(join(out_root, f"ep{gi:06d}.npz"),
                            z=z, acts=acts[:-1], states=states[:-1], x=x[:-1])
        gi += 1
    print(f"  encoded {gi} episodes (skipped {skipped} by wind_filter='{wind_filter}') -> {out_root}")
    return gi


# ---------------------------------------------------------------------------
# LSTM: παράθυρα latents — multi-root, eager
# ---------------------------------------------------------------------------
class LatentSequenceDataset(torch.utils.data.Dataset):
    """ seq_len ΜΕΤΑΒΑΣΕΙΣ latents. Επιστρέφει: z_t (L,latent), action (L,), z_tp1 (L,latent),
    state_t (L,8), state_tp1 (L,8). Multi-root (str ή λίστα). """
    def __init__(self, roots, seq_len=30, stride=1, state_mean=None, state_std=None):
        self.seq_len = seq_len
        self.mean = None if state_mean is None else np.asarray(state_mean, np.float32)
        self.std = None if state_std is None else np.asarray(state_std, np.float32)
        self.eps, self.index = [], []
        for fi, f in enumerate(tqdm(list_npz(roots), desc="latents -> RAM")):
            with np.load(f) as d:
                ep = {k: d[k].astype(np.float32) for k in ("z", "acts", "states", "x")}
            self.eps.append(ep)
            n = ep["z"].shape[0] - (seq_len + 1) + 1
            for s in range(0, max(n, 0), stride):
                self.index.append((fi, s))
        if not self.index:
            raise RuntimeError(f"No latent windows from {roots} (seq_len too large?)")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        fi, s = self.index[i]
        ep = self.eps[fi]; L = self.seq_len
        z = ep["z"][s:s + L + 1]
        z_t = torch.from_numpy(z[:-1]); z_tp1 = torch.from_numpy(z[1:])
        action = torch.from_numpy(ep["acts"][s:s + L])
        state_t = torch.from_numpy(_standardize(ep["x"][s:s + L], self.mean, self.std))
        state_tp1 = torch.from_numpy(_standardize(ep["states"][s + 1:s + L + 1], self.mean, self.std))
        return z_t, action, z_tp1, state_t, state_tp1


# ---------------------------------------------------------------------------
# Transitions (reward selection / MPC) — ΧΩΡΙΣ εικόνες -> ελάχιστη RAM
# ---------------------------------------------------------------------------
class TransitionDataset(torch.utils.data.Dataset):
    """ Per-step: state_t (8), action (), next_state (8), reward (), done ().
    Νέα keys (next_states/rewards/dones)· για παλιά αρχεία παράγει από t+1.
    state_mean/std -> standardized, αλλιώς RAW. wind_filter όπως στο precompute. """
    def __init__(self, roots, shift=0, state_mean=None, state_std=None, wind_filter="all"):
        self.mean = None if state_mean is None else np.asarray(state_mean, np.float32)
        self.std = None if state_std is None else np.asarray(state_std, np.float32)
        self.eps, self.index = [], []
        for f in tqdm(list_npz(roots), desc="transitions -> RAM"):
            with np.load(f) as d:
                if _wind_skip(d, wind_filter):
                    continue
                states = d["states"].astype(np.float32)
                acts = d["acts"].astype(np.float32)
                x = (d[f"noisy_states_{shift}"] if shift in _NOISY else states).astype(np.float32)
                T = states.shape[0]
                nxt = (d["next_states"].astype(np.float32) if "next_states" in d
                       else np.concatenate([states[1:], states[-1:]], 0))
                rew = d["rewards"].astype(np.float32) if "rewards" in d else np.zeros(T, np.float32)
                if "dones" in d:
                    done = d["dones"].astype(np.float32)
                else:
                    done = np.zeros(T, np.float32); done[-1] = 1.0
            fi = len(self.eps)
            self.eps.append({"x": x, "acts": acts, "next": nxt, "rew": rew, "done": done})
            self.index += [(fi, t) for t in range(T)]
        if not self.index:
            raise RuntimeError(f"No transitions from {roots}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        fi, t = self.index[i]
        ep = self.eps[fi]
        state = torch.from_numpy(_standardize(ep["x"][t], self.mean, self.std))
        next_state = torch.from_numpy(_standardize(ep["next"][t], self.mean, self.std))
        return state, torch.tensor(ep["acts"][t]), next_state, torch.tensor(ep["rew"][t]), torch.tensor(ep["done"][t])
