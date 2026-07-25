"""
loader.py — EAGER all-in-RAM loading for LunarLander (state=8D), following
cartpole's loader_final.py (no cache/LRU, zero IO during training).

Rationale (same as cartpole):
  * The whole dataset is loaded ONCE in __init__ (progress bar). After that every
    __getitem__ is a pure RAM access -> the fastest possible training loop.
  * Images stay uint8 in RAM; the /255 happens per sample (cheap). This way we do NOT
    quadruple the memory (float would be 4x).
  * Global flat index (file, position t) -> DataLoader(shuffle=True) gives true
    shuffling ACROSS episodes (not the authors' random sampling).
  * shift in {2,5,10} -> noisy_states_{shift} (same keys as LunaDataCollect.py).
    Standardizing the states with the train norm_stats is optional.
  * With num_workers>0 on Linux, self.eps is shared via fork copy-on-write
    (RAM is not duplicated). Use persistent_workers=True.

DIFFERENCES from cartpole: state 4D -> 8D, and the angle dim (theta) is 4 (not 2).
obs = [x, y, vx, vy, theta, omega, leg1, leg2].

Flow: VaePairDataset -> (train VAE) -> precompute_latents -> LatentSequenceDataset.
"""
from os import listdir, makedirs
from os.path import join, isdir, basename

import numpy as np
import torch
import torch.utils.data
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# LunarLander-specific constants
# ---------------------------------------------------------------------------
STATE_DIM = 8                 # [x, y, vx, vy, theta, omega, leg1, leg2]
ANGLE_DIM = 4                 # index of theta (it was 2 in cartpole) -> used for balancing


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
# Shared base: EAGER load of every episode into RAM + a flat index
# ---------------------------------------------------------------------------
class _BaseRollout(torch.utils.data.Dataset):
    def __init__(self, root, window, stride=1, shift=0,
                 state_mean=None, state_std=None, cache_size=None):
        # cache_size: ignored (kept for backward compatibility with old call sites)
        self.window = window
        self.shift = shift
        self.mean = None if state_mean is None else np.asarray(state_mean, np.float32)
        self.std = None if state_std is None else np.asarray(state_std, np.float32)

        files = list_npz(root)
        if not files:
            raise RuntimeError(f"No .npz files in: {root}")

        self.eps, self.index, angles = [], [], []
        tag = basename(root.rstrip("/")) or root
        for fi, f in enumerate(tqdm(files, desc=f"loading '{tag}' -> RAM")):
            with np.load(f) as d:
                ep = {
                    "imgs": d["imgs"],                                  # uint8 (T,H,W,3) — kept as uint8
                    "acts": d["acts"].astype(np.float32),               # (T,)
                    "states": d["states"].astype(np.float32),           # (T,8) clean
                    "x": (d[f"noisy_states_{shift}"] if shift in (2, 5, 10)
                          else d["states"]).astype(np.float32),         # (T,8) input
                }
            self.eps.append(ep)
            T = ep["states"].shape[0]
            for s in range(0, max(T - window + 1, 0), stride):
                self.index.append((fi, s))
                angles.append(ep["states"][s, ANGLE_DIM])
        self.angles = np.asarray(angles, np.float32)

    def __len__(self):
        return len(self.index)


# ---------------------------------------------------------------------------
# VAE: one sample = a pair (t, t+1)
# ---------------------------------------------------------------------------
class VaePairDataset(_BaseRollout):
    """ Returns: img_t, img_tp1 (3,H,W), action_t, state_t (8), state_tp1 (8). """
    def __init__(self, root, **kw):
        super().__init__(root, window=2, **kw)

    def __getitem__(self, i):
        fi, t = self.index[i]
        ep = self.eps[fi]
        # Return uint8 (C,H,W); the .float()/255 happens on the GPU (run_epoch),
        # so the CPU stays idle and 4x fewer bytes cross the PCIe bus.
        img_t = torch.from_numpy(ep["imgs"][t]).permute(2, 0, 1)        # uint8
        img_tp1 = torch.from_numpy(ep["imgs"][t + 1]).permute(2, 0, 1)  # uint8
        action = torch.tensor(ep["acts"][t])
        state_t = torch.from_numpy(_standardize(ep["x"][t], self.mean, self.std))
        state_tp1 = torch.from_numpy(_standardize(ep["states"][t + 1], self.mean, self.std))
        return img_t, img_tp1, action, state_t, state_tp1


# ---------------------------------------------------------------------------
# LSTM (image-based alternative): a window of seq_len pairs
# ---------------------------------------------------------------------------
class SequenceDataset(_BaseRollout):
    """ seq_len consecutive pairs (t,t+1). For 30 steps -> seq_len=31.
    Returns: img_t (L,3,H,W), img_tp1 (L,3,H,W), actions (L,), states (L,8),
                states_clean (L,8).  Shuffle the WINDOWS (shuffle=True), not within them. """
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
# Pre-encoding: run the (frozen) VAE once -> z sequences
# ---------------------------------------------------------------------------
@torch.no_grad()
def precompute_latents(encode_fn, root, out_root, shift=0, batch=256, device="cuda"):
    """ encode_fn(img_t, img_tp1) -> z. Saves one .npz per episode with z,acts,states,x (N=T-1). """
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
# LSTM (recommended): windows of latents (already all-in-RAM)
# ---------------------------------------------------------------------------
class LatentSequenceDataset(torch.utils.data.Dataset):
    """ seq_len latent TRANSITIONS. For 30 steps -> seq_len=30.
    Returns: z_t (L,latent), action (L,), z_tp1 (L,latent), state_t (L,8), state_tp1 (L,8).
      - state_t   : input state at t  (x -> noisy if shift), standardized
      - state_tp1 : CLEAN state at t+1, standardized (target for physical MSE / hybrid gt) """
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
            raise RuntimeError(f"No windows produced from: {root} (seq_len too large?)")

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
# Optional: rebalance the tails (rare theta angles, dim=ANGLE_DIM)
# ---------------------------------------------------------------------------
def angle_balanced_weights(dataset, bins=40):
    ang = dataset.angles
    hist, edges = np.histogram(ang, bins=bins)
    idx = np.clip(np.digitize(ang, edges[:-1]) - 1, 0, len(hist) - 1)
    return torch.as_tensor(1.0 / (hist[idx] + 1.0), dtype=torch.double)
