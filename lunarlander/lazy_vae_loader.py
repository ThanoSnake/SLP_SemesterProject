"""
lazy_vae_loader.py — LAZY (low-RAM) VAE pair dataset for LARGE datasets (control 8k is ~27GB eager).

Why: `loader.VaePairDataset` loads ALL the imgs into RAM (eager) -> ~27GB for control 8k,
which does NOT fit in a Kaggle GPU session (~13-16GB). Here the imgs stay compressed on disk
(the npz files are ~1GB) and are decompressed PER EPISODE on the fly.

THE SPEED PROBLEM & THE FIX:
  With plain random shuffling, every sample would decompress the WHOLE episode (compressed .npy has no
  partial read) -> painfully slow. Fix: `ChunkedEpisodeSampler` — it shuffles the EPISODES, splits them into
  chunks (e.g. 64 episodes), and shuffles ALL the windows together inside each chunk. So:
    * locality: each episode is decompressed ~once per epoch (it stays in the small LRU cache while
      its chunk is active), but
    * batch diversity: each batch draws windows from ~64 different episodes.
  RAM: cache_size x ~4.3MB per worker (e.g. 72 x 4.3MB ~ 0.3GB/worker).

Compatible 1:1 with `VaePairDataset`: __getitem__ -> (img_t uint8 (3,H,W), img_tp1 uint8, action,
state_t std, state_tp1 std). The uint8 -> float/255 conversion happens on the GPU inside run_epoch.

Import-only; it modifies no existing module. Borrows helpers from loader_control.
"""
import numpy as np
import torch
import torch.utils.data
from collections import OrderedDict
from tqdm.auto import tqdm

from loader_control import list_npz, load_norm_stats, _standardize, _wind_skip, _NOISY


# ---------------------------------------------------------------------------
# Lazy pair dataset
# ---------------------------------------------------------------------------
class VaePairDatasetLazy(torch.utils.data.Dataset):
    """Pairs (frame_t, frame_{t+1}) from a LARGE dataset, without eager RAM.

    roots: str or list of split-dirs (e.g. [control/train])  ;  shift: 0|2|5|10 (weak-sup input)
    cache_size: how many DECOMPRESSED episodes to keep per process (>= the sampler's chunk_size).
    wind_filter: 'all'|'clean'|'wind'.
    """
    def __init__(self, roots, shift=0, state_mean=None, state_std=None,
                 cache_size=72, wind_filter="all"):
        self.shift = shift
        self.cache_size = cache_size
        self.mean = None if state_mean is None else np.asarray(state_mean, np.float32)
        self.std = None if state_std is None else np.asarray(state_std, np.float32)

        # Indexing: reads ONLY 'states' (+'wind_enabled' if needed) -> cheap, low RAM.
        self.files, self.index, self.ep_ranges = [], [], []
        for f in tqdm(list_npz(roots), desc="indexing (lazy)"):
            with np.load(f) as d:
                if _wind_skip(d, wind_filter):
                    continue
                T = d["states"].shape[0]
            if T < 2:
                continue
            fi = len(self.files)
            self.files.append(f)
            start = len(self.index)
            for t in range(T - 1):                      # window=2 -> windows 0..T-2
                self.index.append((fi, t))
            self.ep_ranges.append((start, len(self.index)))   # [start,end) global window indices per episode
        if not self.index:
            raise RuntimeError(f"No pairs from {roots} (wind_filter={wind_filter}?)")
        self._cache = OrderedDict()                     # fi -> dict(imgs,acts,states,x)  (per process, after the fork)

    def __len__(self):
        return len(self.index)

    def _load(self, fi):
        ep = self._cache.get(fi)
        if ep is not None:
            self._cache.move_to_end(fi)
            return ep
        with np.load(self.files[fi]) as d:
            ep = {
                "imgs": d["imgs"],                                       # uint8 (T,H,W,3)
                "acts": d["acts"].astype(np.float32),
                "states": d["states"].astype(np.float32),
                "x": (d[f"noisy_states_{self.shift}"] if self.shift in _NOISY
                      else d["states"]).astype(np.float32),
            }
        self._cache[fi] = ep
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)             # evict LRU
        return ep

    def __getitem__(self, i):
        fi, t = self.index[i]
        ep = self._load(fi)
        img_t = torch.from_numpy(ep["imgs"][t]).permute(2, 0, 1)        # uint8 (3,H,W)
        img_tp1 = torch.from_numpy(ep["imgs"][t + 1]).permute(2, 0, 1)
        action = torch.tensor(ep["acts"][t])
        state_t = torch.from_numpy(_standardize(ep["x"][t], self.mean, self.std))
        state_tp1 = torch.from_numpy(_standardize(ep["states"][t + 1], self.mean, self.std))
        return img_t, img_tp1, action, state_t, state_tp1


# ---------------------------------------------------------------------------
# Sampler: locality (cache hits) + batch diversity
# ---------------------------------------------------------------------------
class ChunkedEpisodeSampler(torch.utils.data.Sampler):
    """Shuffles EPISODES, groups them into chunks, and shuffles ALL the windows together inside
    each chunk. This way each batch sees ~chunk_size different episodes (diversity),
    but only chunk_size episodes are "open" at a time (locality -> cache hits).

    NOTE: the dataset's cache_size must be >= chunk_size."""
    def __init__(self, ep_ranges, chunk_size=64, seed=0):
        self.ep_ranges = list(ep_ranges)
        self.chunk_size = chunk_size
        self.epoch = 0
        self.seed = seed

    def set_epoch(self, e):
        self.epoch = e

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        order = rng.permutation(len(self.ep_ranges))
        for c in range(0, len(order), self.chunk_size):
            chunk = order[c:c + self.chunk_size]
            idxs = []
            for ei in chunk:
                s, e = self.ep_ranges[ei]
                idxs.extend(range(s, e))
            rng.shuffle(idxs)
            yield from idxs

    def __len__(self):
        return sum(e - s for s, e in self.ep_ranges)
