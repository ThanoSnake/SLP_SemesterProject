"""
lazy_vae_loader.py — LAZY (low-RAM) VAE pair dataset για ΜΕΓΑΛΑ datasets (control 8k ~27GB eager).

Γιατί: ο `loader.VaePairDataset` φορτώνει ΟΛΑ τα imgs στη RAM (eager) -> ~27GB για το control 8k,
που ΔΕΝ χωράει σε Kaggle GPU session (~13-16GB). Εδώ τα imgs μένουν συμπιεσμένα στον δίσκο
(τα npz είναι ~1GB) και αποσυμπιέζονται ΑΝΑ ΕΠΕΙΣΟΔΙΟ on-the-fly.

ΠΡΟΒΛΗΜΑ ΤΑΧΥΤΗΤΑΣ & ΛΥΣΗ:
  Με σκέτο random shuffle, κάθε δείγμα θα αποσυμπίεζε ΟΛΟ το επεισόδιο (τα συμπιεσμένα .npy δεν έχουν
  partial read) -> τραγικά αργό. Λύση: `ChunkedEpisodeSampler` — ανακατεύει τα ΕΠΕΙΣΟΔΙΑ, τα χωρίζει σε
  chunks (π.χ. 64 επεισόδια), και μέσα σε κάθε chunk ανακατεύει ΟΛΑ τα παράθυρα μαζί. Έτσι:
    * τοπικότητα: κάθε επεισόδιο αποσυμπιέζεται ~μία φορά/epoch (μένει στο μικρό LRU cache όσο είναι ενεργό
      το chunk του), αλλά
    * διαφορετικότητα batch: κάθε batch τραβάει παράθυρα από ~64 διαφορετικά επεισόδια.
  RAM: cache_size × ~4.3MB ανά worker (π.χ. 72 × 4.3MB ≈ 0.3GB/worker).

Συμβατό 1:1 με τον `VaePairDataset`: __getitem__ -> (img_t uint8 (3,H,W), img_tp1 uint8, action,
state_t std, state_tp1 std). Η μετατροπή uint8 -> float/255 γίνεται στη GPU μέσα στο run_epoch.

Import-form· δεν τροποποιεί κανένα υπάρχον module. Δανείζεται helpers από το loader_control.
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
    """Ζεύγη (frame_t, frame_{t+1}) από ΜΕΓΑΛΟ dataset, χωρίς eager RAM.

    roots: str ή λίστα από split-dirs (π.χ. [control/train])  ·  shift: 0|2|5|10 (weak sup input)
    cache_size: πόσα ΑΠΟΣΥΜΠΙΕΣΜΕΝΑ επεισόδια κρατά ανά process (>= chunk_size του sampler).
    wind_filter: 'all'|'clean'|'wind'.
    """
    def __init__(self, roots, shift=0, state_mean=None, state_std=None,
                 cache_size=72, wind_filter="all"):
        self.shift = shift
        self.cache_size = cache_size
        self.mean = None if state_mean is None else np.asarray(state_mean, np.float32)
        self.std = None if state_std is None else np.asarray(state_std, np.float32)

        # Indexing: διαβάζει ΜΟΝΟ 'states' (+'wind_enabled' αν χρειάζεται) -> φθηνό, χαμηλή RAM.
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
            for t in range(T - 1):                      # window=2 -> παράθυρα 0..T-2
                self.index.append((fi, t))
            self.ep_ranges.append((start, len(self.index)))   # [start,end) global window-indices ανά επεισόδιο
        if not self.index:
            raise RuntimeError(f"No pairs from {roots} (wind_filter={wind_filter}?)")
        self._cache = OrderedDict()                     # fi -> dict(imgs,acts,states,x)  (per-process, μετά το fork)

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
    """Ανακατεύει ΕΠΕΙΣΟΔΙΑ, τα ομαδοποιεί σε chunks, και μέσα σε κάθε chunk ανακατεύει ΟΛΑ τα
    παράθυρα μαζί. Έτσι κάθε batch βλέπει ~chunk_size διαφορετικά επεισόδια (διαφορετικότητα),
    αλλά μόνο chunk_size επεισόδια είναι «ανοιχτά» ταυτόχρονα (τοπικότητα -> cache hits).

    ΣΗΜ.: cache_size του dataset πρέπει να είναι >= chunk_size."""
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
