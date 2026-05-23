"""Dataset classes for CartPole world-model training.

Two datasets are exposed:

  VaeDataset       -- pairs of consecutive frames; for training the VAE.
                      Each item: (frame_t, frame_{t+1}, action_t, state_t, state_{t+1})
                      frames are (3, H, W) float32 in [0, 1].

  SequenceDataset  -- fixed-length frame windows; for training the LSTM
                      latent-dynamics model on top of a frozen VAE.
                      Each item: (frames, actions, states)
                          frames  : (T, 3, H, W)  float32 in [0, 1]
                          actions : (T,)          int64
                          states  : (T, 4)        float32
                      Training code slices frames[:-1] as input and
                      frames[1:] as targets (teacher forcing), or whatever
                      input/horizon split it wants.

Episodes are loaded into memory once at construction time. Frames are stored
pre-resized to (IMG_H, IMG_W) on disk by `dataCollect.py`, so the memory
footprint is small (~1.5 GB for 1000 train episodes at 80x120).
"""
from bisect import bisect_right
from os import listdir
from os.path import isdir, join

import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm


# Must match the values used in dataCollect.py.
IMG_H, IMG_W = 80, 120


def _list_npz(root):
    """Walk `root` (and one level of subdirectories) for .npz files."""
    paths = []
    for name in sorted(listdir(root)):
        full = join(root, name)
        if isdir(full):
            for sub in sorted(listdir(full)):
                if sub.endswith(".npz"):
                    paths.append(join(full, sub))
        elif name.endswith(".npz"):
            paths.append(full)
    if not paths:
        raise RuntimeError(f"No .npz files found under {root}")
    return paths


def _frames_to_chw_float(imgs_uint8):
    """uint8 (N, H, W, 3) -> float32 (N, 3, H, W) in [0, 1]."""
    t = torch.from_numpy(imgs_uint8).float().div_(255.0)   # (N, H, W, 3)
    return t.permute(0, 3, 1, 2).contiguous()              # (N, 3, H, W)


class _RolloutDataset(torch.utils.data.Dataset):
    """Base class: loads episodes into memory and indexes valid windows.

    Subclasses set `window` (frames per sample) and implement `_make_sample`.
    """

    def __init__(self, root, window):
        if window < 2:
            raise ValueError("window must be >= 2 (need input + at least 1 target)")
        self.window = window

        paths = _list_npz(root)

        self._episodes = []          # list of dicts: imgs / acts / states
        self._cum      = [0]         # cumulative number of windows per episode

        pbar = tqdm(paths, desc=f"Loading {root}",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}")
        for p in pbar:
            with np.load(p) as data:
                T = int(data["imgs"].shape[0])
                if T < window:
                    continue
                self._episodes.append({
                    "imgs":   data["imgs"][:],                       # uint8
                    "acts":   data["acts"][:].astype(np.int64),      # int64
                    "states": data["states"][:].astype(np.float32),  # float32
                })
                self._cum.append(self._cum[-1] + (T - window + 1))

        if not self._episodes:
            raise RuntimeError(
                f"No episodes >= window={window} found in {root}"
            )

        print(f"  loaded {len(self._episodes)} episodes "
              f"-> {self._cum[-1]} windows of length {window}")

    def __len__(self):
        return self._cum[-1]

    def _locate(self, idx):
        """Global window index -> (episode_idx, start_within_episode)."""
        if idx < 0:
            idx += self._cum[-1]
        if not 0 <= idx < self._cum[-1]:
            raise IndexError(idx)
        ep = bisect_right(self._cum, idx) - 1
        start = idx - self._cum[ep]
        return ep, start

    def __getitem__(self, idx):
        ep, start = self._locate(idx)
        end = start + self.window
        data = self._episodes[ep]
        imgs   = data["imgs"  ][start:end]   # (window, H, W, 3) uint8
        acts   = data["acts"  ][start:end]   # (window,)         int64
        states = data["states"][start:end]   # (window, 4)       float32
        return self._make_sample(imgs, acts, states)

    def _make_sample(self, imgs, acts, states):
        raise NotImplementedError


class VaeDataset(_RolloutDataset):
    """Pairs of consecutive frames for VAE training."""

    def __init__(self, root):
        super().__init__(root, window=2)

    def _make_sample(self, imgs, acts, states):
        frames = _frames_to_chw_float(imgs)              # (2, 3, H, W)
        return (
            frames[0],                                   # frame_t      (3, H, W)
            frames[1],                                   # frame_{t+1}  (3, H, W)
            int(acts[0]),                                # action_t
            torch.from_numpy(states[0]),                 # state_t      (4,)
            torch.from_numpy(states[1]),                 # state_{t+1}  (4,)
        )


class SequenceDataset(_RolloutDataset):
    """Fixed-length frame windows for LSTM training.

    `window` is the total number of frames in one sample. To train a model
    that predicts horizon H, set window = H + N where N is the number of
    history frames the LSTM consumes before its first prediction.

    A common choice is window=33: feed frames[:-1] (32 frames) to the LSTM
    with teacher forcing, and supervise it against frames[1:] (32 next-state
    targets). At evaluation we autoregress to obtain predictions at horizons
    1..32, matching the x-axis of Figure 3 in the paper.
    """

    def __init__(self, root, window=33):
        super().__init__(root, window=window)

    def _make_sample(self, imgs, acts, states):
        return (
            _frames_to_chw_float(imgs),                  # (T, 3, H, W)
            torch.from_numpy(acts),                      # (T,)
            torch.from_numpy(states),                    # (T, 4)
        )


# --- Quick self-check: `python loader.py --root ./cartpole_data/train` -----
if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument("--root",   type=str, default="./cartpole_data/train")
    parser.add_argument("--window", type=int, default=33)
    args = parser.parse_args()

    print("--- VaeDataset ---")
    vae_ds = VaeDataset(root=args.root)
    f0, f1, a, s0, s1 = vae_ds[0]
    print(f"frame_t={tuple(f0.shape)} dtype={f0.dtype}  "
          f"action={a}  state_t={s0.tolist()}")

    print("\n--- SequenceDataset ---")
    seq_ds = SequenceDataset(root=args.root, window=args.window)
    frames, acts, states = seq_ds[0]
    print(f"frames={tuple(frames.shape)}  acts={tuple(acts.shape)}  "
          f"states={tuple(states.shape)}")

    loader = DataLoader(seq_ds, batch_size=4, shuffle=True, drop_last=True)
    frames, acts, states = next(iter(loader))
    print(f"batched frames={tuple(frames.shape)}  acts={tuple(acts.shape)}  "
          f"states={tuple(states.shape)}")
