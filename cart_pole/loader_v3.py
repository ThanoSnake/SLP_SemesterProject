""" Lazy per-episode data loading — ΧΩΡΙΣ buffers.

Διαφορά από το loader.py:
  - ΔΕΝ προφορτώνει chunks (buffers) στη RAM, δεν κρατάει cum_size/bisect index.
  - Κάθε item = ΕΝΑ ολόκληρο επεισόδιο (.npz), φορτωμένο on-demand στο __getitem__.
  - Με DataLoader(batch_size=None, num_workers>0) η αποσυμπίεση κάθε επεισοδίου
    γίνεται παράλληλα με το GPU compute. Μόνο λίγα επεισόδια στη RAM κάθε στιγμή.

Κάθε .npz (από dataCollect_v3.py) έχει: imgs (T,H,W,3) uint8, acts (T,),
states (T,4), και noisy_states_2/5/10 (T,4).

Σε αντίθεση με το loader.py ΔΕΝ χρειάζεται φιλτράρισμα <33 frames: εδώ δεν
φτιάχνουμε fixed 30-step windows, οπότε αρκεί T>=2 για να βγει ζεύγος (t, t+1).
"""
from os import listdir
from os.path import join, isdir

import numpy as np
import torch
import torch.utils.data


def _list_npz(root):
    """ Όλα τα αρχεία επεισοδίων κάτω από το root (1 επίπεδο υποφακέλων). """
    files = []
    for sd in sorted(listdir(root)):
        p = join(root, sd)
        if isdir(p):
            for ssd in sorted(listdir(p)):
                files.append(join(p, ssd))
        else:
            files.append(p)
    return sorted(files)


class EpisodeDataset(torch.utils.data.Dataset):
    """ Ένα item = ένα επεισόδιο, ως ζεύγη διαδοχικών frames (t, t+1).

    Επιστρέφει (ίδια σημασιολογία keys με το loader.py, αλλά για ΟΛΟ το επεισόδιο):
        img_t      : (N, 3, H, W)  frames t        (N = T-1)
        img_tp1    : (N, 3, H, W)  frames t+1
        action     : (N,)          ενέργειες a_t
        states_t   : (N, 4)        κατάσταση t  (clean ή noisy ανάλογα με shift)
        states_tp1 : (N, 4)        clean κατάσταση t+1

    Χρήση:
      - VAE  -> τα N frames (img_t) ως mini-batch εικόνων.
      - LSTM -> το επεισόδιο ως μία ακολουθία μήκους N: (z_t, a_t) -> z_{t+1}.
    """

    def __init__(self, root, shift=0):
        self.shift = shift
        self._files = _list_npz(root)
        if not self._files:
            raise RuntimeError(f"Κανένα .npz αρχείο στο: {root}")

    def __len__(self):
        return len(self._files)

    def _states_for_shift(self, data):
        """ shift in {2,5,10} -> noisy_states_{shift}· αλλιώς clean states. """
        if self.shift in (2, 5, 10):
            return data[f'noisy_states_{self.shift}']
        return data['states']

    def __getitem__(self, i):
        with np.load(self._files[i]) as data:
            imgs = data['imgs']                  # (T, H, W, 3) uint8
            acts = data['acts']                  # (T,)
            states = data['states']              # (T, 4)
            x = self._states_for_shift(data)     # (T, 4)

        # Εκφυλισμένο επεισόδιο (σχεδόν ποτέ): πάπλωμα ώστε T>=2, χωρίς crash.
        if imgs.shape[0] < 2:
            imgs = np.concatenate([imgs, imgs[-1:]], axis=0)
            acts = np.concatenate([acts, acts[-1:]], axis=0)
            states = np.concatenate([states, states[-1:]], axis=0)
            x = np.concatenate([x, x[-1:]], axis=0)

        imgs = torch.from_numpy(imgs.astype(np.float32) / 255.0).permute(0, 3, 1, 2)  # (T,3,H,W)
        img_t = imgs[:-1].contiguous()       # frames t      (N,3,H,W)
        img_tp1 = imgs[1:].contiguous()      # frames t+1    (N,3,H,W)
        action = torch.from_numpy(acts[:-1].astype(np.float32))      # (N,)
        states_t = torch.from_numpy(x[:-1].astype(np.float32))       # (N,4)
        states_tp1 = torch.from_numpy(states[1:].astype(np.float32)) # (N,4)
        return img_t, img_tp1, action, states_t, states_tp1
