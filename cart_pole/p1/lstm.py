# =============================================================================
#  Principle 1 — LSTM (Memory/Dynamics) πάνω στο P1 latent space
#  Ίδια λογική με lstm_v3.py (PRECOMPUTED LATENTS + cache), αλλά ο encoder είναι
#  ο VAE_P1 (separate encoding): z = [state_part(4) || image_part(60)] = 64 dims.
#
#  ΝΤΕΤΕΡΜΙΝΙΣΤΙΚΟ latent για το LSTM:
#      z = concat( state_head(feat) , fc_mu(feat) )        # (64,)
#  -> τα πρώτα state_dim=4 dims ΕΙΝΑΙ η φυσική κατάσταση [x, ẋ, θ, θ̇].
#  Στο inference, μετά το rollout, διαβάζεις τη φυσική κατάσταση ΚΑΤΕΥΘΕΙΑΝ από
#  ẑ[..., :4] — ΧΩΡΙΣ probe (αυτό είναι το πλεονέκτημα της Αρχής 1 για το Fig 3).
#
#  Προϋπόθεση: εκπαιδευμένο P1 VAE (vae_p1.pth) από το p1/seperate_encoding.py.
#  Για Kaggle Notebook (χωρίς argparse).
# =============================================================================
import os
import pickle
from os import listdir
from os.path import join, isdir

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ------------------------------- CONFIG --------------------------------------
class CFG:
    train        = "/kaggle/working/cartpole_data/train"
    val          = "/kaggle/working/cartpole_data/val"

    vae_weights  = "/kaggle/working/vae_p1.pth"               # ΠΑΓΩΜΕΝΟ P1 VAE
    save         = "/kaggle/working/p1lstm.pth"               # ΚΑΛΥΤΕΡΟ P1 LSTM (έξοδος)
    weights      = "/kaggle/working/p1lstm.pth"               # έτοιμα P1 LSTM βάρη (είσοδος)
    cache_train  = "/kaggle/working/p1_latents_train.pkl"
    cache_val    = "/kaggle/working/p1_latents_val.pkl"
    force_retrain   = False
    force_recompute = False

    # --- model ---
    latent       = 64          # συνολικό latent (state_dim + image_dim)
    state_dim    = 4           # πρώτα dims = φυσική κατάσταση
    hidden       = 64
    num_layers   = 2

    # --- training ---
    seq_len      = 30
    stride       = 1
    enc_batch    = 256
    batch        = 64
    num_workers  = 0
    epochs       = 200
    patience     = 10
    min_delta    = 0.0
    lr           = 1e-3


def _list_npz(root):
    files = []
    for sd in sorted(listdir(root)):
        p = join(root, sd)
        if isdir(p):
            for ssd in sorted(listdir(p)):
                files.append(join(p, ssd))
        else:
            files.append(p)
    return sorted(files)


# ------------------------------- MODELS --------------------------------------
class VAE_P1(nn.Module):
    """ Ίδιο με p1/seperate_encoding.py — φορτώνεται παγωμένο από vae_p1.pth. """

    def __init__(self, latent_size=64, state_dim=4):
        super(VAE_P1, self).__init__()
        self.latent_size = latent_size
        self.state_dim = state_dim
        self.img_dim = latent_size - state_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(inplace=True),
        )
        self.state_head = nn.Linear(64 * 10 * 15, state_dim)
        self.fc_mu = nn.Linear(64 * 10 * 15, self.img_dim)
        self.fc_logvar = nn.Linear(64 * 10 * 15, self.img_dim)
        self.fc_decode = nn.Linear(latent_size, 64 * 10 * 15)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 3, 4, 2, 1), nn.Sigmoid(),
        )


class LSTMTimeSeries(nn.Module):
    """ Memory/Dynamics (M): [z_t || a_t] -> z_{t+1}. input=latent+1, output=latent. """

    def __init__(self, input_size=65, hidden_size=64, num_layers=2, output_size=64):
        super(LSTMTimeSeries, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        b = x.shape[0]
        h0 = torch.zeros(self.num_layers, b, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, b, self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out)


@torch.no_grad()
def encode_latent(vae, frames, device, enc_batch=256):
    """ ΝΤΕΤΕΡΜΙΝΙΣΤΙΚΟ P1 latent: [state_head(feat) || fc_mu(feat)] -> (N, latent).
    Τα πρώτα state_dim dims είναι η φυσική κατάσταση. """
    outs = []
    for s in range(0, frames.shape[0], enc_batch):
        x = frames[s:s + enc_batch].to(device)
        feat = vae.encoder(x).reshape(x.size(0), -1)
        state = vae.state_head(feat)               # (b, state_dim)
        img_mu = vae.fc_mu(feat)                    # (b, img_dim)
        outs.append(torch.cat([state, img_mu], dim=1))
    return torch.cat(outs, dim=0)                   # (N, latent)


def load_vae(cfg, device):
    if not os.path.isfile(cfg.vae_weights):
        raise FileNotFoundError(
            f"Δεν βρέθηκαν βάρη P1 VAE στο '{cfg.vae_weights}'.\n"
            f"Τρέξε πρώτα το p1/seperate_encoding.py.")
    vae = VAE_P1(latent_size=cfg.latent, state_dim=cfg.state_dim).to(device)
    vae.load_state_dict(torch.load(cfg.vae_weights, map_location=device))
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    print(f"[P1-VAE] Παγωμένα βάρη φορτώθηκαν από {cfg.vae_weights}.")
    return vae


# --------------------------- PRECOMPUTE LATENTS ------------------------------
def _vae_signature(vae_weights):
    mtime = os.path.getmtime(vae_weights) if os.path.isfile(vae_weights) else None
    return (vae_weights, mtime)


@torch.no_grad()
def precompute_latents(vae, root, device, cfg, cache_path, force=False):
    """ Κωδικοποιεί όλα τα frames -> P1 latent (μία φορά). Cache σε .pkl. """
    sig = _vae_signature(cfg.vae_weights)
    if (not force) and os.path.isfile(cache_path):
        with open(cache_path, 'rb') as fh:
            blob = pickle.load(fh)
        if blob.get('sig') == sig:
            print(f"[CACHE] Latents από {cache_path} ({len(blob['episodes'])} επεισόδια).")
            return blob['episodes']
        print(f"[CACHE] Το {cache_path} αφορά άλλο/ξανα-εκπαιδευμένο VAE -> recompute.")

    files = _list_npz(root)
    if not files:
        raise RuntimeError(f"Κανένα .npz αρχείο στο: {root}")

    episodes = []
    for f in tqdm(files, desc=f"Encoding P1 latents [{os.path.basename(root)}]"):
        with np.load(f) as d:
            imgs = d['imgs']
            if imgs.shape[0] < 2:
                continue
            acts = d['acts'].astype(np.float32)
            frames = torch.from_numpy(imgs.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
        z = encode_latent(vae, frames, device, cfg.enc_batch).cpu().numpy().astype(np.float32)
        episodes.append({'z': z, 'acts': acts})

    with open(cache_path, 'wb') as fh:
        pickle.dump({'episodes': episodes, 'sig': sig}, fh)
    print(f"[CACHE] Αποθηκεύτηκαν {len(episodes)} επεισόδια latents -> {cache_path}")
    return episodes


# ----------------------------- DATASET (latents) -----------------------------
class LatentSequenceDataset(Dataset):
    """ 30-step windows πάνω στα precomputed P1 latents (όλα στη RAM). """

    def __init__(self, episodes, seq_len=30, stride=1):
        self.episodes = episodes
        self.seq_len = seq_len
        self.index = []
        for ei, ep in enumerate(episodes):
            max_start = ep['z'].shape[0] - (seq_len + 1)
            for s in range(0, max_start + 1, stride):
                self.index.append((ei, s))
        if not self.index:
            raise RuntimeError(f"Κανένα έγκυρο παράθυρο μήκους {seq_len + 1}.")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        ei, s = self.index[i]
        ep = self.episodes[ei]
        L = self.seq_len
        z_in = torch.from_numpy(ep['z'][s:s + L])
        z_out = torch.from_numpy(ep['z'][s + 1:s + L + 1])
        act = torch.from_numpy(ep['acts'][s:s + L])
        return z_in, z_out, act


# ------------------------------- TRAINING ------------------------------------
def train_one_epoch(predictor, loader, optimizer, device):
    predictor.train()
    total_loss, total_n = 0.0, 0
    for z_in, z_out, act in tqdm(loader, desc="Training Progress", leave=False):
        z_in = z_in.to(device)
        z_out = z_out.to(device)
        act = act.to(device).float()
        lstm_in = torch.cat([z_in, act.unsqueeze(-1)], dim=-1)

        optimizer.zero_grad()
        pred = predictor(lstm_in)
        loss = F.mse_loss(pred, z_out, reduction='sum')
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_n += z_in.shape[0] * z_in.shape[1]
    return total_loss / max(total_n, 1)


@torch.no_grad()
def evaluate(predictor, loader, device, state_dim):
    """ Επιστρέφει (συνολικό latent MSE per-step, MSE ΜΟΝΟ στα state dims per-step). """
    predictor.eval()
    total_loss, state_loss, total_n = 0.0, 0.0, 0
    for z_in, z_out, act in loader:
        z_in = z_in.to(device)
        z_out = z_out.to(device)
        act = act.to(device).float()
        lstm_in = torch.cat([z_in, act.unsqueeze(-1)], dim=-1)
        pred = predictor(lstm_in)
        total_loss += F.mse_loss(pred, z_out, reduction='sum').item()
        state_loss += F.mse_loss(pred[..., :state_dim], z_out[..., :state_dim],
                                 reduction='sum').item()
        total_n += z_in.shape[0] * z_in.shape[1]
    return total_loss / max(total_n, 1), state_loss / max(total_n, 1)


def train_lstm(cfg, predictor, train_eps, val_eps, device):
    traindataset = LatentSequenceDataset(train_eps, seq_len=cfg.seq_len, stride=cfg.stride)
    valdataset = LatentSequenceDataset(val_eps, seq_len=cfg.seq_len, stride=cfg.stride)
    print(f"[DATA] train windows: {len(traindataset)} | val windows: {len(valdataset)}")

    TrainLoader = DataLoader(traindataset, batch_size=cfg.batch, shuffle=True,
                             drop_last=True, num_workers=cfg.num_workers)
    ValLoader = DataLoader(valdataset, batch_size=cfg.batch, shuffle=False,
                           drop_last=False, num_workers=cfg.num_workers)

    optimizer = optim.Adam(predictor.parameters(), lr=cfg.lr)

    best_val = float('inf')
    epochs_no_improve = 0
    for epoch in range(cfg.epochs):
        train_loss = train_one_epoch(predictor, TrainLoader, optimizer, device)
        val_loss, val_state = evaluate(predictor, ValLoader, device, cfg.state_dim)
        print(f'Epoch {epoch + 1}: train {train_loss:.4f} | val {val_loss:.4f} '
              f'| val state-MSE {val_state:.4f}')

        if val_loss < best_val - cfg.min_delta:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save(predictor.state_dict(), cfg.save)
            print(f'  ✓ νέο καλύτερο val ({best_val:.4f}) -> {cfg.save}')
        else:
            epochs_no_improve += 1
            print(f'  χωρίς βελτίωση ({epochs_no_improve}/{cfg.patience})')
            if epochs_no_improve >= cfg.patience:
                print(f'Early stopping στο epoch {epoch + 1}.')
                break
    return best_val


def load_or_train_lstm(cfg=CFG):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    vae = load_vae(cfg, device)
    predictor = LSTMTimeSeries(input_size=cfg.latent + 1, hidden_size=cfg.hidden,
                               num_layers=cfg.num_layers, output_size=cfg.latent).to(device)

    if (not cfg.force_retrain) and os.path.isfile(cfg.weights):
        predictor.load_state_dict(torch.load(cfg.weights, map_location=device))
        predictor.eval()
        print(f"[P1-LSTM] Φορτώθηκαν έτοιμα βάρη από {cfg.weights} -> παράλειψη εκπαίδευσης.")
        return vae, predictor, device

    print(f"[P1-LSTM] Δεν βρέθηκε checkpoint (ή force_retrain=True) -> εκπαίδευση.")
    train_eps = precompute_latents(vae, cfg.train, device, cfg, cfg.cache_train, cfg.force_recompute)
    val_eps = precompute_latents(vae, cfg.val, device, cfg, cfg.cache_val, cfg.force_recompute)

    best_val = train_lstm(cfg, predictor, train_eps, val_eps, device)
    predictor.load_state_dict(torch.load(cfg.save, map_location=device))
    predictor.eval()
    print(f"[P1-LSTM] Φορτώθηκε το καλύτερο μοντέλο (val={best_val:.4f}) από {cfg.save}.")
    return vae, predictor, device


# =============================================================================
#  INFERENCE — πρόβλεψη (ΔΕΝ τρέχουν εδώ· κάλεσέ τα σε ΕΠΟΜΕΝΟ cell)
#  Σημείωση P1: μετά από κάθε πρόβλεψη ẑ, η φυσική κατάσταση είναι ẑ[..., :state_dim].
# =============================================================================
def init_hidden(lstm, batch=1, device='cpu'):
    h = torch.zeros(lstm.num_layers, batch, lstm.hidden_size, device=device)
    c = torch.zeros(lstm.num_layers, batch, lstm.hidden_size, device=device)
    return (h, c)


@torch.no_grad()
def predict_next(lstm, z_t, a_t, hc=None):
    """ ΕΝΑ βήμα (stateful): (z_t, a_t) + (h,c) -> (ẑ_{t+1}, νέο (h,c)). """
    lstm.eval()
    device = z_t.device
    z = z_t.reshape(1, 1, -1).float()
    a = torch.as_tensor(a_t, dtype=torch.float32, device=device).reshape(1, 1, 1)
    if hc is None:
        hc = init_hidden(lstm, batch=1, device=device)
    out, hc = lstm.lstm(torch.cat([z, a], dim=-1), hc)
    return lstm.fc(out).reshape(-1), hc


@torch.no_grad()
def rollout_batch(lstm, z0, actions, device):
    """ Open-loop autoregressive rollout, batched (χωρίς warm-up).
        z0:(B,latent) πραγματικό· actions:(B,H). Επιστρέφει (B,H,latent). """
    B, H = actions.shape
    h = torch.zeros(lstm.num_layers, B, lstm.hidden_size, device=device)
    c = torch.zeros_like(h)
    z = z0.unsqueeze(1)
    preds = []
    for k in range(H):
        a = actions[:, k].view(B, 1, 1)
        out, (h, c) = lstm.lstm(torch.cat([z, a], dim=-1), (h, c))
        z = lstm.fc(out)
        preds.append(z)
    return torch.cat(preds, dim=1)


@torch.no_grad()
def decode_latents(vae, z):
    """ z: (K, latent) -> frames (K, 3, H, W) μέσω του P1 VAE decoder. """
    vae.eval()
    z = z.reshape(z.shape[0], -1).float()
    dec = vae.fc_decode(z).view(z.shape[0], 64, 10, 15)
    return vae.decoder(dec)


def latent_to_state(z, state_dim=4):
    """ Φυσική κατάσταση ΚΑΤΕΥΘΕΙΑΝ από το latent (Αρχή 1, χωρίς probe). """
    return z[..., :state_dim]


# Παράδειγμα (επόμενο cell): πρόβλεψη φυσικής κατάστασης 30 βημάτων, χωρίς probe
#   import numpy as np
#   d = np.load("/kaggle/working/cartpole_data/test/0.npz")
#   frames = torch.from_numpy(d["imgs"].astype(np.float32)/255.0).permute(0,3,1,2)
#   z0 = encode_latent(vae_model, frames[:1].to(device), device, 256)        # (1, latent)
#   acts = torch.from_numpy(d["acts"][:30].astype(np.float32)).view(1, 30).to(device)
#   z_pred = rollout_batch(lstm_model, z0, acts, device)                     # (1, 30, latent)
#   state_pred = latent_to_state(z_pred, CFG.state_dim)                      # (1, 30, 4) — φυσική κατάσταση!


# Φορτώνει παγωμένο P1 VAE + (φορτώνει ή εκπαιδεύει) LSTM σε precomputed P1 latents.
vae_model, lstm_model, device = load_or_train_lstm(CFG)
