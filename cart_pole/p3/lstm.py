# =============================================================================
#  Principle 3 — LSTM (Memory/Dynamics) πάνω στο P3 latent space (semi / weak)
#  Ίδια λογική με p1/p2 lstm (precomputed latents + cache), encoder = VAE_P1.
#
#  ΣΤΟΧΟΣ LSTM (ground truth), ανά διάσταση κατάστασης [x, ẋ, θ, θ̇]:
#    - static (0,2)   -> ΑΛΗΘΙΝΟ x_{t+1}, θ_{t+1} (dataset)           [semi & weak]
#    - velocity (1,3) -> semi: encoder latent (ΑΝΕΠΟΠΤΕΥΤΟ) ·
#                         weak: ΕΚΤΙΜΗΣΗ ẋ_est,θ̇_est=Δθέση/dt (dataset positions)
#    - image (4:)     -> encoder latent z_{t+1}
#  Inference: φυσική κατάσταση = ẑ[..., :state_dim] (ΧΩΡΙΣ probe).
#
#  Προϋπόθεση: εκπαιδευμένο P3 VAE (vae_p3_<mode>.pth) από το p3/vae.py.
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
from tqdm.auto import tqdm

CACHE_VERSION = "v3-p3"   # cache schema: z + acts + states (true) + vel_est


# ------------------------------- CONFIG --------------------------------------
class CFG:
    train        = "/kaggle/working/cartpole_data/train"
    val          = "/kaggle/working/cartpole_data/val"

    supervision  = "weak"    # "semi" | "weak" (ΠΡΕΠΕΙ να ταιριάζει με το VAE)
    vae_weights  = None      # None -> "/kaggle/working/vae_p3_<mode>.pth"
    save         = None      # None -> "/kaggle/working/p3lstm_<mode>.pth"
    weights      = None      # None -> ίδιο με save
    cache_train  = None      # None -> "/kaggle/working/p3_latents_<mode>_train.pkl"
    cache_val    = None
    force_retrain   = False
    force_recompute = False

    # --- model ---
    latent       = 64
    state_dim    = 4
    static_dims  = (0, 2)
    vel_dims     = (1, 3)
    hidden       = 64
    num_layers   = 2
    dt           = 0.02      # για την εκτίμηση ταχύτητας (πρέπει να ταιριάζει με το VAE)

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
    state_weight = 10.0      # βάρος των ΕΠΟΠΤΕΥΟΜΕΝΩΝ state dims

    @classmethod
    def resolve(cls):
        m = cls.supervision
        if cls.vae_weights is None:
            cls.vae_weights = f"/kaggle/working/vae_p3_{m}.pth"
        if cls.save is None:
            cls.save = f"/kaggle/working/p3lstm_{m}.pth"
        if cls.weights is None:
            cls.weights = cls.save
        if cls.cache_train is None:
            cls.cache_train = f"/kaggle/working/p3_latents_{m}_train.pkl"
        if cls.cache_val is None:
            cls.cache_val = f"/kaggle/working/p3_latents_{m}_val.pkl"


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
    """ Ίδιο split-encoding VAE — φορτώνεται παγωμένο από vae_p3_<mode>.pth. """

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
    """ ΝΤΕΤΕΡΜΙΝΙΣΤΙΚΟ latent: [state_head(feat) || fc_mu(feat)] -> (N, latent). """
    outs = []
    for s in range(0, frames.shape[0], enc_batch):
        x = frames[s:s + enc_batch].to(device)
        feat = vae.encoder(x).reshape(x.size(0), -1)
        outs.append(torch.cat([vae.state_head(feat), vae.fc_mu(feat)], dim=1))
    return torch.cat(outs, dim=0)


def load_vae(cfg, device):
    if not os.path.isfile(cfg.vae_weights):
        raise FileNotFoundError(
            f"Δεν βρέθηκαν βάρη P3 VAE στο '{cfg.vae_weights}'.\n"
            f"Τρέξε πρώτα το p3/vae.py με ΙΔΙΟ CFG.supervision.")
    vae = VAE_P1(latent_size=cfg.latent, state_dim=cfg.state_dim).to(device)
    vae.load_state_dict(torch.load(cfg.vae_weights, map_location=device))
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    print(f"[P3-VAE] Παγωμένα βάρη φορτώθηκαν από {cfg.vae_weights}.")
    return vae


# --------------------------- PRECOMPUTE LATENTS ------------------------------
def _vae_signature(vae_weights):
    mtime = os.path.getmtime(vae_weights) if os.path.isfile(vae_weights) else None
    return (vae_weights, mtime, CACHE_VERSION)


@torch.no_grad()
def precompute_latents(vae, root, device, cfg, cache_path, force=False):
    """ Encode -> z, ΜΑΖΙ με ΑΛΗΘΙΝΑ states και ΕΚΤΙΜΗΣΗ ταχύτητας (Δθέση/dt). """
    sig = _vae_signature(cfg.vae_weights)
    if (not force) and os.path.isfile(cache_path):
        with open(cache_path, 'rb') as fh:
            blob = pickle.load(fh)
        if blob.get('sig') == sig:
            print(f"[CACHE] Latents από {cache_path} ({len(blob['episodes'])} επεισόδια).")
            return blob['episodes']
        print(f"[CACHE] Το {cache_path} είναι stale -> recompute.")

    files = _list_npz(root)
    if not files:
        raise RuntimeError(f"Κανένα .npz αρχείο στο: {root}")

    episodes = []
    for f in tqdm(files, desc=f"Encoding P3 latents [{os.path.basename(root)}]"):
        with np.load(f) as d:
            imgs = d['imgs']
            if imgs.shape[0] < 2:
                continue
            acts = d['acts'].astype(np.float32)
            states = d['states'].astype(np.float32)            # (T,4) ΑΛΗΘΙΝΟ
            frames = torch.from_numpy(imgs.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
        z = encode_latent(vae, frames, device, cfg.enc_batch).cpu().numpy().astype(np.float32)
        T = states.shape[0]
        vel_est = np.zeros((T, 2), dtype=np.float32)           # [ẋ_est, θ̇_est]
        vel_est[:-1, 0] = (states[1:, 0] - states[:-1, 0]) / cfg.dt
        vel_est[:-1, 1] = (states[1:, 2] - states[:-1, 2]) / cfg.dt
        vel_est[-1] = vel_est[-2]                              # boundary: copy
        episodes.append({'z': z, 'acts': acts, 'states': states, 'vel_est': vel_est})

    with open(cache_path, 'wb') as fh:
        pickle.dump({'episodes': episodes, 'sig': sig}, fh)
    print(f"[CACHE] Αποθηκεύτηκαν {len(episodes)} επεισόδια latents -> {cache_path}")
    return episodes


# ----------------------------- DATASET (latents) -----------------------------
class LatentSequenceDataset(Dataset):
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
        z_out = torch.from_numpy(ep['z'][s + 1:s + L + 1])          # encoder latent (στόχος image/semi-vel)
        act = torch.from_numpy(ep['acts'][s:s + L])
        st_out = torch.from_numpy(ep['states'][s + 1:s + L + 1])    # (L,4) ΑΛΗΘΙΝΟ state_{t+1}
        vel_out = torch.from_numpy(ep['vel_est'][s + 1:s + L + 1])  # (L,2) εκτίμηση ταχύτητας_{t+1}
        return z_in, z_out, act, st_out, vel_out


# ------------------------------- TRAINING ------------------------------------
def _build_target(z_out, st_out, vel_out, cfg):
    """ z_target = encoder latent, με overwrite στις φυσικές dims:
        static(0,2) = ΑΛΗΘΙΝΟ· velocity(1,3) = εκτίμηση (μόνο weak). """
    z_target = z_out.clone()
    z_target[..., 0] = st_out[..., 0]       # true x
    z_target[..., 2] = st_out[..., 2]       # true θ
    if cfg.supervision == "weak":
        z_target[..., 1] = vel_out[..., 0]  # ẋ_est
        z_target[..., 3] = vel_out[..., 1]  # θ̇_est
    return z_target


def _dim_weights(cfg, device):
    """ Βάρος ανά latent dim: state_weight στις ΕΠΟΠΤΕΥΟΜΕΝΕΣ φυσικές dims, αλλιώς 1. """
    w = torch.ones(cfg.latent, device=device)
    for d in cfg.static_dims:
        w[d] = cfg.state_weight
    if cfg.supervision == "weak":
        for d in cfg.vel_dims:
            w[d] = cfg.state_weight
    return w


def train_one_epoch(predictor, loader, optimizer, device, cfg, w):
    predictor.train()
    total_loss, total_n = 0.0, 0
    for z_in, z_out, act, st_out, vel_out in tqdm(loader, desc="Training Progress", leave=False):
        z_in = z_in.to(device)
        z_out = z_out.to(device)
        act = act.to(device).float()
        st_out = st_out.to(device).float()
        vel_out = vel_out.to(device).float()
        lstm_in = torch.cat([z_in, act.unsqueeze(-1)], dim=-1)

        optimizer.zero_grad()
        pred = predictor(lstm_in)
        z_target = _build_target(z_out, st_out, vel_out, cfg)
        loss = (((pred - z_target) ** 2) * w).sum()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_n += z_in.shape[0] * z_in.shape[1]
    return total_loss / max(total_n, 1)


@torch.no_grad()
def evaluate(predictor, loader, device, cfg, w):
    """ Επιστρέφει (total per-step, static-MSE vs true, velocity-MSE vs ΑΛΗΘΙΝΗ). """
    predictor.eval()
    sd_idx = list(cfg.static_dims)
    vd_idx = list(cfg.vel_dims)
    total_loss, static_mse, vel_mse, total_n = 0.0, 0.0, 0.0, 0
    for z_in, z_out, act, st_out, vel_out in loader:
        z_in = z_in.to(device)
        z_out = z_out.to(device)
        act = act.to(device).float()
        st_out = st_out.to(device).float()
        vel_out = vel_out.to(device).float()
        lstm_in = torch.cat([z_in, act.unsqueeze(-1)], dim=-1)
        pred = predictor(lstm_in)
        z_target = _build_target(z_out, st_out, vel_out, cfg)
        total_loss += (((pred - z_target) ** 2) * w).sum().item()
        static_mse += F.mse_loss(pred[..., sd_idx], st_out[..., sd_idx], reduction='sum').item()
        # velocity vs ΑΛΗΘΙΝΗ ταχύτητα (diagnostic) -> δείχνει semi vs weak
        vel_mse += F.mse_loss(pred[..., vd_idx], st_out[..., vd_idx], reduction='sum').item()
        total_n += z_in.shape[0] * z_in.shape[1]
    return (total_loss / max(total_n, 1), static_mse / max(total_n, 1),
            vel_mse / max(total_n, 1))


def train_lstm(cfg, predictor, train_eps, val_eps, device):
    traindataset = LatentSequenceDataset(train_eps, seq_len=cfg.seq_len, stride=cfg.stride)
    valdataset = LatentSequenceDataset(val_eps, seq_len=cfg.seq_len, stride=cfg.stride)
    print(f"[DATA] train windows: {len(traindataset)} | val windows: {len(valdataset)}")

    TrainLoader = DataLoader(traindataset, batch_size=cfg.batch, shuffle=True,
                             drop_last=True, num_workers=cfg.num_workers)
    ValLoader = DataLoader(valdataset, batch_size=cfg.batch, shuffle=False,
                           drop_last=False, num_workers=cfg.num_workers)

    optimizer = optim.Adam(predictor.parameters(), lr=cfg.lr)
    w = _dim_weights(cfg, device)

    best_val = float('inf')
    epochs_no_improve = 0
    for epoch in range(cfg.epochs):
        train_loss = train_one_epoch(predictor, TrainLoader, optimizer, device, cfg, w)
        val_loss, val_static, val_vel = evaluate(predictor, ValLoader, device, cfg, w)
        print(f'Epoch {epoch + 1} [{cfg.supervision}]: train {train_loss:.4f} | '
              f'val {val_loss:.4f} | static-MSE {val_static:.4f} | vel-MSE(vs true) {val_vel:.4f}')

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
    cfg.resolve()
    assert cfg.supervision in ("semi", "weak"), "CFG.supervision ∈ {'semi','weak'}"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | supervision: {cfg.supervision}")

    vae = load_vae(cfg, device)
    predictor = LSTMTimeSeries(input_size=cfg.latent + 1, hidden_size=cfg.hidden,
                               num_layers=cfg.num_layers, output_size=cfg.latent).to(device)

    if (not cfg.force_retrain) and os.path.isfile(cfg.weights):
        predictor.load_state_dict(torch.load(cfg.weights, map_location=device))
        predictor.eval()
        print(f"[P3-LSTM] Φορτώθηκαν έτοιμα βάρη από {cfg.weights} -> παράλειψη εκπαίδευσης.")
        return vae, predictor, device

    print(f"[P3-LSTM] Δεν βρέθηκε checkpoint (ή force_retrain=True) -> εκπαίδευση.")
    train_eps = precompute_latents(vae, cfg.train, device, cfg, cfg.cache_train, cfg.force_recompute)
    val_eps = precompute_latents(vae, cfg.val, device, cfg, cfg.cache_val, cfg.force_recompute)

    best_val = train_lstm(cfg, predictor, train_eps, val_eps, device)
    predictor.load_state_dict(torch.load(cfg.save, map_location=device))
    predictor.eval()
    print(f"[P3-LSTM] Φορτώθηκε το καλύτερο μοντέλο (val={best_val:.4f}) από {cfg.save}.")
    return vae, predictor, device


# =============================================================================
#  INFERENCE — φυσική κατάσταση = ẑ[..., :state_dim] (χωρίς probe).
# =============================================================================
@torch.no_grad()
def rollout_batch(lstm, z0, actions, device):
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


def latent_to_state(z, state_dim=4):
    return z[..., :state_dim]


# Φορτώνει παγωμένο P3 VAE + (φορτώνει ή εκπαιδεύει) LSTM σε precomputed P3 latents.
vae_model, lstm_model, device = load_or_train_lstm(CFG)
