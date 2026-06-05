# =============================================================================
#  LSTM — Memory/Dynamics component (M) — έκδοση v2 (ΧΩΡΙΣ buffers)
#  Προβλέπει το επόμενο latent:  (z_t, a_t) -> z_{t+1}
#  Φορτώνει ένα επεισόδιο τη φορά (loader_v3.EpisodeDataset).
#  Έκδοση για Kaggle Notebook (χωρίς argparse / CLI parsers).
#
#  Διαφορά από lstm.py: αντί για buffers + fixed 30-step windows, ΚΑΘΕ επεισόδιο
#  τρέχει ως ΜΙΑ ακολουθία μήκους N=T-1. Ο DataLoader (batch_size=None,
#  num_workers>0) δίνει ένα επεισόδιο τη φορά (παράλληλη φόρτωση).
#
#  Προϋπόθεση: εκπαιδευμένο & αποθηκευμένο VAE (vae.pth). Φορτώνεται ΠΑΓΩΜΕΝΟ.
#
#  Τρόπος χρήσης:
#    1) Τρέξε πρώτα το cell με το loader_v3.py (ορίζει το EpisodeDataset).
#    2) Βεβαιώσου ότι υπάρχει το VAE checkpoint στο CFG.vae_weights.
#    3) Άλλαξε ό,τι θες στο CFG και τρέξε αυτό το cell.
#       Στο τέλος καλείται load_or_train_lstm(CFG).
#
#  Kaggle persistence: το /kaggle/working αδειάζει ανά session — κράτα τα .pth
#  μέσω Notebook Persistence ("Files only") ή ως Kaggle Dataset (/kaggle/input/...).
# =============================================================================
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Το EpisodeDataset ορίζεται στο loader_v3.py (import αν είναι αρχείο/utility
# script, αλλιώς υπάρχει ήδη από προηγούμενο cell).
try:
    from loader_v3 import EpisodeDataset
except (ImportError, ModuleNotFoundError):
    pass


# ------------------------------- CONFIG --------------------------------------
class CFG:
    """ Ρυθμίσεις εκπαίδευσης (άλλαξέ τες εδώ αντί για command-line arguments). """
    train       = "/kaggle/working/cartpole_data/train"   # φάκελος train (.npz)
    val         = "/kaggle/working/cartpole_data/val"     # φάκελος val

    # --- checkpoint paths ---
    vae_weights = "/kaggle/working/vae.pth"   # ΠΑΓΩΜΕΝΟ VAE (πρέπει να υπάρχει)
    save        = "/kaggle/working/lstm.pth"  # ΠΟΥ γράφεται το ΚΑΛΥΤΕΡΟ LSTM (έξοδος)
    weights     = "/kaggle/working/lstm.pth"  # ΑΠΟ ΠΟΥ φορτώνονται έτοιμα LSTM βάρη (είσοδος)
    force_retrain = False                     # True -> εκπαίδευσε ξανά ακόμη κι αν υπάρχει checkpoint

    # --- model ---
    latent      = 64       # latent size του VAE (= μέγεθος z)
    hidden      = 64       # hidden size του LSTM
    num_layers  = 2

    # --- training ---
    shift       = 0        # επίπεδο θορύβου: 0 (clean) | 2 | 5 | 10
    enc_batch   = 256      # GPU sub-batch για το (frozen) encode των frames -> z
    num_workers = 2        # παράλληλη φόρτωση επεισοδίων (overlap με GPU)
    epochs      = 200
    patience    = 10
    min_delta   = 0.0
    lr          = 1e-3


class VAE(nn.Module):
    """ Ίδιο VAE με το vae.py/vae_v2.py — εδώ φορτώνεται παγωμένο από το vae.pth. """

    def __init__(self, latent_size=64):
        super(VAE, self).__init__()
        self.latent_size = latent_size
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(64 * 10 * 15, latent_size)
        self.fc_logvar = nn.Linear(64 * 10 * 15, latent_size)
        self.fc_decode = nn.Linear(latent_size, 64 * 10 * 15)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        batch_size = x.size(0)
        encoded = self.encoder(x).reshape(batch_size, -1)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        z = self.reparameterize(mu, logvar)
        decoded = self.fc_decode(z).view(batch_size, 64, 10, 15)
        reconstructed = self.decoder(decoded)
        return reconstructed, mu, logvar


class LSTMTimeSeries(nn.Module):
    """ Memory/Dynamics component (M): προβλέπει το επόμενο latent.

    Είσοδος ανά βήμα: [z_t (latent) || a_t (action)]  -> input_size = latent + 1
    Έξοδος ανά βήμα:  z_{t+1} (latent)                -> output_size = latent
    """

    def __init__(self, input_size=65, hidden_size=64, num_layers=2, output_size=64):
        super(LSTMTimeSeries, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out)


@torch.no_grad()
def encode_mu(vae, frames, device, enc_batch):
    """ Frozen VAE encoder -> mu (ντετερμινιστικό latent). Σε sub-batches για τη μνήμη. """
    outs = []
    for s in range(0, frames.shape[0], enc_batch):
        x = frames[s:s + enc_batch].to(device)
        enc = vae.encoder(x).reshape(x.size(0), -1)
        outs.append(vae.fc_mu(enc))
    return torch.cat(outs, dim=0)  # (N, latent)


def episode_to_io(vae, frames_in, frames_out, action, device, enc_batch):
    """ Ένα επεισόδιο -> (lstm_in, z_next) για ακολουθία μήκους N=T-1.
        lstm_in : (1, N, latent+1)   z_next : (1, N, latent) """
    z_t = encode_mu(vae, frames_in, device, enc_batch)      # (N, latent)
    z_next = encode_mu(vae, frames_out, device, enc_batch)  # (N, latent)
    action = action.to(device).float().unsqueeze(-1)        # (N, 1)
    lstm_in = torch.cat([z_t, action], dim=-1).unsqueeze(0)  # (1, N, latent+1)
    return lstm_in, z_next.unsqueeze(0)                      # (1, N, latent)


def load_vae(cfg, device):
    """ Φορτώνει το ΠΑΓΩΜΕΝΟ VAE από cfg.vae_weights (απαιτείται να υπάρχει). """
    if not os.path.isfile(cfg.vae_weights):
        raise FileNotFoundError(
            f"Δεν βρέθηκαν βάρη VAE στο '{cfg.vae_weights}'.\n"
            f"Τρέξε πρώτα το vae_v2.py (ή vae.py), ή διόρθωσε το CFG.vae_weights "
            f"(π.χ. /kaggle/input/<dataset>/vae.pth).")
    vae = VAE(latent_size=cfg.latent).to(device)
    vae.load_state_dict(torch.load(cfg.vae_weights, map_location=device))
    vae.eval()
    for p in vae.parameters():          # πάγωμα: το VAE δεν εκπαιδεύεται εδώ
        p.requires_grad_(False)
    print(f"[VAE] Παγωμένα βάρη φορτώθηκαν από {cfg.vae_weights}.")
    return vae


@torch.no_grad()
def evaluate(predictor, vae, val_loader, device, enc_batch):
    """ Validation σε ΟΛΟ το val set (όλα τα επεισόδια). Per-step latent MSE. """
    predictor.eval()
    total_loss, total_n = 0.0, 0
    for frames_in, frames_out, action, _s1, _s2 in val_loader:
        lstm_in, z_next = episode_to_io(vae, frames_in, frames_out, action, device, enc_batch)
        pred = predictor(lstm_in)
        total_loss += F.mse_loss(pred, z_next, reduction='sum').item()
        total_n += z_next.shape[1]   # N βήματα
    return total_loss / max(total_n, 1)


def train_lstm(cfg, vae, predictor, device):
    """ Βρόχος εκπαίδευσης του LSTM με validation + early stopping (ΧΩΡΙΣ buffers).
    Γράφει το ΚΑΛΥΤΕΡΟ LSTM στο cfg.save. Επιστρέφει το best val loss. """
    traindataset = EpisodeDataset(root=cfg.train, shift=cfg.shift)
    valdataset = EpisodeDataset(root=cfg.val, shift=cfg.shift)

    # batch_size=None -> ο DataLoader δίνει ΕΝΑ επεισόδιο τη φορά (χωρίς collation).
    TrainLoader = DataLoader(traindataset, batch_size=None, shuffle=True,
                             num_workers=cfg.num_workers, pin_memory=True)
    ValLoader = DataLoader(valdataset, batch_size=None, shuffle=False,
                           num_workers=cfg.num_workers, pin_memory=True)

    optimizer = optim.Adam(predictor.parameters(), lr=cfg.lr)

    best_val = float('inf')
    epochs_no_improve = 0
    for epoch in range(cfg.epochs):
        vae.eval()
        predictor.train()
        epoch_loss, epoch_n = 0.0, 0
        for frames_in, frames_out, action, _s1, _s2 in tqdm(
                TrainLoader, desc=f"Epoch {epoch + 1}", leave=False):
            lstm_in, z_next = episode_to_io(vae, frames_in, frames_out, action, device, cfg.enc_batch)

            optimizer.zero_grad()
            pred = predictor(lstm_in)                           # (1, N, latent)
            loss = F.mse_loss(pred, z_next, reduction='sum')
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_n += z_next.shape[1]
        train_loss = epoch_loss / max(epoch_n, 1)

        val_loss = evaluate(predictor, vae, ValLoader, device, cfg.enc_batch)
        print(f'Epoch {epoch + 1}: train {train_loss:.4f} | val {val_loss:.4f}')

        if val_loss < best_val - cfg.min_delta:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save(predictor.state_dict(), cfg.save)  # κρατάμε το ΚΑΛΥΤΕΡΟ LSTM
            print(f'  ✓ νέο καλύτερο val ({best_val:.4f}) -> αποθήκευση στο {cfg.save}')
        else:
            epochs_no_improve += 1
            print(f'  χωρίς βελτίωση ({epochs_no_improve}/{cfg.patience})')
            if epochs_no_improve >= cfg.patience:
                print(f'Early stopping στο epoch {epoch + 1}. Καλύτερο val loss: {best_val:.4f}')
                break

    return best_val


def load_or_train_lstm(cfg=CFG):
    """ Επιστρέφει (vae_model, lstm_model, device). Φορτώνει ΠΑΝΤΑ παγωμένο VAE.
    Για το LSTM: φορτώνει checkpoint αν υπάρχει, αλλιώς εκπαιδεύει & σώζει το best. """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    vae = load_vae(cfg, device)
    predictor = LSTMTimeSeries(input_size=cfg.latent + 1, hidden_size=cfg.hidden,
                               num_layers=cfg.num_layers, output_size=cfg.latent).to(device)

    if (not cfg.force_retrain) and os.path.isfile(cfg.weights):
        predictor.load_state_dict(torch.load(cfg.weights, map_location=device))
        predictor.eval()
        print(f"[LSTM] Φορτώθηκαν έτοιμα βάρη από {cfg.weights} -> παράλειψη εκπαίδευσης.")
        return vae, predictor, device

    print(f"[LSTM] Δεν βρέθηκε checkpoint στο {cfg.weights} (ή force_retrain=True) -> εκπαίδευση.")
    best_val = train_lstm(cfg, vae, predictor, device)
    predictor.load_state_dict(torch.load(cfg.save, map_location=device))  # best (early stopping)
    predictor.eval()
    print(f"[LSTM] Φορτώθηκε το καλύτερο μοντέλο (val={best_val:.4f}) από {cfg.save}.")
    return vae, predictor, device


# =============================================================================
#  INFERENCE — πρόβλεψη μετά την εκπαίδευση
#  (ΔΕΝ τρέχουν εδώ· κάλεσέ τα σε ΕΠΟΜΕΝΟ cell με τα vae_model/lstm_model/device)
# =============================================================================
def init_hidden(lstm, batch=1, device='cpu'):
    """ Μηδενική αρχική κρυφή κατάσταση (h0, c0): σχήμα (num_layers, batch, hidden). """
    h = torch.zeros(lstm.num_layers, batch, lstm.hidden_size, device=device)
    c = torch.zeros(lstm.num_layers, batch, lstm.hidden_size, device=device)
    return (h, c)


@torch.no_grad()
def predict_next(lstm, z_t, a_t, hc=None):
    """ ΕΝΑ βήμα (stateful): (z_t, a_t) + κρυφή κατάσταση -> (ẑ_{t+1}, νέα κρυφή κατάσταση).

    Χρησιμοποιεί ΑΠΕΥΘΕΙΑΣ τα lstm.lstm / lstm.fc ώστε να ΚΡΑΤΑΕΙ το (h,c)
    (το lstm.forward θα μηδένιζε το state σε κάθε κλήση).
        z_t : (latent,) ή (1, latent)      a_t : scalar ή tensor
        hc  : (h, c) ή None (αρχικοποιείται στο 0)
    """
    lstm.eval()
    device = z_t.device
    z = z_t.reshape(1, 1, -1).float()
    a = torch.as_tensor(a_t, dtype=torch.float32, device=device).reshape(1, 1, 1)
    if hc is None:
        hc = init_hidden(lstm, batch=1, device=device)
    out, hc = lstm.lstm(torch.cat([z, a], dim=-1), hc)   # (1,1,latent+1) -> (1,1,hidden)
    z_next = lstm.fc(out).reshape(-1)                    # (latent,)
    return z_next, hc


@torch.no_grad()
def warmup(lstm, z_seq, a_seq, hc=None):
    """ 'Φορτίζει' το hidden state τρέχοντας το LSTM σε ΠΡΑΓΜΑΤΙΚΑ (z, a).
        z_seq : (L, latent)    a_seq : (L,)
    Επιστρέφει (τελευταία πρόβλεψη ẑ, hc). """
    lstm.eval()
    device = z_seq.device
    L = z_seq.shape[0]
    z = z_seq.reshape(1, L, -1).float()
    a = torch.as_tensor(a_seq, dtype=torch.float32, device=device).reshape(1, L, 1)
    if hc is None:
        hc = init_hidden(lstm, batch=1, device=device)
    out, hc = lstm.lstm(torch.cat([z, a], dim=-1), hc)
    return lstm.fc(out)[:, -1].reshape(-1), hc


@torch.no_grad()
def rollout(lstm, z_start, actions, hc=None):
    """ Autoregressive rollout K βημάτων (τροφοδοτεί τις ΔΙΚΕΣ του προβλέψεις).
        z_start : (latent,) το ΠΡΑΓΜΑΤΙΚΟ z από όπου ξεκινάμε
        actions : (K,) οι ενέργειες των K βημάτων (η 1η συνδυάζεται με το z_start)
    Επιστρέφει (z_preds (K, latent), hc).  Τα λάθη ΣΥΣΣΩΡΕΥΟΝΤΑΙ όσο μεγαλώνει το K. """
    lstm.eval()
    device = z_start.device
    if hc is None:
        hc = init_hidden(lstm, batch=1, device=device)
    actions = torch.as_tensor(actions, dtype=torch.float32, device=device).reshape(-1)
    z = z_start
    preds = []
    for k in range(actions.shape[0]):
        z, hc = predict_next(lstm, z, actions[k], hc)   # η πρόβλεψη γίνεται είσοδος του επόμενου
        preds.append(z)
    return torch.stack(preds, dim=0), hc


@torch.no_grad()
def decode_latents(vae, z):
    """ z: (K, latent) -> frames (K, 3, H, W) μέσω του VAE decoder. """
    vae.eval()
    z = z.reshape(z.shape[0], -1).float()
    dec = vae.fc_decode(z).view(z.shape[0], 64, 10, 15)
    return vae.decoder(dec)


@torch.no_grad()
def imagine(vae, lstm, context_frames, actions, horizon, device, enc_batch=256):
    """ Warm-up σε πραγματικά context frames + autoregressive rollout 'horizon' βημάτων.

        context_frames : (M, 3, H, W) πραγματικά frames (M >= 1)
        actions        : (M-1+horizon,) όλες οι ενέργειες a_0 .. a_{M-2+horizon}
                         - οι πρώτες M-1 'φορτίζουν' το hidden (μεταβάσεις του context)
                         - οι υπόλοιπες 'horizon' οδηγούν το rollout
        horizon (K)    : πόσα μελλοντικά latents/frames να προβλεφθούν

    Επιστρέφει:
        z_preds     : (K, latent)   ẑ_M .. ẑ_{M+K-1}
        pred_frames : (K, 3, H, W)  τα decoded προβλεπόμενα frames
    """
    vae.eval()
    lstm.eval()
    M = context_frames.shape[0]
    actions = torch.as_tensor(actions, dtype=torch.float32, device=device).reshape(-1)
    need = M - 1 + horizon
    assert actions.shape[0] == need, \
        f"Χρειάζονται M-1+horizon = {need} ενέργειες, δόθηκαν {actions.shape[0]}."

    z_ctx = encode_mu(vae, context_frames, device, enc_batch)        # (M, latent) πραγματικά
    hc = init_hidden(lstm, batch=1, device=device)
    if M >= 2:                                                       # warm-up: μεταβάσεις 0..M-2
        _z_last, hc = warmup(lstm, z_ctx[:M - 1], actions[:M - 1], hc)
    # rollout ξεκινώντας από το ΠΡΑΓΜΑΤΙΚΟ z_{M-1}, με actions a_{M-1}..a_{M+K-2}
    z_preds, _hc = rollout(lstm, z_ctx[M - 1], actions[M - 1:], hc)  # (K, latent)
    pred_frames = decode_latents(vae, z_preds)                       # (K, 3, H, W)
    return z_preds, pred_frames


# Παράδειγμα χρήσης (σε ΕΠΟΜΕΝΟ cell, αφού τρέξει το load_or_train_lstm):
#   import numpy as np
#   d = np.load("/kaggle/working/cartpole_data/test/0.npz")
#   frames = torch.from_numpy(d["imgs"].astype(np.float32) / 255.0).permute(0, 3, 1, 2)
#   M, K = 10, 20
#   ctx  = frames[:M].to(device)
#   acts = torch.from_numpy(d["acts"][:M - 1 + K].astype(np.float32))   # M-1+K ενέργειες
#   z_pred, pred_frames = imagine(vae_model, lstm_model, ctx, acts, horizon=K, device=device)
#   # pred_frames: (K, 3, 80, 120) -> pred_frames.permute(0, 2, 3, 1).cpu().numpy() για matplotlib
#
# Streaming/βήμα-βήμα:
#   z = encode_mu(vae_model, frames[:1].to(device), device, 256)[0]    # πραγματικό z_0
#   hc = None
#   z, hc = predict_next(lstm_model, z, a_0, hc)   # ẑ_1
#   z, hc = predict_next(lstm_model, z, a_1, hc)   # ẑ_2 (autoregressive) ...


# Φορτώνει παγωμένο VAE + (φορτώνει ή εκπαιδεύει) LSTM, χωρίς περιττό re-train.
vae_model, lstm_model, device = load_or_train_lstm(CFG)
