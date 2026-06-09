# =============================================================================
#  Principle 1 — Separate Encoding VAE — έκδοση v2 (ΧΩΡΙΣ buffers)
#  Ίδια ιδέα με p1/seperate_encoding.py (split: state part + image part), αλλά:
#    - ΧΩΡΙΣ chunks/buffers -> ένα επεισόδιο τη φορά (loader_v3.EpisodeDataset),
#      DataLoader(batch_size=None, num_workers>0) -> παράλληλη φόρτωση, πιο γρήγορο.
#    - Πιο απλός κώδικας (όπως vae_v2.py του baseline).
#
#  STATE PART (πρώτα state_dim=4 dims): ντετερμινιστικό, ΕΠΟΠΤΕΥΕΤΑΙ να ισούται με
#  την πραγματική φυσική κατάσταση [x, ẋ, θ, θ̇] (το EpisodeDataset το δίνει ως
#  states_t). IMAGE PART (60 dims): κανονικό VAE (reparameterize + KL).
#  Loss = recon(MSE, sum) + β·KL(image dims) + λ·MSE(state_part, αληθινή κατάσταση).
#
#  Προϋπόθεση: τρέξε πρώτα το cell του loader_v3.py (ορίζει EpisodeDataset).
#  Σώζει σε ΞΕΧΩΡΙΣΤΟ vae_p1.pth (συμβατό με p1/lstm.py & p1/seperate_encoding.py).
# =============================================================================
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm   # .auto -> σε Kaggle/Jupyter μία ενιαία μπάρα-widget (όχι νέα γραμμή ανά update)

try:
    from loader_v3 import EpisodeDataset
except (ImportError, ModuleNotFoundError):
    pass


# ------------------------------- CONFIG --------------------------------------
class CFG:
    train      = "/kaggle/working/cartpole_data/train"
    val        = "/kaggle/working/cartpole_data/val"
    test       = "/kaggle/working/cartpole_data/test"

    save       = "/kaggle/working/vae_p1.pth"   # ΞΕΧΩΡΙΣΤΟ checkpoint (όχι το baseline vae.pth)
    weights    = "/kaggle/working/vae_p1.pth"
    force_retrain = False

    latent       = 64       # ίδιο συνολικό latent με το baseline
    state_dim    = 4        # πρώτα dims = φυσική κατάσταση [x, ẋ, θ, θ̇]
    shift        = 0        # 0 (clean εποπτεία) | 2 | 5 | 10 (noisy)
    batch        = 64       # GPU sub-batch: πόσα frames του επεισοδίου ανά forward/step
    num_workers  = 2        # παράλληλη φόρτωση επεισοδίων
    beta         = 1.0      # βάρος KL (μόνο image dims)
    state_weight = 100.0    # λ: βάρος εποπτείας φυσικής κατάστασης (ΚΡΙΣΙΜΟ — δοκίμασε 10–1000)
    epochs       = 200
    patience     = 10
    min_delta    = 0.0
    lr           = 1e-3


class VAE_P1(nn.Module):
    """ VAE με διαχωρισμένο encoding (Principle 1).

    encoder (shared conv) -> feat
      ├── state_head      : feat -> state_dim   (ντετερμινιστικό, εποπτευόμενο)
      └── fc_mu/fc_logvar : feat -> (latent - state_dim)   (image VAE part)
    z = [state_part || image_part] -> fc_decode -> decoder
    """

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

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        b = x.size(0)
        feat = self.encoder(x).reshape(b, -1)
        state = self.state_head(feat)                  # (B, state_dim) — ντετερμινιστικό
        img_mu = self.fc_mu(feat)                      # (B, img_dim)
        img_logvar = self.fc_logvar(feat)
        img_z = self.reparameterize(img_mu, img_logvar)
        z = torch.cat([state, img_z], dim=1)           # (B, latent)
        decoded = self.fc_decode(z).view(b, 64, 10, 15)
        recon = self.decoder(decoded)
        return recon, state, img_mu, img_logvar

    @torch.no_grad()
    def encode_state(self, x):
        """ Inference: φυσική κατάσταση ΚΑΤΕΥΘΕΙΑΝ από το state part (χωρίς probe). """
        feat = self.encoder(x).reshape(x.size(0), -1)
        return self.state_head(feat)


def p1_loss(recon, x, state_pred, state_target, img_mu, img_logvar, beta, state_weight):
    """ recon (sum) + β·KL(image dims) + λ·state MSE (sum). Επιστρέφει (total, components). """
    recon_loss = F.mse_loss(recon, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + img_logvar - img_mu.pow(2) - img_logvar.exp())
    state_loss = F.mse_loss(state_pred, state_target, reduction='sum')
    total = recon_loss + beta * kl_div + state_weight * state_loss
    return total, (recon_loss.item(), kl_div.item(), state_loss.item())


def train_one_episode(model, frames, states, optimizer, device, cfg):
    """ Εκπαίδευση στα frames + states ΕΝΟΣ επεισοδίου, σε sub-batches.
    Επιστρέφει (άθροισμα loss, πλήθος frames). """
    n = frames.shape[0]
    total_loss = 0.0
    for s in range(0, n, cfg.batch):
        x = frames[s:s + cfg.batch].to(device)
        target = states[s:s + cfg.batch].to(device).float()
        optimizer.zero_grad()
        recon, state_pred, img_mu, img_logvar = model(x)
        loss, _ = p1_loss(recon, x, state_pred, target, img_mu, img_logvar,
                          cfg.beta, cfg.state_weight)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss, n


@torch.no_grad()
def evaluate(model, loader, device, cfg):
    """ Validation σε ΟΛΟ το val. Επιστρέφει (total per-frame, state MSE per-frame). """
    model.eval()
    tot, state_se, n = 0.0, 0.0, 0
    for frames, _img2, _act, states, _s2 in loader:
        nf = frames.shape[0]
        for s in range(0, nf, cfg.batch):
            x = frames[s:s + cfg.batch].to(device)
            target = states[s:s + cfg.batch].to(device).float()
            recon, state_pred, img_mu, img_logvar = model(x)
            _, comps = p1_loss(recon, x, state_pred, target, img_mu, img_logvar,
                               cfg.beta, cfg.state_weight)
            tot += comps[0] + cfg.beta * comps[1] + cfg.state_weight * comps[2]
            state_se += comps[2]
        n += nf
    return tot / max(n, 1), state_se / max(n, 1)


def train_vae(cfg, model, device):
    traindataset = EpisodeDataset(root=cfg.train, shift=cfg.shift)
    valdataset = EpisodeDataset(root=cfg.val, shift=cfg.shift)
    # persistent_workers=True -> τα workers μένουν ζωντανά ανάμεσα στα epochs
    # (φεύγει το "AssertionError: can only test a child process" στο shutdown).
    persistent = cfg.num_workers > 0
    TrainLoader = DataLoader(traindataset, batch_size=None, shuffle=True,
                             num_workers=cfg.num_workers, pin_memory=True,
                             persistent_workers=persistent)
    ValLoader = DataLoader(valdataset, batch_size=None, shuffle=False,
                           num_workers=cfg.num_workers, pin_memory=True,
                           persistent_workers=persistent)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    best_val = float('inf')
    epochs_no_improve = 0
    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss, epoch_n = 0.0, 0
        for frames, _img2, _act, states, _s2 in tqdm(
                TrainLoader, desc=f"Epoch {epoch + 1}", leave=False):
            el, en = train_one_episode(model, frames, states, optimizer, device, cfg)
            epoch_loss += el
            epoch_n += en
        train_loss = epoch_loss / max(epoch_n, 1)

        val_loss, val_state = evaluate(model, ValLoader, device, cfg)
        print(f'Epoch {epoch + 1}: train {train_loss:.2f} | val {val_loss:.2f} '
              f'| val state-MSE {val_state:.4f}')

        if val_loss < best_val - cfg.min_delta:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), cfg.save)
            print(f'  ✓ νέο καλύτερο val ({best_val:.2f}) -> {cfg.save}')
        else:
            epochs_no_improve += 1
            print(f'  χωρίς βελτίωση ({epochs_no_improve}/{cfg.patience})')
            if epochs_no_improve >= cfg.patience:
                print(f'Early stopping στο epoch {epoch + 1}.')
                break
    return best_val


def load_or_train_vae(cfg=CFG):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    model = VAE_P1(latent_size=cfg.latent, state_dim=cfg.state_dim).to(device)

    if (not cfg.force_retrain) and os.path.isfile(cfg.weights):
        model.load_state_dict(torch.load(cfg.weights, map_location=device))
        model.eval()
        print(f"[P1-VAE] Φορτώθηκαν έτοιμα βάρη από {cfg.weights} -> παράλειψη εκπαίδευσης.")
        return model, device

    print(f"[P1-VAE] Δεν βρέθηκε checkpoint (ή force_retrain=True) -> εκπαίδευση.")
    best_val = train_vae(cfg, model, device)
    model.load_state_dict(torch.load(cfg.save, map_location=device))
    model.eval()
    print(f"[P1-VAE] Φορτώθηκε το καλύτερο μοντέλο (val={best_val:.2f}) από {cfg.save}.")
    return model, device


# Φορτώνει έτοιμα βάρη αν υπάρχουν, αλλιώς εκπαιδεύει & σώζει το καλύτερο μοντέλο.
vae_model, device = load_or_train_vae(CFG)
