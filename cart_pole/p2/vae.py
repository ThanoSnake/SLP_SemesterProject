# =============================================================================
#  Principle 2 — Equivariance (translation/position) — Kaggle-ready
#  Χτίζει πάνω στην Αρχή 1 (split: state part + image part) και προσθέτει
#  ΣΩΣΤΟ equivariance constraint ως προς γνωστό μετασχηματισμό g.
#
#  EQUIVARIANCE (paper): "g shifts position, ḡ shifts the latent state".
#  Εδώ για CartPole:
#    g  = οριζόντια μετατόπιση εικόνας κατά dpx pixels (border replicate)
#    ḡ  = μετατόπιση του x-state dim κατά Δx = dpx * px_to_x, τα υπόλοιπα state
#         dims (θ, ẋ, θ̇) ΑΜΕΤΑΒΛΗΤΑ.
#  px_to_x: gym render 600px / world_width 4.8 = 125 px/μονάδα· μετά το resize
#           σε IMG_W=120 -> 25 px/μονάδα -> 0.04 μονάδες/pixel (= 4.8/120).
#  INVARIANCE χρώματος: αλλαγή φωτεινότητας/αντίθεσης/χρώματος -> ΚΑΜΙΑ αλλαγή στο
#    state (δ=0 σε ΟΛΕΣ τις state dims)· μόνο οι image dims επιτρέπεται να αλλάξουν.
#  Loss = recon(sum) + β·KL(image) + λ·state_MSE(P1) + μ·equiv_MSE + ν·color_inv_MSE.
#
#  Σημείωση: η equivariance ΠΕΡΙΣΤΡΟΦΗΣ (θ) θα απαιτούσε re-render της εικόνας με
#  τον simulator (περιστροφή ΟΛΗΣ της εικόνας είναι φυσικά λάθος). Με στατικά
#  frames υλοποιούμε equivariance ΘΕΣΗΣ (translation). Το paper σημειώνει ότι η
#  Αρχή 2 έχει μικρότερη επίδραση στο cart pole.
#
#  Προϋπόθεση: τρέξε πρώτα το cell του loader_v3.py (ορίζει EpisodeDataset).
#  Σώζει σε ΞΕΧΩΡΙΣΤΟ vae_p2.pth (συμβατό αρχιτεκτονικά με p1).
# =============================================================================
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

try:
    from loader_v3 import EpisodeDataset
except (ImportError, ModuleNotFoundError):
    pass


# ------------------------------- CONFIG --------------------------------------
class CFG:
    train      = "/kaggle/working/cartpole_data/train"
    val        = "/kaggle/working/cartpole_data/val"
    test       = "/kaggle/working/cartpole_data/test"

    save       = "/kaggle/working/vae_p2.pth"
    weights    = "/kaggle/working/vae_p2.pth"
    force_retrain = False

    latent       = 64
    state_dim    = 4         # [x, ẋ, θ, θ̇] — το x (index 0) είναι το position dim
    shift        = 0
    batch        = 64
    num_workers  = 2
    beta         = 1.0       # KL (image dims)
    state_weight = 100.0     # λ: εποπτεία φυσικής κατάστασης (P1)
    equiv_weight = 100.0     # μ: equivariance θέσης (P2)
    color_weight = 100.0     # ν: invariance χρώματος (P2)

    # --- equivariance (translation) ---
    max_shift_px = 12        # τυχαία οριζόντια μετατόπιση ∈ [-12, 12] px (κρατά το κάρο εντός)
    x_threshold  = 2.4       # gym CartPole: world_width = 2*x_threshold = 4.8
    img_width    = 120       # IMG_W (μετά το resize)
    # px_to_x = world_width / img_width = 4.8 / 120 = 0.04 μονάδες x ανά pixel
    px_to_x      = (2 * 2.4) / 120

    # --- color invariance (εύρη τυχαίου jitter) ---
    color_brightness = 0.3   # ±εύρος φωτεινότητας
    color_contrast   = 0.3   # ±εύρος αντίθεσης
    color_gain       = 0.3   # ±εύρος ανά-κανάλι gain (χρωματική απόχρωση)

    epochs       = 200
    patience     = 10
    min_delta    = 0.0
    lr           = 1e-3


class VAE_P1(nn.Module):
    """ Ίδιο split-encoding VAE με την Αρχή 1 (state part + image part). """

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

    def state_of(self, x):
        """ Ντετερμινιστικό state part (ΜΕ gradient — για supervision & equivariance). """
        feat = self.encoder(x).reshape(x.size(0), -1)
        return self.state_head(feat)

    def forward(self, x):
        b = x.size(0)
        feat = self.encoder(x).reshape(b, -1)
        state = self.state_head(feat)
        img_mu = self.fc_mu(feat)
        img_logvar = self.fc_logvar(feat)
        img_z = self.reparameterize(img_mu, img_logvar)
        z = torch.cat([state, img_z], dim=1)
        decoded = self.fc_decode(z).view(b, 64, 10, 15)
        recon = self.decoder(decoded)
        return recon, state, img_mu, img_logvar

    @torch.no_grad()
    def encode_state(self, x):
        return self.state_of(x)


def shift_image(x, dpx):
    """ Οριζόντια μετατόπιση κατά dpx pixels με border-replicate (όχι wrap).
    dpx>0 -> περιεχόμενο/κάρο δεξιά (x αυξάνεται). x: (B,3,H,W). """
    W = x.shape[-1]
    if dpx == 0 or abs(dpx) >= W:
        return x
    if dpx > 0:
        left = x[..., :1].expand(*x.shape[:-1], dpx)        # επανάληψη αριστερής στήλης
        return torch.cat([left, x[..., :W - dpx]], dim=-1)
    d = -dpx
    right = x[..., -1:].expand(*x.shape[:-1], d)            # επανάληψη δεξιάς στήλης
    return torch.cat([x[..., d:], right], dim=-1)


def equivariance_loss(model, x, state_pred, cfg):
    """ Translation equivariance: state(g(x)) ≈ state(x) + [Δx, 0, 0, 0].
    Διαλέγει τυχαίο dpx, μετατοπίζει, και επιβάλλει τη γνωστή μετατόπιση. """
    dpx = random.randint(1, cfg.max_shift_px) * random.choice([-1, 1])
    x_shift = shift_image(x, dpx)
    state_shift = model.state_of(x_shift)
    delta = torch.zeros_like(state_pred)
    delta[:, 0] = dpx * cfg.px_to_x        # μόνο το position dim μετατοπίζεται
    return F.mse_loss(state_shift, state_pred + delta, reduction='sum')


def color_jitter(x, cfg):
    """ Τυχαία αλλαγή φωτεινότητας/αντίθεσης/χρώματος (per-sample). x: (B,3,H,W) σε [0,1]. """
    B, dev = x.shape[0], x.device
    bright = 1.0 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * cfg.color_brightness
    contrast = 1.0 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * cfg.color_contrast
    gain = 1.0 + (torch.rand(B, 3, 1, 1, device=dev) * 2 - 1) * cfg.color_gain
    out = x * bright
    mean = out.mean(dim=(1, 2, 3), keepdim=True)       # per-image μέση φωτεινότητα
    out = (out - mean) * contrast + mean
    return (out * gain).clamp(0, 1)


def color_invariance_loss(model, x, state_pred, cfg):
    """ Color invariance: state(color(x)) ≈ state(x) (δ=0 σε ΟΛΕΣ τις state dims).
    Η φυσική κατάσταση ΔΕΝ αλλάζει με το χρώμα· μόνο οι (μη-περιορισμένες εδώ)
    image dims επιτρέπεται να την απορροφήσουν. """
    x_color = color_jitter(x, cfg)
    state_color = model.state_of(x_color)
    return F.mse_loss(state_color, state_pred, reduction='sum')


def train_one_episode(model, frames, states, optimizer, device, cfg):
    n = frames.shape[0]
    total_loss = 0.0
    for s in range(0, n, cfg.batch):
        x = frames[s:s + cfg.batch].to(device)
        target = states[s:s + cfg.batch].to(device).float()
        optimizer.zero_grad()

        recon, state_pred, img_mu, img_logvar = model(x)
        recon_loss = F.mse_loss(recon, x, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + img_logvar - img_mu.pow(2) - img_logvar.exp())
        state_loss = F.mse_loss(state_pred, target, reduction='sum')          # P1 supervision
        equiv_loss = equivariance_loss(model, x, state_pred, cfg)             # P2 equivariance (θέση)
        color_loss = color_invariance_loss(model, x, state_pred, cfg)         # P2 invariance (χρώμα)

        loss = (recon_loss + cfg.beta * kl_div + cfg.state_weight * state_loss
                + cfg.equiv_weight * equiv_loss + cfg.color_weight * color_loss)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss, n


@torch.no_grad()
def evaluate(model, loader, device, cfg):
    """ Validation: total (recon+β·KL+λ·state) per-frame, state MSE per-frame,
    και equivariance error per-frame (διαγνωστικό). """
    model.eval()
    tot, state_se, equiv_se, color_se, n = 0.0, 0.0, 0.0, 0.0, 0
    for frames, _img2, _act, states, _s2 in loader:
        nf = frames.shape[0]
        for s in range(0, nf, cfg.batch):
            x = frames[s:s + cfg.batch].to(device)
            target = states[s:s + cfg.batch].to(device).float()
            recon, state_pred, img_mu, img_logvar = model(x)
            recon_loss = F.mse_loss(recon, x, reduction='sum')
            kl_div = -0.5 * torch.sum(1 + img_logvar - img_mu.pow(2) - img_logvar.exp())
            state_loss = F.mse_loss(state_pred, target, reduction='sum')
            tot += (recon_loss + cfg.beta * kl_div + cfg.state_weight * state_loss).item()
            state_se += state_loss.item()
            equiv_se += equivariance_loss(model, x, state_pred, cfg).item()
            color_se += color_invariance_loss(model, x, state_pred, cfg).item()
        n += nf
    return (tot / max(n, 1), state_se / max(n, 1),
            equiv_se / max(n, 1), color_se / max(n, 1))


def train_vae(cfg, model, device):
    traindataset = EpisodeDataset(root=cfg.train, shift=cfg.shift)
    valdataset = EpisodeDataset(root=cfg.val, shift=cfg.shift)
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

        val_loss, val_state, val_equiv, val_color = evaluate(model, ValLoader, device, cfg)
        print(f'Epoch {epoch + 1}: train {train_loss:.2f} | val {val_loss:.2f} '
              f'| state {val_state:.4f} | equiv {val_equiv:.4f} | color {val_color:.4f}')

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
        print(f"[P2-VAE] Φορτώθηκαν έτοιμα βάρη από {cfg.weights} -> παράλειψη εκπαίδευσης.")
        return model, device

    print(f"[P2-VAE] Δεν βρέθηκε checkpoint (ή force_retrain=True) -> εκπαίδευση.")
    best_val = train_vae(cfg, model, device)
    model.load_state_dict(torch.load(cfg.save, map_location=device))
    model.eval()
    print(f"[P2-VAE] Φορτώθηκε το καλύτερο μοντέλο (val={best_val:.2f}) από {cfg.save}.")
    return model, device


# Φορτώνει έτοιμα βάρη αν υπάρχουν, αλλιώς εκπαιδεύει & σώζει το καλύτερο μοντέλο.
vae_model, device = load_or_train_vae(CFG)
