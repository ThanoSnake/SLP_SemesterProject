# =============================================================================
#  Principle 3 — Multi-level supervision (semi / weak) — VAE — Kaggle-ready
#  Βασική ιδέα του paper για CartPole:
#    (1) semi : εποπτεύεται ΜΟΝΟ το ΣΤΑΤΙΚΟ (θέση x, γωνία θ)· η ΤΑΧΥΤΗΤΑ άγνωστη.
#    (2) weak : + εποπτεία ταχύτητας με ΕΚΤΙΜΗΣΗ από φυσική γνώση (finite diff):
#               ẋ_est=(x_{t+1}-x_t)/dt , θ̇_est=(θ_{t+1}-θ_t)/dt  (dt=tau=0.02).
#
#  Δείκτες state [x, ẋ, θ, θ̇]: static = [0, 2], velocity = [1, 3].
#  Διακόπτης: CFG.supervision ∈ {"semi", "weak"}.
#
#  Single-frame encoder (όπως p1/p2 -> δίκαιη σύγκριση Fig 3). Το ζεύγος (t,t+1)
#  που δίνει το EpisodeDataset το χρησιμοποιούμε ΜΟΝΟ για το label ταχύτητας.
#  Εποπτεία σε VAE + LSTM (Option I), συνεπές με p1/p2.
#
#  Αρχιτεκτονική = VAE_P1 (4 state dims + 60 image). Inference: state = z[:, :4]
#  (χωρίς probe). Σώζει σε vae_p3_<mode>.pth.
#
#  Προϋπόθεση: τρέξε πρώτα το cell του loader_v3.py (ορίζει EpisodeDataset).
# =============================================================================
import os

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

    supervision = "weak"     # "semi" (μόνο static) | "weak" (+ εκτιμώμενη ταχύτητα)
    save       = None        # αν None -> "/kaggle/working/vae_p3_<supervision>.pth"
    weights    = None        # αν None -> ίδιο με save
    force_retrain = False

    latent       = 64
    state_dim    = 4
    static_dims  = (0, 2)    # x, θ
    vel_dims     = (1, 3)    # ẋ, θ̇
    dt           = 0.02      # gym CartPole tau -> για την εκτίμηση ταχύτητας
    shift        = 0
    batch        = 64
    num_workers  = 2
    beta         = 1.0       # KL (image dims)
    state_weight = 100.0     # λ: εποπτεία φυσικής κατάστασης
    epochs       = 200
    patience     = 10
    min_delta    = 0.0
    lr           = 1e-3

    @classmethod
    def resolve(cls):
        if cls.save is None:
            cls.save = f"/kaggle/working/vae_p3_{cls.supervision}.pth"
        if cls.weights is None:
            cls.weights = cls.save


class VAE_P1(nn.Module):
    """ Ίδιο split-encoding VAE με p1/p2 (state part + image part). """

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
        feat = self.encoder(x).reshape(x.size(0), -1)
        return self.state_head(feat)


_LOSS_KEYS = ('recon', 'kl', 'state', 'vel_true')   # vel_true = ΔΙΑΓΝΩΣΤΙΚΟ (όχι στο total)


def _state_mask(cfg, device):
    """ Ποιες state dims εποπτεύονται: semi -> static μόνο· weak -> όλες. """
    m = [0.0, 0.0, 0.0, 0.0]
    for d in cfg.static_dims:
        m[d] = 1.0
    if cfg.supervision == "weak":
        for d in cfg.vel_dims:
            m[d] = 1.0
    return torch.tensor(m, device=device)


def _state_target(states_t, states_tp1, cfg):
    """ Στόχος state head για frame t: static = ΑΛΗΘΙΝΟ· velocity = ΕΚΤΙΜΗΣΗ (weak). """
    target = states_t.clone()
    if cfg.supervision == "weak":
        # ẋ_est = Δx/dt , θ̇_est = Δθ/dt  (από τις θέσεις/γωνίες του ζεύγους t,t+1)
        target[:, 1] = (states_tp1[:, 0] - states_t[:, 0]) / cfg.dt
        target[:, 3] = (states_tp1[:, 2] - states_t[:, 2]) / cfg.dt
    return target


def _component_losses(model, x, states_t, states_tp1, cfg):
    recon, state_pred, img_mu, img_logvar = model(x)
    recon_loss = F.mse_loss(recon, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + img_logvar - img_mu.pow(2) - img_logvar.exp())
    target = _state_target(states_t, states_tp1, cfg)
    mask = _state_mask(cfg, x.device)
    state_loss = (((state_pred - target) ** 2) * mask).sum()           # εποπτευόμενες dims μόνο
    # ΔΙΑΓΝΩΣΤΙΚΟ: σφάλμα ΠΡΟΒΛΕΠΟΜΕΝΗΣ ταχύτητας vs ΑΛΗΘΙΝΗ (states_t) — δείχνει
    # γιατί η weak βοηθάει· ΔΕΝ μπαίνει στο objective.
    vel_idx = list(cfg.vel_dims)
    vel_true = F.mse_loss(state_pred[:, vel_idx], states_t[:, vel_idx], reduction='sum')
    return {'recon': recon_loss, 'kl': kl_div, 'state': state_loss, 'vel_true': vel_true}


def _weighted_total(L, cfg):
    """ Το πραγματικό objective (το vel_true είναι μόνο διαγνωστικό). """
    return L['recon'] + cfg.beta * L['kl'] + cfg.state_weight * L['state']


def _report(sums, n, cfg):
    c = {k: v / max(n, 1) for k, v in sums.items()}
    return _weighted_total(c, cfg), c


def _fmt(c):
    return (f"recon {c['recon']:8.2f} | kl {c['kl']:7.3f} | state {c['state']:.4f} "
            f"| vel_true(diag) {c['vel_true']:.4f}")


def train_one_episode(model, frames, states_t, states_tp1, optimizer, device, cfg):
    n = frames.shape[0]
    sums = {k: 0.0 for k in _LOSS_KEYS}
    for s in range(0, n, cfg.batch):
        x = frames[s:s + cfg.batch].to(device)
        st = states_t[s:s + cfg.batch].to(device).float()
        stp = states_tp1[s:s + cfg.batch].to(device).float()
        optimizer.zero_grad()
        L = _component_losses(model, x, st, stp, cfg)
        loss = _weighted_total(L, cfg)
        loss.backward()
        optimizer.step()
        for k in _LOSS_KEYS:
            sums[k] += L[k].item()
    return sums, n


@torch.no_grad()
def evaluate(model, loader, device, cfg):
    model.eval()
    sums = {k: 0.0 for k in _LOSS_KEYS}
    n = 0
    for frames, _img2, _act, states_t, states_tp1 in loader:
        nf = frames.shape[0]
        for s in range(0, nf, cfg.batch):
            x = frames[s:s + cfg.batch].to(device)
            st = states_t[s:s + cfg.batch].to(device).float()
            stp = states_tp1[s:s + cfg.batch].to(device).float()
            L = _component_losses(model, x, st, stp, cfg)
            for k in _LOSS_KEYS:
                sums[k] += L[k].item()
        n += nf
    return sums, n


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
        train_sums = {k: 0.0 for k in _LOSS_KEYS}
        train_n = 0
        for frames, _img2, _act, states_t, states_tp1 in tqdm(
                TrainLoader, desc=f"Epoch {epoch + 1}", leave=False):
            sums, en = train_one_episode(model, frames, states_t, states_tp1,
                                         optimizer, device, cfg)
            for k in _LOSS_KEYS:
                train_sums[k] += sums[k]
            train_n += en
        train_total, tr = _report(train_sums, train_n, cfg)

        val_sums, val_n = evaluate(model, ValLoader, device, cfg)
        val_loss, va = _report(val_sums, val_n, cfg)

        print(f'Epoch {epoch + 1} [{cfg.supervision}]:')
        print(f'  train  total {train_total:10.2f}  | {_fmt(tr)}')
        print(f'  val    total {val_loss:10.2f}  | {_fmt(va)}')

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
    cfg.resolve()
    assert cfg.supervision in ("semi", "weak"), "CFG.supervision ∈ {'semi','weak'}"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | supervision: {cfg.supervision}")
    model = VAE_P1(latent_size=cfg.latent, state_dim=cfg.state_dim).to(device)

    if (not cfg.force_retrain) and os.path.isfile(cfg.weights):
        model.load_state_dict(torch.load(cfg.weights, map_location=device))
        model.eval()
        print(f"[P3-VAE] Φορτώθηκαν έτοιμα βάρη από {cfg.weights} -> παράλειψη εκπαίδευσης.")
        return model, device

    print(f"[P3-VAE] Δεν βρέθηκε checkpoint (ή force_retrain=True) -> εκπαίδευση.")
    best_val = train_vae(cfg, model, device)
    model.load_state_dict(torch.load(cfg.save, map_location=device))
    model.eval()
    print(f"[P3-VAE] Φορτώθηκε το καλύτερο μοντέλο (val={best_val:.2f}) από {cfg.save}.")
    return model, device


# Φορτώνει έτοιμα βάρη αν υπάρχουν, αλλιώς εκπαιδεύει & σώζει το καλύτερο μοντέλο.
vae_model, device = load_or_train_vae(CFG)
