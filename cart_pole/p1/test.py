# =============================================================================
#  Principle 1 — test.py : αναπαραγωγή Figure 3 (MSE φυσικής κατάστασης ανά
#  ορίζοντα πρόβλεψης) για το μοντέλο της Αρχής 1.
#
#  ΚΛΕΙΔΙ: στην Αρχή 1 το state part είναι ΕΠΟΠΤΕΥΟΜΕΝΟ -> η φυσική κατάσταση
#  διαβάζεται ΚΑΤΕΥΘΕΙΑΝ από το latent: state = z[..., :state_dim].
#  Άρα ΧΩΡΙΣ probe και ΧΩΡΙΣ decode σε εικόνα (σε αντίθεση με το baseline test.py).
#
#  Πρωτόκολλο (open-loop "dreaming", ΧΩΡΙΣ warm-up):
#    - z_0 από ΕΝΑ πραγματικό frame (P1 encoder, ντετερμινιστικό latent).
#    - Το LSTM ονειρεύεται 30 βήματα τρώγοντας τις ΔΙΚΕΣ του προβλέψεις,
#      με ΠΡΑΓΜΑΤΙΚΕΣ ενέργειες.
#    - Σε κάθε ορίζοντα h: predicted state = ẑ_h[:, :state_dim], MSE vs ground-truth.
#
#  Επειδή δεν χρειάζεται decode, χρησιμοποιούμε stride=1 (πυκνά anchors).
#  Προ-υπολογίζουμε ΟΛΟ το test set σε latents (z_t) μία φορά στην αρχή.
#
#  Απαιτεί έτοιμα vae_p1.pth & p1lstm.pth. Για Kaggle Notebook (χωρίς argparse).
# =============================================================================
import os
from os import listdir
from os.path import join, isdir

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


# ------------------------------- CONFIG --------------------------------------
class CFG:
    test         = "/kaggle/working/cartpole_data/test"

    vae_weights  = "/kaggle/working/vae_p1.pth"            # P1 VAE (separate encoding)
    lstm_weights = "/kaggle/working/p1lstm.pth"            # P1 LSTM
    out_dir      = "/kaggle/working/p1_test_results"

    latent       = 64
    state_dim    = 4       # πρώτα dims = φυσική κατάσταση [x, ẋ, θ, θ̇]
    hidden       = 64
    num_layers   = 2

    horizon      = 30      # ορίζοντας πρόβλεψης (βήματα)
    start_stride = 1       # πυκνά anchors (δεν χρειάζεται decode -> φθηνό)
    enc_batch    = 256     # GPU sub-batch για encode frames -> z


STATE_NAMES = ['x', 'x_dot', 'theta', 'theta_dot']


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
    """ Ίδιο με p1/seperate_encoding.py / p1/vae.py — φορτώνεται παγωμένο. """

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
    """ ΝΤΕΤΕΡΜΙΝΙΣΤΙΚΟ P1 latent: [state_head(feat) || fc_mu(feat)] -> (N, latent). """
    outs = []
    for s in range(0, frames.shape[0], enc_batch):
        x = frames[s:s + enc_batch].to(device)
        feat = vae.encoder(x).reshape(x.size(0), -1)
        outs.append(torch.cat([vae.state_head(feat), vae.fc_mu(feat)], dim=1))
    return torch.cat(outs, dim=0)


def load_models(cfg, device):
    for name, path in [("P1-VAE", cfg.vae_weights), ("P1-LSTM", cfg.lstm_weights)]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Δεν βρέθηκε checkpoint {name}: '{path}'.")
    vae = VAE_P1(latent_size=cfg.latent, state_dim=cfg.state_dim).to(device)
    vae.load_state_dict(torch.load(cfg.vae_weights, map_location=device))
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    lstm = LSTMTimeSeries(input_size=cfg.latent + 1, hidden_size=cfg.hidden,
                          num_layers=cfg.num_layers, output_size=cfg.latent).to(device)
    lstm.load_state_dict(torch.load(cfg.lstm_weights, map_location=device))
    lstm.eval()
    print(f"[OK] Φορτώθηκαν P1 VAE ({cfg.vae_weights}) & P1 LSTM ({cfg.lstm_weights}).")
    return vae, lstm


# ------------------------------- ROLLOUT -------------------------------------
@torch.no_grad()
def rollout_batch(lstm, z0, actions, device):
    """ Open-loop autoregressive rollout (χωρίς warm-up), batched.
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


# --------------------------- PRECOMPUTE TEST LATENTS -------------------------
@torch.no_grad()
def precompute_test(vae, cfg, device):
    """ Όλο το test set -> latents (z_t) έτοιμα στη RAM.
    Λίστα από {'z': (T,latent), 'states': (T,4), 'acts': (T,)}. """
    files = _list_npz(cfg.test)
    if not files:
        raise RuntimeError(f"Κανένα test .npz στο: {cfg.test}")
    episodes = []
    for f in tqdm(files, desc="Encoding test latents"):
        with np.load(f) as d:
            imgs = d['imgs']
            if imgs.shape[0] < 2:
                continue
            states = d['states'].astype(np.float32)
            acts = d['acts'].astype(np.float32)
            frames = torch.from_numpy(imgs.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
        z = encode_latent(vae, frames, device, cfg.enc_batch).cpu().numpy().astype(np.float32)
        episodes.append({'z': z, 'states': states, 'acts': acts})
    print(f"[OK] Προ-υπολογίστηκαν latents για {len(episodes)} test επεισόδια.")
    return episodes


# ------------------------------ EVALUATION -----------------------------------
@torch.no_grad()
def evaluate_test(lstm, episodes, cfg, device):
    """ Rollouts 30 βημάτων· MSE φυσικής κατάστασης (από dims :state_dim) ανά ορίζοντα. """
    H = cfg.horizon
    sd = cfg.state_dim
    se = np.zeros((H, sd), dtype=np.float64)   # άθροισμα squared error ανά ορίζοντα/μέγεθος
    count = 0
    floor_se = np.zeros(sd, dtype=np.float64)  # encoding floor: z[:, :sd] vs αληθινό state
    floor_n = 0
    all_states = []

    for ep in tqdm(episodes, desc=f"Test rollouts (H={H})"):
        z_np = ep['z']
        states = ep['states']
        acts = ep['acts']
        T = z_np.shape[0]
        all_states.append(states)

        z_t = torch.from_numpy(z_np).to(device)            # (T, latent)

        # encoding floor: άμεση εκτίμηση κατάστασης από το VAE (h=0, χωρίς δυναμική)
        s_enc = z_np[:, :sd]
        floor_se += ((s_enc - states[:, :sd]) ** 2).sum(axis=0)
        floor_n += T

        if T < H + 1:
            continue
        starts = list(range(0, T - H, cfg.start_stride))
        if not starts:
            continue
        B = len(starts)
        starts_t = torch.tensor(starts, device=device)
        z0 = z_t[starts_t]                                  # (B, latent)
        actions = torch.empty(B, H, device=device)
        gt = np.empty((B, H, sd), dtype=np.float32)
        for i, s in enumerate(starts):
            actions[i] = torch.from_numpy(acts[s:s + H]).to(device)
            gt[i] = states[s + 1:s + H + 1, :sd]

        z_pred = rollout_batch(lstm, z0, actions, device)   # (B, H, latent)
        s_pred = z_pred[:, :, :sd].cpu().numpy()            # (B, H, sd) — φυσική κατάσταση!
        se += ((s_pred - gt) ** 2).sum(axis=0)              # (H, sd)
        count += B

    if count == 0:
        raise RuntimeError(f"Κανένα έγκυρο rollout (επεισόδια με >= {H+1} frames).")

    mse_per_comp = se / count
    floor = floor_se / max(floor_n, 1)
    state_var = np.concatenate(all_states, axis=0)[:, :sd].var(axis=0) + 1e-8
    metrics = {
        'horizon': np.arange(1, H + 1),
        'mse_per_comp': mse_per_comp,
        'mse_total': mse_per_comp.mean(axis=1),
        'floor': floor,
        'n_rollouts': count,
        'n_episodes': len(episodes),
    }
    metrics['nmse_per_comp'] = mse_per_comp / state_var
    metrics['nmse_total'] = (mse_per_comp / state_var).mean(axis=1)
    metrics['nfloor'] = floor / state_var
    return metrics


# --------------------------- TABLES & PLOTS ----------------------------------
def show_and_save(metrics, cfg):
    os.makedirs(cfg.out_dir, exist_ok=True)
    h = metrics['horizon']
    sd = cfg.state_dim
    names = STATE_NAMES[:sd]

    df = pd.DataFrame(metrics['mse_per_comp'], columns=[f'MSE_{n}' for n in names])
    df.insert(0, 'horizon', h)
    df['MSE_total'] = metrics['mse_total']
    df['NMSE_total'] = metrics['nmse_total']

    pick = [hh for hh in [1, 5, 10, 15, 20, 25, 30] if hh <= cfg.horizon]
    print("\n==========  P1 — MSE φυσικής κατάστασης ανά ορίζοντα (από dims 0:4)  ==========")
    print(f"(rollouts={metrics['n_rollouts']}, episodes={metrics['n_episodes']}, "
          f"open-loop, χωρίς probe)\n")
    print(df[df['horizon'].isin(pick)].to_string(index=False,
          float_format=lambda v: f"{v:.4f}"))

    csv_path = join(cfg.out_dir, "p1_fig3_mse_per_horizon.csv")
    df.to_csv(csv_path, index=False)
    np.savez(join(cfg.out_dir, "p1_fig3_metrics.npz"),
             horizon=h, mse_per_comp=metrics['mse_per_comp'], mse_total=metrics['mse_total'],
             nmse_per_comp=metrics['nmse_per_comp'], nmse_total=metrics['nmse_total'],
             floor=metrics['floor'])
    print(f"\n[SAVE] CSV -> {csv_path}")
    print(f"[SAVE] arrays -> {join(cfg.out_dir, 'p1_fig3_metrics.npz')}")

    # --- PLOT 1: Fig-3 repro (aggregate) ---
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(h, metrics['mse_total'], marker='o', color='#0F244F', label='P1 (dreaming)')
    ax[0].axhline(metrics['floor'].mean(), ls='--', color='gray', label='encoding floor')
    ax[0].set_title("Principle 1 — MSE φυσικής κατάστασης (raw, μέσος 4)")
    ax[0].set_xlabel("ορίζοντας πρόβλεψης (βήματα)")
    ax[0].set_ylabel("MSE")
    ax[0].legend(); ax[0].grid(alpha=0.3)

    ax[1].plot(h, metrics['nmse_total'], marker='o', color='#C9A227', label='P1')
    ax[1].axhline(metrics['nfloor'].mean(), ls='--', color='gray', label='encoding floor')
    ax[1].set_title("Principle 1 — Normalized MSE")
    ax[1].set_xlabel("ορίζοντας πρόβλεψης (βήματα)")
    ax[1].set_ylabel("NMSE")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    p1 = join(cfg.out_dir, "p1_fig3_aggregate.png")
    fig.savefig(p1, dpi=140)
    print(f"[SAVE] plot -> {p1}")

    # --- PLOT 2: per-component ---
    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    for c, name in enumerate(names):
        ax2.plot(h, metrics['mse_per_comp'][:, c], marker='.', label=name)
    ax2.set_title("Principle 1 — MSE ανά φυσικό μέγεθος vs ορίζοντα")
    ax2.set_xlabel("ορίζοντας πρόβλεψης (βήματα)")
    ax2.set_ylabel("MSE")
    ax2.legend(); ax2.grid(alpha=0.3)
    fig2.tight_layout()
    p2 = join(cfg.out_dir, "p1_fig3_per_component.png")
    fig2.savefig(p2, dpi=140)
    print(f"[SAVE] plot -> {p2}")

    plt.show()
    return df


def run_test(cfg=CFG):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    vae, lstm = load_models(cfg, device)
    episodes = precompute_test(vae, cfg, device)        # όλο το test -> z_t έτοιμα
    metrics = evaluate_test(lstm, episodes, cfg, device)
    df = show_and_save(metrics, cfg)
    return metrics, df


# Τρέχει όλη την αξιολόγηση όταν εκτελείται το cell.
metrics, results_df = run_test(CFG)
