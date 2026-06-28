"""
hnn.py — Forced Hamiltonian Neural Network as a latent dynamics model, compared to the LSTM.

Idea: if the supervised latent dims are physical (q = positions, p = velocity-like momenta),
the dynamics should derive from a single learned energy H_theta(q, p):

    q_dot =  dH/dp
    p_dot = -dH/dq + g_theta(q) * u        (u = control / action)

integrated one step (symplectic / semi-implicit Euler). Energy structure -> bounded, stable
rollouts (like SINDy) but learned and more expressive than a fixed sparse library.

This script reuses sindy.py's data/encode/eval helpers, TRAINS an HNN per model (Baseline, P1),
LOADS the trained LSTM checkpoints, and compares them visually:
  * per-horizon physical MSE (HNN vs LSTM),
  * a trajectory overlay (GT vs LSTM vs HNN) for one test window,
  * the learned energy H along the HNN vs LSTM rollouts (physical-consistency check).
"""
import os
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from loader import load_norm_stats
from lstm import LatentPredictor
from sindy import (
    DATA_ROOT, NORM_STATS, N_SUP, N_ACTIONS, LATENT_SIZE, HIDDEN, LAYERS,
    SEQ_LEN, TEST_STRIDE, BATCH, TRAIN_STRIDE, TRAIN_BATCH, NUM_WORKERS,
    EPOCHS, LR, CLIP, W_PHYS, P_START, P_END, P_DECAY_EPOCHS, L_START,
    CURRICULUM_EPOCHS, EARLY_STOP_PATIENCE, SCHED_PATIENCE, SEED,
    CONTROL_MODE, LOG_Y, SHOW_PROGRESS, MODELS,
    NOISE_TYPE, NOISE_LEVELS, noise_tag,
    action_to_control, control_dim, encode_latents, make_loader, make_test_loader,
    _train_rollout,
)

#
#  Config
#
# Baseline only (drop Principle 1). MODEL provides BOTH the VAE (make_vae + vae_ckpt, used to
# encode latents) and the pretrained LSTM (lstm_ckpt) — same selection mechanism for both.
MODEL = next(m for m in MODELS if m["label"] == "Baseline")

H_HIDDEN = 64            # width of the Hamiltonian MLP H_theta
G_HIDDEN = 32            # width of the input-map MLP g_theta
DT = 1.0                 # integration step (latent frame interval; the net absorbs scaling)
SAVE_DIR = "/kaggle/working/cartpole_hnn"
STATE_NAMES = ["x", "x_dot", "theta", "theta_dot"]
EVAL_LEVEL = 0.0         # noise level used for the trajectory/energy visuals (0.0 = clean)
TRAIN_HNN = True         # True -> train a fresh HNN (+save to HNN_CKPT); False -> load HNN_CKPT
HNN_CKPT = os.path.join(SAVE_DIR, "hnn_baseline.pth")


def load_lstm(cfg, device):
    """Load the pretrained LSTM from cfg['lstm_ckpt'] (same source as the VAE's vae_ckpt)."""
    model = LatentPredictor(LATENT_SIZE, N_ACTIONS, HIDDEN, LAYERS).to(device)
    model.load_state_dict(torch.load(cfg["lstm_ckpt"], map_location=device))
    model.eval()
    print(f"  LSTM loaded from {cfg['lstm_ckpt']}")
    return model


def build_hnn(device):
    return HNNPredictor(N_SUP, N_ACTIONS, H_HIDDEN, G_HIDDEN, CONTROL_MODE, DT).to(device)


def load_hnn(device):
    """Load a previously trained HNN from HNN_CKPT (skips retraining)."""
    model = build_hnn(device)
    model.load_state_dict(torch.load(HNN_CKPT, map_location=device))
    model.eval()
    print(f"  HNN loaded from {HNN_CKPT}")
    return model


#
#  Forced Hamiltonian predictor (same init_hidden / step interface as LatentPredictor)
#
class HNNPredictor(nn.Module):
    """Markov predictor. Physical dims are laid out interleaved as [x, x_dot, theta, theta_dot],
    i.e. even indices = q (positions), odd indices = p (velocity-like momenta). Style dims are
    carried through untouched (the metric never reads them)."""
    def __init__(self, n_sup=4, n_actions=2, h_hidden=64, g_hidden=32,
                 control_mode="signed", dt=1.0):
        super().__init__()
        assert n_sup % 2 == 0, "n_sup must be even (q,p pairs)"
        self.n_sup = n_sup
        self.n = n_sup // 2                 # degrees of freedom (2 for CartPole)
        self.n_actions = n_actions
        self.control_mode = control_mode
        self.dt = dt
        self.n_ctrl = control_dim(n_actions, control_mode)

        # Scalar energy H(q, p) over the concatenation [q, p]
        self.H = nn.Sequential(
            nn.Linear(n_sup, h_hidden), nn.Softplus(),
            nn.Linear(h_hidden, h_hidden), nn.Softplus(),
            nn.Linear(h_hidden, 1),
        )
        # Input map g(q): how the control force couples to each momentum dim
        self.g = nn.Sequential(
            nn.Linear(self.n, g_hidden), nn.Tanh(),
            nn.Linear(g_hidden, self.n * self.n_ctrl),
        )

    def init_hidden(self, b, device):
        return None

    def _split(self, z):
        zp = z[:, :self.n_sup]
        return zp[:, 0::2], zp[:, 1::2]    # q (even dims), p (odd dims)

    def _merge(self, z, q, p):
        # interleave back to [q0, p0, q1, p1, ...] without in-place ops
        phys = torch.stack([q, p], dim=2).reshape(q.shape[0], self.n_sup)
        return torch.cat([phys, z[:, self.n_sup:]], dim=1)

    def _dH(self, q, p):
        """Gradients of H w.r.t (q, p). enable_grad so it works even under outer no_grad;
        create_graph during training so the loss can backprop into H's parameters."""
        with torch.enable_grad():
            qp = torch.cat([q, p], dim=1).detach().requires_grad_(True)
            H = self.H(qp).sum()
            grads = torch.autograd.grad(H, qp, create_graph=self.training)[0]
        return grads[:, :self.n], grads[:, self.n:]   # dH/dq, dH/dp

    def _force(self, q, u):
        g = self.g(q).view(q.shape[0], self.n, self.n_ctrl)
        return (g * u.unsqueeze(1)).sum(dim=-1)        # (B, n)

    def step(self, z, a_onehot, hidden):
        q, p = self._split(z)
        u = action_to_control(a_onehot.argmax(dim=-1), self.n_actions, self.control_mode).to(z.dtype)
        # symplectic (semi-implicit) Euler: update p first, then q with the new p
        dHdq, _ = self._dH(q, p)
        p_new = p + self.dt * (-dHdq + self._force(q, u))
        _, dHdp = self._dH(q, p_new)
        q_new = q + self.dt * dHdp
        return self._merge(z, q_new, p_new), hidden

    def energy(self, z):
        q, p = self._split(z)
        return self.H(torch.cat([q, p], dim=1)).squeeze(-1)   # (B,) learned energy


#
#  Rollout + evaluation (grad stays enabled internally for HNN; states detached between steps)
#
def free_run(model, batch):
    """Free-running rollout (seed z_0, feed own prediction). Works for any model with the
    init_hidden/step interface. Not wrapped in no_grad because HNN.step needs autograd
    internally; each step is detached so no graph accumulates."""
    z_t, action, z_tp1, state_t, state_tp1 = batch
    B, L, _ = z_t.shape
    z_in = z_t[:, 0]
    hidden = model.init_hidden(B, z_t.device)
    preds = []
    for k in range(L):
        a = F.one_hot(action[:, k].long(), N_ACTIONS).float()
        z_pred, hidden = model.step(z_in, a, hidden)
        preds.append(z_pred.detach())
        z_in = z_pred.detach()
    return torch.stack(preds, dim=1), state_tp1


def eval_per_horizon(model, loader, device, std4):
    """Per-horizon physical MSE (de-standardized), averaged over the test set."""
    model.eval()
    se, n = None, 0
    for batch in tqdm(loader, desc="eval", leave=False, disable=not SHOW_PROGRESS):
        batch = [b.to(device, non_blocking=True) for b in batch]
        preds, state_tp1 = free_run(model, batch)
        err = (preds[..., :N_SUP] - state_tp1) * std4
        s = (err ** 2).sum(dim=0)
        se = s if se is None else se + s
        n += preds.size(0)
    return (se / n).mean(dim=1).cpu().numpy()


#
#  Train the HNN (reuses sindy's scheduled-sampling + curriculum rollout)
#
def train_hnn(model, cfg, mean, std, std4, device):
    train_dl = make_loader(os.path.join(cfg["latent_root"], "train"), mean, std,
                           TRAIN_STRIDE, TRAIN_BATCH, shuffle=True)
    val_dl = make_loader(os.path.join(cfg["latent_root"], "val"), mean, std,
                         TRAIN_STRIDE, BATCH, shuffle=False)
    opt = optim.Adam(model.parameters(), lr=LR)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=SCHED_PATIENCE)

    best, bad, best_state = float("inf"), 0, None
    print(f"  training {cfg['label']} HNN (max {EPOCHS} epochs) ...")
    for epoch in range(1, EPOCHS + 1):
        p_tf = max(P_END, P_START - (P_START - P_END) * (epoch - 1) / max(P_DECAY_EPOCHS, 1))
        cur_len = int(round(min(SEQ_LEN, L_START + (SEQ_LEN - L_START)
                                * (epoch - 1) / max(CURRICULUM_EPOCHS, 1))))
        model.train()
        for batch in train_dl:
            batch = [b.to(device, non_blocking=True) for b in batch]
            preds, z_gt = _train_rollout(model, batch, p_tf, cur_len, device)
            loss = (F.mse_loss(preds, z_gt)
                    + W_PHYS * F.mse_loss(preds[..., :N_SUP], z_gt[..., :N_SUP]))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()

        mse_h = eval_per_horizon(model, val_dl, device, std4)
        val_mean = float(mse_h.mean())
        sched.step(val_mean)
        improved = val_mean < best - 1e-6
        if improved:
            best, bad, best_state = val_mean, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
        print(f"  E{epoch:02d}/{EPOCHS} | p_tf={p_tf:.2f} H={cur_len:2d} | "
              f"val phys-MSE={val_mean:.4f}{'  *best' if improved else ''}")
        if bad >= EARLY_STOP_PATIENCE:
            print(f"  early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    os.makedirs(os.path.dirname(HNN_CKPT), exist_ok=True)
    torch.save(model.state_dict(), HNN_CKPT)
    print(f"  [{cfg['label']}] HNN trained: best val phys-MSE={best:.4f} -> {HNN_CKPT}")
    return model


#
#  Visual comparisons
#
def plot_per_horizon(results_level, save_dir, level):
    """Per-horizon physical MSE at one noise level: LSTM vs HNN (Baseline)."""
    os.makedirs(save_dir, exist_ok=True)
    horizons = np.arange(1, SEQ_LEN + 1)
    styles = {"LSTM": ("C0", "-"), "HNN": ("C4", "-.")}
    plt.figure(figsize=(6.6, 4.6))
    for name in ("LSTM", "HNN"):
        mse_h = results_level[name]
        c, ls = styles[name]
        plt.plot(horizons, mse_h, color=c, ls=ls, lw=2, label=f"{name} (mean={mse_h.mean():.4f})")
    if LOG_Y:
        plt.yscale("log")
    plt.title(f"Baseline: LSTM vs HNN per horizon ({noise_tag(level)})")
    plt.xlabel("Prediction horizon"); plt.ylabel("State MSE (physical units)")
    plt.xlim(1, SEQ_LEN); plt.grid(alpha=0.3, which="both"); plt.legend()
    plt.tight_layout()
    path = os.path.join(save_dir, f"hnn_vs_lstm_mse_{noise_tag(level)}.png")
    plt.savefig(path, dpi=150); plt.show(); print("saved figure ->", path)


def plot_degradation(results, save_dir):
    """Mean state-MSE (over horizon) vs noise level: LSTM vs HNN (Baseline)."""
    os.makedirs(save_dir, exist_ok=True)
    styles = {"LSTM": ("C0", "-"), "HNN": ("C4", "-.")}
    plt.figure(figsize=(6.6, 4.6))
    for name in ("LSTM", "HNN"):
        ys = [results[nl][name].mean() for nl in NOISE_LEVELS]
        c, ls = styles[name]
        plt.plot(NOISE_LEVELS, ys, color=c, ls=ls, lw=2, marker="o", label=name)
    if LOG_Y:
        plt.yscale("log")
    plt.title(f"Baseline: robustness to {NOISE_TYPE} noise (LSTM vs HNN)")
    plt.xlabel(f"noise level ({NOISE_TYPE})"); plt.ylabel("mean state MSE (over horizon)")
    plt.grid(alpha=0.3, which="both"); plt.legend()
    plt.tight_layout()
    path = os.path.join(save_dir, "hnn_vs_lstm_degradation.png")
    plt.savefig(path, dpi=150); plt.show(); print("saved figure ->", path)


def plot_trajectory(models, batch, save_dir, label):
    """Overlay GT vs each model's free-running rollout for one test window, per physical dim."""
    os.makedirs(save_dir, exist_ok=True)
    horizons = np.arange(1, SEQ_LEN + 1)
    gt = batch[4][0, :, :N_SUP].cpu().numpy()    # state_tp1 (standardized) for window 0
    colors = {"LSTM": "C0", "HNN": "C4"}

    fig, axes = plt.subplots(1, N_SUP, figsize=(4.0 * N_SUP, 3.6), squeeze=False)
    for name, m in models.items():
        preds, _ = free_run(m, batch)
        pr = preds[0, :, :N_SUP].cpu().numpy()
        for d in range(N_SUP):
            axes[0][d].plot(horizons, pr[:, d], color=colors[name], lw=2, label=name)
    for d in range(N_SUP):
        ax = axes[0][d]
        ax.plot(horizons, gt[:, d], "k--", lw=2, label="GT")
        ax.set_title(STATE_NAMES[d])
        ax.set_xlabel("horizon")
        ax.set_xlim(1, SEQ_LEN)
        ax.grid(alpha=0.3)
        if d == 0:
            ax.set_ylabel("standardized state")
            ax.legend(fontsize=8)
    plt.suptitle(f"{label}: trajectory rollout (GT vs LSTM vs HNN)", y=1.02)
    plt.tight_layout()
    slug = label.lower().replace(" ", "_")
    path = os.path.join(save_dir, f"hnn_trajectory_{slug}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print("saved figure ->", path)


def plot_energy(hnn, lstm, batch, save_dir, label):
    """The HNN's learned energy H evaluated along the HNN rollout vs the LSTM rollout.
    A physically consistent model keeps H bounded; the black-box LSTM drifts/explodes."""
    os.makedirs(save_dir, exist_ok=True)
    horizons = np.arange(1, SEQ_LEN + 1)

    def energy_curve(model):
        z_t, action = batch[0], batch[1]
        B, L, _ = z_t.shape
        z_in = z_t[:, 0]
        hidden = model.init_hidden(B, z_t.device)
        es = []
        for k in range(L):
            es.append(hnn.energy(z_in).mean().item())   # always the HNN's H
            a = F.one_hot(action[:, k].long(), N_ACTIONS).float()
            z_in, hidden = model.step(z_in, a, hidden)
            z_in = z_in.detach()
        return np.array(es)

    plt.figure(figsize=(6.4, 4.4))
    plt.plot(horizons, energy_curve(hnn), color="C4", lw=2, label="along HNN rollout")
    plt.plot(horizons, energy_curve(lstm), color="C0", lw=2, label="along LSTM rollout")
    plt.title(f"{label}: learned energy $H_\\theta$ along rollout")
    plt.xlabel("Prediction horizon")
    plt.ylabel(r"$H_\theta(z)$ (mean over batch)")
    plt.xlim(1, SEQ_LEN)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    slug = label.lower().replace(" ", "_")
    path = os.path.join(save_dir, f"hnn_energy_{slug}.png")
    plt.savefig(path, dpi=150)
    plt.show()
    print("saved figure ->", path)


#
#  Main
#
if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    mean, std = load_norm_stats(NORM_STATS)
    std4 = torch.tensor(std[:N_SUP], device=device)

    cfg = MODEL
    encode_latents(cfg, device)

    # HNN: train fresh (+save) or load a previous run; LSTM: load pretrained from cfg.
    if TRAIN_HNN:
        hnn = build_hnn(device)
        train_hnn(hnn, cfg, mean, std, std4, device)
    else:
        hnn = load_hnn(device)
    lstm = load_lstm(cfg, device)

    # Noise sweep: evaluate both at every level (test re-encoded under noise upstream)
    results = {}   # results[noise_level] = {"LSTM": mse_h, "HNN": mse_h}
    for nl in NOISE_LEVELS:
        test_dl = make_test_loader(cfg, mean, std, nl)
        results[nl] = {
            "LSTM": eval_per_horizon(lstm, test_dl, device, std4),
            "HNN":  eval_per_horizon(hnn, test_dl, device, std4),
        }

    # Visual comparisons on the clean test split
    clean_batch = [b.to(device) for b in next(iter(make_test_loader(cfg, mean, std, EVAL_LEVEL)))]
    plot_trajectory({"LSTM": lstm, "HNN": hnn}, clean_batch, SAVE_DIR, "Baseline")
    plot_energy(hnn, lstm, clean_batch, SAVE_DIR, "Baseline")

    # Robustness table: mean state-MSE per noise level
    HS = [h for h in (1, 10, 20, SEQ_LEN) if h <= SEQ_LEN]
    print(f"\n{'='*54}")
    print(f"Mean state-MSE vs {NOISE_TYPE} noise — LSTM vs HNN (Baseline)")
    print(f"{'='*54}")
    print(f"{'noise':<8}{'LSTM':>12}{'HNN':>12}")
    print("-" * 32)
    for nl in NOISE_LEVELS:
        print(f"{nl:<8.2f}{results[nl]['LSTM'].mean():>12.4f}{results[nl]['HNN'].mean():>12.4f}")

    plot_per_horizon(results[EVAL_LEVEL], SAVE_DIR, EVAL_LEVEL)
    plot_degradation(results, SAVE_DIR)
