"""
extension_control_sindy.py

Extension 4 variant: keep the VAE encoder as the pixel -> physical-state sensor,
but replace the LSTM dream model with an action-conditioned SINDy-style sparse
polynomial dynamics model over the 8 interpretable physical dimensions.

The SINDy model is fitted at script startup from DATA_ROOT/train:
    s_{t+1} - s_t = f_a(s_t), one sparse polynomial model per discrete action.

Closed-loop controllers:
    enc_pid       : PID/PD heuristic on encoder-estimated physical state.
    guided_mpc_*  : PID-guided CEM MPC using SINDy rollouts instead of LSTM rollouts.

This is intentionally a separate file from extension4_control_alt.py so the LSTM
experiment stays untouched.
"""
import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataCollect import resize_frame
from loader import list_npz, load_norm_stats
from vae import VAE
from vae_p1 import VAE_P1
from vae_p2 import VAE_P2
from vae_p3 import VAE_P3


# ---------------------------------------------------------------------------
# CONFIG - placeholders <...> are filled by the Kaggle bootstrap patcher
# ---------------------------------------------------------------------------
DATA_ROOT = "<lunarlander-dataset>"
NORM_STATS = os.path.join(DATA_ROOT, "norm_stats.npz")
SAVE_DIR = "/kaggle/working/lunarlander_ext4_control_sindy"

LATENT_SIZE, N_SUP, N_IMG = 64, 8, 56
N_ACTIONS = 4
IMG_H, IMG_W = 80, 120

# P1 was better as an encoder-as-sensor; SINDy only replaces the dynamics model.
MODEL = "baseline"
MODEL_REGISTRY = {
    "baseline": (lambda: VAE(latent_size=LATENT_SIZE), "<lunarlander-baseline-vae>"),
    "p1": (lambda: VAE_P1(n_sup=N_SUP, n_img=N_IMG), "<lunarlander-p1-vae>"),
    "p2": (lambda: VAE_P2(latent_size=LATENT_SIZE), "<lunarlander-p2-vae>"),
    "p3_semi": (lambda: VAE_P3(latent_size=LATENT_SIZE), "<lunarlander-p3-semi-vae>"),
    "p3_weak": (lambda: VAE_P3(latent_size=LATENT_SIZE), "<lunarlander-p3-weak-vae>"),
}

N_EPISODES = 20
MAX_STEPS = 400
SEED = 0
ENABLE_WIND = False
WIND_POWER, TURBULENCE_POWER = 15.0, 1.5
CONTROLLERS = ["enc_pid", "guided_mpc"]

RECORD_GIF = False
GIF_FPS = 30

# --- SINDy fitting ---
SINDY_DEGREE = 2
SINDY_ALPHA = 1e-4
SINDY_THRESHOLD = 1e-3
SINDY_ITERS = 8
SINDY_MAX_TRANSITIONS_PER_ACTION = 120_000
SINDY_EVAL_MAX_TRANSITIONS = 80_000
SINDY_SEED = 0

# --- State estimator trust gate ---
USE_FILTER = False
W_MODEL = np.array([0.15, 0.15, 0.50, 0.50, 0.20, 0.45, 0.00, 0.00], dtype=np.float64)
RESID_SCALE0 = 1.0
FILTER_WARMUP = 5
TRUST_FACTOR = 1.2
TRUST_FLOOR = 0.05
TRUST_ABS_MAX = 0.75
MPC_MIN_HISTORY = 8

# --- Guided MPC (CEM shield/corrector) ---
MPC_HORIZON = 5
MPC_REPEAT = 1
MPC_SAMPLES = 256
CEM_ITERS = 3
CEM_ELITE = 32
CEM_LR = 0.7
PID_BIAS = 0.75
MPC_SEED = 0

# --- Objective weights ---
FUEL_W = 0.30
TERM_W = 1.0
LAND_LEG = 20.0
LAND_CRASH = 100.0
SAFE_SPEED = 0.5
MPC_MARGIN = 5.0
RISK_TRIGGER = 2.0
RISK_MARGIN = 1.0

GUIDED_GRID = [
    {"label": "med_full", "mode": "shield", "mpc_margin": 5.0, "risk_trigger": 2.0, "risk_margin": 1.0, "pid_bias": 0.75},
    {"label": "value_only", "mode": "value_only", "mpc_margin": 5.0, "risk_trigger": 2.0, "risk_margin": 1.0, "pid_bias": 0.75},
    {"label": "risk_only", "mode": "risk_only", "mpc_margin": 5.0, "risk_trigger": 2.0, "risk_margin": 1.0, "pid_bias": 0.75},
    {"label": "force_diff", "mode": "force_diff", "mpc_margin": 5.0, "risk_trigger": 2.0, "risk_margin": 1.0, "pid_bias": 0.75},
]

FUEL_COST = np.array([0.0, 0.03, 0.30, 0.03], dtype=np.float64)
DIM_NAMES = ["x", "y", "vx", "vy", "theta", "omega", "leg1", "leg2"]


def cfg_get(cfg, key, default):
    return default if cfg is None else cfg.get(key, default)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_env():
    last_err = None
    for env_id in ("LunarLander-v3", "LunarLander-v2"):
        try:
            kw = dict(render_mode="rgb_array")
            if ENABLE_WIND:
                kw.update(enable_wind=True, wind_power=WIND_POWER, turbulence_power=TURBULENCE_POWER)
            return gym.make(env_id, **kw)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"LunarLander not found. Try: pip install 'gymnasium[box2d]'. {last_err}")


def heuristic_control(s):
    """s = physical [x,y,vx,vy,theta,omega,leg1,leg2] -> action in {0,1,2,3}."""
    x, y, vx, vy, theta, omega = float(s[0]), float(s[1]), float(s[2]), float(s[3]), float(s[4]), float(s[5])
    leg1, leg2 = float(s[6]) > 0.5, float(s[7]) > 0.5
    angle_targ = float(np.clip(x * 0.5 + vx * 1.0, -0.4, 0.4))
    hover_targ = 0.55 * abs(x)
    angle_todo = (angle_targ - theta) * 0.5 - omega * 1.0
    hover_todo = (hover_targ - y) * 0.5 - vy * 0.5
    if leg1 or leg2:
        angle_todo, hover_todo = 0.0, -vy * 0.5
    if hover_todo > abs(angle_todo) and hover_todo > 0.05:
        return 2
    if angle_todo < -0.05:
        return 3
    if angle_todo > 0.05:
        return 1
    return 0


def _to_tensor(frame, device):
    t = torch.from_numpy(frame.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return t.to(device)


@torch.no_grad()
def encode_pair(vae, f_prev, f_cur, device):
    x = torch.cat([_to_tensor(f_prev, device), _to_tensor(f_cur, device)], dim=1)
    mu, _ = vae.encode(x)
    return mu


def to_phys_np(z8_std, mean_np, std_np):
    return np.asarray(z8_std, dtype=np.float64) * std_np[:N_SUP] + mean_np[:N_SUP]


def save_gif(frames, path, fps=GIF_FPS):
    from PIL import Image
    if not frames:
        print("[gif] no frames to save:", path)
        return
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / max(fps, 1)), loop=0)
    print("saved:", path)


# ---------------------------------------------------------------------------
# SINDy model
# ---------------------------------------------------------------------------
def sindy_feature_names():
    names = ["1"]
    names += DIM_NAMES
    if SINDY_DEGREE >= 2:
        for i, ni in enumerate(DIM_NAMES):
            for j, nj in enumerate(DIM_NAMES[i:], start=i):
                names.append(f"{ni}*{nj}")
    return names


def sindy_library(x):
    """Polynomial library on standardized states. x: (N,8) -> Theta: (N,F)."""
    x = np.asarray(x, dtype=np.float64)
    cols = [np.ones((x.shape[0], 1), dtype=np.float64), x]
    if SINDY_DEGREE >= 2:
        quad = []
        for i in range(N_SUP):
            for j in range(i, N_SUP):
                quad.append((x[:, i] * x[:, j])[:, None])
        cols.append(np.concatenate(quad, axis=1))
    return np.concatenate(cols, axis=1)


def ridge_solve(theta, y, alpha):
    n_feat = theta.shape[1]
    a = theta.T @ theta
    a.flat[:: n_feat + 1] += alpha
    b = theta.T @ y
    return np.linalg.solve(a, b)


def stlsq(theta, y, alpha=SINDY_ALPHA, threshold=SINDY_THRESHOLD, iters=SINDY_ITERS):
    """Sequential thresholded least squares, fitted independently per output dim."""
    coef = ridge_solve(theta, y, alpha)
    for _ in range(iters):
        small = np.abs(coef) < threshold
        coef[small] = 0.0
        for d in range(y.shape[1]):
            active = ~small[:, d]
            if active.sum() == 0:
                continue
            coef[active, d] = ridge_solve(theta[:, active], y[:, d:d + 1], alpha).ravel()
            coef[~active, d] = 0.0
    return coef


class SindyDynamics:
    def __init__(self, coef, mean, std, state_lo, state_hi, feature_names):
        self.coef = np.asarray(coef, dtype=np.float64)        # (A,F,8), predicts delta in standardized units
        self.mean = np.asarray(mean[:N_SUP], dtype=np.float64)
        self.std = np.asarray(std[:N_SUP], dtype=np.float64)
        self.state_lo = np.asarray(state_lo, dtype=np.float64)
        self.state_hi = np.asarray(state_hi, dtype=np.float64)
        self.feature_names = np.asarray(feature_names)

    def clip_phys(self, s):
        s = np.clip(s, self.state_lo, self.state_hi)
        s[:, 6:8] = np.clip(s[:, 6:8], 0.0, 1.0)
        return s

    def step_phys(self, states, actions):
        states = np.asarray(states, dtype=np.float64)
        actions = np.asarray(actions, dtype=np.int64).reshape(-1)
        if states.ndim == 1:
            states = states[None, :]
        out = np.empty_like(states)
        for a in range(N_ACTIONS):
            mask = actions == a
            if not np.any(mask):
                continue
            x_std = (states[mask] - self.mean) / self.std
            delta_std = sindy_library(x_std) @ self.coef[a]
            next_phys = (x_std + delta_std) * self.std + self.mean
            out[mask] = next_phys
        return self.clip_phys(out)

    def nonzero_counts(self):
        return np.count_nonzero(np.abs(self.coef) > 0.0, axis=(1, 2))


def _sample_rows(x, y, max_rows, rng):
    if x.shape[0] <= max_rows:
        return x, y
    idx = rng.choice(x.shape[0], size=max_rows, replace=False)
    return x[idx], y[idx]


def collect_transitions(root, max_per_action=None, seed=SINDY_SEED):
    rng = np.random.default_rng(seed)
    xs, xps = [[] for _ in range(N_ACTIONS)], [[] for _ in range(N_ACTIONS)]
    all_states = []
    files = list_npz(root)
    if not files:
        raise RuntimeError(f"No .npz files found in {root}")
    for f in files:
        with np.load(f) as d:
            states = d["states"].astype(np.float64)
            acts = d["acts"].astype(np.int64)
        s = states[:-1]
        sp = states[1:]
        a = acts[:-1]
        all_states.append(s)
        for act in range(N_ACTIONS):
            mask = a == act
            if np.any(mask):
                xs[act].append(s[mask])
                xps[act].append(sp[mask])

    out_x, out_xp = [], []
    for act in range(N_ACTIONS):
        if not xs[act]:
            raise RuntimeError(f"No transitions for action {act} in {root}")
        x = np.concatenate(xs[act], axis=0)
        xp = np.concatenate(xps[act], axis=0)
        if max_per_action is not None:
            x, xp = _sample_rows(x, xp, max_per_action, rng)
        out_x.append(x)
        out_xp.append(xp)
    return out_x, out_xp, np.concatenate(all_states, axis=0)


def fit_sindy_dynamics(train_root, mean_np, std_np):
    print(f"\n[SINDy] fitting action-conditioned sparse dynamics from: {train_root}")
    xs, xps, all_states = collect_transitions(
        train_root,
        max_per_action=SINDY_MAX_TRANSITIONS_PER_ACTION,
        seed=SINDY_SEED,
    )
    feature_names = sindy_feature_names()
    coefs, rmses = [], []
    mean8 = mean_np[:N_SUP].astype(np.float64)
    std8 = std_np[:N_SUP].astype(np.float64)
    for act in range(N_ACTIONS):
        x_std = (xs[act] - mean8) / std8
        xp_std = (xps[act] - mean8) / std8
        theta = sindy_library(x_std)
        y = xp_std - x_std
        coef = stlsq(theta, y)
        pred_delta = theta @ coef
        rmse_std = np.sqrt(np.mean((pred_delta - y) ** 2, axis=0))
        coefs.append(coef)
        rmses.append(rmse_std)
        print(
            f"[SINDy] action {act}: n={len(xs[act]):6d}  "
            f"nonzero={np.count_nonzero(coef):4d}/{coef.size}  "
            f"rmse_std_mean={rmse_std.mean():.4f}"
        )

    lo = np.percentile(all_states, 0.1, axis=0)
    hi = np.percentile(all_states, 99.9, axis=0)
    pad = 0.35 * np.maximum(hi - lo, 1e-3)
    state_lo = lo - pad
    state_hi = hi + pad
    state_lo[6:8] = 0.0
    state_hi[6:8] = 1.0
    state_lo[1] = min(state_lo[1], -0.25)
    state_hi[1] = max(state_hi[1], 2.0)

    sindy = SindyDynamics(np.stack(coefs, axis=0), mean8, std8, state_lo, state_hi, feature_names)
    rmses = np.stack(rmses, axis=0)
    print("[SINDy] train one-step RMSE std by dim:")
    print("        " + " ".join(f"{n:>7s}" for n in DIM_NAMES))
    for act in range(N_ACTIONS):
        print(f"  a={act}: " + " ".join(f"{v:7.3f}" for v in rmses[act]))
    return sindy


def evaluate_sindy(sindy, root, max_rows=SINDY_EVAL_MAX_TRANSITIONS, seed=SINDY_SEED):
    if not os.path.isdir(root):
        return
    xs, xps, _ = collect_transitions(root, max_per_action=max_rows // N_ACTIONS, seed=seed)
    x = np.concatenate(xs, axis=0)
    xp = np.concatenate(xps, axis=0)
    acts = np.concatenate([np.full(len(xs[a]), a, dtype=np.int64) for a in range(N_ACTIONS)])
    pred = sindy.step_phys(x, acts)
    rmse = np.sqrt(np.mean((pred - xp) ** 2, axis=0))
    print("[SINDy] validation one-step RMSE physical:")
    print("        " + " ".join(f"{n:>7s}" for n in DIM_NAMES))
    print("        " + " ".join(f"{v:7.3f}" for v in rmse))


# ---------------------------------------------------------------------------
# State estimator: encoder plus optional SINDy one-step prediction
# ---------------------------------------------------------------------------
class StateEstimator:
    def __init__(self, sindy, mean_np, std_np, mpc_cfg=None):
        self.sindy = sindy
        self.mean_np = mean_np
        self.std_np = std_np
        self.mpc_cfg = mpc_cfg
        self.reset()

    def reset(self):
        self.phys_fused = None
        self.a_prev = None
        self.resid = 0.0
        self.resid_hist = []

    def _scale(self):
        return RESID_SCALE0 if len(self.resid_hist) < FILTER_WARMUP else max(np.median(self.resid_hist), 1e-6)

    def trust_threshold(self):
        min_history = int(cfg_get(self.mpc_cfg, "mpc_min_history", MPC_MIN_HISTORY))
        if len(self.resid_hist) < min_history:
            return 0.0
        med = float(np.median(self.resid_hist))
        trust_factor = float(cfg_get(self.mpc_cfg, "trust_factor", TRUST_FACTOR))
        trust_floor = float(cfg_get(self.mpc_cfg, "trust_floor", TRUST_FLOOR))
        trust_abs_max = float(cfg_get(self.mpc_cfg, "trust_abs_max", TRUST_ABS_MAX))
        return min(trust_abs_max, max(trust_floor, med * trust_factor))

    def model_trusted(self):
        min_history = int(cfg_get(self.mpc_cfg, "mpc_min_history", MPC_MIN_HISTORY))
        return len(self.resid_hist) >= min_history and self.resid <= self.trust_threshold()

    def update(self, mu):
        enc_std = mu[0, :N_SUP].detach().cpu().numpy()
        enc_phys = to_phys_np(enc_std, self.mean_np, self.std_np)
        if self.phys_fused is not None and self.a_prev is not None:
            pred_phys = self.sindy.step_phys(self.phys_fused[None, :], np.array([self.a_prev]))[0]
            self.resid = float(np.linalg.norm(pred_phys - enc_phys))
            self.resid_hist.append(self.resid)
            if USE_FILTER:
                gate = float(np.exp(-self.resid / self._scale()))
                w = W_MODEL * gate
                fused_phys = (1.0 - w) * enc_phys + w * pred_phys
            else:
                fused_phys = enc_phys
        else:
            self.resid = 0.0
            fused_phys = enc_phys
        self.phys_fused = fused_phys
        return fused_phys

    def set_action(self, a):
        self.a_prev = int(a)


# ---------------------------------------------------------------------------
# SINDy dream rollout and objectives
# ---------------------------------------------------------------------------
def shaping_phys_np(phys):
    x, y, vx, vy, theta = phys[..., 0], phys[..., 1], phys[..., 2], phys[..., 3], phys[..., 4]
    leg1 = np.clip(phys[..., 6], 0.0, 1.0)
    leg2 = np.clip(phys[..., 7], 0.0, 1.0)
    return (
        -100.0 * np.sqrt(x * x + y * y + 1e-8)
        -100.0 * np.sqrt(vx * vx + vy * vy + 1e-8)
        -100.0 * np.abs(theta)
        +10.0 * leg1
        +10.0 * leg2
    )


def dream_rollout(sindy, phys0, macro_actions):
    macro_actions = np.asarray(macro_actions, dtype=np.int64)
    n, k = macro_actions.shape
    states = np.repeat(np.asarray(phys0, dtype=np.float64)[None, :], n, axis=0)
    phys = [states.copy()]
    prim = []
    for j in range(k):
        a = macro_actions[:, j]
        for _ in range(MPC_REPEAT):
            states = sindy.step_phys(states, a)
            phys.append(states.copy())
            prim.append(a.copy())
    return np.stack(phys, axis=1), np.stack(prim, axis=1)


def dream_value(phys_traj, prim_acts):
    sh = shaping_phys_np(phys_traj)
    prog = sh[:, -1] - sh[:, 0]
    fuel = FUEL_COST[prim_acts].sum(axis=1)
    last = phys_traj[:, -1]
    legs = np.clip(last[:, 6:8], 0.0, 1.0).sum(axis=1)
    speed = np.sqrt(last[:, 2] ** 2 + last[:, 3] ** 2 + 1e-8)
    term = LAND_LEG * legs - LAND_CRASH * np.maximum(speed - SAFE_SPEED, 0.0)
    return prog - FUEL_W * fuel + TERM_W * term


def dream_risk(phys_traj):
    x = phys_traj[:, :, 0]
    y = phys_traj[:, :, 1]
    vx = phys_traj[:, :, 2]
    vy = phys_traj[:, :, 3]
    theta = phys_traj[:, :, 4]
    speed = np.sqrt(vx * vx + vy * vy + 1e-8)
    low_alt = (y < 0.25).astype(np.float64)
    return (
        80.0 * np.maximum(np.abs(x).max(axis=1) - 1.0, 0.0)
        +80.0 * np.maximum(np.abs(theta).max(axis=1) - 0.75, 0.0)
        +120.0 * np.maximum((speed * low_alt).max(axis=1) - SAFE_SPEED, 0.0)
        +80.0 * np.maximum((-y).max(axis=1), 0.0)
    )


def pid_nominal_dream(sindy, phys0):
    s = np.asarray(phys0, dtype=np.float64).copy()
    macro = []
    for _ in range(MPC_HORIZON):
        a = heuristic_control(s)
        macro.append(a)
        for _ in range(MPC_REPEAT):
            s = sindy.step_phys(s[None, :], np.array([a]))[0]
    macro = np.asarray(macro, dtype=np.int64)
    phys_traj, prim = dream_rollout(sindy, phys0, macro[None, :])
    return macro, float(dream_value(phys_traj, prim)[0]), float(dream_risk(phys_traj)[0])


def cem_plan(sindy, phys0, nominal_macro, rng, pid_bias=PID_BIAS):
    k = MPC_HORIZON
    probs = np.full((k, N_ACTIONS), (1.0 - pid_bias) / N_ACTIONS)
    probs[np.arange(k), nominal_macro] += pid_bias
    best_v, best_seq = -1e18, nominal_macro.copy()

    for _ in range(CEM_ITERS):
        cdf = probs.cumsum(axis=1)
        u = rng.random((MPC_SAMPLES, k))
        samples = np.clip((u[:, :, None] >= cdf[None, :, :]).sum(axis=2), 0, N_ACTIONS - 1)
        phys_traj, prim = dream_rollout(sindy, phys0, samples)
        scores = dream_value(phys_traj, prim)

        elite_idx = np.argsort(scores)[-CEM_ELITE:]
        elite = samples[elite_idx]
        freq = np.stack([(elite == a).mean(axis=0) for a in range(N_ACTIONS)], axis=1)
        probs = (1.0 - CEM_LR) * probs + CEM_LR * freq
        probs /= probs.sum(axis=1, keepdims=True)

        top = elite_idx[-1]
        if scores[top] > best_v:
            best_v, best_seq = float(scores[top]), samples[top].copy()

    return int(best_seq[0]), best_v, best_seq


# ---------------------------------------------------------------------------
# Closed-loop episode
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_episode(controller, env, vae, sindy, mean_np, std_np, device, ep_seed, mpc_rng, record=False, mpc_cfg=None):
    obs, _ = env.reset(seed=ep_seed)
    f_cur = resize_frame(env.render())
    f_prev = f_cur
    est = StateEstimator(sindy, mean_np, std_np, mpc_cfg=mpc_cfg)
    total_r, fuel, last_r = 0.0, 0.0, 0.0
    dist_log, frames = [], []
    n_override, n_mpc, n_steps = 0, 0, 0
    n_diff_action, n_risky_pid, n_safer_mpc, n_better_mpc, n_all_gates = 0, 0, 0, 0, 0

    for _ in range(MAX_STEPS):
        n_steps += 1
        raw = env.render()
        f_cur = resize_frame(raw)
        if record:
            frames.append(raw)

        mu = encode_pair(vae, f_prev, f_cur, device)
        phys_est = est.update(mu)
        if est.resid:
            dist_log.append(est.resid)

        if controller == "true_pid":
            a = heuristic_control(obs)
        elif controller == "enc_pid":
            a = heuristic_control(phys_est)
        elif controller == "guided_mpc":
            a_pid = heuristic_control(phys_est)
            if not est.model_trusted():
                a = a_pid
            else:
                nominal_macro, v_pid, risk_pid = pid_nominal_dream(sindy, phys_est)
                pid_bias = float(cfg_get(mpc_cfg, "pid_bias", PID_BIAS))
                a_mpc, v_mpc, mpc_macro = cem_plan(sindy, phys_est, nominal_macro, mpc_rng, pid_bias=pid_bias)
                phys_mpc, _ = dream_rollout(sindy, phys_est, mpc_macro[None, :])
                risk_mpc = float(dream_risk(phys_mpc)[0])
                n_mpc += 1

                risk_trigger = float(cfg_get(mpc_cfg, "risk_trigger", RISK_TRIGGER))
                risk_margin = float(cfg_get(mpc_cfg, "risk_margin", RISK_MARGIN))
                mpc_margin = float(cfg_get(mpc_cfg, "mpc_margin", MPC_MARGIN))
                mode = cfg_get(mpc_cfg, "mode", "shield")

                diff_action = a_mpc != a_pid
                risky_pid = risk_pid > risk_trigger
                safer_mpc = risk_mpc + risk_margin < risk_pid
                better_mpc = v_mpc > v_pid + mpc_margin
                all_gates = diff_action and risky_pid and safer_mpc and better_mpc

                n_diff_action += int(diff_action)
                n_risky_pid += int(risky_pid)
                n_safer_mpc += int(safer_mpc)
                n_better_mpc += int(better_mpc)
                n_all_gates += int(all_gates)

                if mode == "shield":
                    do_override = all_gates
                elif mode == "value_only":
                    do_override = diff_action and better_mpc
                elif mode == "risk_only":
                    do_override = diff_action and risky_pid and safer_mpc
                elif mode == "force_diff":
                    do_override = diff_action
                else:
                    raise ValueError(f"Unknown guided_mpc mode: {mode}")

                if do_override:
                    a = a_mpc
                    n_override += 1
                else:
                    a = a_pid
        else:
            raise ValueError(controller)

        obs, r, terminated, truncated, _ = env.step(a)
        total_r += r
        last_r = r
        fuel += (0.30 if a == 2 else 0.03 if a in (1, 3) else 0.0)
        est.set_action(a)
        f_prev = f_cur
        if terminated or truncated:
            break

    landed = last_r >= 100.0
    crashed = last_r <= -100.0
    override_pct = (100.0 * n_override / n_steps) if n_steps else 0.0
    mpc_attempt_pct = (100.0 * n_mpc / n_steps) if n_steps else 0.0
    gate_pct = lambda n: (100.0 * n / n_mpc) if n_mpc else 0.0
    return {
        "return": total_r,
        "landed": landed,
        "crashed": crashed,
        "fuel": fuel,
        "dist": dist_log,
        "frames": frames,
        "override_pct": override_pct,
        "mpc_attempt_pct": mpc_attempt_pct,
        "diff_action_pct": gate_pct(n_diff_action),
        "risky_pid_pct": gate_pct(n_risky_pid),
        "safer_mpc_pct": gate_pct(n_safer_mpc),
        "better_mpc_pct": gate_pct(n_better_mpc),
        "all_gates_pct": gate_pct(n_all_gates),
    }


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = get_device()
    assert MODEL in MODEL_REGISTRY, f"MODEL must be one of {list(MODEL_REGISTRY)}"
    make_vae, vae_ckpt = MODEL_REGISTRY[MODEL]
    print("device:", device, "| model:", MODEL, "| dynamics: SINDy | wind:", ENABLE_WIND)

    mean_np, std_np = load_norm_stats(NORM_STATS)
    sindy = fit_sindy_dynamics(os.path.join(DATA_ROOT, "train"), mean_np, std_np)
    evaluate_sindy(sindy, os.path.join(DATA_ROOT, "val"))

    vae = make_vae().to(device)
    vae.load_state_dict(torch.load(vae_ckpt, map_location=device))
    vae.eval()

    env = make_env()
    run_specs = []
    for c in CONTROLLERS:
        if c == "guided_mpc":
            for cfg in GUIDED_GRID:
                run_specs.append({"label": f"guided_mpc_{cfg['label']}", "controller": c, "mpc_cfg": cfg})
        else:
            run_specs.append({"label": c, "controller": c, "mpc_cfg": None})

    results = {s["label"]: [] for s in run_specs}
    dist_example = {}
    for spec in run_specs:
        label, c, mpc_cfg = spec["label"], spec["controller"], spec["mpc_cfg"]
        mpc_rng = np.random.default_rng(MPC_SEED)
        print(f"\n{'=' * 72}\n  CONTROLLER: {label}\n{'=' * 72}")
        if mpc_cfg:
            print("  config:", "  ".join(f"{k}={v}" for k, v in mpc_cfg.items() if k != "label"))
        for ep in range(N_EPISODES):
            rec = RECORD_GIF and ep == 0
            res = run_episode(c, env, vae, sindy, mean_np, std_np, device, SEED + ep, mpc_rng, record=rec, mpc_cfg=mpc_cfg)
            results[label].append(res)
            if ep == 0:
                dist_example[label] = res["dist"]
                if rec:
                    save_gif(res["frames"], os.path.join(SAVE_DIR, f"ext4sindy_{MODEL}_{label}.gif"))
            res["frames"] = []
            extra = (
                f"  override={res['override_pct']:.0f}%  mpc_used={res['mpc_attempt_pct']:.0f}%"
                if c == "guided_mpc"
                else ""
            )
            print(
                f"  ep{ep:02d}  return={res['return']:8.1f}  "
                f"{'LAND' if res['landed'] else 'CRASH' if res['crashed'] else 'timeout':6}  "
                f"fuel={res['fuel']:.1f}{extra}"
            )
    env.close()

    print(f"\n{'=' * 104}")
    print(
        f"{'controller':<22}{'mean return':>13}{'success %':>11}{'crash %':>9}"
        f"{'mean fuel':>11}{'MPC override %':>16}{'MPC used %':>12}"
    )
    print("-" * 104)
    labels = [s["label"] for s in run_specs]
    for spec in run_specs:
        label, c = spec["label"], spec["controller"]
        rvals = np.array([r["return"] for r in results[label]])
        succ = 100.0 * np.mean([r["landed"] for r in results[label]])
        crash = 100.0 * np.mean([r["crashed"] for r in results[label]])
        fuel = np.mean([r["fuel"] for r in results[label]])
        ovr = np.mean([r["override_pct"] for r in results[label]])
        used = np.mean([r["mpc_attempt_pct"] for r in results[label]])
        ovr_s = f"{ovr:>15.1f}%" if c == "guided_mpc" else f"{'-':>16}"
        used_s = f"{used:>11.1f}%" if c == "guided_mpc" else f"{'-':>12}"
        print(f"{label:<22}{rvals.mean():>13.1f}{succ:>11.0f}{crash:>9.0f}{fuel:>11.1f}{ovr_s}{used_s}")
    print("=" * 104)

    guided_labels = [s["label"] for s in run_specs if s["controller"] == "guided_mpc"]
    if guided_labels:
        print(f"\n{'=' * 104}")
        print("GATE DIAGNOSTICS (% of MPC-used steps)")
        print(f"{'controller':<22}{'diff action':>13}{'risky PID':>12}{'safer MPC':>12}{'better MPC':>13}{'all gates':>12}")
        print("-" * 104)
        for label in guided_labels:
            diff = np.mean([r["diff_action_pct"] for r in results[label]])
            risky = np.mean([r["risky_pid_pct"] for r in results[label]])
            safer = np.mean([r["safer_mpc_pct"] for r in results[label]])
            better = np.mean([r["better_mpc_pct"] for r in results[label]])
            allg = np.mean([r["all_gates_pct"] for r in results[label]])
            print(f"{label:<22}{diff:>12.1f}%{risky:>11.1f}%{safer:>11.1f}%{better:>12.1f}%{allg:>11.1f}%")
        print("=" * 104)

    plt.figure(figsize=(max(7.6, 1.8 * len(labels)), 4.8))
    data = [[r["return"] for r in results[label]] for label in labels]
    plt.boxplot(data, tick_labels=labels, showmeans=True)
    plt.xticks(rotation=20, ha="right")
    plt.axhline(200, color="g", ls="--", lw=1, label="solved (>=200)")
    plt.axhline(0, color="0.6", lw=0.8)
    plt.ylabel("episode return")
    plt.title(f"Closed-loop control with SINDy dynamics (model={MODEL}, wind={ENABLE_WIND})")
    plt.grid(alpha=0.3, axis="y")
    plt.legend()
    plt.tight_layout()
    p1 = os.path.join(SAVE_DIR, f"ext4sindy_{MODEL}_returns.png")
    plt.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close()
    print("saved:", p1)

    plt.figure(figsize=(7.6, 4.2))
    for label in labels:
        d = dist_example.get(label, [])
        if d:
            plt.plot(np.arange(1, len(d) + 1), d, lw=1.3, label=label)
    plt.xlabel("t (step)")
    plt.ylabel("encoder - SINDy one-step residual")
    plt.title(f"SINDy model-trust signal (model={MODEL}, wind={ENABLE_WIND})")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    p2 = os.path.join(SAVE_DIR, f"ext4sindy_{MODEL}_disturbance.png")
    plt.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close()
    print("saved:", p2)

    np.savez(
        os.path.join(SAVE_DIR, f"ext4sindy_{MODEL}_results.npz"),
        model=MODEL,
        dynamics="sindy",
        controllers=np.array(labels),
        returns=np.array([[r["return"] for r in results[label]] for label in labels]),
        landed=np.array([[r["landed"] for r in results[label]] for label in labels]),
        override_pct=np.array([[r["override_pct"] for r in results[label]] for label in labels]),
        mpc_attempt_pct=np.array([[r["mpc_attempt_pct"] for r in results[label]] for label in labels]),
        diff_action_pct=np.array([[r["diff_action_pct"] for r in results[label]] for label in labels]),
        risky_pid_pct=np.array([[r["risky_pid_pct"] for r in results[label]] for label in labels]),
        safer_mpc_pct=np.array([[r["safer_mpc_pct"] for r in results[label]] for label in labels]),
        better_mpc_pct=np.array([[r["better_mpc_pct"] for r in results[label]] for label in labels]),
        all_gates_pct=np.array([[r["all_gates_pct"] for r in results[label]] for label in labels]),
        grid_mode=np.array([cfg_get(s["mpc_cfg"], "mode", "") for s in run_specs]),
        grid_mpc_margin=np.array([cfg_get(s["mpc_cfg"], "mpc_margin", np.nan) for s in run_specs]),
        grid_risk_trigger=np.array([cfg_get(s["mpc_cfg"], "risk_trigger", np.nan) for s in run_specs]),
        grid_risk_margin=np.array([cfg_get(s["mpc_cfg"], "risk_margin", np.nan) for s in run_specs]),
        grid_pid_bias=np.array([cfg_get(s["mpc_cfg"], "pid_bias", np.nan) for s in run_specs]),
        sindy_coef=sindy.coef,
        sindy_feature_names=sindy.feature_names,
        sindy_state_lo=sindy.state_lo,
        sindy_state_hi=sindy.state_hi,
        sindy_nonzero=sindy.nonzero_counts(),
        wind=ENABLE_WIND,
    )
    print(f"\nsaved figures + ext4sindy_{MODEL}_results.npz -> {SAVE_DIR}")


if __name__ == "__main__":
    main()
