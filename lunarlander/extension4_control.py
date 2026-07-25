"""
extension4_control.py — Extension 4: connecting to classical control (LunarLander).

Closed-loop control of the REAL LunarLander, FROM PIXELS, using our own
interpretable models (baseline VAE encoder + encoded LSTM world model). It implements:

  (A) The ENCODER as a STATE ESTIMATOR -> a classical PID/heuristic:
      pixels -> VAE.encode -> mu[:8] (physical state) -> PD controller -> action.
  (B) The LSTM as an MPC SAFETY SHIELD (it "dreams" K steps):
      receding horizon: sample N action sequences, roll them out in latent
      space, score with LunarLander's reward shaping, execute the first action of the best one.
  (C) A DISTURBANCE SIGNAL ("it senses external noise"):
      residual = || encoder(t) − LSTM_pred(t-1 -> t) || in physical units. It grows under
      an external disturbance (e.g. ENABLE_WIND=True).

CONTROLLER COMPARISON (the same episode seeds):
   true_pid  : PD on the TRUE obs                 (upper bound)
   enc_pid   : PD on the encoder's ESTIMATE       (the encoder as a sensor)
   latent_mpc: MPC on the world model             (control purely from pixels + LSTM)

Imports from the canonical lunarlander/ modules. cwd: lunarlander/. Requires gymnasium[box2d].
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import gymnasium as gym
from dataCollect import resize_frame                 # the SAME frame pipeline as training
from vae import VAE
from vae_p1 import VAE_P1
from vae_p2 import VAE_P2
from vae_p3 import VAE_P3
from lstm import LatentPredictor
from loader import load_norm_stats

from paths import BASELINE_LSTM, BASELINE_VAE, DATA_ROOT, P1_LSTM, P1_VAE, P2_LSTM, P2_VAE, P3_SEMI_LSTM, P3_SEMI_VAE, P3_WEAK_LSTM, P3_WEAK_VAE, outputs

# ---------------------------------------------------------------------------
# CONFIG  (paths from config.py via paths.py)
# ---------------------------------------------------------------------------
NORM_STATS = os.path.join(DATA_ROOT, "norm_stats.npz")
SAVE_DIR = outputs("lunarlander_ext4_control")

LATENT_SIZE, N_SUP, N_IMG = 64, 8, 56
N_ACTIONS, HIDDEN, LAYERS = 4, 64, 2
IMG_H, IMG_W = 80, 120

# --- WHICH model drives the encoder/MPC (all have a supervised mu[:8] -> interpretable) ---
#   baseline: full supervision (the most faithful physical encoding)
#   p1: decoupled encoders (more robust to visual noise)
#   p2: brightness/contrast invariance
#   p3_semi/p3_weak: reduced velocity supervision
# Run with different MODEL values to compare which principle controls best (especially under wind).
MODEL = "p1"
MODEL_REGISTRY = {
    "baseline": (lambda: VAE(latent_size=LATENT_SIZE),       BASELINE_VAE, BASELINE_LSTM),
    "p1":       (lambda: VAE_P1(n_sup=N_SUP, n_img=N_IMG),    P1_VAE,       P1_LSTM),
    "p2":       (lambda: VAE_P2(latent_size=LATENT_SIZE),     P2_VAE,       P2_LSTM),
    "p3_semi":  (lambda: VAE_P3(latent_size=LATENT_SIZE),     P3_SEMI_VAE,  P3_SEMI_LSTM),
    "p3_weak":  (lambda: VAE_P3(latent_size=LATENT_SIZE),     P3_WEAK_VAE,  P3_WEAK_LSTM),
}

N_EPISODES = 20                  # episodes per controller (the same seeds for a fair comparison)
MAX_STEPS = 400
SEED = 0
ENABLE_WIND = False              # LunarLander-v3 wind -> a demo for the disturbance signal
CONTROLLERS = ["true_pid", "enc_pid", "latent_mpc"]

RECORD_GIF = True                # save a GIF of the 1st episode per controller (full-res render)
GIF_FPS = 30

# --- MPC (safety shield): heuristic-guided + first-action enumeration ---
MPC_HORIZON = 5                  # K steps of "dream" (short -> less compounding error)
MPC_SAMPLES_PER_ACTION = 64      # stochastic "tails" per immediate action (N = 4 x this)
EPS_EXPLORE = 0.20               # exploration in the tail (perturbation around the heuristic)
FUEL_W = 0.30                    # fuel penalty
BOUND_W = 50.0                   # penalty for |x| out of bounds (out-of-frame)
SMOOTH_W = 2.0                   # penalty for switching action (smoother control)
X_BOUND = 1.0                    # the |x| bound beyond which we penalize
LAND_W = 50.0                    # bonus for leg contact at the terminal state (anti-hover, pro-landing)
MPC_SEED = 0
RUN_DIAGNOSTIC = True            # print the action-sensitivity check before the episodes

DIM_NAMES = ["x", "y", "vx", "vy", "theta", "omega", "leg1", "leg2"]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_env():
    """LunarLander discrete, rgb_array render. ENABLE_WIND -> an external disturbance (v3)."""
    last_err = None
    for env_id in ("LunarLander-v3", "LunarLander-v2"):
        try:
            kw = dict(render_mode="rgb_array")
            if ENABLE_WIND:
                kw.update(enable_wind=True, wind_power=15.0, turbulence_power=1.5)
            return gym.make(env_id, **kw)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"LunarLander not found (pip install 'gymnasium[box2d]'). {last_err}")


# ---------------------------------------------------------------------------
# (A) Classical controller — a DETERMINISTIC PD heuristic (no ε-greedy)
# ---------------------------------------------------------------------------
def heuristic_control(s):
    """ s = [x, y, vx, vy, theta, omega, leg1, leg2] (physical units) -> action ∈ {0,1,2,3}. """
    x, y, vx, vy, theta, omega = float(s[0]), float(s[1]), float(s[2]), float(s[3]), float(s[4]), float(s[5])
    leg1, leg2 = float(s[6]) > 0.5, float(s[7]) > 0.5
    angle_targ = float(np.clip(x * 0.5 + vx * 1.0, -0.4, 0.4))
    hover_targ = 0.55 * abs(x)
    angle_todo = (angle_targ - theta) * 0.5 - omega * 1.0
    hover_todo = (hover_targ - y) * 0.5 - vy * 0.5
    if leg1 or leg2:                                  # contact -> stop the angle control, gentle braking only
        angle_todo, hover_todo = 0.0, -vy * 0.5
    if hover_todo > abs(angle_todo) and hover_todo > 0.05:
        return 2                                      # main engine
    if angle_todo < -0.05:
        return 3                                      # right engine
    if angle_todo > 0.05:
        return 1                                      # left engine
    return 0                                          # noop


# ---------------------------------------------------------------------------
# Encoder helpers — pixels -> latent / physical estimate
# ---------------------------------------------------------------------------
def _to_tensor(frame, device):
    """uint8 (H,W,3) -> float (1,3,H,W) [0,1]."""
    t = torch.from_numpy(frame.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return t.to(device)


@torch.no_grad()
def encode_pair(vae, f_prev, f_cur, device):
    """stack(prev,cur) -> mu (1,64). mu[:8] is the estimated physical state (standardized)."""
    x = torch.cat([_to_tensor(f_prev, device), _to_tensor(f_cur, device)], dim=1)   # (1,6,H,W)
    mu, _ = vae.encode(x)
    return mu


def to_phys(z8_std, mean, std):
    """standardized (...,8) -> physical units."""
    return z8_std * std[:N_SUP] + mean[:N_SUP]


def save_gif(frames, path, fps=GIF_FPS):
    """frames: a list of full-res uint8 (H,W,3) renders -> an animated GIF."""
    from PIL import Image
    if not frames:
        print("[gif] no frames to save:", path); return
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / max(fps, 1)), loop=0)
    print("saved:", path)


# ---------------------------------------------------------------------------
# LunarLander's reward shaping (as a cost-to-go proxy for the MPC)
# ---------------------------------------------------------------------------
def shaping_phys(phys):
    """ phys: (...,8) tensor in physical units -> shaping (...). Higher = better. """
    x, y, vx, vy, theta = phys[..., 0], phys[..., 1], phys[..., 2], phys[..., 3], phys[..., 4]
    leg1, leg2 = phys[..., 6], phys[..., 7]
    return (-100.0 * torch.sqrt(x * x + y * y + 1e-8)
            - 100.0 * torch.sqrt(vx * vx + vy * vy + 1e-8)
            - 100.0 * torch.abs(theta)
            + 10.0 * leg1 + 10.0 * leg2)


# ---------------------------------------------------------------------------
# (B) Latent MPC — heuristic-guided, first-action enumeration (ON-distribution)
# ---------------------------------------------------------------------------
def heuristic_action_batch(phys):
    """ Vectorized PD heuristic: phys (N,8) physical units -> actions (N,) long.
    The SAME logic as heuristic_control but batched -> keeps the "tails" ON-distribution. """
    x, vx, y, vy, theta, omega = (phys[:, 0], phys[:, 2], phys[:, 1],
                                  phys[:, 3], phys[:, 4], phys[:, 5])
    contact = (phys[:, 6] > 0.5) | (phys[:, 7] > 0.5)
    angle_targ = torch.clamp(x * 0.5 + vx * 1.0, -0.4, 0.4)
    hover_targ = 0.55 * torch.abs(x)
    angle_todo = torch.where(contact, torch.zeros_like(x), (angle_targ - theta) * 0.5 - omega * 1.0)
    hover_todo = torch.where(contact, -vy * 0.5, (hover_targ - y) * 0.5 - vy * 0.5)
    a = torch.zeros(phys.size(0), dtype=torch.long, device=phys.device)
    a = torch.where(angle_todo > 0.05, torch.full_like(a, 1), a)          # left
    a = torch.where(angle_todo < -0.05, torch.full_like(a, 3), a)         # right
    main = (hover_todo > torch.abs(angle_todo)) & (hover_todo > 0.05)
    a = torch.where(main, torch.full_like(a, 2), a)                       # main (takes priority)
    return a


@torch.no_grad()
def mpc_plan(lstm, z_t, mean_t, std_t, device):
    """ Heuristic-guided MPC: enumerate the immediate action ∈ {0..3}, roll out the "tail" with the
    heuristic (ON-distribution) + a little exploration.
    OBJECTIVE = PROGRESS (env-style reward = Δshaping): hovering -> 0, descending toward the pad -> positive
    (anti-hover). score = Σ Δshaping − fuel − bounds − switch + LAND_W·(terminal legs).
    Returns the immediate action with the best MEAN score (robust to model exploitation). """
    K, M = MPC_HORIZON, MPC_SAMPLES_PER_ACTION
    N = N_ACTIONS * M
    first = torch.arange(N_ACTIONS, device=device).repeat_interleave(M)   # the immediate action per candidate
    z = z_t.expand(N, -1).contiguous()
    hidden = lstm.init_hidden(N, device)
    fuel_cost = torch.tensor([0.0, 0.03, 0.30, 0.03], device=device)
    std8, mean8 = std_t[:N_SUP], mean_t[:N_SUP]
    score = torch.zeros(N, device=device)
    prev_shaping = shaping_phys(z[:, :N_SUP] * std8 + mean8)              # shaping_0 (the same for all)
    prev_a, phys = None, z[:, :N_SUP] * std8 + mean8
    for k in range(K):
        if k == 0:
            a = first                                                     # the action under evaluation
        else:
            a = heuristic_action_batch(phys)                             # ON-distribution tail
            explore = torch.rand(N, device=device) < EPS_EXPLORE
            a = torch.where(explore, torch.randint(0, N_ACTIONS, (N,), device=device), a)
        z, hidden = lstm.step(z, F.one_hot(a, N_ACTIONS).float(), hidden)
        phys = z[:, :N_SUP] * std8 + mean8
        sh = shaping_phys(phys)
        step_cost = ((sh - prev_shaping)                                  # PROGRESS (anti-hover)
                     - FUEL_W * fuel_cost[a]
                     - BOUND_W * torch.relu(torch.abs(phys[:, 0]) - X_BOUND))
        if prev_a is not None:
            step_cost = step_cost - SMOOTH_W * (a != prev_a).float()
        score += step_cost
        prev_shaping, prev_a = sh, a
    score += LAND_W * (phys[:, 6] + phys[:, 7])                           # terminal landing bonus (legs)
    score_per_action = score.view(N_ACTIONS, M).mean(dim=1)               # the MEAN per immediate action
    return int(torch.argmax(score_per_action).item())


# ---------------------------------------------------------------------------
# Diagnostic — how much the world model "listens" to the actions
# ---------------------------------------------------------------------------
@torch.no_grad()
def _seed_z_from_env(vae, env, device, warmup=20):
    """ Run a few heuristic steps -> take z_t from a pair of frames (airborne). """
    obs, _ = env.reset(seed=SEED)
    frames = [resize_frame(env.render())]
    for _ in range(warmup):
        obs, _, term, trunc, _ = env.step(heuristic_control(obs))
        frames.append(resize_frame(env.render()))
        if term or trunc:
            break
    return encode_pair(vae, frames[-2], frames[-1], device)


@torch.no_grad()
def action_sensitivity(vae, lstm, env, mean_t, std_t, device, K=10):
    """ From z_t (airborne) -> a K-step rollout with a CONSTANT action. If 'main' does not give a noticeably
    larger Δy/Δvy than 'noop' (or left/right do not change x/theta), the model does not 'listen'
    to the actions -> bad for planning. """
    z0 = _seed_z_from_env(vae, env, device)
    std8, mean8 = std_t[:N_SUP], mean_t[:N_SUP]
    p0 = (z0[0, :N_SUP] * std8 + mean8).cpu().numpy()
    print(f"\n{'='*66}\n  ACTION-SENSITIVITY DIAGNOSTIC (rollout K={K}, constant action)\n{'='*66}")
    print(f"  start: x={p0[0]:+.3f} y={p0[1]:+.3f} vx={p0[2]:+.3f} vy={p0[3]:+.3f} theta={p0[4]:+.3f}")
    print(f"  {'action':<8}{'Δx':>9}{'Δy':>9}{'Δvx':>9}{'Δvy':>9}{'Δtheta':>9}")
    names = {0: "noop", 1: "left", 2: "main", 3: "right"}
    for act in (0, 2, 1, 3):
        z = z0.clone(); hidden = lstm.init_hidden(1, device)
        a_oh = F.one_hot(torch.tensor([act], device=device), N_ACTIONS).float()
        for _ in range(K):
            z, hidden = lstm.step(z, a_oh, hidden)
        d = (z[0, :N_SUP] * std8 + mean8).cpu().numpy() - p0
        print(f"  {names[act]:<8}{d[0]:>+9.3f}{d[1]:>+9.3f}{d[2]:>+9.3f}{d[3]:>+9.3f}{d[4]:>+9.3f}")
    print("  -> 'main' should give Δy/Δvy clearly larger than 'noop'.  'left'/'right': should change x & theta.")


# ---------------------------------------------------------------------------
# A closed-loop episode with a given controller
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_episode(controller, env, vae, lstm, mean_t, std_t, device, ep_seed, record=False):
    obs, _ = env.reset(seed=ep_seed)
    f_cur = resize_frame(env.render())
    f_prev = f_cur                                    # t=0: (f0,f0) -> velocities ~ 0 (1 step)
    z_prev, a_prev = None, None
    total_r, fuel, last_r = 0.0, 0.0, 0.0
    dist_log, frames = [], []
    for _ in range(MAX_STEPS):
        raw = env.render()
        f_cur = resize_frame(raw)
        if record:
            frames.append(raw)
        mu = encode_pair(vae, f_prev, f_cur, device)              # (1,64)
        phys_est = to_phys(mu[0, :N_SUP], mean_t, std_t)          # (8,) physical units

        # --- disturbance signal: encoder(t) vs the 1-step LSTM pred from (z_{t-1}, a_{t-1}) ---
        if z_prev is not None:
            a_oh = F.one_hot(torch.tensor([a_prev], device=device), N_ACTIONS).float()
            z_pred, _ = lstm.step(z_prev, a_oh, lstm.init_hidden(1, device))
            resid = torch.norm(to_phys(z_pred[0, :N_SUP], mean_t, std_t) - phys_est).item()
            dist_log.append(resid)

        # --- action selection per controller ---
        if controller == "true_pid":
            a = heuristic_control(obs)
        elif controller == "enc_pid":
            a = heuristic_control(phys_est.cpu().numpy())
        elif controller == "latent_mpc":
            a = mpc_plan(lstm, mu, mean_t, std_t, device)
        else:
            raise ValueError(controller)

        obs, r, terminated, truncated, _ = env.step(a)
        total_r += r; last_r = r
        fuel += (0.30 if a == 2 else 0.03 if a in (1, 3) else 0.0)
        z_prev, a_prev, f_prev = mu, a, f_cur
        if terminated or truncated:
            break

    landed = last_r >= 100.0                          # gym: +100 land / -100 crash on the last step
    crashed = last_r <= -100.0
    return {"return": total_r, "landed": landed, "crashed": crashed,
            "fuel": fuel, "dist": dist_log, "frames": frames}


# ---------------------------------------------------------------------------
# Main — run all the controllers on the SAME seeds and compare
# ---------------------------------------------------------------------------
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = get_device()
    assert MODEL in MODEL_REGISTRY, f"MODEL ∈ {list(MODEL_REGISTRY)}"
    make_vae, vae_ckpt, lstm_ckpt = MODEL_REGISTRY[MODEL]
    print("device:", device, "| model:", MODEL, "| wind:", ENABLE_WIND)

    mean_np, std_np = load_norm_stats(NORM_STATS)
    mean_t = torch.tensor(mean_np, device=device)
    std_t = torch.tensor(std_np, device=device)

    vae = make_vae().to(device)
    vae.load_state_dict(torch.load(vae_ckpt, map_location=device)); vae.eval()
    lstm = LatentPredictor(LATENT_SIZE, N_ACTIONS, HIDDEN, LAYERS).to(device)
    lstm.load_state_dict(torch.load(lstm_ckpt, map_location=device)); lstm.eval()

    env = make_env()
    if RUN_DIAGNOSTIC:
        torch.manual_seed(MPC_SEED)
        action_sensitivity(vae, lstm, env, mean_t, std_t, device)

    results = {c: [] for c in CONTROLLERS}
    dist_example = {}                                 # one disturbance trace per controller
    for c in CONTROLLERS:
        torch.manual_seed(MPC_SEED)                   # the same MPC exploration per controller (reproducible)
        print(f"\n{'='*56}\n  CONTROLLER: {c}\n{'='*56}")
        for ep in range(N_EPISODES):
            rec = RECORD_GIF and ep == 0
            res = run_episode(c, env, vae, lstm, mean_t, std_t, device, SEED + ep, record=rec)
            results[c].append(res)
            if ep == 0:
                dist_example[c] = res["dist"]
                if rec:
                    save_gif(res["frames"], os.path.join(SAVE_DIR, f"ext4_{MODEL}_{c}.gif"))
            res["frames"] = []                        # free the memory (full-res frames)
            print(f"  ep{ep:02d}  return={res['return']:8.1f}  "
                  f"{'LAND' if res['landed'] else 'CRASH' if res['crashed'] else 'timeout':6}  "
                  f"fuel={res['fuel']:.1f}")
    env.close()

    # ---- summary table ----
    print(f"\n{'='*72}")
    print(f"{'controller':<14}{'mean return':>14}{'success %':>12}{'crash %':>10}{'mean fuel':>12}")
    print("-" * 72)
    summary = {}
    for c in CONTROLLERS:
        R = np.array([r["return"] for r in results[c]])
        succ = 100.0 * np.mean([r["landed"] for r in results[c]])
        crash = 100.0 * np.mean([r["crashed"] for r in results[c]])
        fuel = np.mean([r["fuel"] for r in results[c]])
        summary[c] = (R.mean(), succ, crash, fuel)
        print(f"{c:<14}{R.mean():>14.1f}{succ:>12.0f}{crash:>10.0f}{fuel:>12.1f}")
    print("=" * 72)

    # ---- plot 1: return distribution per controller ----
    plt.figure(figsize=(7, 4.6))
    data = [[r["return"] for r in results[c]] for c in CONTROLLERS]
    plt.boxplot(data, tick_labels=CONTROLLERS, showmeans=True)
    plt.axhline(200, color="g", ls="--", lw=1, label="solved (≥200)")
    plt.axhline(0, color="0.6", lw=0.8)
    plt.ylabel("episode return")
    plt.title(f"Closed-loop control — return per controller (model={MODEL}, wind={ENABLE_WIND})")
    plt.grid(alpha=0.3, axis="y"); plt.legend()
    plt.tight_layout()
    p1 = os.path.join(SAVE_DIR, f"ext4_{MODEL}_returns.png")
    plt.savefig(p1, dpi=150, bbox_inches="tight"); plt.close()
    print("saved:", p1)

    # ---- plot 2: disturbance signal (encoder vs LSTM-pred residual) on the 1st episode ----
    plt.figure(figsize=(7.5, 4.2))
    for c in CONTROLLERS:
        d = dist_example.get(c, [])
        if d:
            plt.plot(np.arange(1, len(d) + 1), d, lw=1.4, label=c)
    plt.xlabel("t (step)"); plt.ylabel("‖encoder(t) − LSTM_pred(t)‖  (physical units)")
    plt.title(f"Disturbance signal — residual encoder vs world-model (model={MODEL}, wind={ENABLE_WIND})")
    plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout()
    p2 = os.path.join(SAVE_DIR, f"ext4_{MODEL}_disturbance.png")
    plt.savefig(p2, dpi=150, bbox_inches="tight"); plt.close()
    print("saved:", p2)

    np.savez(os.path.join(SAVE_DIR, f"ext4_{MODEL}_results.npz"),
             model=MODEL, controllers=np.array(CONTROLLERS),
             returns=np.array([[r["return"] for r in results[c]] for c in CONTROLLERS]),
             landed=np.array([[r["landed"] for r in results[c]] for c in CONTROLLERS]),
             wind=ENABLE_WIND)
    print(f"\nsaved figures + ext4_{MODEL}_results.npz -> {SAVE_DIR}")


if __name__ == "__main__":
    main()
