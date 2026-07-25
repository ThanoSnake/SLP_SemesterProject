"""
control.py — Solves LunarLander FROM PIXELS and brings out the BENEFIT of the world model over
classical control (PID). Future Direction E of the paper: world-model components COMPLEMENTARY to
classical autonomy.

All the controllers use the SAME perception (P1 VAE encoder -> physical state mu[:8]) so that
the only difference is whether they exploit the LSTM's DYNAMICS KNOWLEDGE:

  true_pid    : PD on the TRUE obs                              (upper bound of the heuristic itself)
  enc_pid     : PD on the encoder's ESTIMATE                    (classical control, SAME perception)
  mpc_cem     : FREE MPC (PID-guided CEM)                       (NAIVE — it fails, see below)
  rollout     : PID base policy + a model 1-step lookahead      (THE FIX — uses the model correctly)

FINDING (control_diag.py): the model/encoder are GOOD (dream RMSE@h10~0.35, encoder R²>0.97 on the
velocities), but FREE MPC fails because of the OPTIMIZER'S CURSE: the argmax over hundreds of
sequences systematically picks the ones the model OVER-estimates (dream-vs-real corr only +0.45) ->
in closed loop this compounds into catastrophe. THE FIX: rollout/policy improvement — only 4 candidates with
a PID continuation (in-distribution) -> provably >= PID when the model is accurate.

COST: telescoping potential-DIFFERENCE shaping + terminal landing/crash − fuel (gym-faithful weights;
D4 showed that the velocities are encoded excellently, so NO down-weighting).

Evaluation: the same seeds, WITH & WITHOUT wind. Imports from the canonical modules; cwd: lunarlander/.
Run:  !python3 lunarlander/control.py   (requires gymnasium[box2d])
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import gymnasium as gym
from dataCollect import resize_frame                 # the SAME frame pipeline as training
from vae_p1 import VAE_P1
from lstm import LatentPredictor

from paths import CONTROL_LSTM, CONTROL_VAE, DATA_ROOT, outputs

# ---------------------------------------------------------------------------
# CONFIG  (paths from config.py via paths.py)
# ---------------------------------------------------------------------------
NORM_STATS = os.path.join(DATA_ROOT, "norm_stats.npz")
VAE_CKPT = CONTROL_VAE               # P1 VAE (decoupled encoders)
LSTM_CKPT = CONTROL_LSTM             # LSTM on the NEW large dataset (point this at the new .pth)
SAVE_DIR = outputs("lunarlander_control")

LATENT_SIZE, N_SUP, N_IMG = 64, 8, 56
N_ACTIONS, HIDDEN, LAYERS = 4, 64, 2
IMG_H, IMG_W = 80, 120

N_EPISODES = 20                  # episodes per (controller, wind) — the same seeds
MAX_STEPS = 400
SEED = 0
CONTROLLERS = ["true_pid", "enc_pid", "mpc_cem", "rollout"]   # mpc_cem = naive (fails); rollout = the fix
WIND_CONDITIONS = [("no_wind", False), ("wind", True)]
WIND_POWER, TURBULENCE_POWER = 10.0, 1.0         # moderate wind (the PID struggles but works)

RECORD_GIF = True
GIF_FPS = 30

# --- MPC (shared) ---
MPC_SEED = 0
# random shooting
RS_HORIZON = 10                  # length of the random sequence
RS_SAMPLES = 512
# PID-guided CEM
CEM_HORIZON = 8                  # macro decisions
CEM_REPEAT = 3                   # action-repeat -> effective horizon = 24
CEM_SAMPLES = 256
CEM_ITERS = 3
CEM_ELITE = 32
CEM_LR = 0.7
PID_BIAS = 0.5                   # warm-start mass on the PID action per macro-step

# --- COST weights — gym-faithful (D4: the encoder is excellent on the velocities, R²>0.97; NO down-weighting) ---
W_POS, W_VEL, W_ANG, W_LEG = 100.0, 100.0, 100.0, 10.0
FUEL_W = 0.30
TERM_W, LAND_LEG, LAND_CRASH, SAFE_SPEED = 1.0, 20.0, 100.0, 0.5   # terminal landing/crash proxy

# --- rollout corrector (PID base policy + model 1-step lookahead; see control_diag) ---
# Instead of free MPC (which falls into the optimizer's curse), we evaluate "candidate action now, then
# FOLLOW the PID" -> only 4 candidates, an in-distribution continuation, >= PID if the model is accurate.
ROLLOUT_HORIZON = 30             # steps of PID continuation in the dream (the model is reliable to ~h24)
ROLLOUT_MARGIN = 2.0             # override the PID only if clearly better (avoids noisy flips)

FUEL_COST = [0.0, 0.03, 0.30, 0.03]
DIM_NAMES = ["x", "y", "vx", "vy", "theta", "omega", "leg1", "leg2"]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_env(enable_wind):
    last_err = None
    for env_id in ("LunarLander-v3", "LunarLander-v2"):
        try:
            kw = dict(render_mode="rgb_array")
            if enable_wind:
                kw.update(enable_wind=True, wind_power=WIND_POWER, turbulence_power=TURBULENCE_POWER)
            return gym.make(env_id, **kw)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"LunarLander not found (pip install 'gymnasium[box2d]'). {last_err}")


# ---------------------------------------------------------------------------
# Classical controller — the standard LunarLander PD heuristic (same as dataCollect)
# ---------------------------------------------------------------------------
def heuristic_control(s):
    """ s = [x,y,vx,vy,theta,omega,leg1,leg2] (physical units) -> action ∈ {0,1,2,3}. """
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


# ---------------------------------------------------------------------------
# Encoder helpers — pixels -> latent / physical estimate (P1 VAE)
# ---------------------------------------------------------------------------
def _to_tensor(frame, device):
    return torch.from_numpy(frame.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)


@torch.no_grad()
def encode_pair(vae, f_prev, f_cur, device):
    """stack(f_prev, f_cur) -> mu (1,64). mu[:8] = the estimated physical state (standardized).
    NOTE: by the training convention this corresponds ~to f_prev (1-step lag; the same for ALL -> fair)."""
    x = torch.cat([_to_tensor(f_prev, device), _to_tensor(f_cur, device)], dim=1)
    mu, _ = vae.encode(x)
    return mu


def to_phys(z8_std, mean_t, std_t):
    return z8_std * std_t[:N_SUP] + mean_t[:N_SUP]


def save_gif(frames, path, fps=GIF_FPS):
    from PIL import Image
    if not frames:
        return
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / max(fps, 1)), loop=0)
    print("saved:", path)


# ---------------------------------------------------------------------------
# Dream rollout & COST (telescoping shaping + terminal − fuel, down-weighted velocities)
# ---------------------------------------------------------------------------
def shaping_w(phys):
    """phys (...,8) physical -> weighted potential (...). Velocities DOWN-WEIGHTED (W_VEL<W_POS)."""
    x, y, vx, vy, th = phys[..., 0], phys[..., 1], phys[..., 2], phys[..., 3], phys[..., 4]
    l1, l2 = phys[..., 6].clamp(0, 1), phys[..., 7].clamp(0, 1)
    return (-W_POS * torch.sqrt(x * x + y * y + 1e-8)
            - W_VEL * torch.sqrt(vx * vx + vy * vy + 1e-8)
            - W_ANG * torch.abs(th) + W_LEG * (l1 + l2))


@torch.no_grad()
def dream_rollout(lstm, z0, prim, mean_t, std_t, device):
    """z0 (1,64); prim (N,H) primitive actions. -> phys_traj (N,H+1,8) in physical units."""
    N, H = prim.shape
    z = z0.expand(N, -1).contiguous()
    hidden = lstm.init_hidden(N, device)
    phys = [z[:, :N_SUP] * std_t[:N_SUP] + mean_t[:N_SUP]]
    for k in range(H):
        z, hidden = lstm.step(z, F.one_hot(prim[:, k], N_ACTIONS).float(), hidden)
        phys.append(z[:, :N_SUP] * std_t[:N_SUP] + mean_t[:N_SUP])
    return torch.stack(phys, dim=1)


def dream_value(phys_traj, prim, device):
    """ -> (N,) score = telescoping shaping − FUEL_W·fuel + TERM_W·(legs − crash-penalty)."""
    sh = shaping_w(phys_traj)
    prog = sh[:, -1] - sh[:, 0]
    fc = torch.tensor(FUEL_COST, device=device)
    fuel = fc[prim].sum(dim=1)
    last = phys_traj[:, -1]
    legs = last[:, 6:8].clamp(0, 1).sum(dim=1)
    speed = torch.sqrt(last[:, 2] ** 2 + last[:, 3] ** 2 + 1e-8)
    term = LAND_LEG * legs - LAND_CRASH * torch.relu(speed - SAFE_SPEED)
    return prog - FUEL_W * fuel + TERM_W * term


# ---------------------------------------------------------------------------
# MPC variants
# ---------------------------------------------------------------------------
@torch.no_grad()
def mpc_random(lstm, z0, mean_t, std_t, device, rng):
    """Random shooting: RS_SAMPLES random sequences of length RS_HORIZON. -> (first_action, best_value)."""
    prim = torch.from_numpy(rng.integers(0, N_ACTIONS, size=(RS_SAMPLES, RS_HORIZON))).long().to(device)
    pt = dream_rollout(lstm, z0, prim, mean_t, std_t, device)
    sc = dream_value(pt, prim, device)
    best = int(torch.argmax(sc).item())
    return int(prim[best, 0].item()), float(sc[best].item())


@torch.no_grad()
def pid_nominal_dream(lstm, z0, mean_t, std_t, device):
    """Roll the PID INSIDE the dream -> (nominal macro sequence, dream value)."""
    z = z0.clone()
    hidden = lstm.init_hidden(1, device)
    macro = []
    for _ in range(CEM_HORIZON):
        a = heuristic_control(to_phys(z[0, :N_SUP], mean_t, std_t).cpu().numpy())
        macro.append(a)
        a_oh = F.one_hot(torch.tensor([a], device=device), N_ACTIONS).float()
        for _ in range(CEM_REPEAT):
            z, hidden = lstm.step(z, a_oh, hidden)
    macro = np.array(macro)
    prim = torch.from_numpy(np.repeat(macro, CEM_REPEAT)[None]).long().to(device)
    v = float(dream_value(dream_rollout(lstm, z0, prim, mean_t, std_t, device), prim, device)[0].item())
    return macro, v


@torch.no_grad()
def mpc_cem(lstm, z0, nominal_macro, mean_t, std_t, device, rng):
    """PID-guided Cross-Entropy Method (per-macro-step categorical). -> (first_action, best_value)."""
    K = CEM_HORIZON
    probs = np.full((K, N_ACTIONS), (1.0 - PID_BIAS) / N_ACTIONS)
    probs[np.arange(K), nominal_macro] += PID_BIAS
    best_v, best_a0 = -1e18, int(nominal_macro[0])
    for _ in range(CEM_ITERS):
        cdf = probs.cumsum(axis=1)
        u = rng.random((CEM_SAMPLES, K))
        macro = np.clip((u[:, :, None] >= cdf[None, :, :]).sum(axis=2), 0, N_ACTIONS - 1)
        prim = torch.from_numpy(np.repeat(macro, CEM_REPEAT, axis=1)).long().to(device)
        sc = dream_value(dream_rollout(lstm, z0, prim, mean_t, std_t, device), prim, device).cpu().numpy()
        elite = macro[np.argsort(sc)[-CEM_ELITE:]]
        freq = np.stack([(elite == a).mean(axis=0) for a in range(N_ACTIONS)], axis=1)
        probs = (1.0 - CEM_LR) * probs + CEM_LR * freq
        probs /= probs.sum(axis=1, keepdims=True)
        top = int(np.argmax(sc))
        if sc[top] > best_v:
            best_v, best_a0 = float(sc[top]), int(macro[top, 0])
    return int(np.argmax(probs[0])), best_v


# ---------------------------------------------------------------------------
# Rollout corrector — base policy = PID, the model does a 1-step lookahead improvement
# ---------------------------------------------------------------------------
@torch.no_grad()
def mpc_rollout(lstm, z0, mean_t, std_t, device):
    """For each candidate first action a∈{0..3}: dream "a now, then the PID" for ROLLOUT_HORIZON
    steps and score it. -> (best_first_action, best_value, pid_first_value).
    Only 4 candidates + an in-distribution PID continuation -> no optimizer's curse / exploitation."""
    H = ROLLOUT_HORIZON
    a_pid0 = heuristic_control(to_phys(z0[0, :N_SUP], mean_t, std_t).cpu().numpy())
    vals = np.zeros(N_ACTIONS)
    for a0 in range(N_ACTIONS):
        z = z0.clone()
        hidden = lstm.init_hidden(1, device)
        phys = [z[0, :N_SUP] * std_t[:N_SUP] + mean_t[:N_SUP]]
        acts, a = [], a0
        for _ in range(H):
            z, hidden = lstm.step(z, F.one_hot(torch.tensor([a], device=device), N_ACTIONS).float(), hidden)
            cur = z[0, :N_SUP] * std_t[:N_SUP] + mean_t[:N_SUP]
            phys.append(cur); acts.append(a)
            a = heuristic_control(cur.cpu().numpy())           # continuation = the PID base policy
        traj = torch.stack(phys).unsqueeze(0)                  # (1,H+1,8)
        prim = torch.tensor(acts, device=device).unsqueeze(0)  # (1,H)
        vals[a0] = float(dream_value(traj, prim, device)[0].item())
    best_a = int(np.argmax(vals))
    return best_a, float(vals[best_a]), float(vals[a_pid0])


# ---------------------------------------------------------------------------
# Closed-loop episode
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_episode(controller, env, vae, lstm, mean_t, std_t, device, ep_seed, mpc_rng, record=False):
    obs, _ = env.reset(seed=ep_seed)
    f_cur = resize_frame(env.render())
    f_prev = f_cur
    frames = []
    total_r, fuel, last_r = 0.0, 0.0, 0.0
    n_override, n_mpc = 0, 0

    for _ in range(MAX_STEPS):
        raw = env.render()
        f_cur = resize_frame(raw)
        if record:
            frames.append(raw)
        mu = encode_pair(vae, f_prev, f_cur, device)                  # (1,64)
        phys = to_phys(mu[0, :N_SUP], mean_t, std_t).cpu().numpy()

        if controller == "true_pid":
            a = heuristic_control(obs)
        elif controller == "enc_pid":
            a = heuristic_control(phys)
        elif controller == "mpc_random":
            a, _ = mpc_random(lstm, mu, mean_t, std_t, device, mpc_rng)
        elif controller == "mpc_cem":
            nominal, _ = pid_nominal_dream(lstm, mu, mean_t, std_t, device)
            a, _ = mpc_cem(lstm, mu, nominal, mean_t, std_t, device, mpc_rng)
        elif controller == "rollout":
            a_pid = heuristic_control(phys)
            best_a, v_best, v_pid = mpc_rollout(lstm, mu, mean_t, std_t, device)
            n_mpc += 1
            if v_best > v_pid + ROLLOUT_MARGIN:
                a = best_a
                n_override += int(best_a != a_pid)
            else:
                a = a_pid
        else:
            raise ValueError(controller)

        obs, r, terminated, truncated, _ = env.step(a)
        total_r += r; last_r = r
        fuel += (0.30 if a == 2 else 0.03 if a in (1, 3) else 0.0)
        f_prev = f_cur
        if terminated or truncated:
            break

    return {"return": total_r, "landed": last_r >= 100.0, "crashed": last_r <= -100.0,
            "fuel": fuel, "override_pct": (100.0 * n_override / n_mpc) if n_mpc else 0.0,
            "frames": frames}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = get_device()
    print("device:", device)
    z = np.load(NORM_STATS)
    mean_t = torch.tensor(z["mean"], device=device, dtype=torch.float32)
    std_t = torch.tensor(z["std"], device=device, dtype=torch.float32)
    std8 = np.asarray(z["std"][:N_SUP], np.float64)

    vae = VAE_P1(n_sup=N_SUP, n_img=N_IMG).to(device)
    vae.load_state_dict(torch.load(VAE_CKPT, map_location=device)); vae.eval()
    lstm = LatentPredictor(LATENT_SIZE, N_ACTIONS, HIDDEN, LAYERS).to(device)
    lstm.load_state_dict(torch.load(LSTM_CKPT, map_location=device)); lstm.eval()

    results = {}                       # (wind_tag, controller) -> list of episode dicts
    for wind_tag, enable_wind in WIND_CONDITIONS:
        env = make_env(enable_wind)
        print(f"\n{'='*64}\n  WIND: {wind_tag}\n{'='*64}")
        for c in CONTROLLERS:
            mpc_rng = np.random.default_rng(MPC_SEED)
            print(f"\n--- {c} | {wind_tag} ---")
            eps = []
            for ep in range(N_EPISODES):
                rec = RECORD_GIF and ep == 0
                res = run_episode(c, env, vae, lstm, mean_t, std_t, device, SEED + ep, mpc_rng, record=rec)
                if rec:
                    save_gif(res["frames"], os.path.join(SAVE_DIR, f"ctrl_{wind_tag}_{c}.gif"))
                res["frames"] = []
                eps.append(res)
                tag = "LAND" if res["landed"] else "CRASH" if res["crashed"] else "timeout"
                extra = f" override={res['override_pct']:.0f}%" if c == "rollout" else ""
                print(f"  ep{ep:02d} return={res['return']:8.1f}  {tag:7}{extra}")
            results[(wind_tag, c)] = eps
        env.close()

    # ---- summary ----
    print(f"\n{'='*86}")
    print(f"{'wind':<9}{'controller':<13}{'mean return':>13}{'success %':>11}{'crash %':>9}{'mean fuel':>11}{'override%':>11}")
    print("-" * 86)
    summ = {}
    for wind_tag, _ in WIND_CONDITIONS:
        for c in CONTROLLERS:
            eps = results[(wind_tag, c)]
            R = np.array([e["return"] for e in eps])
            succ = 100.0 * np.mean([e["landed"] for e in eps])
            crash = 100.0 * np.mean([e["crashed"] for e in eps])
            fu = np.mean([e["fuel"] for e in eps])
            ov = np.mean([e["override_pct"] for e in eps])
            summ[(wind_tag, c)] = (R.mean(), succ, crash, fu, ov)
            ov_s = f"{ov:>10.0f}%" if c == "rollout" else f"{'—':>11}"
            print(f"{wind_tag:<9}{c:<13}{R.mean():>13.1f}{succ:>11.0f}{crash:>9.0f}{fu:>11.1f}{ov_s}")
    print("=" * 86)

    # ---- plot: returns per controller, one subplot per wind setting ----
    fig, axes = plt.subplots(1, len(WIND_CONDITIONS), figsize=(7.0 * len(WIND_CONDITIONS), 5.0), squeeze=False)
    for j, (wind_tag, _) in enumerate(WIND_CONDITIONS):
        ax = axes[0][j]
        data = [[e["return"] for e in results[(wind_tag, c)]] for c in CONTROLLERS]
        ax.boxplot(data, labels=CONTROLLERS, showmeans=True)
        ax.axhline(200, color="g", ls="--", lw=1, label="solved (≥200)")
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_title(f"returns | {wind_tag}"); ax.set_ylabel("episode return")
        ax.tick_params(axis="x", rotation=20); ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)
    plt.suptitle("Closed-loop control — model-based vs classical (P1 VAE + LSTM)")
    plt.tight_layout()
    p = os.path.join(SAVE_DIR, "control_returns.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); print("saved:", p)

    np.savez(os.path.join(SAVE_DIR, "control_results.npz"),
             controllers=np.array(CONTROLLERS),
             winds=np.array([w for w, _ in WIND_CONDITIONS]),
             returns=np.array([[[e["return"] for e in results[(w, c)]] for c in CONTROLLERS]
                               for w, _ in WIND_CONDITIONS]),
             landed=np.array([[[e["landed"] for e in results[(w, c)]] for c in CONTROLLERS]
                              for w, _ in WIND_CONDITIONS]))
    print(f"\nsaved figures + control_results.npz -> {SAVE_DIR}")


if __name__ == "__main__":
    main()
