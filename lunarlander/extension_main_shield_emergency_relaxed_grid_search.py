"""
extension_main_shield_emergency_relaxed_grid_search.py — Extension 4: a looser ADDITIVE vertical shield.

WHY: the diagnostic (mpc_model_sanity.py) showed that the world model predicts the main->vy
relation (vertical) RELIABLY, but has ~DEAD action conditioning on the side engines (left/right->vx).
So we do NOT do full MPC. Instead:
  * PID  -> horizontal/angular control (left/right/noop) — it works (enc_pid ~80% landing).
  * MPC  -> ONLY the main engine (vertical), on the 1-D subproblem where the model is trustworthy.

Main control = an additive crash shield: enc_pid stays the default controller. If the PID asks for main,
we always keep it. If it does NOT, the world model tries "start main at step j"
(j=0..K), runs a vertical dream through the LSTM, and may ONLY add an emergency main in
very specific states: PID noop, low altitude, fast descent.

Arbitration: the MPC never suppresses the PID's main and never replaces the side engines.

Imports from the canonical modules; cwd: lunarlander/. Requires gymnasium[box2d].
"""
import os
import csv
from itertools import product

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import gymnasium as gym
from dataCollect import resize_frame
from vae import VAE
from vae_p1 import VAE_P1
from vae_p2 import VAE_P2
from vae_p3 import VAE_P3
from lstm import LatentPredictor
from loader import load_norm_stats

from paths import BASELINE_LSTM, BASELINE_VAE, CONTROL_LSTM, DATA_ROOT, P1_VAE, P2_LSTM, P2_VAE, P3_SEMI_LSTM, P3_SEMI_VAE, P3_WEAK_LSTM, P3_WEAK_VAE, outputs

# ---------------------------------------------------------------------------
# CONFIG  (paths from config.py via paths.py)
# ---------------------------------------------------------------------------
NORM_STATS = os.path.join(DATA_ROOT, "norm_stats.npz")
SAVE_DIR = outputs("lunarlander_ext4_main_shield_emergency_relaxed")

LATENT_SIZE, N_SUP, N_IMG = 64, 8, 56
N_ACTIONS, HIDDEN, LAYERS = 4, 64, 2

MODEL = "p1"
MODEL_REGISTRY = {
    "baseline": (lambda: VAE(latent_size=LATENT_SIZE),    BASELINE_VAE, BASELINE_LSTM),
    "p1":       (lambda: VAE_P1(n_sup=N_SUP, n_img=N_IMG), P1_VAE,       CONTROL_LSTM),
    "p2":       (lambda: VAE_P2(latent_size=LATENT_SIZE),  P2_VAE,       P2_LSTM),
    "p3_semi":  (lambda: VAE_P3(latent_size=LATENT_SIZE),  P3_SEMI_VAE,  P3_SEMI_LSTM),
    "p3_weak":  (lambda: VAE_P3(latent_size=LATENT_SIZE),  P3_WEAK_VAE,  P3_WEAK_LSTM),
}

N_EPISODES = 20
MAX_STEPS = 400
SEED = 0
ENABLE_WIND = True
WIND_POWER, TURBULENCE_POWER = 15.0, 1.5
BASELINE_CONTROLLERS = ["true_pid", "enc_pid"]
SWEEP_CONTROLLER = "emergency_shield_relaxed"
RECORD_GIF = False
GIF_FPS = 30

# --- Vertical MPC (main-only additive shield): default/fallback values ---
VERT_HORIZON = 10                 # <= the train window (the model is reliable for ~10 steps)
Y_GROUND_SCALE = 0.60             # more proactive than 0.40, but still near-ground focused
VERT_FUEL_W = 0.02                # lower fuel penalty so it does not block emergency braking
EMERGENCY_Y_MAX = 1.20            # looser: give main more time before the ground
EMERGENCY_VY_MAX = -0.10          # looser: catch the downward trend earlier
EMERGENCY_COST_MARGIN = 0.00      # accept any predicted gain, since we already have a gate

# --- Grid search around the region that looked best in the previous runs ---
Y_GROUND_SCALE_GRID = [0.60]
VERT_FUEL_W_GRID = [0.02, 0.05]
EMERGENCY_Y_MAX_GRID = [0.90, 1.00, 1.10]
EMERGENCY_VY_MAX_GRID = [-0.25, -0.20, -0.15]
EMERGENCY_COST_MARGIN_GRID = [0.00, 0.01, 0.02, 0.04]
ADD_PCT_TARGET_MAX = 15.0         # above this the shield starts becoming the controller

DIM_NAMES = ["x", "y", "vx", "vy", "theta", "omega", "leg1", "leg2"]


def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
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
    raise RuntimeError(f"LunarLander not found (pip install 'gymnasium[box2d]'). {last_err}")


# ---------------------------------------------------------------------------
# PD heuristic (same as dataCollect) + encoder helpers
# ---------------------------------------------------------------------------
def heuristic_control(s):
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
    return torch.from_numpy(frame.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)


@torch.no_grad()
def encode_pair(vae, f_prev, f_cur, device):
    x = torch.cat([_to_tensor(f_prev, device), _to_tensor(f_cur, device)], dim=1)
    mu, _ = vae.encode(x)
    return mu


def to_phys(z8_std, mean_t, std_t):
    return z8_std * std_t[:N_SUP] + mean_t[:N_SUP]


def save_gif(frames, path, fps=GIF_FPS):
    from PIL import Image
    if not frames:
        print("[gif] no frames:", path); return
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / max(fps, 1)), loop=0)
    print("saved:", path)


# ---------------------------------------------------------------------------
# Vertical MPC (main-only) — additive emergency-main enumeration on the reliable main->vy axis
# ---------------------------------------------------------------------------
@torch.no_grad()
def vertical_main_decision(lstm, z0, mean_t, std_t, device, cfg):
    """ Tries 'start main at step j' (j=0..K; j=K -> never). Vertical dream -> penalty
    Σ w_ground(y)·vy² + fuel. Returns the first action and the improvement over no-main. """
    K = int(cfg["horizon"])
    seqs = torch.zeros(K + 1, K, dtype=torch.long, device=device)        # (K+1, K)
    for j in range(K + 1):
        seqs[j, j:] = 2                                                  # main from j onwards
    N = K + 1
    z = z0.expand(N, -1).contiguous()
    hid = lstm.init_hidden(N, device)
    std8, mean8 = std_t[:N_SUP], mean_t[:N_SUP]
    fc = torch.tensor([0.0, 0.03, 0.30, 0.03], device=device)
    cost = torch.zeros(N, device=device)
    for k in range(K):
        a = seqs[:, k]
        z, hid = lstm.step(z, F.one_hot(a, N_ACTIONS).float(), hid)
        phys = z[:, :N_SUP] * std8 + mean8
        y, vy = phys[:, 1], phys[:, 3]
        w_ground = torch.exp(-torch.relu(y) / cfg["y_ground_scale"])     # ~1 close to the ground
        cost += w_ground * (vy * vy) + cfg["fuel_w"] * fc[a]
    best = int(torch.argmin(cost).item())
    no_main_cost = cost[-1]
    improvement = no_main_cost - cost[best]
    return int(seqs[best, 0].item()), float(improvement.item())          # 2 ή 0, gain


def emergency_gate(phys, cfg):
    y, vy = float(phys[1]), float(phys[3])
    return (
        y < cfg["y_max"]
        and vy < cfg["vy_max"]
    )


# ---------------------------------------------------------------------------
# Closed-loop episode
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_episode(controller, env, vae, lstm, mean_t, std_t, device, ep_seed, record=False, cfg=None):
    obs, _ = env.reset(seed=ep_seed)
    f_cur = resize_frame(env.render()); f_prev = f_cur
    total_r, fuel, last_r = 0.0, 0.0, 0.0
    frames = []
    n_add_main, n_add_opportunities, n_gate_pass, n_pid_main = 0, 0, 0, 0
    for _ in range(MAX_STEPS):
        raw = env.render(); f_cur = resize_frame(raw)
        if record:
            frames.append(raw)
        mu = encode_pair(vae, f_prev, f_cur, device)
        phys = to_phys(mu[0, :N_SUP], mean_t, std_t).cpu().numpy()

        if controller == "true_pid":
            a = heuristic_control(obs)
        elif controller == "enc_pid":
            a = heuristic_control(phys)
        elif controller == SWEEP_CONTROLLER:
            if cfg is None:
                raise ValueError("emergency_shield_relaxed requires a cfg")
            a_pid = heuristic_control(phys)
            if a_pid == 2:
                a = 2                                                     # never suppress the PID's main
                n_pid_main += 1
            elif a_pid in (1, 3):
                a = a_pid                                                 # never override the side engines
            else:
                n_add_opportunities += 1                                  # only a PID noop can become an emergency main
                if not emergency_gate(phys, cfg):
                    a = a_pid                                             # noop, but not a real emergency
                else:
                    n_gate_pass += 1
                    a_vert, gain = vertical_main_decision(lstm, mu, mean_t, std_t, device, cfg)
                    if a_vert == 2 and gain >= cfg["cost_margin"]:
                        a = 2
                        n_add_main += 1
                    else:
                        a = a_pid                                         # keep the PID's noop
        else:
            raise ValueError(controller)

        obs, r, terminated, truncated, _ = env.step(a)
        total_r += r; last_r = r
        fuel += (0.30 if a == 2 else 0.03 if a in (1, 3) else 0.0)
        f_prev = f_cur
        if terminated or truncated:
            break

    landed = last_r >= 100.0; crashed = last_r <= -100.0
    add_pct = (100.0 * n_add_main / n_add_opportunities) if n_add_opportunities else 0.0
    gate_pct = (100.0 * n_gate_pass / n_add_opportunities) if n_add_opportunities else 0.0
    return {"return": total_r, "landed": landed, "crashed": crashed, "fuel": fuel,
            "frames": frames, "add_pct": add_pct, "gate_pct": gate_pct,
            "add_main": n_add_main, "gate_count": n_gate_pass,
            "noop_opportunities": n_add_opportunities, "pid_main": n_pid_main}


# ---------------------------------------------------------------------------
# Grid-search helpers
# ---------------------------------------------------------------------------
def build_sweep_configs():
    configs = []
    for scale, fuel_w, y_max, vy_max, margin in product(
        Y_GROUND_SCALE_GRID,
        VERT_FUEL_W_GRID,
        EMERGENCY_Y_MAX_GRID,
        EMERGENCY_VY_MAX_GRID,
        EMERGENCY_COST_MARGIN_GRID,
    ):
        configs.append({
            "name": f"y={y_max:.2f}|vy={vy_max:.2f}|m={margin:.2f}|fw={fuel_w:.2f}|s={scale:.2f}",
            "horizon": VERT_HORIZON,
            "y_ground_scale": float(scale),
            "fuel_w": float(fuel_w),
            "y_max": float(y_max),
            "vy_max": float(vy_max),
            "cost_margin": float(margin),
        })
    return configs


def summarize_results(name, results, cfg=None):
    returns = np.array([r["return"] for r in results], dtype=np.float32)
    landed = np.array([r["landed"] for r in results], dtype=bool)
    crashed = np.array([r["crashed"] for r in results], dtype=bool)
    fuel = np.array([r["fuel"] for r in results], dtype=np.float32)
    add_count = int(sum(r.get("add_main", 0) for r in results))
    gate_count = int(sum(r.get("gate_count", 0) for r in results))
    noop_count = int(sum(r.get("noop_opportunities", 0) for r in results))
    pid_main_count = int(sum(r.get("pid_main", 0) for r in results))
    row = {
        "name": name,
        "mean_return": float(returns.mean()),
        "median_return": float(np.median(returns)),
        "success_pct": float(100.0 * landed.mean()),
        "crash_pct": float(100.0 * crashed.mean()),
        "mean_fuel": float(fuel.mean()),
        "add_pct": float(100.0 * add_count / noop_count) if noop_count else 0.0,
        "gate_pct": float(100.0 * gate_count / noop_count) if noop_count else 0.0,
        "add_count": add_count,
        "gate_count": gate_count,
        "noop_count": noop_count,
        "pid_main_count": pid_main_count,
        "returns": returns,
    }
    if cfg is not None:
        row.update(cfg)
    else:
        row.update({
            "horizon": 0,
            "y_ground_scale": np.nan,
            "fuel_w": np.nan,
            "y_max": np.nan,
            "vy_max": np.nan,
            "cost_margin": np.nan,
        })
    return row


def run_many(controller, env, vae, lstm, mean_t, std_t, device, cfg=None):
    results = []
    for ep in range(N_EPISODES):
        res = run_episode(
            controller, env, vae, lstm, mean_t, std_t, device,
            SEED + ep, record=False, cfg=cfg,
        )
        res["frames"] = []
        results.append(res)
    return results


def print_header(title):
    print(f"\n{title}")
    print("=" * 144)
    print(
        f"{'#':>3} {'config':<48}{'mean':>9}{'median':>9}{'succ%':>8}{'crash%':>8}"
        f"{'fuel':>8}{'add%':>8}{'gate%':>8}{'add#':>8}{'gate#':>8}{'noop#':>8}"
    )
    print("-" * 144)


def print_row(row, idx=None):
    left = "-" if idx is None else str(idx)
    print(
        f"{left:>3} {row['name']:<48}{row['mean_return']:>9.1f}{row['median_return']:>9.1f}"
        f"{row['success_pct']:>8.0f}{row['crash_pct']:>8.0f}{row['mean_fuel']:>8.1f}"
        f"{row['add_pct']:>8.1f}{row['gate_pct']:>8.1f}{row['add_count']:>8}"
        f"{row['gate_count']:>8}{row['noop_count']:>8}"
    )


def save_sweep_outputs(rows, baseline_rows):
    csv_path = os.path.join(SAVE_DIR, f"emsr_{MODEL}_sweep_results.csv")
    keys = [
        "kind", "rank", "name", "mean_return", "median_return", "success_pct", "crash_pct",
        "mean_fuel", "add_pct", "gate_pct", "add_count", "gate_count", "noop_count",
        "pid_main_count", "horizon", "y_ground_scale", "fuel_w", "y_max", "vy_max",
        "cost_margin", "delta_return_vs_enc", "qualified",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in baseline_rows + rows:
            writer.writerow({k: row.get(k, "") for k in keys})
    print("saved:", csv_path)

    npz_path = os.path.join(SAVE_DIR, f"emsr_{MODEL}_sweep_results.npz")
    np.savez(
        npz_path,
        names=np.array([r["name"] for r in rows]),
        mean_return=np.array([r["mean_return"] for r in rows], dtype=np.float32),
        median_return=np.array([r["median_return"] for r in rows], dtype=np.float32),
        success_pct=np.array([r["success_pct"] for r in rows], dtype=np.float32),
        crash_pct=np.array([r["crash_pct"] for r in rows], dtype=np.float32),
        mean_fuel=np.array([r["mean_fuel"] for r in rows], dtype=np.float32),
        add_pct=np.array([r["add_pct"] for r in rows], dtype=np.float32),
        gate_pct=np.array([r["gate_pct"] for r in rows], dtype=np.float32),
        add_count=np.array([r["add_count"] for r in rows], dtype=np.int32),
        gate_count=np.array([r["gate_count"] for r in rows], dtype=np.int32),
        noop_count=np.array([r["noop_count"] for r in rows], dtype=np.int32),
        y_ground_scale=np.array([r["y_ground_scale"] for r in rows], dtype=np.float32),
        fuel_w=np.array([r["fuel_w"] for r in rows], dtype=np.float32),
        y_max=np.array([r["y_max"] for r in rows], dtype=np.float32),
        vy_max=np.array([r["vy_max"] for r in rows], dtype=np.float32),
        cost_margin=np.array([r["cost_margin"] for r in rows], dtype=np.float32),
        qualified=np.array([r["qualified"] for r in rows], dtype=bool),
        baseline_names=np.array([r["name"] for r in baseline_rows]),
        baseline_mean_return=np.array([r["mean_return"] for r in baseline_rows], dtype=np.float32),
        baseline_success_pct=np.array([r["success_pct"] for r in baseline_rows], dtype=np.float32),
    )
    print("saved:", npz_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = get_device()
    assert MODEL in MODEL_REGISTRY, f"MODEL ∈ {list(MODEL_REGISTRY)}"
    make_vae, vae_ckpt, lstm_ckpt = MODEL_REGISTRY[MODEL]
    print("device:", device, "| model:", MODEL, "| wind:", ENABLE_WIND)
    print(f"episodes/config: {N_EPISODES} | grid configs: {len(build_sweep_configs())} | gifs: {RECORD_GIF}")

    mean_np, std_np = load_norm_stats(NORM_STATS)
    mean_t = torch.tensor(mean_np, device=device)
    std_t = torch.tensor(std_np, device=device)

    vae = make_vae().to(device)
    vae.load_state_dict(torch.load(vae_ckpt, map_location=device))
    vae.eval()
    lstm = LatentPredictor(LATENT_SIZE, N_ACTIONS, HIDDEN, LAYERS).to(device)
    lstm.load_state_dict(torch.load(lstm_ckpt, map_location=device))
    lstm.eval()

    env = make_env()

    baseline_rows = []
    print_header("BASELINES")
    for i, controller in enumerate(BASELINE_CONTROLLERS, 1):
        results = run_many(controller, env, vae, lstm, mean_t, std_t, device)
        row = summarize_results(controller, results)
        row["kind"] = "baseline"
        row["rank"] = i
        row["delta_return_vs_enc"] = 0.0
        row["qualified"] = False
        baseline_rows.append(row)
        print_row(row, i)

    enc_row = next((r for r in baseline_rows if r["name"] == "enc_pid"), baseline_rows[-1])
    enc_return = enc_row["mean_return"]
    enc_success = enc_row["success_pct"]

    sweep_rows = []
    configs = build_sweep_configs()
    print_header("GRID SEARCH")
    for i, cfg in enumerate(configs, 1):
        results = run_many(SWEEP_CONTROLLER, env, vae, lstm, mean_t, std_t, device, cfg=cfg)
        row = summarize_results(cfg["name"], results, cfg=cfg)
        row["kind"] = "sweep"
        row["rank"] = i
        row["delta_return_vs_enc"] = row["mean_return"] - enc_return
        row["qualified"] = bool(row["success_pct"] >= enc_success and row["add_pct"] <= ADD_PCT_TARGET_MAX)
        sweep_rows.append(row)
        print_row(row, i)

    env.close()

    ranked = sorted(sweep_rows, key=lambda r: r["mean_return"], reverse=True)
    qualified = [r for r in ranked if r["qualified"]]

    print_header("TOP 10 BY MEAN RETURN")
    for i, row in enumerate(ranked[:10], 1):
        print_row(row, i)

    print_header(f"TOP QUALIFIED (success >= enc_pid {enc_success:.0f}%, add% <= {ADD_PCT_TARGET_MAX:.0f})")
    if qualified:
        for i, row in enumerate(qualified[:10], 1):
            print_row(row, i)
    else:
        print("No qualified config found. Use TOP 10 BY MEAN RETURN and inspect add%/fuel/crash.")

    best = qualified[0] if qualified else ranked[0]
    print("\nBEST CONFIG:")
    print_row(best, 1)
    print(
        f"\nΔmean_return vs enc_pid = {best['delta_return_vs_enc']:+.1f} | "
        f"enc_pid mean={enc_return:.1f}, success={enc_success:.0f}%"
    )

    save_sweep_outputs(sweep_rows, baseline_rows)
    print(f"\nsaved -> {SAVE_DIR}")


if __name__ == "__main__":
    main()
