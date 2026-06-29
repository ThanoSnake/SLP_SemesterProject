"""
extension_main_shield_emergency_relaxed.py — Επέκταση 4: πιο χαλαρή ΠΡΟΣΘΕΤΗ κάθετη ασπίδα.

ΓΙΑΤΙ: το διαγνωστικό (mpc_model_sanity.py) έδειξε ότι το world-model προβλέπει ΑΞΙΟΠΙΣΤΑ τη
σχέση main→vy (κάθετα), αλλά έχει ~ΝΕΚΡΟ action-conditioning στα πλευρικά engines (left/right→vx).
Άρα ΔΕΝ κάνουμε full MPC. Αντ' αυτού:
  * PID  -> οριζόντιος/γωνιακός έλεγχος (left/right/noop) — δουλεύει (enc_pid ~80% landing).
  * MPC  -> ΜΟΝΟ το main (κάθετο), στο 1-D υποπρόβλημα όπου το μοντέλο είναι έμπιστο.

Έλεγχος main = additive crash shield: ο enc_pid παραμένει ο default ελεγκτής. Αν ο PID ζητά main,
το κρατάμε πάντα. Αν ΔΕΝ ζητά main, τότε το world-model δοκιμάζει «ξεκίνα main από βήμα j»
(j=0..K), κάνει vertical dream στο LSTM, και μπορεί ΜΟΝΟ να προσθέσει emergency main σε
πολύ συγκεκριμένες καταστάσεις: PID noop, χαμηλά, γρήγορη κάθοδος.

Arbitration: το MPC δεν καταστέλλει ποτέ main του PID και δεν αντικαθιστά ποτέ side engines.

Imports από canonical modules· cwd: lunarlander/. Απαιτεί gymnasium[box2d].
"""
import os
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

# ---------------------------------------------------------------------------
# CONFIG — placeholders <...> τα συμπληρώνει ο patcher
# ---------------------------------------------------------------------------
DATA_ROOT = "<lunarlander-dataset>"
NORM_STATS = os.path.join(DATA_ROOT, "norm_stats.npz")
SAVE_DIR = "/kaggle/working/lunarlander_ext4_main_shield_emergency_relaxed"

LATENT_SIZE, N_SUP, N_IMG = 64, 8, 56
N_ACTIONS, HIDDEN, LAYERS = 4, 64, 2

MODEL = "p1"
MODEL_REGISTRY = {
    "baseline": (lambda: VAE(latent_size=LATENT_SIZE),    "<lunarlander-baseline-vae>", "<lunarlander-baseline-lstm>"),
    "p1":       (lambda: VAE_P1(n_sup=N_SUP, n_img=N_IMG), "<lunarlander-p1-vae>",       "<lunarlander-p1-lstm>"),
    "p2":       (lambda: VAE_P2(latent_size=LATENT_SIZE),  "<lunarlander-p2-vae>",       "<lunarlander-p2-lstm>"),
    "p3_semi":  (lambda: VAE_P3(latent_size=LATENT_SIZE),  "<lunarlander-p3-semi-vae>",  "<lunarlander-p3-semi-lstm>"),
    "p3_weak":  (lambda: VAE_P3(latent_size=LATENT_SIZE),  "<lunarlander-p3-weak-vae>",  "<lunarlander-p3-weak-lstm>"),
}

N_EPISODES = 20
MAX_STEPS = 400
SEED = 0
ENABLE_WIND = True
WIND_POWER, TURBULENCE_POWER = 15.0, 1.5
CONTROLLERS = ["true_pid", "enc_pid", "emergency_shield_relaxed"]
RECORD_GIF = True
GIF_FPS = 30

# --- Vertical MPC (main-only additive shield) ---
VERT_HORIZON = 10                 # ≤ train window (το μοντέλο είναι αξιόπιστο ~10 βήματα)
Y_GROUND_SCALE = 0.60             # πιο proactive από 0.40, αλλά ακόμα near-ground focused
VERT_FUEL_W = 0.02                # χαμηλότερη ποινή fuel ώστε να μη μπλοκάρει emergency braking
EMERGENCY_Y_MAX = 1.20            # πιο χαλαρό: δώσε περισσότερο χρόνο στο main πριν το έδαφος
EMERGENCY_VY_MAX = -0.10          # πιο χαλαρό: πιάσε πιο νωρίς καθοδική τάση
EMERGENCY_COST_MARGIN = 0.00      # δέξου οποιοδήποτε predicted gain, αφού ήδη έχουμε gate

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
    raise RuntimeError(f"LunarLander δεν βρέθηκε (pip install 'gymnasium[box2d]'). {last_err}")


# ---------------------------------------------------------------------------
# PD heuristic (ίδιο με dataCollect) + encoder helpers
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
# Vertical MPC (main-only) — additive emergency-main enumeration στο αξιόπιστο main→vy
# ---------------------------------------------------------------------------
@torch.no_grad()
def vertical_main_decision(lstm, z0, mean_t, std_t, device):
    """ Δοκιμάζει 'ξεκίνα main από βήμα j' (j=0..K· j=K -> ποτέ). Vertical dream -> ποινή
    Σ w_ground(y)·vy² + fuel. Επιστρέφει 1η ενέργεια και improvement έναντι no-main. """
    K = VERT_HORIZON
    seqs = torch.zeros(K + 1, K, dtype=torch.long, device=device)        # (K+1, K)
    for j in range(K + 1):
        seqs[j, j:] = 2                                                  # main από j και μετά
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
        w_ground = torch.exp(-torch.relu(y) / Y_GROUND_SCALE)            # ~1 κοντά στο έδαφος
        cost += w_ground * (vy * vy) + VERT_FUEL_W * fc[a]
    best = int(torch.argmin(cost).item())
    no_main_cost = cost[-1]
    improvement = no_main_cost - cost[best]
    return int(seqs[best, 0].item()), float(improvement.item())          # 2 ή 0, gain


def emergency_gate(phys):
    y, vy = float(phys[1]), float(phys[3])
    return (
        y < EMERGENCY_Y_MAX
        and vy < EMERGENCY_VY_MAX
    )


# ---------------------------------------------------------------------------
# Closed-loop episode
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_episode(controller, env, vae, lstm, mean_t, std_t, device, ep_seed, record=False):
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
        elif controller == "emergency_shield_relaxed":
            a_pid = heuristic_control(phys)
            if a_pid == 2:
                a = 2                                                     # ποτέ suppress PID-main
                n_pid_main += 1
            elif a_pid in (1, 3):
                a = a_pid                                                 # ποτέ override σε side engines
            else:
                n_add_opportunities += 1                                  # μόνο PID-noop μπορεί να γίνει emergency main
                if not emergency_gate(phys):
                    a = a_pid                                             # noop, αλλά όχι πραγματικό emergency
                else:
                    n_gate_pass += 1
                    a_vert, gain = vertical_main_decision(lstm, mu, mean_t, std_t, device)
                    if a_vert == 2 and gain >= EMERGENCY_COST_MARGIN:
                        a = 2
                        n_add_main += 1
                    else:
                        a = a_pid                                         # κράτα noop του PID
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
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = get_device()
    assert MODEL in MODEL_REGISTRY, f"MODEL ∈ {list(MODEL_REGISTRY)}"
    make_vae, vae_ckpt, lstm_ckpt = MODEL_REGISTRY[MODEL]
    print("device:", device, "| model:", MODEL, "| wind:", ENABLE_WIND)

    mean_np, std_np = load_norm_stats(NORM_STATS)
    mean_t = torch.tensor(mean_np, device=device); std_t = torch.tensor(std_np, device=device)

    vae = make_vae().to(device)
    vae.load_state_dict(torch.load(vae_ckpt, map_location=device)); vae.eval()
    lstm = LatentPredictor(LATENT_SIZE, N_ACTIONS, HIDDEN, LAYERS).to(device)
    lstm.load_state_dict(torch.load(lstm_ckpt, map_location=device)); lstm.eval()

    env = make_env()
    results = {c: [] for c in CONTROLLERS}
    for c in CONTROLLERS:
        print(f"\n{'='*56}\n  CONTROLLER: {c}\n{'='*56}")
        for ep in range(N_EPISODES):
            rec = RECORD_GIF and ep == 0
            res = run_episode(c, env, vae, lstm, mean_t, std_t, device, SEED + ep, record=rec)
            results[c].append(res)
            if rec:
                save_gif(res["frames"], os.path.join(SAVE_DIR, f"emsr_{MODEL}_{c}.gif"))
            res["frames"] = []
            extra = (
                f"  add_main={res['add_pct']:.0f}% ({res['add_main']}/{res['noop_opportunities']})"
                f" gate={res['gate_pct']:.0f}% ({res['gate_count']}/{res['noop_opportunities']})"
                f" pid_main={res['pid_main']}"
            ) if c == "emergency_shield_relaxed" else ""
            print(f"  ep{ep:02d}  return={res['return']:8.1f}  "
                  f"{'LAND' if res['landed'] else 'CRASH' if res['crashed'] else 'timeout':6}  fuel={res['fuel']:.1f}{extra}")
    env.close()

    print(f"\n{'='*118}")
    print(f"{'controller':<24}{'mean return':>13}{'success %':>11}{'crash %':>10}{'mean fuel':>11}{'add %':>8}{'gate %':>8}{'add#':>8}{'gate#':>8}{'noop#':>8}")
    print("-" * 118)
    for c in CONTROLLERS:
        R = np.array([r["return"] for r in results[c]])
        succ = 100.0 * np.mean([r["landed"] for r in results[c]])
        crash = 100.0 * np.mean([r["crashed"] for r in results[c]])
        fuelm = np.mean([r["fuel"] for r in results[c]])
        mpc = np.mean([r["add_pct"] for r in results[c]]) if c == "emergency_shield_relaxed" else 0.0
        gate = np.mean([r["gate_pct"] for r in results[c]]) if c == "emergency_shield_relaxed" else 0.0
        add_n = np.mean([r.get("add_main", 0) for r in results[c]]) if c == "emergency_shield_relaxed" else 0.0
        gate_n = np.mean([r.get("gate_count", 0) for r in results[c]]) if c == "emergency_shield_relaxed" else 0.0
        noop_n = np.mean([r.get("noop_opportunities", 0) for r in results[c]]) if c == "emergency_shield_relaxed" else 0.0
        print(f"{c:<24}{R.mean():>13.1f}{succ:>11.0f}{crash:>10.0f}{fuelm:>11.1f}{mpc:>8.0f}{gate:>8.0f}{add_n:>8.1f}{gate_n:>8.1f}{noop_n:>8.1f}")
    print("=" * 118)

    plt.figure(figsize=(7, 4.6))
    plt.boxplot([[r["return"] for r in results[c]] for c in CONTROLLERS], tick_labels=CONTROLLERS, showmeans=True)
    plt.axhline(200, color="g", ls="--", lw=1, label="solved (≥200)"); plt.axhline(0, color="0.6", lw=0.8)
    plt.ylabel("episode return"); plt.title(f"Relaxed emergency main-shield — return (model={MODEL}, wind={ENABLE_WIND})")
    plt.grid(alpha=0.3, axis="y"); plt.legend(); plt.tight_layout()
    p = os.path.join(SAVE_DIR, f"emsr_{MODEL}_returns.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close(); print("saved:", p)

    np.savez(os.path.join(SAVE_DIR, f"emsr_{MODEL}_results.npz"),
             controllers=np.array(CONTROLLERS),
             returns=np.array([[r["return"] for r in results[c]] for c in CONTROLLERS]),
             landed=np.array([[r["landed"] for r in results[c]] for c in CONTROLLERS]),
             crashed=np.array([[r["crashed"] for r in results[c]] for c in CONTROLLERS]),
             fuel=np.array([[r["fuel"] for r in results[c]] for c in CONTROLLERS]),
             add_pct=np.array([[r.get("add_pct", 0.0) for r in results[c]] for c in CONTROLLERS]),
             gate_pct=np.array([[r.get("gate_pct", 0.0) for r in results[c]] for c in CONTROLLERS]),
             add_count=np.array([[r.get("add_main", 0) for r in results[c]] for c in CONTROLLERS]),
             gate_count=np.array([[r.get("gate_count", 0) for r in results[c]] for c in CONTROLLERS]),
             noop_opportunities=np.array([[r.get("noop_opportunities", 0) for r in results[c]] for c in CONTROLLERS]),
             pid_main_count=np.array([[r.get("pid_main", 0) for r in results[c]] for c in CONTROLLERS]))
    print(f"\nsaved -> {SAVE_DIR}")


if __name__ == "__main__":
    main()
  
