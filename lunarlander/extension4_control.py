"""
extension4_control.py — Επέκταση 4: Σύνδεση με κλασικό έλεγχο (LunarLander).

Closed-loop έλεγχος του ΠΡΑΓΜΑΤΙΚΟΥ LunarLander, ΑΠΟ PIXELS, χρησιμοποιώντας τα δικά σου
ερμηνεύσιμα μοντέλα (baseline VAE encoder + encoded LSTM world-model). Υλοποιεί:

  (A) Ο ENCODER ως STATE ESTIMATOR -> κλασικός PID/heuristic:
      pixels -> VAE.encode -> mu[:8] (φυσική κατάσταση) -> PD controller -> action.
  (B) Το LSTM ως MPC SAFETY SHIELD ("ονειρεύεται" K βήματα):
      receding-horizon: δειγματίζει N ακολουθίες ενεργειών, τις κάνει rollout στον latent
      χώρο, βαθμολογεί με το reward-shaping του LunarLander, εκτελεί την 1η ενέργεια της best.
  (C) DISTURBANCE SIGNAL ("αντιλαμβάνεται εξωτερικό θόρυβο"):
      residual = || encoder(t) − LSTM_pred(t-1 -> t) || σε φυσικές μονάδες. Μεγαλώνει υπό
      εξωτερική διαταραχή (π.χ. ENABLE_WIND=True).

ΣΥΓΚΡΙΣΗ controllers (ίδια seeds επεισοδίων):
   true_pid  : PD πάνω στο ΑΛΗΘΙΝΟ obs            (upper bound)
   enc_pid   : PD πάνω στην ΕΚΤΙΜΗΣΗ του encoder  (ο encoder ως αισθητήρας)
   latent_mpc: MPC πάνω στο world-model           (έλεγχος καθαρά από pixels + LSTM)

Imports από τα canonical modules του lunarlander/. cwd: lunarlander/. Απαιτεί gymnasium[box2d].
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import gymnasium as gym
from dataCollect import resize_frame                 # ΙΔΙΟ frame-pipeline με το training
from vae import VAE
from vae_p1 import VAE_P1
from vae_p2 import VAE_P2
from vae_p3 import VAE_P3
from lstm import LatentPredictor
from loader import load_norm_stats

# ---------------------------------------------------------------------------
# CONFIG — placeholders <...> τα συμπληρώνει το bootstrap patcher (CONFIG_PATHS)
# ---------------------------------------------------------------------------
DATA_ROOT = "<lunarlander-dataset>"
NORM_STATS = os.path.join(DATA_ROOT, "norm_stats.npz")
SAVE_DIR = "/kaggle/working/lunarlander_ext4_control"

LATENT_SIZE, N_SUP, N_IMG = 64, 8, 56
N_ACTIONS, HIDDEN, LAYERS = 4, 64, 2
IMG_H, IMG_W = 80, 120

# --- ΠΟΙΟ μοντέλο οδηγεί τον encoder/MPC (όλα έχουν supervised mu[:8] -> ερμηνεύσιμα) ---
#   baseline: full supervision (πιο πιστό physical encoding)
#   p1: decoupled encoders (πιο robust σε visual θόρυβο)
#   p2: brightness/contrast invariance
#   p3_semi/p3_weak: μειωμένη εποπτεία ταχυτήτων
# Τρέξε με διαφορετικά MODEL για να συγκρίνεις ποια αρχή ελέγχει καλύτερα (ειδικά υπό wind).
MODEL = "p1"
MODEL_REGISTRY = {
    "baseline": (lambda: VAE(latent_size=LATENT_SIZE),       "<lunarlander-baseline-vae>", "<lunarlander-baseline-lstm>"),
    "p1":       (lambda: VAE_P1(n_sup=N_SUP, n_img=N_IMG),    "<lunarlander-p1-vae>",       "<lunarlander-p1-lstm>"),
    "p2":       (lambda: VAE_P2(latent_size=LATENT_SIZE),     "<lunarlander-p2-vae>",       "<lunarlander-p2-lstm>"),
    "p3_semi":  (lambda: VAE_P3(latent_size=LATENT_SIZE),     "<lunarlander-p3-semi-vae>",  "<lunarlander-p3-semi-lstm>"),
    "p3_weak":  (lambda: VAE_P3(latent_size=LATENT_SIZE),     "<lunarlander-p3-weak-vae>",  "<lunarlander-p3-weak-lstm>"),
}

N_EPISODES = 20                  # επεισόδια ανά controller (ίδια seeds για fair σύγκριση)
MAX_STEPS = 400
SEED = 0
ENABLE_WIND = False              # LunarLander-v3 wind -> demo για το disturbance signal
CONTROLLERS = ["true_pid", "enc_pid", "latent_mpc"]

RECORD_GIF = True                # σώσε GIF του 1ου επεισοδίου ανά controller (full-res render)
GIF_FPS = 30

# --- MPC (safety shield): heuristic-guided + first-action enumeration ---
MPC_HORIZON = 5                  # K βήματα "ονείρου" (κοντινός -> λιγότερο compounding error)
MPC_SAMPLES_PER_ACTION = 64      # στοχαστικές "ουρές" ανά immediate action (N = 4 × αυτό)
EPS_EXPLORE = 0.20               # εξερεύνηση στην ουρά (perturbation γύρω από τον heuristic)
FUEL_W = 0.30                    # ποινή καυσίμου
BOUND_W = 50.0                   # ποινή για |x| εκτός ορίων (out-of-frame)
SMOOTH_W = 2.0                   # ποινή εναλλαγής ενέργειας (smoother control)
X_BOUND = 1.0                    # όριο |x| πέρα από το οποίο τιμωρούμε
LAND_W = 50.0                    # bonus για επαφή ποδιών στο τερματικό (anti-hover, υπέρ landing)
MPC_SEED = 0
RUN_DIAGNOSTIC = True            # τύπωσε action-sensitivity check πριν τα επεισόδια

DIM_NAMES = ["x", "y", "vx", "vy", "theta", "omega", "leg1", "leg2"]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_env():
    """LunarLander discrete, rgb_array render. ENABLE_WIND -> εξωτερική διαταραχή (v3)."""
    last_err = None
    for env_id in ("LunarLander-v3", "LunarLander-v2"):
        try:
            kw = dict(render_mode="rgb_array")
            if ENABLE_WIND:
                kw.update(enable_wind=True, wind_power=15.0, turbulence_power=1.5)
            return gym.make(env_id, **kw)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"LunarLander δεν βρέθηκε (pip install 'gymnasium[box2d]'). {last_err}")


# ---------------------------------------------------------------------------
# (A) Κλασικός controller — ΝΤΕΤΕΡΜΙΝΙΣΤΙΚΟ PD heuristic (χωρίς ε-greedy)
# ---------------------------------------------------------------------------
def heuristic_control(s):
    """ s = [x, y, vx, vy, theta, omega, leg1, leg2] (φυσικές μονάδες) -> action ∈ {0,1,2,3}. """
    x, y, vx, vy, theta, omega = float(s[0]), float(s[1]), float(s[2]), float(s[3]), float(s[4]), float(s[5])
    leg1, leg2 = float(s[6]) > 0.5, float(s[7]) > 0.5
    angle_targ = float(np.clip(x * 0.5 + vx * 1.0, -0.4, 0.4))
    hover_targ = 0.55 * abs(x)
    angle_todo = (angle_targ - theta) * 0.5 - omega * 1.0
    hover_todo = (hover_targ - y) * 0.5 - vy * 0.5
    if leg1 or leg2:                                  # επαφή -> σταμάτα γωνία, μόνο μαλακό φρενάρισμα
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
    """stack(prev,cur) -> mu (1,64). Το mu[:8] είναι η εκτιμώμενη φυσική κατάσταση (standardized)."""
    x = torch.cat([_to_tensor(f_prev, device), _to_tensor(f_cur, device)], dim=1)   # (1,6,H,W)
    mu, _ = vae.encode(x)
    return mu


def to_phys(z8_std, mean, std):
    """standardized (...,8) -> φυσικές μονάδες."""
    return z8_std * std[:N_SUP] + mean[:N_SUP]


def save_gif(frames, path, fps=GIF_FPS):
    """frames: λίστα από full-res uint8 (H,W,3) renders -> animated GIF."""
    from PIL import Image
    if not frames:
        print("[gif] no frames to save:", path); return
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / max(fps, 1)), loop=0)
    print("saved:", path)


# ---------------------------------------------------------------------------
# Reward shaping του LunarLander (ως cost-to-go proxy για το MPC)
# ---------------------------------------------------------------------------
def shaping_phys(phys):
    """ phys: (...,8) tensor σε φυσικές μονάδες -> shaping (...). Υψηλότερο = καλύτερο. """
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
    """ Vectorized PD heuristic: phys (N,8) φυσικές μονάδες -> actions (N,) long.
    ΙΔΙΑ λογική με το heuristic_control αλλά batched -> κρατάει τις "ουρές" ON-distribution. """
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
    a = torch.where(main, torch.full_like(a, 2), a)                       # main (προτεραιότητα)
    return a


@torch.no_grad()
def mpc_plan(lstm, z_t, mean_t, std_t, device):
    """ Heuristic-guided MPC: enumerate immediate action ∈ {0..3}, roll out την "ουρά" με τον
    heuristic (ON-distribution) + μικρή εξερεύνηση.
    OBJECTIVE = ΠΡΟΟΔΟΣ (env-style reward = Δshaping): hovering -> 0, κάθοδος προς pad -> θετικό
    (anti-hover). score = Σ Δshaping − fuel − bounds − switch + LAND_W·(legs τερματικά).
    Επιστρέφει το immediate action με το καλύτερο ΜΕΣΟ score (robust σε model exploitation). """
    K, M = MPC_HORIZON, MPC_SAMPLES_PER_ACTION
    N = N_ACTIONS * M
    first = torch.arange(N_ACTIONS, device=device).repeat_interleave(M)   # immediate action ανά candidate
    z = z_t.expand(N, -1).contiguous()
    hidden = lstm.init_hidden(N, device)
    fuel_cost = torch.tensor([0.0, 0.03, 0.30, 0.03], device=device)
    std8, mean8 = std_t[:N_SUP], mean_t[:N_SUP]
    score = torch.zeros(N, device=device)
    prev_shaping = shaping_phys(z[:, :N_SUP] * std8 + mean8)              # shaping_0 (ίδιο σε όλους)
    prev_a, phys = None, z[:, :N_SUP] * std8 + mean8
    for k in range(K):
        if k == 0:
            a = first                                                     # ενέργεια υπό αξιολόγηση
        else:
            a = heuristic_action_batch(phys)                             # ON-distribution ουρά
            explore = torch.rand(N, device=device) < EPS_EXPLORE
            a = torch.where(explore, torch.randint(0, N_ACTIONS, (N,), device=device), a)
        z, hidden = lstm.step(z, F.one_hot(a, N_ACTIONS).float(), hidden)
        phys = z[:, :N_SUP] * std8 + mean8
        sh = shaping_phys(phys)
        step_cost = ((sh - prev_shaping)                                  # ΠΡΟΟΔΟΣ (anti-hover)
                     - FUEL_W * fuel_cost[a]
                     - BOUND_W * torch.relu(torch.abs(phys[:, 0]) - X_BOUND))
        if prev_a is not None:
            step_cost = step_cost - SMOOTH_W * (a != prev_a).float()
        score += step_cost
        prev_shaping, prev_a = sh, a
    score += LAND_W * (phys[:, 6] + phys[:, 7])                           # τερματικό landing bonus (legs)
    score_per_action = score.view(N_ACTIONS, M).mean(dim=1)               # ΜΕΣΟ ανά immediate action
    return int(torch.argmax(score_per_action).item())


# ---------------------------------------------------------------------------
# Διαγνωστικό — πόσο "ακούει" το world-model τις ενέργειες
# ---------------------------------------------------------------------------
@torch.no_grad()
def _seed_z_from_env(vae, env, device, warmup=20):
    """ Τρέξε λίγα βήματα heuristic -> πάρε z_t από ζεύγος frames (airborne). """
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
    """ Από z_t (airborne) -> rollout K βημάτων με ΣΤΑΘΕΡΟ action. Αν 'main' δεν δίνει αισθητά
    μεγαλύτερο Δy/Δvy από 'noop' (ή left/right δεν αλλάζουν x/theta), το μοντέλο δεν 'ακούει'
    τις ενέργειες -> κακό για planning. """
    z0 = _seed_z_from_env(vae, env, device)
    std8, mean8 = std_t[:N_SUP], mean_t[:N_SUP]
    p0 = (z0[0, :N_SUP] * std8 + mean8).cpu().numpy()
    print(f"\n{'='*66}\n  ACTION-SENSITIVITY DIAGNOSTIC (rollout K={K}, σταθερό action)\n{'='*66}")
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
    print("  -> 'main' πρέπει: Δy/Δvy αρκετά > 'noop'.  'left'/'right': να αλλάζουν x & theta.")


# ---------------------------------------------------------------------------
# Closed-loop episode με δοσμένο controller
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_episode(controller, env, vae, lstm, mean_t, std_t, device, ep_seed, record=False):
    obs, _ = env.reset(seed=ep_seed)
    f_cur = resize_frame(env.render())
    f_prev = f_cur                                    # t=0: (f0,f0) -> ταχύτητες ≈ 0 (1 βήμα)
    z_prev, a_prev = None, None
    total_r, fuel, last_r = 0.0, 0.0, 0.0
    dist_log, frames = [], []
    for _ in range(MAX_STEPS):
        raw = env.render()
        f_cur = resize_frame(raw)
        if record:
            frames.append(raw)
        mu = encode_pair(vae, f_prev, f_cur, device)              # (1,64)
        phys_est = to_phys(mu[0, :N_SUP], mean_t, std_t)          # (8,) φυσικές μονάδες

        # --- disturbance signal: encoder(t) vs 1-step LSTM pred από (z_{t-1}, a_{t-1}) ---
        if z_prev is not None:
            a_oh = F.one_hot(torch.tensor([a_prev], device=device), N_ACTIONS).float()
            z_pred, _ = lstm.step(z_prev, a_oh, lstm.init_hidden(1, device))
            resid = torch.norm(to_phys(z_pred[0, :N_SUP], mean_t, std_t) - phys_est).item()
            dist_log.append(resid)

        # --- επιλογή ενέργειας ανά controller ---
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

    landed = last_r >= 100.0                          # gym: +100 land / -100 crash στο τελευταίο βήμα
    crashed = last_r <= -100.0
    return {"return": total_r, "landed": landed, "crashed": crashed,
            "fuel": fuel, "dist": dist_log, "frames": frames}


# ---------------------------------------------------------------------------
# Main — τρέξε όλους τους controllers σε ΙΔΙΑ seeds, σύγκρινε
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
    dist_example = {}                                 # ένα disturbance trace ανά controller
    for c in CONTROLLERS:
        torch.manual_seed(MPC_SEED)                   # ίδια εξερεύνηση MPC ανά controller (reproducible)
        print(f"\n{'='*56}\n  CONTROLLER: {c}\n{'='*56}")
        for ep in range(N_EPISODES):
            rec = RECORD_GIF and ep == 0
            res = run_episode(c, env, vae, lstm, mean_t, std_t, device, SEED + ep, record=rec)
            results[c].append(res)
            if ep == 0:
                dist_example[c] = res["dist"]
                if rec:
                    save_gif(res["frames"], os.path.join(SAVE_DIR, f"ext4_{MODEL}_{c}.gif"))
            res["frames"] = []                        # ελευθέρωσε τη μνήμη (full-res frames)
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

    # ---- plot 1: return distribution ανά controller ----
    plt.figure(figsize=(7, 4.6))
    data = [[r["return"] for r in results[c]] for c in CONTROLLERS]
    plt.boxplot(data, tick_labels=CONTROLLERS, showmeans=True)
    plt.axhline(200, color="g", ls="--", lw=1, label="solved (≥200)")
    plt.axhline(0, color="0.6", lw=0.8)
    plt.ylabel("episode return")
    plt.title(f"Closed-loop control — return ανά controller (model={MODEL}, wind={ENABLE_WIND})")
    plt.grid(alpha=0.3, axis="y"); plt.legend()
    plt.tight_layout()
    p1 = os.path.join(SAVE_DIR, f"ext4_{MODEL}_returns.png")
    plt.savefig(p1, dpi=150, bbox_inches="tight"); plt.close()
    print("saved:", p1)

    # ---- plot 2: disturbance signal (encoder vs LSTM-pred residual) στο 1ο επεισόδιο ----
    plt.figure(figsize=(7.5, 4.2))
    for c in CONTROLLERS:
        d = dist_example.get(c, [])
        if d:
            plt.plot(np.arange(1, len(d) + 1), d, lw=1.4, label=c)
    plt.xlabel("t (step)"); plt.ylabel("‖encoder(t) − LSTM_pred(t)‖  (φυσικές μονάδες)")
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
