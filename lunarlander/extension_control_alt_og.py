"""
extension4_control_alt.py — Επέκταση 4 (ΒΕΛΤΙΩΜΕΝΗ, wind-oriented): world-model ΣΥΜΠΛΗΡΩΜΑΤΙΚΑ
στον κλασικό έλεγχο, ΟΧΙ ως αποκλειστικός greedy planner.

ΓΙΑΤΙ άλλαξε σε σχέση με το extension4_control.py (που το latent_mpc «δεν προσγείωνε ποτέ»):
  Το αρχικό latent_mpc απέτυχε για πολλούς λόγους ταυτόχρονα — εδώ διορθώνονται όλοι:

  (1) ΣΩΣΤΟ OBJECTIVE. Το αρχικό άθροιζε ABSOLUTE shaping πάνω στον ορίζοντα -> μεγιστοποιείται
      με HOVER κοντά στο (0,0), ΟΧΙ με προσγείωση. Εδώ χρησιμοποιούμε το ΠΡΑΓΜΑΤΙΚΟ gym objective:
      potential-DIFFERENCE (telescoping) shaping  +  terminal landing/crash value  −  fuel.
      Το hovering πλέον δίνει ~0 shaping-gain και μόνο fuel cost -> αποθαρρύνεται.

  (2) PID-GUIDED CEM (αντί random shooting). Το μοντέλο εκπαιδεύτηκε ΜΟΝΟ σε heuristic+ε=0.2 data
      (dataCollect.py) -> uniform-random ακολουθίες είναι OOD -> «όνειρο» σκουπίδια. Εδώ δειγματίζουμε
      ΓΥΡΩ από τον heuristic (warm-start) και «σφίγγουμε» την κατανομή με Cross-Entropy Method.
      In-distribution rollouts -> ο MPC ψάχνει μόνο ΤΟΠΙΚΕΣ βελτιώσεις. + ACTION-REPEAT για μακρύτερο
      effective horizon (συνεκτικά bursts, που χρειάζεται η προσγείωση).

  (3) MPC ως SHIELD/CORRECTOR. Default = PID· ο MPC κάνει override ΜΟΝΟ όταν (α) το μοντέλο είναι
      έμπιστο (χαμηλό disturbance residual) ΚΑΙ (β) το «όνειρό» του είναι σαφώς καλύτερο από το
      PID-plan. Εγγύηση ≥ PID, αποφυγή model-exploitation. Υπό WIND (υψηλό residual) -> PID.

  (4) COMPLEMENTARY STATE FILTER. Ο encoder είναι θορυβώδης στις ΤΑΧΥΤΗΤΕΣ. Συνδυάζουμε encoder +
      1-step model-prediction (gated από το residual: υψηλό residual -> εμπιστεύσου τον encoder).
      Βελτιώνει ΚΑΙ το enc_pid ΚΑΙ τον MPC seed. (Ίδια ιδέα με το Kalman fusion του cartpole.)

CONTROLLERS (ίδια seeds):
   true_pid        : PD στο ΑΛΗΘΙΝΟ obs                         (upper bound)
   enc_pid         : PD στο ΦΙΛΤΡΑΡΙΣΜΕΝΟ encoder-estimate      (encoder ως αισθητήρας + filter)
   latent_mpc_rand : ΠΑΛΙΟ random-shooting MPC                  (baseline «πριν», για contrast)
   guided_mpc      : PID-guided CEM shield/corrector            (η βελτιωμένη πρόταση)

Imports από τα canonical modules του lunar_lander/. Απαιτεί gymnasium[box2d].
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
# CONFIG — placeholders <...> τα συμπληρώνει το bootstrap patcher (CONFIG_PATHS)
# ---------------------------------------------------------------------------
DATA_ROOT = "<lunarlander-dataset>"
NORM_STATS = os.path.join(DATA_ROOT, "norm_stats.npz")
SAVE_DIR = "/kaggle/working/lunarlander_ext4_control_alt"

LATENT_SIZE, N_SUP, N_IMG = 64, 8, 56
N_ACTIONS, HIDDEN, LAYERS = 4, 64, 2
IMG_H, IMG_W = 80, 120

MODEL = "p1"
MODEL_REGISTRY = {
    "baseline": (lambda: VAE(latent_size=LATENT_SIZE),       "<lunarlander-baseline-vae>", "<lunarlander-baseline-lstm>"),
    "p1":       (lambda: VAE_P1(n_sup=N_SUP, n_img=N_IMG),    "<lunarlander-p1-vae>",       "<lunarlander-p1-lstm>"),
    "p2":       (lambda: VAE_P2(latent_size=LATENT_SIZE),     "<lunarlander-p2-vae>",       "<lunarlander-p2-lstm>"),
    "p3_semi":  (lambda: VAE_P3(latent_size=LATENT_SIZE),     "<lunarlander-p3-semi-vae>",  "<lunarlander-p3-semi-lstm>"),
    "p3_weak":  (lambda: VAE_P3(latent_size=LATENT_SIZE),     "<lunarlander-p3-weak-vae>",  "<lunarlander-p3-weak-lstm>"),
}

N_EPISODES = 20
MAX_STEPS = 400
SEED = 0
ENABLE_WIND = False               # το πείραμα ΕΧΕΙ wind -> εδώ ο corrector/shield έχει νόημα
WIND_POWER, TURBULENCE_POWER = 15.0, 1.5
CONTROLLERS = ["true_pid", "enc_pid", "latent_mpc_rand", "guided_mpc"]

RECORD_GIF = True
GIF_FPS = 30

# --- Complementary state filter (encoder + model, gated από το residual) ---
USE_FILTER = True
#                x     y     vx    vy    theta omega leg1  leg2   (βάρος του ΜΟΝΤΕΛΟΥ ανά dim)
W_MODEL = np.array([0.15, 0.15, 0.50, 0.50, 0.20, 0.45, 0.00, 0.00], dtype=np.float64)
RESID_SCALE0 = 1.0               # αρχικό scale του gate πριν μαζευτεί ιστορικό residual
FILTER_WARMUP = 5                # βήματα πριν ενεργοποιηθεί το adaptive scaling/threshold

# --- Guided MPC (CEM shield/corrector) ---
MPC_HORIZON = 8                  # K macro-βήματα απόφασης
MPC_REPEAT = 3                   # action-repeat -> effective horizon = K*REPEAT (=24)
MPC_SAMPLES = 256                # ακολουθίες ανά CEM iteration
CEM_ITERS = 3                    # iterations του Cross-Entropy Method
CEM_ELITE = 32                   # πλήθος elites
CEM_LR = 0.7                     # refit rate της κατανομής προς τα elites
PID_BIAS = 0.5                   # warm-start: μάζα πιθανότητας στο PID-nominal action ανά βήμα
MPC_SEED = 0

# --- Cost weights (tunable· οι τιμές είναι σε «φυσικές μονάδες» του LunarLander shaping) ---
GAMMA = 0.93                     # discount στο dream rollout: γ^10≈0.48 -> ο αξιόπιστος (≤SEQ_LEN)
                                 # ορίζοντας κυριαρχεί· down-weights τον αναξιόπιστο μακρινό ορίζοντα
                                 # (το LSTM εκπαιδεύεται free-running ~10 βήματα· το MPC ονειρεύεται 24)
FUEL_W = 0.30                    # ποινή καυσίμου
TERM_W = 1.0                     # βάρος terminal value
LAND_CRASH = 100.0               # ποινή για υψηλή ταχύτητα στο τέλος (proxy του −100 crash)
SAFE_SPEED = 0.5                 # «ασφαλής» ταχύτητα προσγείωσης (πάνω από αυτή -> crash risk)
# Landing-funnel terminal σε ΚΑΛΑ προβλέψιμα dims (x,y,θ,speed) — ΟΧΙ predicted legs (το μοντέλο
# δεν τα προβλέπει: W_MODEL[legs]=0 και τα leg-flags είναι ~0 μέχρι την επαφή). Peak + widths^2.
FUNNEL_BONUS = 40.0              # peak terminal bonus (over-pad, upright, αργό) — proxy του +100 landing
FUNNEL_X2, FUNNEL_Y2, FUNNEL_TH2 = 0.25, 0.25, 0.04

# --- Shield / corrector ---
USE_SHIELD = True
TRUST_FACTOR = 2.0               # distrust όταν residual > TRUST_FACTOR * median(residual)
MPC_MARGIN = 5.0                 # override μόνο αν dream-value(MPC) > dream-value(PID) + MARGIN

FUEL_COST = [0.0, 0.03, 0.30, 0.03]   # ανά action {noop,left,main,right}
DIM_NAMES = ["x", "y", "vx", "vy", "theta", "omega", "leg1", "leg2"]


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
    raise RuntimeError(f"LunarLander δεν βρέθηκε (pip install 'gymnasium[box2d]'). {last_err}")


# ---------------------------------------------------------------------------
# (A) Κλασικός controller — ντετερμινιστικό PD heuristic (ίδιο με dataCollect)
# ---------------------------------------------------------------------------
def heuristic_control(s):
    """ s = [x, y, vx, vy, theta, omega, leg1, leg2] (φυσικές μονάδες) -> action ∈ {0,1,2,3}. """
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
# Encoder helpers — pixels -> latent / physical estimate
# ---------------------------------------------------------------------------
def _to_tensor(frame, device):
    t = torch.from_numpy(frame.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return t.to(device)


@torch.no_grad()
def encode_pair(vae, f_prev, f_cur, device):
    x = torch.cat([_to_tensor(f_prev, device), _to_tensor(f_cur, device)], dim=1)
    mu, _ = vae.encode(x)
    return mu


def to_phys(z8_std, mean_t, std_t):
    """standardized (...,8) -> φυσικές μονάδες."""
    return z8_std * std_t[:N_SUP] + mean_t[:N_SUP]


def save_gif(frames, path, fps=GIF_FPS):
    from PIL import Image
    if not frames:
        print("[gif] no frames to save:", path); return
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / max(fps, 1)), loop=0)
    print("saved:", path)


# ---------------------------------------------------------------------------
# Complementary state estimator: encoder + 1-step model prediction (gated)
# ---------------------------------------------------------------------------
class StateEstimator:
    """fused_phys = (1-w_eff)·encoder + w_eff·model_pred, με w_eff = W_MODEL·exp(-residual/scale).
       Υψηλό residual (π.χ. wind/OOD) -> w_eff->0 -> εμπιστεύσου τον encoder. Κρατά ιστορικό
       residual για adaptive scale & trust-threshold (για το shield)."""

    def __init__(self, lstm, mean_t, std_t, device):
        self.lstm, self.mean_t, self.std_t, self.device = lstm, mean_t, std_t, device
        self.w_model = torch.tensor(W_MODEL, device=device, dtype=torch.float32)
        self.reset()

    def reset(self):
        self.z_fused = None       # προηγούμενο fused full latent (για το model predict)
        self.a_prev = None
        self.resid = 0.0
        self.resid_hist = []

    def _scale(self):
        return RESID_SCALE0 if len(self.resid_hist) < FILTER_WARMUP else max(np.median(self.resid_hist), 1e-6)

    def trust_threshold(self):
        if len(self.resid_hist) < FILTER_WARMUP:
            return float("inf")   # warmup -> πάντα έμπιστο (MPC ενεργό)
        return float(np.median(self.resid_hist) * TRUST_FACTOR)

    @torch.no_grad()
    def update(self, mu):
        """mu: (1,64) encoder output. -> fused_mu (1,64) με denoised [:8]."""
        enc_phys = mu[:, :N_SUP]
        if self.z_fused is not None and self.a_prev is not None:
            a_oh = F.one_hot(torch.tensor([self.a_prev], device=self.device), N_ACTIONS).float()
            z_pred, _ = self.lstm.step(self.z_fused, a_oh, self.lstm.init_hidden(1, self.device))
            pred_phys = z_pred[:, :N_SUP]
            self.resid = torch.norm(to_phys(pred_phys[0], self.mean_t, self.std_t)
                                    - to_phys(enc_phys[0], self.mean_t, self.std_t)).item()
            self.resid_hist.append(self.resid)
            if USE_FILTER:
                gate = float(np.exp(-self.resid / self._scale()))
                w = self.w_model * gate                       # (8,)
                fused_phys = (1.0 - w) * enc_phys + w * pred_phys
            else:
                fused_phys = enc_phys
        else:
            self.resid = 0.0
            fused_phys = enc_phys
        fused_mu = mu.clone()
        fused_mu[:, :N_SUP] = fused_phys
        self.z_fused = fused_mu
        return fused_mu

    def set_action(self, a):
        self.a_prev = int(a)


# ---------------------------------------------------------------------------
# Reward shaping (gym LunarLander potential) — clamp legs για ασφάλεια στο dreamed state
# ---------------------------------------------------------------------------
def shaping_phys(phys):
    """phys: (...,8) φυσικές μονάδες -> shaping potential (...). Ίδιοι συντελεστές με το gym."""
    x, y, vx, vy, theta = phys[..., 0], phys[..., 1], phys[..., 2], phys[..., 3], phys[..., 4]
    leg1 = phys[..., 6].clamp(0.0, 1.0)
    leg2 = phys[..., 7].clamp(0.0, 1.0)
    return (-100.0 * torch.sqrt(x * x + y * y + 1e-8)
            - 100.0 * torch.sqrt(vx * vx + vy * vy + 1e-8)
            - 100.0 * torch.abs(theta)
            + 10.0 * leg1 + 10.0 * leg2)


# ---------------------------------------------------------------------------
# Dream rollout & ΣΩΣΤΟ cost (telescoping shaping + terminal landing/crash − fuel)
# ---------------------------------------------------------------------------
@torch.no_grad()
def dream_rollout(lstm, z0, macro_actions, mean_t, std_t, device):
    """z0 (1,64)· macro_actions (N,K). Κάθε macro-action κρατιέται MPC_REPEAT βήματα.
       -> phys_traj (N, K*REPEAT+1, 8), prim_acts (N, K*REPEAT)."""
    N, K = macro_actions.shape
    z = z0.expand(N, -1).contiguous()
    hidden = lstm.init_hidden(N, device)
    phys = [z[:, :N_SUP] * std_t[:N_SUP] + mean_t[:N_SUP]]
    prim = []
    for k in range(K):
        a = macro_actions[:, k]
        a_oh = F.one_hot(a, N_ACTIONS).float()
        for _ in range(MPC_REPEAT):
            z, hidden = lstm.step(z, a_oh, hidden)
            phys.append(z[:, :N_SUP] * std_t[:N_SUP] + mean_t[:N_SUP])
            prim.append(a)
    return torch.stack(phys, dim=1), torch.stack(prim, dim=1)


def dream_value(phys_traj, prim_acts, device):
    """ -> (N,) score. DISCOUNTED gym-faithful objective:
         Σ_t γ^t (Δshaping_t − FUEL_W·fuel_t)  +  γ^T·TERM_W·(landing-funnel − crash).
       Δshaping_t = potential difference per step == gym's shaping reward (telescopes to the
       gym return when γ=1). Discount: το μοντέλο είναι αξιόπιστο ~SEQ_LEN(10) βήματα αλλά το MPC
       ονειρεύεται K·REPEAT(=24) -> down-weight τον αναξιόπιστο μακρινό ορίζοντα (συμβατό με το
       uncertainty work: το σφάλμα συσσωρεύεται με τον ορίζοντα). Terminal σε ΚΑΛΑ προβλέψιμα dims
       (x,y,θ,speed)· ΟΧΙ predicted legs (το αρχικό leg-bonus σχεδόν ποτέ δεν ενεργοποιούνταν στο όνειρο)."""
    sh = shaping_phys(phys_traj)                                # (N, T+1)
    dsh = sh[:, 1:] - sh[:, :-1]                                # (N, T) per-step potential gain (gym shaping reward)
    fc = torch.tensor(FUEL_COST, device=device)
    fuel = fc[prim_acts]                                        # (N, T)
    T = dsh.shape[1]
    disc = (GAMMA ** torch.arange(T, device=device, dtype=dsh.dtype)).unsqueeze(0)   # (1, T)
    step_r = ((dsh - FUEL_W * fuel) * disc).sum(dim=1)          # (N,) discounted per-step return
    last = phys_traj[:, -1]
    x, y = last[:, 0], last[:, 1]
    speed = torch.sqrt(last[:, 2] ** 2 + last[:, 3] ** 2 + 1e-8)
    tilt = last[:, 4].abs()
    funnel = FUNNEL_BONUS * torch.exp(-(x * x / FUNNEL_X2 + y * y / FUNNEL_Y2 + tilt * tilt / FUNNEL_TH2))
    crash = LAND_CRASH * torch.relu(speed - SAFE_SPEED)
    term = (GAMMA ** T) * (funnel - crash)
    return step_r + TERM_W * term


@torch.no_grad()
def pid_nominal_dream(lstm, z0, mean_t, std_t, device):
    """Roll τον PID ΜΕΣΑ στο όνειρο -> nominal macro-ακολουθία + dream-value της (warm-start & shield)."""
    z = z0.clone()
    hidden = lstm.init_hidden(1, device)
    macro = []
    for _ in range(MPC_HORIZON):
        a = heuristic_control(to_phys(z[0, :N_SUP], mean_t, std_t).cpu().numpy())
        macro.append(a)
        a_oh = F.one_hot(torch.tensor([a], device=device), N_ACTIONS).float()
        for _ in range(MPC_REPEAT):
            z, hidden = lstm.step(z, a_oh, hidden)
    macro_t = torch.tensor(macro, device=device).unsqueeze(0)          # (1,K)
    phys_traj, prim = dream_rollout(lstm, z0, macro_t, mean_t, std_t, device)
    v = float(dream_value(phys_traj, prim, device)[0].item())
    return np.array(macro), v


@torch.no_grad()
def cem_plan(lstm, z0, nominal_macro, mean_t, std_t, device, rng):
    """PID-guided Cross-Entropy Method πάνω σε per-step categorical actions.
       Warm-start γύρω από το PID-nominal -> in-distribution. -> (first_action, best_dream_value)."""
    K = MPC_HORIZON
    probs = np.full((K, N_ACTIONS), (1.0 - PID_BIAS) / N_ACTIONS)
    probs[np.arange(K), nominal_macro] += PID_BIAS                      # μάζα στο PID action ανά βήμα
    best_v, best_a0 = -1e18, int(nominal_macro[0])

    for _ in range(CEM_ITERS):
        cdf = probs.cumsum(axis=1)
        u = rng.random((MPC_SAMPLES, K))
        samples = np.clip((u[:, :, None] >= cdf[None, :, :]).sum(axis=2), 0, N_ACTIONS - 1)
        macro_t = torch.from_numpy(samples).long().to(device)
        phys_traj, prim = dream_rollout(lstm, z0, macro_t, mean_t, std_t, device)
        sc = dream_value(phys_traj, prim, device).cpu().numpy()        # (N,)

        elite_idx = np.argsort(sc)[-CEM_ELITE:]
        elite = samples[elite_idx]                                     # (E,K)
        freq = np.stack([(elite == a).mean(axis=0) for a in range(N_ACTIONS)], axis=1)  # (K,4)
        probs = (1.0 - CEM_LR) * probs + CEM_LR * freq
        probs /= probs.sum(axis=1, keepdims=True)

        top = elite_idx[-1]
        if sc[top] > best_v:
            best_v, best_a0 = float(sc[top]), int(samples[top, 0])

    a0 = int(np.argmax(probs[0]))                                      # mode της refined κατανομής
    return a0, best_v


# ---------------------------------------------------------------------------
# ΠΑΛΙΟ random-shooting MPC (baseline «πριν») — absolute shaping, uniform actions
# ---------------------------------------------------------------------------
@torch.no_grad()
def mpc_plan_random(lstm, z_t, mean_t, std_t, device, rng):
    N, K = MPC_SAMPLES, MPC_HORIZON * MPC_REPEAT
    seqs = torch.from_numpy(rng.integers(0, N_ACTIONS, size=(N, K))).to(device)
    z = z_t.expand(N, -1).contiguous()
    hidden = lstm.init_hidden(N, device)
    score = torch.zeros(N, device=device)
    fc = torch.tensor(FUEL_COST, device=device)
    for k in range(K):
        a = seqs[:, k]
        z, hidden = lstm.step(z, F.one_hot(a, N_ACTIONS).float(), hidden)
        phys = z[:, :N_SUP] * std_t[:N_SUP] + mean_t[:N_SUP]
        score += shaping_phys(phys) - FUEL_W * fc[a]               # absolute shaping (το «λάθος»)
    return int(seqs[int(torch.argmax(score).item()), 0].item())


# ---------------------------------------------------------------------------
# Closed-loop episode
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_episode(controller, env, vae, lstm, mean_t, std_t, device, ep_seed, mpc_rng, record=False):
    obs, _ = env.reset(seed=ep_seed)
    f_cur = resize_frame(env.render())
    f_prev = f_cur
    est = StateEstimator(lstm, mean_t, std_t, device)
    total_r, fuel, last_r = 0.0, 0.0, 0.0
    dist_log, frames = [], []
    n_override, n_mpc = 0, 0

    for _ in range(MAX_STEPS):
        raw = env.render()
        f_cur = resize_frame(raw)
        if record:
            frames.append(raw)

        mu = encode_pair(vae, f_prev, f_cur, device)              # (1,64)
        fused_mu = est.update(mu)                                 # filter + disturbance residual
        if est.resid:
            dist_log.append(est.resid)
        phys_est = to_phys(fused_mu[0, :N_SUP], mean_t, std_t).cpu().numpy()

        # --- επιλογή ενέργειας ανά controller ---
        if controller == "true_pid":
            a = heuristic_control(obs)
        elif controller == "enc_pid":
            a = heuristic_control(phys_est)
        elif controller == "latent_mpc_rand":
            a = mpc_plan_random(lstm, fused_mu, mean_t, std_t, device, mpc_rng)
        elif controller == "guided_mpc":
            a_pid = heuristic_control(phys_est)
            if USE_SHIELD and est.resid > est.trust_threshold():
                a = a_pid                                         # μοντέλο αναξιόπιστο (wind/OOD) -> PID
            else:
                nominal_macro, v_pid = pid_nominal_dream(lstm, fused_mu, mean_t, std_t, device)
                a_mpc, v_mpc = cem_plan(lstm, fused_mu, nominal_macro, mean_t, std_t, device, mpc_rng)
                n_mpc += 1
                if v_mpc > v_pid + MPC_MARGIN:                    # override μόνο αν σαφώς καλύτερο
                    a = a_mpc
                    n_override += int(a_mpc != a_pid)
                else:
                    a = a_pid
        else:
            raise ValueError(controller)

        obs, r, terminated, truncated, _ = env.step(a)
        total_r += r; last_r = r
        fuel += (0.30 if a == 2 else 0.03 if a in (1, 3) else 0.0)
        est.set_action(a)
        f_prev = f_cur
        if terminated or truncated:
            break

    landed = last_r >= 100.0
    crashed = last_r <= -100.0
    override_pct = (100.0 * n_override / n_mpc) if n_mpc else 0.0
    return {"return": total_r, "landed": landed, "crashed": crashed, "fuel": fuel,
            "dist": dist_log, "frames": frames, "override_pct": override_pct}


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
    mean_t = torch.tensor(mean_np, device=device)
    std_t = torch.tensor(std_np, device=device)

    vae = make_vae().to(device)
    vae.load_state_dict(torch.load(vae_ckpt, map_location=device)); vae.eval()
    lstm = LatentPredictor(LATENT_SIZE, N_ACTIONS, HIDDEN, LAYERS).to(device)
    lstm.load_state_dict(torch.load(lstm_ckpt, map_location=device)); lstm.eval()

    env = make_env()
    results = {c: [] for c in CONTROLLERS}
    dist_example = {}
    for c in CONTROLLERS:
        mpc_rng = np.random.default_rng(MPC_SEED)
        print(f"\n{'='*56}\n  CONTROLLER: {c}\n{'='*56}")
        for ep in range(N_EPISODES):
            rec = RECORD_GIF and ep == 0
            res = run_episode(c, env, vae, lstm, mean_t, std_t, device, SEED + ep, mpc_rng, record=rec)
            results[c].append(res)
            if ep == 0:
                dist_example[c] = res["dist"]
                if rec:
                    save_gif(res["frames"], os.path.join(SAVE_DIR, f"ext4alt_{MODEL}_{c}.gif"))
            res["frames"] = []
            extra = f"  override={res['override_pct']:.0f}%" if c in ("guided_mpc",) else ""
            print(f"  ep{ep:02d}  return={res['return']:8.1f}  "
                  f"{'LAND' if res['landed'] else 'CRASH' if res['crashed'] else 'timeout':6}  "
                  f"fuel={res['fuel']:.1f}{extra}")
    env.close()

    # ---- summary ----
    print(f"\n{'='*84}")
    print(f"{'controller':<16}{'mean return':>13}{'success %':>11}{'crash %':>9}{'mean fuel':>11}{'MPC override %':>16}")
    print("-" * 84)
    summary = {}
    for c in CONTROLLERS:
        R = np.array([r["return"] for r in results[c]])
        succ = 100.0 * np.mean([r["landed"] for r in results[c]])
        crash = 100.0 * np.mean([r["crashed"] for r in results[c]])
        fuel = np.mean([r["fuel"] for r in results[c]])
        ovr = np.mean([r["override_pct"] for r in results[c]])
        summary[c] = (R.mean(), succ, crash, fuel, ovr)
        ovr_s = f"{ovr:>15.0f}%" if c in ("guided_mpc",) else f"{'—':>16}"
        print(f"{c:<16}{R.mean():>13.1f}{succ:>11.0f}{crash:>9.0f}{fuel:>11.1f}{ovr_s}")
    print("=" * 84)

    # ---- plot 1: return distribution ----
    plt.figure(figsize=(7.6, 4.6))
    data = [[r["return"] for r in results[c]] for c in CONTROLLERS]
    plt.boxplot(data, labels=CONTROLLERS, showmeans=True)
    plt.axhline(200, color="g", ls="--", lw=1, label="solved (≥200)")
    plt.axhline(0, color="0.6", lw=0.8)
    plt.ylabel("episode return")
    plt.title(f"Closed-loop control (ALT) — return ανά controller (model={MODEL}, wind={ENABLE_WIND})")
    plt.grid(alpha=0.3, axis="y"); plt.legend()
    plt.tight_layout()
    p1 = os.path.join(SAVE_DIR, f"ext4alt_{MODEL}_returns.png")
    plt.savefig(p1, dpi=150, bbox_inches="tight"); plt.close()
    print("saved:", p1)

    # ---- plot 2: disturbance signal (now ALSO the trust/gating signal) ----
    plt.figure(figsize=(7.6, 4.2))
    for c in CONTROLLERS:
        d = dist_example.get(c, [])
        if d:
            plt.plot(np.arange(1, len(d) + 1), d, lw=1.3, label=c)
    plt.xlabel("t (step)"); plt.ylabel("‖encoder − model‖  (residual· trust gate)")
    plt.title(f"Disturbance / model-trust signal (model={MODEL}, wind={ENABLE_WIND})")
    plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout()
    p2 = os.path.join(SAVE_DIR, f"ext4alt_{MODEL}_disturbance.png")
    plt.savefig(p2, dpi=150, bbox_inches="tight"); plt.close()
    print("saved:", p2)

    np.savez(os.path.join(SAVE_DIR, f"ext4alt_{MODEL}_results.npz"),
             model=MODEL, controllers=np.array(CONTROLLERS),
             returns=np.array([[r["return"] for r in results[c]] for c in CONTROLLERS]),
             landed=np.array([[r["landed"] for r in results[c]] for c in CONTROLLERS]),
             override_pct=np.array([[r["override_pct"] for r in results[c]] for c in CONTROLLERS]),
             wind=ENABLE_WIND)
    print(f"\nsaved figures + ext4alt_{MODEL}_results.npz -> {SAVE_DIR}")


if __name__ == "__main__":
    main()
