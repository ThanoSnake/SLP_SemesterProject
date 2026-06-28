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
      έμπιστο (χαμηλό disturbance residual), (β) το PID-plan προβλέπεται επικίνδυνο, ΚΑΙ
      (γ) το «όνειρο» του MPC είναι σαφώς ασφαλέστερο. Υπό WIND/υψηλό residual -> PID.

  (4) COMPLEMENTARY STATE FILTER. Ο encoder είναι θορυβώδης στις ΤΑΧΥΤΗΤΕΣ. Συνδυάζουμε encoder +
      1-step model-prediction (gated από το residual: υψηλό residual -> εμπιστεύσου τον encoder).
      Βελτιώνει ΚΑΙ το enc_pid ΚΑΙ τον MPC seed. (Ίδια ιδέα με το Kalman fusion του cartpole.)

CONTROLLERS (ίδια seeds):
   enc_pid         : PD στο ΦΙΛΤΡΑΡΙΣΜΕΝΟ encoder-estimate      (encoder ως αισθητήρας + filter)
   guided_mpc_*    : PID-guided CEM shield/corrector με diagnostic modes πάνω στα override gates

Για γρήγορο tuning, τα true_pid / latent_mpc_rand είναι προσωρινά εκτός CONTROLLERS.

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
ENABLE_WIND = False              # grid search πρώτα χωρίς wind· μετά ξανατρέχουμε το καλύτερο με wind
WIND_POWER, TURBULENCE_POWER = 15.0, 1.5
CONTROLLERS = ["enc_pid", "guided_mpc"]

RECORD_GIF = False               # grid search -> μην γράφουμε GIF για κάθε threshold setting
GIF_FPS = 30

# --- Complementary state filter (encoder + model, gated από το residual) ---
USE_FILTER = False               # wind/OOD: μην ανακατεύεις model prediction μέσα στον sensor estimate
#                x     y     vx    vy    theta omega leg1  leg2   (βάρος του ΜΟΝΤΕΛΟΥ ανά dim)
W_MODEL = np.array([0.15, 0.15, 0.50, 0.50, 0.20, 0.45, 0.00, 0.00], dtype=np.float64)
RESID_SCALE0 = 1.0               # αρχικό scale του gate πριν μαζευτεί ιστορικό residual
FILTER_WARMUP = 5                # βήματα πριν ενεργοποιηθεί το adaptive scaling/threshold

# --- Guided MPC (CEM shield/corrector) ---
MPC_HORIZON = 5                  # κοντός ορίζοντας ασφάλειας -> λιγότερο model exploitation
MPC_REPEAT = 1                   # no macro-repeat: κρατάμε τις ουρές κοντά στο training distribution
MPC_SAMPLES = 256                # ακολουθίες ανά CEM iteration
CEM_ITERS = 3                    # iterations του Cross-Entropy Method
CEM_ELITE = 32                   # πλήθος elites
CEM_LR = 0.7                     # refit rate της κατανομής προς τα elites
PID_BIAS = 0.85                  # warm-start: ισχυρή μάζα στο PID-nominal action ανά βήμα
MPC_SEED = 0

# --- Cost weights (tunable· οι τιμές είναι σε «φυσικές μονάδες» του LunarLander shaping) ---
FUEL_W = 0.30                    # ποινή καυσίμου
TERM_W = 1.0                     # βάρος terminal value
LAND_LEG = 20.0                  # bonus ανά leg-contact στο τέλος (proxy του +100 landing)
LAND_CRASH = 100.0               # ποινή για υψηλή ταχύτητα στο τέλος (proxy του −100 crash)
SAFE_SPEED = 0.5                 # «ασφαλής» ταχύτητα προσγείωσης (πάνω από αυτή -> crash risk)

# --- Shield / corrector ---
USE_SHIELD = True
TRUST_FACTOR = 1.2               # distrust όταν residual > TRUST_FACTOR * median(residual)
TRUST_FLOOR = 0.05               # αποφυγή μηδενικού threshold όταν το median είναι πολύ μικρό
TRUST_ABS_MAX = 0.75             # absolute cap σε φυσικές μονάδες· persistent wind δεν γίνεται "normal"
MPC_MIN_HISTORY = 8              # πριν έχουμε residual history, default PID
MPC_MARGIN = 25.0                # override μόνο αν dream-value(MPC) > dream-value(PID) + MARGIN
RISK_TRIGGER = 10.0              # override μόνο αν το PID dream έχει ουσιαστικό predicted risk
RISK_MARGIN = 5.0                # και ο MPC μειώνει το risk τουλάχιστον τόσο

# --- Diagnostics γύρω από το καλύτερο αρχικό setting: "medium" ---
# Κάθε config τρέχει σαν ξεχωριστός "guided_mpc_<label>" controller, στα ίδια episode seeds.
# Τα mode δείχνουν ποιο gate κόβει τα overrides και αν το MPC βοηθά όταν το αφήνουμε πιο ελεύθερο.
GUIDED_GRID = [
    # κανονικό safety shield: όλα τα gates ενεργά
    {"label": "med_full",   "mode": "shield",     "mpc_margin": 5.0, "risk_trigger": 2.0, "risk_margin": 1.0, "pid_bias": 0.75},
    # diagnostic: αγνοεί το risk gate, κρατά μόνο value improvement
    {"label": "value_only", "mode": "value_only", "mpc_margin": 5.0, "risk_trigger": 2.0, "risk_margin": 1.0, "pid_bias": 0.75},
    # diagnostic: αγνοεί το value gate, κρατά μόνο risk improvement
    {"label": "risk_only",  "mode": "risk_only",  "mpc_margin": 5.0, "risk_trigger": 2.0, "risk_margin": 1.0, "pid_bias": 0.75},
    # diagnostic stress test: αν το MPC προτείνει άλλο action, το παίρνουμε
    {"label": "force_diff", "mode": "force_diff", "mpc_margin": 5.0, "risk_trigger": 2.0, "risk_margin": 1.0, "pid_bias": 0.75},
]

FUEL_COST = [0.0, 0.03, 0.30, 0.03]   # ανά action {noop,left,main,right}
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

    def __init__(self, lstm, mean_t, std_t, device, mpc_cfg=None):
        self.lstm, self.mean_t, self.std_t, self.device = lstm, mean_t, std_t, device
        self.mpc_cfg = mpc_cfg
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
        min_history = int(cfg_get(self.mpc_cfg, "mpc_min_history", MPC_MIN_HISTORY))
        if len(self.resid_hist) < min_history:
            return 0.0             # warmup -> όχι MPC, μόνο PID
        med = float(np.median(self.resid_hist))
        trust_factor = float(cfg_get(self.mpc_cfg, "trust_factor", TRUST_FACTOR))
        trust_floor = float(cfg_get(self.mpc_cfg, "trust_floor", TRUST_FLOOR))
        trust_abs_max = float(cfg_get(self.mpc_cfg, "trust_abs_max", TRUST_ABS_MAX))
        return min(trust_abs_max, max(trust_floor, med * trust_factor))

    def model_trusted(self):
        min_history = int(cfg_get(self.mpc_cfg, "mpc_min_history", MPC_MIN_HISTORY))
        return len(self.resid_hist) >= min_history and self.resid <= self.trust_threshold()

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
    """ -> (N,) score. ΣΩΣΤΟ objective:
         telescoping shaping-gain  −  FUEL_W·fuel  +  TERM_W·(leg-bonus − crash-penalty).
       (Το άθροισμα ABSOLUTE shaping του αρχικού ευνοούσε hover· εδώ μετράμε ΚΕΡΔΟΣ potential.)"""
    sh = shaping_phys(phys_traj)                                # (N, T+1)
    prog = sh[:, -1] - sh[:, 0]                                 # potential gain (telescoping)
    fc = torch.tensor(FUEL_COST, device=device)
    fuel = fc[prim_acts].sum(dim=1)                             # (N,)
    last = phys_traj[:, -1]
    legs = last[:, 6:8].clamp(0.0, 1.0).sum(dim=1)
    speed = torch.sqrt(last[:, 2] ** 2 + last[:, 3] ** 2 + 1e-8)
    term = LAND_LEG * legs - LAND_CRASH * torch.relu(speed - SAFE_SPEED)
    return prog - FUEL_W * fuel + TERM_W * term


def dream_risk(phys_traj):
    """Predicted danger proxy. 0≈safe, larger=worse.
    Το χρησιμοποιούμε μόνο ως shield gate, όχι ως reward προς βελτιστοποίηση."""
    x = phys_traj[:, :, 0]
    y = phys_traj[:, :, 1]
    vx = phys_traj[:, :, 2]
    vy = phys_traj[:, :, 3]
    theta = phys_traj[:, :, 4]
    speed = torch.sqrt(vx * vx + vy * vy + 1e-8)
    low_alt = (y < 0.25).float()
    return (
        80.0 * torch.relu(torch.abs(x).max(dim=1).values - 1.0)
        + 80.0 * torch.relu(torch.abs(theta).max(dim=1).values - 0.75)
        + 120.0 * torch.relu((speed * low_alt).max(dim=1).values - SAFE_SPEED)
        + 80.0 * torch.relu((-y).max(dim=1).values)
    )


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
    risk = float(dream_risk(phys_traj)[0].item())
    return np.array(macro), v, risk


@torch.no_grad()
def cem_plan(lstm, z0, nominal_macro, mean_t, std_t, device, rng, pid_bias=PID_BIAS):
    """PID-guided Cross-Entropy Method πάνω σε per-step categorical actions.
       Warm-start γύρω από το PID-nominal -> in-distribution. -> (first_action, best_dream_value)."""
    K = MPC_HORIZON
    probs = np.full((K, N_ACTIONS), (1.0 - pid_bias) / N_ACTIONS)
    probs[np.arange(K), nominal_macro] += pid_bias                      # μάζα στο PID action ανά βήμα
    best_v, best_seq = -1e18, nominal_macro.copy()

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
            best_v, best_seq = float(sc[top]), samples[top].copy()

    # Επιστρέφουμε την 1η ενέργεια της best-valued sequence. Πριν γυρνούσε το mode των
    # refined probabilities, αλλά σύγκρινε με το value άλλης sequence -> ψεύτικα overrides.
    return int(best_seq[0]), best_v, best_seq


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
def run_episode(controller, env, vae, lstm, mean_t, std_t, device, ep_seed,
                mpc_rng, record=False, mpc_cfg=None):
    obs, _ = env.reset(seed=ep_seed)
    f_cur = resize_frame(env.render())
    f_prev = f_cur
    est = StateEstimator(lstm, mean_t, std_t, device, mpc_cfg=mpc_cfg)
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
            if USE_SHIELD and not est.model_trusted():
                a = a_pid                                         # μοντέλο αναξιόπιστο (wind/OOD) -> PID
            else:
                nominal_macro, v_pid, risk_pid = pid_nominal_dream(lstm, fused_mu, mean_t, std_t, device)
                pid_bias = float(cfg_get(mpc_cfg, "pid_bias", PID_BIAS))
                a_mpc, v_mpc, mpc_macro = cem_plan(lstm, fused_mu, nominal_macro, mean_t, std_t,
                                                   device, mpc_rng, pid_bias=pid_bias)
                mpc_t = torch.tensor(mpc_macro, device=device).long().unsqueeze(0)
                phys_mpc, _ = dream_rollout(lstm, fused_mu, mpc_t, mean_t, std_t, device)
                risk_mpc = float(dream_risk(phys_mpc)[0].item())
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
        total_r += r; last_r = r
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
    return {"return": total_r, "landed": landed, "crashed": crashed, "fuel": fuel,
            "dist": dist_log, "frames": frames, "override_pct": override_pct,
            "mpc_attempt_pct": mpc_attempt_pct,
            "diff_action_pct": gate_pct(n_diff_action),
            "risky_pid_pct": gate_pct(n_risky_pid),
            "safer_mpc_pct": gate_pct(n_safer_mpc),
            "better_mpc_pct": gate_pct(n_better_mpc),
            "all_gates_pct": gate_pct(n_all_gates)}


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
    run_specs = []
    for c in CONTROLLERS:
        if c == "guided_mpc":
            for cfg in GUIDED_GRID:
                label = f"guided_mpc_{cfg['label']}"
                run_specs.append({"label": label, "controller": c, "mpc_cfg": cfg})
        else:
            run_specs.append({"label": c, "controller": c, "mpc_cfg": None})

    results = {s["label"]: [] for s in run_specs}
    dist_example = {}
    for spec in run_specs:
        label, c, mpc_cfg = spec["label"], spec["controller"], spec["mpc_cfg"]
        mpc_rng = np.random.default_rng(MPC_SEED)
        print(f"\n{'='*72}\n  CONTROLLER: {label}\n{'='*72}")
        if mpc_cfg:
            print("  config:", "  ".join(f"{k}={v}" for k, v in mpc_cfg.items() if k != "label"))
        for ep in range(N_EPISODES):
            rec = RECORD_GIF and ep == 0
            res = run_episode(c, env, vae, lstm, mean_t, std_t, device, SEED + ep,
                              mpc_rng, record=rec, mpc_cfg=mpc_cfg)
            results[label].append(res)
            if ep == 0:
                dist_example[label] = res["dist"]
                if rec:
                    save_gif(res["frames"], os.path.join(SAVE_DIR, f"ext4alt_{MODEL}_{label}.gif"))
            res["frames"] = []
            extra = (f"  override={res['override_pct']:.0f}%  mpc_used={res['mpc_attempt_pct']:.0f}%"
                     if c == "guided_mpc" else "")
            print(f"  ep{ep:02d}  return={res['return']:8.1f}  "
                  f"{'LAND' if res['landed'] else 'CRASH' if res['crashed'] else 'timeout':6}  "
                  f"fuel={res['fuel']:.1f}{extra}")
    env.close()

    # ---- summary ----
    print(f"\n{'='*104}")
    print(f"{'controller':<22}{'mean return':>13}{'success %':>11}{'crash %':>9}"
          f"{'mean fuel':>11}{'MPC override %':>16}{'MPC used %':>12}")
    print("-" * 104)
    summary = {}
    labels = [s["label"] for s in run_specs]
    for spec in run_specs:
        label, c = spec["label"], spec["controller"]
        R = np.array([r["return"] for r in results[label]])
        succ = 100.0 * np.mean([r["landed"] for r in results[label]])
        crash = 100.0 * np.mean([r["crashed"] for r in results[label]])
        fuel = np.mean([r["fuel"] for r in results[label]])
        ovr = np.mean([r["override_pct"] for r in results[label]])
        used = np.mean([r["mpc_attempt_pct"] for r in results[label]])
        summary[label] = (R.mean(), succ, crash, fuel, ovr, used)
        ovr_s = f"{ovr:>15.1f}%" if c == "guided_mpc" else f"{'—':>16}"
        used_s = f"{used:>11.1f}%" if c == "guided_mpc" else f"{'—':>12}"
        print(f"{label:<22}{R.mean():>13.1f}{succ:>11.0f}{crash:>9.0f}"
              f"{fuel:>11.1f}{ovr_s}{used_s}")
    print("=" * 104)

    guided_labels = [s["label"] for s in run_specs if s["controller"] == "guided_mpc"]
    if guided_labels:
        print(f"\n{'='*104}")
        print("GATE DIAGNOSTICS (% of MPC-used steps)")
        print(f"{'controller':<22}{'diff action':>13}{'risky PID':>12}{'safer MPC':>12}"
              f"{'better MPC':>13}{'all gates':>12}")
        print("-" * 104)
        for label in guided_labels:
            diff = np.mean([r["diff_action_pct"] for r in results[label]])
            risky = np.mean([r["risky_pid_pct"] for r in results[label]])
            safer = np.mean([r["safer_mpc_pct"] for r in results[label]])
            better = np.mean([r["better_mpc_pct"] for r in results[label]])
            allg = np.mean([r["all_gates_pct"] for r in results[label]])
            print(f"{label:<22}{diff:>12.1f}%{risky:>11.1f}%{safer:>11.1f}%"
                  f"{better:>12.1f}%{allg:>11.1f}%")
        print("=" * 104)

    # ---- plot 1: return distribution ----
    plt.figure(figsize=(max(7.6, 1.8 * len(labels)), 4.8))
    data = [[r["return"] for r in results[label]] for label in labels]
    plt.boxplot(data, tick_labels=labels, showmeans=True)
    plt.xticks(rotation=20, ha="right")
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
    for label in labels:
        d = dist_example.get(label, [])
        if d:
            plt.plot(np.arange(1, len(d) + 1), d, lw=1.3, label=label)
    plt.xlabel("t (step)"); plt.ylabel("‖encoder − model‖  (residual· trust gate)")
    plt.title(f"Disturbance / model-trust signal (model={MODEL}, wind={ENABLE_WIND})")
    plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout()
    p2 = os.path.join(SAVE_DIR, f"ext4alt_{MODEL}_disturbance.png")
    plt.savefig(p2, dpi=150, bbox_inches="tight"); plt.close()
    print("saved:", p2)

    np.savez(os.path.join(SAVE_DIR, f"ext4alt_{MODEL}_results.npz"),
             model=MODEL, controllers=np.array(labels),
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
             wind=ENABLE_WIND)
    print(f"\nsaved figures + ext4alt_{MODEL}_results.npz -> {SAVE_DIR}")


if __name__ == "__main__":
    main()
