"""
validate_cost_function.py — Does the MPC `dream_value` cost rank trajectories like the TRUE gym return?

A cost function is only a good MPC objective if, on RECORDED trajectories, it agrees with the
actual episode return and separates landings from crashes. This script computes the proposed cost
on the recorded (physical) states+actions — replay, NOT dreamed, so no VAE/LSTM needed — and
compares it against the saved per-step `rewards` / `episode_return` / terminal outcome.

What it validates:
  * the SHAPING + FUEL + TERMINAL structure reconstructs the gym return  (undiscounted cost ~ return)
  * the funnel terminal separates landings (+100) from crashes (-100)
  * (reference) the discounted cost — the actual MPC objective — still ranks consistently

What it does NOT validate: the leg->funnel terminal swap is justified by the *model's* inability to
predict legs in a DREAM (W_MODEL[legs]=0); on recorded data the legs are real, so both terminals
work there. The swap is a dream-robustness choice, not a recorded-data-fidelity one.

Reads the control + elite datasets (raw .npz). Env-overridable roots (same as lstm_p1_control.py):
  CONTROL_ROOT=~/lunarlander_control_data
  ELITE_ROOT=<this_dir>/lunarlander_elite_recovery_4000
Run:  python lunarlander/validate_cost_function.py
"""
import os
import glob

import numpy as np

# --- cost params — keep in sync with extension_control_alt_og.py ---
FUEL_COST = np.array([0.0, 0.03, 0.30, 0.03])     # noop, left, main, right
FUEL_W = 0.30
SAFE_SPEED = 0.5
LAND_CRASH = 100.0
FUNNEL_BONUS = 40.0
FUNNEL_X2, FUNNEL_Y2, FUNNEL_TH2 = 0.25, 0.25, 0.04
TERM_W = 1.0
GAMMA = 0.93
LAND_LEG = 20.0                                    # for the OLD leg-terminal (reference only)

HERE = os.path.dirname(os.path.abspath(__file__))
CONTROL_ROOT = os.environ.get("CONTROL_ROOT", os.path.expanduser("~/lunarlander_control_data"))
ELITE_ROOT = os.environ.get("ELITE_ROOT", os.path.join(HERE, "lunarlander_elite_recovery_4000"))
MAX_EPISODES = int(os.environ.get("MAX_EPISODES", "0"))   # 0 = all


def shaping(s):
    """s (...,8) physical -> gym shaping potential (...). Same coefficients as the env."""
    x, y, vx, vy, th = s[..., 0], s[..., 1], s[..., 2], s[..., 3], s[..., 4]
    leg1 = np.clip(s[..., 6], 0.0, 1.0)
    leg2 = np.clip(s[..., 7], 0.0, 1.0)
    return (-100.0 * np.sqrt(x * x + y * y + 1e-8)
            - 100.0 * np.sqrt(vx * vx + vy * vy + 1e-8)
            - 100.0 * np.abs(th) + 10.0 * leg1 + 10.0 * leg2)


def _funnel_terminal(last):
    speed = np.sqrt(last[2] ** 2 + last[3] ** 2 + 1e-8)
    tilt = abs(last[4])
    funnel = FUNNEL_BONUS * np.exp(-(last[0] ** 2 / FUNNEL_X2 + last[1] ** 2 / FUNNEL_Y2 + tilt ** 2 / FUNNEL_TH2))
    crash = LAND_CRASH * max(speed - SAFE_SPEED, 0.0)
    return funnel - crash


def _leg_terminal(last):
    legs = float(np.clip(last[6], 0, 1) + np.clip(last[7], 0, 1))
    speed = np.sqrt(last[2] ** 2 + last[3] ** 2 + 1e-8)
    return LAND_LEG * legs - LAND_CRASH * max(speed - SAFE_SPEED, 0.0)


def cost_on_episode(states, acts, gamma=1.0, terminal=_funnel_terminal):
    """Replay cost over a recorded episode. states (T,8) physical, acts (T,)."""
    sh = shaping(states)                              # (T,)
    dsh = sh[1:] - sh[:-1]                            # (T-1,)
    n = len(dsh)
    a = acts[:n].astype(int)
    fuel = FUEL_COST[a]
    disc = gamma ** np.arange(n)
    step_r = float(((dsh - FUEL_W * fuel) * disc).sum())
    term = (gamma ** n) * terminal(states[-1])
    return step_r + TERM_W * term


def _outcome(term_reward, total_return):
    if not np.isnan(term_reward):
        if term_reward >= 50:
            return "land"
        if term_reward <= -50:
            return "crash"
    if total_return >= 200:
        return "land"
    if total_return <= -50:
        return "crash"
    return "other"


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    files = []
    for root in (CONTROL_ROOT, ELITE_ROOT):
        for split in ("train", "val", "test"):
            files += glob.glob(os.path.join(root, split, "*.npz"))
    files = sorted(files)
    if MAX_EPISODES > 0:
        files = files[:MAX_EPISODES]
    if not files:
        raise SystemExit(f"No .npz found. Set CONTROL_ROOT / ELITE_ROOT.\n  CONTROL_ROOT={CONTROL_ROOT}\n  ELITE_ROOT={ELITE_ROOT}")

    returns, c_undisc, c_disc, c_legs, term_proxy, term_true, outcomes = [], [], [], [], [], [], []
    for f in files:
        with np.load(f) as d:
            if "states" not in d or "acts" not in d:
                continue
            states = d["states"].astype(np.float64)
            acts = d["acts"].astype(np.int64)
            if "episode_return" in d:
                ret = float(np.asarray(d["episode_return"]).item())
            elif "rewards" in d:
                ret = float(d["rewards"].sum())
            else:
                continue
            term_r = float(d["rewards"][-1]) if "rewards" in d and len(d["rewards"]) else np.nan
        if len(states) < 3:
            continue
        returns.append(ret)
        c_undisc.append(cost_on_episode(states, acts, gamma=1.0))
        c_disc.append(cost_on_episode(states, acts, gamma=GAMMA))
        c_legs.append(cost_on_episode(states, acts, gamma=1.0, terminal=_leg_terminal))
        term_proxy.append(_funnel_terminal(states[-1]))
        term_true.append(term_r)
        outcomes.append(_outcome(term_r, ret))

    returns = np.array(returns)
    c_undisc = np.array(c_undisc)
    c_disc = np.array(c_disc)
    c_legs = np.array(c_legs)
    outcomes = np.array(outcomes)
    n = len(returns)
    print(f"episodes scored: {n}  (control + elite)\n")

    print("=== Rank-correlation with true episode return (Spearman) ===")
    print(f"  proposed cost (funnel, undiscounted) : {spearman(c_undisc, returns):+.3f}   <- validates the structure")
    print(f"  proposed cost (funnel, γ={GAMMA})       : {spearman(c_disc, returns):+.3f}   <- the actual MPC objective")
    print(f"  old cost      (leg terminal)         : {spearman(c_legs, returns):+.3f}   <- reference")

    valid = ~np.isnan(term_true)
    if valid.sum() > 2:
        print(f"\n  funnel terminal vs TRUE terminal (±100) : {spearman(np.array(term_proxy)[valid], np.array(term_true)[valid]):+.3f}")

    print("\n=== Mean cost per outcome (proposed, undiscounted) — should be land > other > crash ===")
    for o in ("land", "other", "crash"):
        m = outcomes == o
        if m.any():
            print(f"  {o:<6} n={m.sum():4d} | mean cost={c_undisc[m].mean():9.1f} | mean return={returns[m].mean():8.1f}")

    # optional scatter
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6.4, 5.0))
        col = {"land": "C2", "crash": "C3", "other": "C0"}
        for o in ("land", "other", "crash"):
            m = outcomes == o
            if m.any():
                plt.scatter(returns[m], c_undisc[m], s=10, alpha=0.4, c=col[o], label=f"{o} (n={m.sum()})")
        plt.xlabel("true episode return")
        plt.ylabel("proposed cost (funnel, undiscounted)")
        plt.title(f"Cost vs true return  |  Spearman={spearman(c_undisc, returns):+.3f}")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        out = os.path.join(HERE, "cost_validation.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nsaved: {out}")
    except Exception as e:
        print(f"\n[plot skipped: {e}]")


if __name__ == "__main__":
    main()
