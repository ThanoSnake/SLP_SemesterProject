"""
control_diag2.py — Why does the rollout/override make the (good) PID WORSE? Data, not guesses.

control_diag.py showed: the model/encoder are GOOD, but free MPC = the optimizer's curse. We built rollout
(policy improvement) — BUT it fails too, with ~98% of overrides harmful. Three tests here:

  R1) OVERRIDE-vs-REALITY (the critical one). At real mid-flight states (reached with the PID, deterministically
      via seed+replay), where the model WANTS an override (best_a != a_pid): we execute IN REALITY
      both "best_a then PID" AND "a_pid then PID" from the SAME state, and measure the real return.
      -> Does the override help or hurt? Does the model gap correlate with the real improvement?

  R2) CLOSED-LOOP DRIFT. We run the rollout controller and measure the model's 1-step error
      UNDER THE ROLLOUT'S ACTIONS (possibly OOD) — vs D1 (0.31 with the data policy). A large increase
      = the rollout drives the system OOD and the model breaks.

  R3) OVERRIDE BIAS. Which action does the override pick (histogram) + the distribution of the value gap.
      -> a systematic bias (e.g. always noop/main) or noise.

Run:  !python3 lunarlander/control_diag2.py
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import control as C

N_SUP, N_ACTIONS, SEED, MAX_STEPS = C.N_SUP, C.N_ACTIONS, C.SEED, C.MAX_STEPS
SAVE_DIR = os.path.join(C.SAVE_DIR, "diag2")
DIM_NAMES = C.DIM_NAMES

N_OVERRIDE_CASES = 30        # how many override states to try in R1
MAX_TRIALS = 200             # upper bound on trials while collecting the overrides
BRANCH_CAP = 160             # steps until termination for the branch return
R2_EPS = 4


def get_models(device):
    vae = C.VAE_P1(n_sup=N_SUP, n_img=C.N_IMG).to(device)
    vae.load_state_dict(torch.load(C.VAE_CKPT, map_location=device)); vae.eval()
    lstm = C.LatentPredictor(C.LATENT_SIZE, N_ACTIONS, C.HIDDEN, C.LAYERS).to(device)
    lstm.load_state_dict(torch.load(C.LSTM_CKPT, map_location=device)); lstm.eval()
    return vae, lstm


@torch.no_grad()
def enc_pid_action(vae, f_prev, f_cur, mean_t, std_t, device):
    mu = C.encode_pair(vae, f_prev, f_cur, device)
    a = C.heuristic_control(C.to_phys(mu[0, :N_SUP], mean_t, std_t).cpu().numpy())
    return a, mu


@torch.no_grad()
def real_branch_return(env, seed, prefix_actions, first_a, vae, mean_t, std_t, device):
    """reset(seed) -> replay prefix -> first_a -> then enc_pid until termination. -> cumulative reward."""
    env.reset(seed=seed)
    for a in prefix_actions:
        env.step(a)
    f_prev = C.resize_frame(env.render())
    total, a = 0.0, first_a
    for _ in range(BRANCH_CAP):
        _, r, term, trunc, _ = env.step(a); total += r
        if term or trunc:
            break
        f_cur = C.resize_frame(env.render())
        a, _ = enc_pid_action(vae, f_prev, f_cur, mean_t, std_t, device)
        f_prev = f_cur
    return total


@torch.no_grad()
def r1_override_vs_reality(vae, lstm, mean_t, std_t, device, enable_wind=False):
    env = C.make_env(enable_wind)
    rng = np.random.default_rng(0)
    gaps, d_real, ov_acts = [], [], []
    trials = 0
    while len(gaps) < N_OVERRIDE_CASES and trials < MAX_TRIALS:
        trials += 1
        seed = SEED + trials
        ckpt = int(rng.integers(12, 110))
        env.reset(seed=seed)
        f_prev = C.resize_frame(env.render())
        prefix, mu_ck = [], None
        ok = True
        for t in range(ckpt):
            f_cur = C.resize_frame(env.render())
            a, mu = enc_pid_action(vae, f_prev, f_cur, mean_t, std_t, device)
            prefix.append(a)
            _, _, term, trunc, _ = env.step(a); f_prev = f_cur
            mu_ck = mu
            if term or trunc:
                ok = False; break
        if not ok or mu_ck is None:
            continue
        a_pid = C.heuristic_control(C.to_phys(mu_ck[0, :N_SUP], mean_t, std_t).cpu().numpy())
        best_a, v_best, v_pid = C.mpc_rollout(lstm, mu_ck, mean_t, std_t, device)
        if best_a == a_pid:
            continue                                    # no override here
        r_over = real_branch_return(env, seed, prefix, best_a, vae, mean_t, std_t, device)
        r_pid = real_branch_return(env, seed, prefix, a_pid, vae, mean_t, std_t, device)
        gaps.append(v_best - v_pid); d_real.append(r_over - r_pid); ov_acts.append(best_a)
    env.close()
    gaps, d_real, ov_acts = np.array(gaps), np.array(d_real), np.array(ov_acts)
    if len(gaps) == 0:
        print("[R1] no override case (is best_a always == a_pid?)"); return

    helped = float((d_real > 0).mean())
    corr = float(np.corrcoef(gaps, d_real)[0, 1]) if len(gaps) > 2 else float("nan")
    print(f"\n[R1] OVERRIDE-vs-REALITY  ({len(gaps)} override states, wind={enable_wind})")
    print(f"  fraction of overrides that HELPED (real_over>real_pid) = {helped:.2f}   (we want >0.5)")
    print(f"  mean(real_override − real_pid) = {d_real.mean():+.1f}   (we want >0; <0 = THEY HURT)")
    print(f"  corr(model_gap, real_improvement) = {corr:+.3f}   (~0 = the model gap is USELESS)")
    hist = np.bincount(ov_acts, minlength=N_ACTIONS)
    names = {0: "noop", 1: "left", 2: "MAIN", 3: "right"}
    print("  override action histogram: " + "  ".join(f"{names[i]}={hist[i]}" for i in range(N_ACTIONS)))
    print(f"  mean model-gap = {gaps.mean():.2f}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    ax[0].scatter(gaps, d_real, s=24, alpha=0.7)
    ax[0].axhline(0, color="k", lw=1); ax[0].set_xlabel("model value-gap (v_best − v_pid)")
    ax[0].set_ylabel("real Δreturn (override − pid)"); ax[0].set_title(f"R1 — corr={corr:+.2f}"); ax[0].grid(alpha=0.3)
    ax[1].bar([names[i] for i in range(N_ACTIONS)], hist, color="C1")
    ax[1].set_title("override action histogram"); ax[1].grid(alpha=0.3, axis="y")
    plt.tight_layout()
    p = os.path.join(SAVE_DIR, "r1_override_vs_reality.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); print("saved:", p)


@torch.no_grad()
def r2_closed_loop_drift(vae, lstm, mean_t, std_t, std8, device, enable_wind=False):
    env = C.make_env(enable_wind)
    se = np.zeros(N_SUP); cnt = 0
    ov_acts = []
    for ep in range(R2_EPS):
        env.reset(seed=SEED + ep)
        f_prev = C.resize_frame(env.render())
        prev_mu, prev_a = None, None
        for _ in range(MAX_STEPS):
            f_cur = C.resize_frame(env.render())
            mu = C.encode_pair(vae, f_prev, f_cur, device)
            if prev_mu is not None:
                z_pred, _ = lstm.step(prev_mu, F.one_hot(torch.tensor([prev_a], device=device),
                                                         N_ACTIONS).float(), lstm.init_hidden(1, device))
                pred = C.to_phys(z_pred[0, :N_SUP], mean_t, std_t).cpu().numpy()
                real = C.to_phys(mu[0, :N_SUP], mean_t, std_t).cpu().numpy()
                se += ((pred - real) / std8) ** 2; cnt += 1
            a_pid = C.heuristic_control(C.to_phys(mu[0, :N_SUP], mean_t, std_t).cpu().numpy())
            best_a, v_best, v_pid = C.mpc_rollout(lstm, mu, mean_t, std_t, device)
            a = best_a if v_best > v_pid + C.ROLLOUT_MARGIN else a_pid
            if a != a_pid:
                ov_acts.append(a)
            _, _, term, trunc, _ = env.step(a)
            prev_mu, prev_a, f_prev = mu, a, f_cur
            if term or trunc:
                break
    env.close()
    rmse = np.sqrt(se / max(cnt, 1))
    print(f"\n[R2] CLOSED-LOOP 1-step error UNDER rollout actions  ({cnt} steps)")
    print("  dim     " + "  ".join(f"{DIM_NAMES[d][:5]:>5}" for d in range(N_SUP)))
    print("  RMSE    " + "  ".join(f"{rmse[d]:>5.2f}" for d in range(N_SUP)))
    print(f"  MEAN = {rmse.mean():.2f}   (compare with D1 h1~0.31; much larger = OOD drift)")


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = C.get_device()
    print("device:", device)
    z = np.load(C.NORM_STATS)
    mean, std = z["mean"].astype(np.float64), z["std"].astype(np.float64)
    mean_t = torch.tensor(mean, device=device, dtype=torch.float32)
    std_t = torch.tensor(std, device=device, dtype=torch.float32)
    std8 = std[:N_SUP]
    vae, lstm = get_models(device)

    r1_override_vs_reality(vae, lstm, mean_t, std_t, device)
    r2_closed_loop_drift(vae, lstm, mean_t, std_t, std8, device)

    print(f"\n{'='*70}\nHOW TO READ IT:")
    print("  R1: if fraction-helped < 0.5 or mean Δ < 0 -> the overrides HURT.")
    print("      if corr(gap,Δreal) ~ 0 -> the cost does NOT separate nearby policies (noise).")
    print("      if the histogram shows 1 action -> a systematic cost bias.")
    print("  R2: if the 1-step RMSE >> 0.31 -> the rollout drives OOD & the model breaks.")
    print(f"{'='*70}\nsaved -> {SAVE_DIR}")


if __name__ == "__main__":
    main()
