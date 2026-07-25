"""
control_diag3.py — Why does est_pid help in wind but hurt in calm, and why does the shield break?

  G1) ESTIMATOR QUALITY vs GT (the critical one). We drive with true_pid (a clean trajectory); at every step
      we compare BOTH the RAW encoder state (mu) AND the model-FILTERED est-state against the TRUE obs.
      Per-dim standardized RMSE, calm vs wind. -> Does the model improve the CURRENT state? WHERE
      (which dims) and WHEN (calm/wind)? -> this determines the est_pid fix (velocity-only/adaptive).

  G2) SHIELD PRECISION. We run est_pid WITHOUT the shield, record when the shield WOULD have fired,
      and check whether the danger actually appears in the REAL (unshielded) trajectory (y<Y_LOW &
      speed>S_DANGER) within SHIELD_HORIZON -> precision + lead time. Low precision = false alarms;
      a large lead = it brakes far too early.

Run:  !python3 lunarlander/control_diag3.py
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

import control as C

N_SUP, SEED, MAX_STEPS = C.N_SUP, C.SEED, C.MAX_STEPS
DIM_NAMES = C.DIM_NAMES
SAVE_DIR = os.path.join(C.SAVE_DIR, "diag3")
N_EPS = 8


def get_models(device):
    vae = C.VAE_P1(n_sup=N_SUP, n_img=C.N_IMG).to(device)
    vae.load_state_dict(torch.load(C.VAE_CKPT, map_location=device)); vae.eval()
    lstm = C.LatentPredictor(C.LATENT_SIZE, C.N_ACTIONS, C.HIDDEN, C.LAYERS).to(device)
    lstm.load_state_dict(torch.load(C.LSTM_CKPT, map_location=device)); lstm.eval()
    return vae, lstm


@torch.no_grad()
def g1_estimator_quality(vae, lstm, mean_t, std_t, std8, device, enable_wind):
    """ -> (rmse_enc (8,), rmse_est (8,)) standardized, vs the TRUE obs. Driven with true_pid."""
    env = C.make_env(enable_wind)
    se_enc = np.zeros(N_SUP); se_est = np.zeros(N_SUP); cnt = 0
    for ep in range(N_EPS):
        obs, _ = env.reset(seed=SEED + ep)
        f_prev = C.resize_frame(env.render())
        est = C.StateEstimator(lstm, mean_t, std_t, device)
        for t in range(MAX_STEPS):
            f_cur = C.resize_frame(env.render())
            mu = C.encode_pair(vae, f_prev, f_cur, device)
            enc_state = C.to_phys(mu[0, :N_SUP], mean_t, std_t).cpu().numpy()
            z_cur = est.estimate(mu)
            est_state = C.to_phys(z_cur[0, :N_SUP], mean_t, std_t).cpu().numpy()
            if t >= 1:                                       # t=0: lag undefined
                se_enc += ((enc_state - obs) / std8) ** 2
                se_est += ((est_state - obs) / std8) ** 2
                cnt += 1
            a = C.heuristic_control(obs)                     # true_pid drive (a clean trajectory)
            est.set_action(a)
            obs, r, term, trunc, _ = env.step(a); f_prev = f_cur
            if term or trunc:
                break
    env.close()
    return np.sqrt(se_enc / max(cnt, 1)), np.sqrt(se_est / max(cnt, 1))


@torch.no_grad()
def g2_shield_precision(vae, lstm, mean_t, std_t, device, enable_wind):
    """Runs est_pid WITHOUT the shield; precision = P(real danger within SHIELD_HORIZON | shield fired)."""
    env = C.make_env(enable_wind)
    n_fire, n_true, leads = 0, 0, []
    for ep in range(N_EPS):
        obs, _ = env.reset(seed=SEED + ep)
        f_prev = C.resize_frame(env.render())
        est = C.StateEstimator(lstm, mean_t, std_t, device)
        obs_traj, fire_traj = [], []
        for t in range(MAX_STEPS):
            f_cur = C.resize_frame(env.render())
            mu = C.encode_pair(vae, f_prev, f_cur, device)
            z_cur = est.estimate(mu)
            fire = C.shield_predicts_crash(lstm, z_cur, mean_t, std_t, device)
            a = C.heuristic_control(C.to_phys(z_cur[0, :N_SUP], mean_t, std_t).cpu().numpy())  # NO shield
            obs_traj.append(obs.copy()); fire_traj.append(bool(fire))
            est.set_action(a)
            obs, r, term, trunc, _ = env.step(a); f_prev = f_cur
            if term or trunc:
                break
        T = len(obs_traj)
        for t in range(T):
            if not fire_traj[t]:
                continue
            n_fire += 1
            for k in range(t, min(t + C.SHIELD_HORIZON, T)):
                o = obs_traj[k]
                if o[1] < C.Y_LOW and (o[2] ** 2 + o[3] ** 2) ** 0.5 > C.S_DANGER:
                    n_true += 1; leads.append(k - t); break
    env.close()
    prec = n_true / max(n_fire, 1)
    lead = float(np.mean(leads)) if leads else float("nan")
    return n_fire, prec, lead


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

    print(f"\n[G1] ESTIMATOR QUALITY vs GT obs (standardized RMSE; LOWER=better)")
    for tag, wind in (("no_wind", False), ("wind", True)):
        rmse_enc, rmse_est = g1_estimator_quality(vae, lstm, mean_t, std_t, std8, device, wind)
        print(f"\n  --- {tag} ---")
        print(f"  {'dim':<8}{'enc(raw)':>10}{'est(model)':>12}{'Δ%':>8}")
        for d in range(N_SUP):
            imp = 100.0 * (rmse_enc[d] - rmse_est[d]) / (rmse_enc[d] + 1e-8)
            print(f"  {DIM_NAMES[d]:<8}{rmse_enc[d]:>10.3f}{rmse_est[d]:>12.3f}{imp:>+7.0f}%")
        imp_m = 100.0 * (rmse_enc.mean() - rmse_est.mean()) / (rmse_enc.mean() + 1e-8)
        print(f"  {'MEAN':<8}{rmse_enc.mean():>10.3f}{rmse_est.mean():>12.3f}{imp_m:>+7.0f}%   (>0 = the model improves it)")

    print(f"\n[G2] SHIELD PRECISION (est_pid without the shield)")
    print(f"  {'cond':<9}{'#fires':>8}{'precision':>11}{'mean lead':>11}")
    for tag, wind in (("no_wind", False), ("wind", True)):
        nf, prec, lead = g2_shield_precision(vae, lstm, mean_t, std_t, device, wind)
        print(f"  {tag:<9}{nf:>8}{prec:>11.2f}{lead:>11.1f}")

    print(f"\n{'='*70}\nHOW TO READ IT:")
    print("  G1: if est only improves in wind/only on velocities -> a velocity-only or adaptive estimator.")
    print("      if est is WORSE in calm -> the lag removal injects noise where it is not needed.")
    print("  G2: low precision -> false alarms (a stricter trigger); a large lead -> it brakes far too early.")
    print(f"{'='*70}\nsaved -> {SAVE_DIR}")


if __name__ == "__main__":
    main()
