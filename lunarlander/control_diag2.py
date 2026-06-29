"""
control_diag2.py — Γιατί το rollout/override ΧΕΙΡΟΤΕΡΕΥΕΙ τον (καλό) PID; Δεδομένα, όχι εικασίες.

To control_diag.py έδειξε: μοντέλο/encoder ΚΑΛΑ, αλλά free-MPC = optimizer's curse. Φτιάξαμε rollout
(policy-improvement) — ΑΛΛΑ πάλι αποτυγχάνει, με override~98% επιβλαβές. Τρία tests εδώ:

  R1) OVERRIDE-vs-REALITY (το κρίσιμο). Σε πραγματικά mid-flight states (φτάνουμε με PID, ντετερμινιστικά
      μέσω seed+replay), όπου το μοντέλο ΘΕΛΕΙ override (best_a ≠ a_pid): εκτελούμε ΣΤΗΝ ΠΡΑΓΜΑΤΙΚΟΤΗΤΑ
      και «best_a μετά PID» ΚΑΙ «a_pid μετά PID» από το ΙΔΙΟ state, μετράμε πραγματικό return.
      -> Βοηθάει το override ή βλάπτει; Συσχετίζεται το model-gap με την πραγματική βελτίωση;

  R2) CLOSED-LOOP DRIFT. Τρέχουμε το rollout controller και μετράμε το 1-step σφάλμα του μοντέλου
      ΥΠΟ ΤΙΣ ΕΝΕΡΓΕΙΕΣ ΤΟΥ ROLLOUT (πιθανώς OOD) — vs το D1 (0.31 με data-policy). Μεγάλη αύξηση
      = το rollout οδηγεί το σύστημα OOD και το μοντέλο σπάει.

  R3) OVERRIDE BIAS. Ποια ενέργεια διαλέγει το override (histogram) + κατανομή του value-gap.
      -> systematic bias (π.χ. πάντα noop/main) ή θόρυβος.

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

N_OVERRIDE_CASES = 30        # πόσα override-states να δοκιμάσουμε στο R1
MAX_TRIALS = 200             # πάνω όριο trials για να μαζέψουμε τα overrides
BRANCH_CAP = 160             # βήματα μέχρι τερματισμό για το branch return
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
    """reset(seed) -> replay prefix -> first_a -> μετά enc_pid μέχρι τερματισμό. -> cumulative reward."""
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
            continue                                    # κανένα override εδώ
        r_over = real_branch_return(env, seed, prefix, best_a, vae, mean_t, std_t, device)
        r_pid = real_branch_return(env, seed, prefix, a_pid, vae, mean_t, std_t, device)
        gaps.append(v_best - v_pid); d_real.append(r_over - r_pid); ov_acts.append(best_a)
    env.close()
    gaps, d_real, ov_acts = np.array(gaps), np.array(d_real), np.array(ov_acts)
    if len(gaps) == 0:
        print("[R1] κανένα override case (best_a πάντα == a_pid;)"); return

    helped = float((d_real > 0).mean())
    corr = float(np.corrcoef(gaps, d_real)[0, 1]) if len(gaps) > 2 else float("nan")
    print(f"\n[R1] OVERRIDE-vs-REALITY  ({len(gaps)} override states, wind={enable_wind})")
    print(f"  fraction overrides που ΒΟΗΘΗΣΑΝ (real_over>real_pid) = {helped:.2f}   (θέλουμε >0.5)")
    print(f"  mean(real_override − real_pid) = {d_real.mean():+.1f}   (θέλουμε >0· <0 = ΒΛΑΠΤΟΥΝ)")
    print(f"  corr(model_gap, real_improvement) = {corr:+.3f}   (~0 = το model-gap είναι ΑΧΡΗΣΤΟ)")
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
    print(f"\n[R2] CLOSED-LOOP 1-step σφάλμα ΥΠΟ rollout actions  ({cnt} βήματα)")
    print("  dim     " + "  ".join(f"{DIM_NAMES[d][:5]:>5}" for d in range(N_SUP)))
    print("  RMSE    " + "  ".join(f"{rmse[d]:>5.2f}" for d in range(N_SUP)))
    print(f"  MEAN = {rmse.mean():.2f}   (σύγκρινε με D1 h1≈0.31· πολύ μεγαλύτερο = OOD drift)")


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

    print(f"\n{'='*70}\nΠΩΣ ΔΙΑΒΑΖΕΤΑΙ:")
    print("  R1: αν fraction-helped < 0.5 ή mean Δ < 0 -> τα overrides ΒΛΑΠΤΟΥΝ.")
    print("      αν corr(gap,Δreal) ~ 0 -> το cost ΔΕΝ ξεχωρίζει κοντινές πολιτικές (θόρυβος).")
    print("      αν το histogram δείχνει 1 ενέργεια -> systematic cost bias.")
    print("  R2: αν το 1-step RMSE >> 0.31 -> το rollout οδηγεί OOD & το μοντέλο σπάει.")
    print(f"{'='*70}\nsaved -> {SAVE_DIR}")


if __name__ == "__main__":
    main()
