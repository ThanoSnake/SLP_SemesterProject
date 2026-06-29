"""
control_diag.py — Στοχευμένα διαγνωστικά για το ΓΙΑΤΙ αποτυγχάνει το model-based control του
control.py (ενώ ο enc_pid προσγειώνεται). ΚΑΜΙΑ τυφλή αλλαγή — πρώτα αποδείξεις με δεδομένα.

Τέσσερα tests (τρέξε & δες τα prints/plots):

  D1) DREAM ACCURACY — open-loop LSTM rollout από ΑΛΗΘΙΝΟ seed + ΑΛΗΘΙΝΑ actions σε test τροχιές,
      vs GT, standardized RMSE ανά horizon & dim. -> Πόσο μακριά εμπιστεύεσαι το «όνειρο»; (Η1)

  D2) ACTION RESPONSE — από ΑΛΗΘΙΝΟ mid-flight seed, ονειρέψου ΣΤΑΘΕΡΗ action {0,1,2,3} και δες αν
      η απόκριση έχει νόημα (main engine -> vy ανεβαίνει· side -> ω αλλάζει πρόσημο). -> bug/σημασία.

  D3) DREAM-VALUE vs REALITY — από ΙΔΙΟ start state, για M τυχαίες ακολουθίες: dream_value (MPC cost)
      ΕΝΑΝΤΙ του ΠΡΑΓΜΑΤΙΚΟΥ gym return (εκτέλεση στο env). Correlation. -> Αν ≤0, ο planner
      βελτιστοποιεί ΛΑΘΟΣ πράγμα = exploitation (Η2). Το «καπνίζον όπλο».

  D4) ENCODER R² ανά dim — επιβεβαιώνει ότι η αντίληψη (ειδικά ταχύτητες) είναι/δεν είναι ο ένοχος.

Επαναχρησιμοποιεί τα config/functions του control.py (ίδια checkpoints/paths· patched μαζί).
Run:  !python3 lunarlander/control_diag.py
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import control as C                       # ίδιο config/functions· import ΔΕΝ τρέχει το main()
from loader import list_npz

N_SUP, N_ACTIONS = C.N_SUP, C.N_ACTIONS
SEED = C.SEED
SAVE_DIR = os.path.join(C.SAVE_DIR, "diag")
DIM_NAMES = C.DIM_NAMES

# diag config
N_EPS_D1 = 8
H_LIST = [1, 5, 10, 24]
H_MAX = 24
STRIDE = 10
M_D3 = 96            # τυχαίες ακολουθίες για το value-vs-reality
H_D3 = 24
WIND_D3 = False


def get_models(device):
    vae = C.VAE_P1(n_sup=N_SUP, n_img=C.N_IMG).to(device)
    vae.load_state_dict(torch.load(C.VAE_CKPT, map_location=device)); vae.eval()
    lstm = C.LatentPredictor(C.LATENT_SIZE, N_ACTIONS, C.HIDDEN, C.LAYERS).to(device)
    lstm.load_state_dict(torch.load(C.LSTM_CKPT, map_location=device)); lstm.eval()
    return vae, lstm


@torch.no_grad()
def encode_episode(vae, imgs, device, batch=256):
    """imgs (T,H,W,3) uint8 -> z (T-1,64) (pairs frame_t,frame_t+1)."""
    t = torch.from_numpy(imgs.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
    img_t, img_tp1 = t[:-1], t[1:]
    zs = []
    for b in range(0, img_t.shape[0], batch):
        x = torch.cat([img_t[b:b + batch], img_tp1[b:b + batch]], dim=1).to(device)
        zs.append(vae.encode(x)[0].cpu())
    return torch.cat(zs).numpy() if zs else np.zeros((0, 64), np.float32)


# ---------------------------------------------------------------------------
# D1 — dream accuracy vs GT
# ---------------------------------------------------------------------------
@torch.no_grad()
def d1_dream_accuracy(vae, lstm, mean_t, std_t, std8, device, test_dir):
    files = list_npz(test_dir)
    if not files:
        print("[D1] no test episodes:", test_dir); return
    # διάλεξε τα ΜΕΓΑΛΥΤΕΡΑ επεισόδια (πλούσια δυναμική)
    lens = [(np.load(f)["states"].shape[0], f) for f in files[:200]]
    pick = [f for _, f in sorted(lens, reverse=True)[:N_EPS_D1]]
    se = np.zeros((H_MAX, N_SUP)); cnt = 0
    for f in pick:
        d = np.load(f); imgs, acts, states = d["imgs"], d["acts"], d["states"].astype(np.float64)
        z = encode_episode(vae, imgs, device)
        T = z.shape[0]
        for s in range(0, max(T - H_MAX - 1, 0), STRIDE):
            z0 = torch.from_numpy(z[s:s + 1]).to(device)
            prim = torch.from_numpy(acts[s:s + H_MAX].astype(np.int64))[None].to(device)
            traj = C.dream_rollout(lstm, z0, prim, mean_t, std_t, device)[0, 1:].cpu().numpy()
            gt = states[s + 1:s + 1 + H_MAX]
            se += ((traj - gt) / std8) ** 2; cnt += 1
    rmse = np.sqrt(se / max(cnt, 1))                  # (H_MAX, 8) standardized
    print(f"\n[D1] DREAM ACCURACY (standardized RMSE· {cnt} windows)")
    print("  dim         " + "  ".join(f"h{h:>2}" for h in H_LIST))
    for d in range(N_SUP):
        print(f"  {DIM_NAMES[d]:<10} " + "  ".join(f"{rmse[h-1, d]:>5.2f}" for h in H_LIST))
    print(f"  {'MEAN':<10} " + "  ".join(f"{rmse[h-1].mean():>5.2f}" for h in H_LIST))
    h = np.arange(1, H_MAX + 1)
    plt.figure(figsize=(7.2, 4.8))
    for d in range(N_SUP):
        plt.plot(h, rmse[:, d], lw=1.6, label=DIM_NAMES[d])
    plt.axhline(1.0, color="k", ls=":", lw=1, label="RMSE=1 (≈ τυχαίο)")
    plt.title("D1 — dream RMSE vs horizon (standardized· >1 = άχρηστη πρόβλεψη)")
    plt.xlabel("horizon"); plt.ylabel("standardized RMSE"); plt.grid(alpha=0.3); plt.legend(fontsize=8)
    plt.tight_layout()
    p = os.path.join(SAVE_DIR, "d1_dream_accuracy.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close(); print("saved:", p)


# ---------------------------------------------------------------------------
# D2 — action response sanity
# ---------------------------------------------------------------------------
@torch.no_grad()
def d2_action_response(vae, lstm, mean_t, std_t, device):
    env = C.make_env(False)
    env.reset(seed=SEED)
    for _ in range(15):                                # πήγαινε σε mid-flight
        env.step(0)
    f_prev = C.resize_frame(env.render()); env.step(0); f_cur = C.resize_frame(env.render())
    env.close()
    z0 = C.encode_pair(vae, f_prev, f_cur, device)
    labels = {0: "noop", 1: "left", 2: "MAIN", 3: "right"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    show = [(1, "y"), (3, "vy"), (4, "theta"), (5, "omega")]
    for a in range(N_ACTIONS):
        prim = torch.full((1, H_MAX), a, dtype=torch.long, device=device)
        traj = C.dream_rollout(lstm, z0, prim, mean_t, std_t, device)[0].cpu().numpy()   # (H+1,8)
        for j, (dim, name) in enumerate(show):
            axes[j // 2][j % 2].plot(traj[:, dim], lw=1.8, label=labels[a])
    for j, (dim, name) in enumerate(show):
        ax = axes[j // 2][j % 2]
        ax.set_title(f"dreamed {name} υπό σταθερή action"); ax.set_xlabel("horizon"); ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(fontsize=8)
    plt.suptitle("D2 — action response (sanity: MAIN engine πρέπει να ↑ vy· left/right αντίθετο ω)")
    plt.tight_layout()
    p = os.path.join(SAVE_DIR, "d2_action_response.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); print("saved:", p)
    print("[D2] saved action-response plot — έλεγξε ότι MAIN κρατά/ανεβάζει το y & vy· side αλλάζει ω.")


# ---------------------------------------------------------------------------
# D3 — dream-value vs real return  (το «καπνίζον όπλο» για exploitation)
# ---------------------------------------------------------------------------
@torch.no_grad()
def d3_value_vs_reality(vae, lstm, mean_t, std_t, device):
    env = C.make_env(WIND_D3)
    rng = np.random.default_rng(0)
    seqs = rng.integers(0, N_ACTIONS, size=(M_D3, H_D3))
    dv, rr = [], []
    for m in range(M_D3):
        env.reset(seed=SEED)
        f_prev = C.resize_frame(env.render())
        env.step(0)                                    # ένα βήμα -> ζεύγος + mid-state (ίδιο για όλα)
        f_cur = C.resize_frame(env.render())
        z0 = C.encode_pair(vae, f_prev, f_cur, device)
        prim = torch.from_numpy(seqs[m:m + 1]).to(device)
        v = C.dream_value(C.dream_rollout(lstm, z0, prim, mean_t, std_t, device), prim, device)[0].item()
        real = 0.0
        for k in range(H_D3):
            _, r, term, trunc, _ = env.step(int(seqs[m, k])); real += r
            if term or trunc:
                break
        dv.append(v); rr.append(real)
    env.close()
    dv, rr = np.array(dv), np.array(rr)
    corr = float(np.corrcoef(dv, rr)[0, 1])
    # πόσο καλή είναι η ΕΠΙΛΟΓΗ του MPC: το real-return της ακολουθίας με το ΜΕΓΑΛΥΤΕΡΟ dream_value
    best_dream = int(np.argmax(dv))
    print(f"\n[D3] DREAM-VALUE vs REAL RETURN  (M={M_D3}, H={H_D3}, wind={WIND_D3})")
    print(f"  correlation(dream_value, real_return) = {corr:+.3f}   (θέλουμε >0· ≤0 = exploitation)")
    print(f"  real_return της 'best-dream' ακολουθίας = {rr[best_dream]:+.1f}  "
          f"(median real = {np.median(rr):+.1f}, max real = {rr.max():+.1f})")
    plt.figure(figsize=(6.4, 5.0))
    plt.scatter(dv, rr, s=18, alpha=0.6)
    plt.scatter(dv[best_dream], rr[best_dream], s=90, c="C3", marker="*", label="MPC pick (max dream)")
    plt.title(f"D3 — dream_value vs real return  (corr={corr:+.2f})")
    plt.xlabel("dream_value (MPC cost)"); plt.ylabel("real gym return"); plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout()
    p = os.path.join(SAVE_DIR, "d3_value_vs_reality.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close(); print("saved:", p)


# ---------------------------------------------------------------------------
# D4 — encoder R² per dim
# ---------------------------------------------------------------------------
@torch.no_grad()
def d4_encoder_r2(vae, mean, std, device, test_dir, n_eps=10):
    files = list_npz(test_dir)[:n_eps]
    mus, gts = [], []
    for f in files:
        d = np.load(f); imgs, states = d["imgs"], d["states"].astype(np.float64)
        z = encode_episode(vae, imgs, device)
        mus.append(z[:, :N_SUP]); gts.append((states[:-1] - mean[:N_SUP]) / std[:N_SUP])
    mu = np.concatenate(mus); gt = np.concatenate(gts)
    ss_res = ((mu - gt) ** 2).sum(0); ss_tot = ((gt - gt.mean(0)) ** 2).sum(0) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    rmse = np.sqrt(((mu - gt) ** 2).mean(0))
    print(f"\n[D4] ENCODER quality (test· {mu.shape[0]} frames)")
    print(f"  {'dim':<10}{'R^2':>8}{'RMSE(std)':>12}")
    for d in range(N_SUP):
        print(f"  {DIM_NAMES[d]:<10}{r2[d]:>8.3f}{rmse[d]:>12.3f}")


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = C.get_device()
    print("device:", device)
    z = np.load(C.NORM_STATS)
    mean, std = z["mean"].astype(np.float64), z["std"].astype(np.float64)
    mean_t = torch.tensor(mean, device=device, dtype=torch.float32)
    std_t = torch.tensor(std, device=device, dtype=torch.float32)
    std8 = std[:N_SUP]
    test_dir = os.path.join(C.DATA_ROOT, "test")

    vae, lstm = get_models(device)
    d1_dream_accuracy(vae, lstm, mean_t, std_t, std8, device, test_dir)
    d2_action_response(vae, lstm, mean_t, std_t, device)
    d3_value_vs_reality(vae, lstm, mean_t, std_t, device)
    d4_encoder_r2(vae, mean, std, device, test_dir)

    print(f"\n{'='*70}")
    print("ΠΩΣ ΔΙΑΒΑΖΕΤΑΙ:")
    print("  D1: αν το mean RMSE ξεπερνά ~1 πριν τον h=10 -> το ΟΝΕΙΡΟ είναι άχρηστο (Η1).")
    print("  D3: αν corr ≤ 0 ή το 'best-dream' real_return είναι χάλια -> EXPLOITATION (Η2).")
    print("  D4: αν τα x/y/theta έχουν καλό R² αλλά vx/vy/omega όχι -> seed-velocity bottleneck.")
    print(f"  -> εστιάζουμε τη διόρθωση εκεί που δείχνουν τα νούμερα.\n{'='*70}")
    print("saved diag ->", SAVE_DIR)


if __name__ == "__main__":
    main()
