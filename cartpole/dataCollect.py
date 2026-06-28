import os

import gymnasium as gym
import numpy as np
from PIL import Image

# ----------------------------------------------------------------------------
# Ρυθμίσεις συλλογής
# ----------------------------------------------------------------------------
NUM_EPISODES = 5000
MAX_STEPS = 200                 # μήκος ακολουθίας-στόχος ανά επεισόδιο
IMG_H, IMG_W = 80, 120          # resize ώστε να ταιριάζει στο VAE (encoder ÷8 -> 10x15)
SEED = 0                        # για αναπαραγωγιμότητα (σημαντικό σε reproduction paper)

# Διαχωρισμός σε train / val / test (πρέπει να αθροίζουν σε 1.0)
TRAIN_FRACTION = 0.8
VAL_FRACTION = 0.1
TEST_FRACTION = 0.1

# PD / linear state-feedback controller με ε-greedy τυχαιότητα.
KP_THETA, KD_THETA = 12.0, 2.0  # κέρδη γωνίας / γωνιακής ταχύτητας
KP_X, KD_X = 0.0, 0.0           # κέρδη θέσης / ταχύτητας κάρου (0 = ανενεργά)
EPSILON = 0.5                  # πιθανότητα τυχαίας ενέργειας (exploration)

# Θόρυβος στην ΚΑΤΑΣΤΑΣΗ (μόνο θέση & γωνία). Labels -> noisy_states_2/_5/_10.
NOISE_LEVELS = [(0.025, 2), (0.05, 5), (0.10, 10)]
POSITION_REF = 4.8 * 2          # πλήρες εύρος obs-space για x
ANGLE_REF = 0.84                # πλήρες εύρος obs-space για theta


def pd_action(obs, action_space, rng):
    """ PD/linear controller με ε-greedy τυχαιότητα.
    obs = [x, x_dot, theta, theta_dot]. Επιστρέφει 0 (αριστερά) ή 1 (δεξιά). """
    if rng.random() < EPSILON:
        return int(action_space.sample())
    x, x_dot, theta, theta_dot = obs
    u = KP_THETA * theta + KD_THETA * theta_dot + KP_X * x + KD_X * x_dot
    return 1 if u > 0 else 0


def resize_frame(img):
    """ RGB render -> (IMG_H, IMG_W, 3) uint8. ΔΕΝ κανονικοποιούμε εδώ· το /255
    γίνεται στο dataloader για να μη 4πλασιάσουμε το μέγεθος στον δίσκο. """
    return np.asarray(Image.fromarray(img).resize((IMG_W, IMG_H)), dtype=np.uint8)


def collect_episode(env, rng, ep_seed, max_steps=MAX_STEPS):
    obs, _ = env.reset(seed=ep_seed)
    imgs, acts, states = [], [], []
    for _ in range(max_steps):
        img = resize_frame(env.render())
        action = pd_action(obs, env.action_space, rng)

        imgs.append(img)
        acts.append(action)
        states.append(obs)

        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    return (np.array(imgs),
            np.array(acts),
            np.array(states, dtype=np.float32))


def save_episode(save_dir, run_id, imgs, acts, states, rng):
    if len(states) == 0:
        return 0

    noisy_versions = {}
    for level, label in NOISE_LEVELS:
        noise = np.zeros_like(states)
        noise[:, 0] = rng.normal(0, level * POSITION_REF, size=states.shape[0])
        noise[:, 2] = rng.normal(0, level * ANGLE_REF, size=states.shape[0])
        noisy_versions[f"noisy_states_{label}"] = (states + noise).astype(np.float32)

    save_path = os.path.join(save_dir, f"{run_id}.npz")
    np.savez_compressed(save_path, imgs=imgs, acts=acts, states=states, **noisy_versions)
    return len(states)


def choose_split(rng):
    r = rng.random()
    if r < TRAIN_FRACTION:
        return "train"
    elif r < TRAIN_FRACTION + VAL_FRACTION:
        return "val"
    return "test"


def compute_and_save_norm_stats(train_dir, out_path):
    """ mean/std ΜΟΝΟ από το train set (αποφυγή leakage). Τα states μένουν raw
    στον δίσκο· το standardize γίνεται στο dataloader με αυτά τα stats. """
    chunks = [np.load(os.path.join(train_dir, f))["states"]
              for f in os.listdir(train_dir) if f.endswith(".npz")]
    all_states = np.concatenate(chunks, axis=0)
    mean = all_states.mean(axis=0)
    std = all_states.std(axis=0) + 1e-8
    np.savez(out_path, mean=mean, std=std)
    print("State mean:", mean)
    print("State std :", std)


if __name__ == '__main__':
    assert abs(TRAIN_FRACTION + VAL_FRACTION + TEST_FRACTION - 1.0) < 1e-6

    base_dir = "/kaggle/working/cartpole_data"
    dirs = {s: os.path.join(base_dir, s) for s in ("train", "val", "test")}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    rng = np.random.default_rng(SEED)
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env.action_space.seed(SEED)

    lengths = []
    for i in range(NUM_EPISODES):
        if i % 100 == 0:
            print(f"Συλλογή επεισοδίου: {i}/{NUM_EPISODES}")

        imgs, acts, states = collect_episode(env, rng, ep_seed=SEED + i)
        split = choose_split(rng)
        n = save_episode(dirs[split], i, imgs, acts, states, rng)
        lengths.append(n)

    env.close()

    lengths = np.array(lengths)
    print(f"\nΕπεισόδια: {len(lengths)} | mean={lengths.mean():.1f} "
          f"median={np.median(lengths):.0f} min={lengths.min()} max={lengths.max()} "
          f"| % που φτάνουν MAX_STEPS: {(lengths >= MAX_STEPS).mean() * 100:.1f}%")

    compute_and_save_norm_stats(dirs["train"], os.path.join(base_dir, "norm_stats.npz"))
    print("Η συλλογή δεδομένων ολοκληρώθηκε επιτυχώς!")
