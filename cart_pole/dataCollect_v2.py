import os

import gymnasium as gym
import numpy as np
from PIL import Image

# ----------------------------------------------------------------------------
# Ρυθμίσεις συλλογής
# ----------------------------------------------------------------------------
NUM_EPISODES = 5000
MAX_STEPS = 300                 # μήκος ακολουθίας-στόχος ανά επεισόδιο
IMG_H, IMG_W = 80, 120          # resize ώστε να ταιριάζει στο VAE (encoder ÷8 -> 10x15)
TEST_FRACTION = 0.1             # ποσοστό επεισοδίων που πάνε στο test set

# PD / linear state-feedback controller με ε-greedy τυχαιότητα.
# Στόχος: μεγαλύτερα επεισόδια ΚΑΙ ποικιλία μεταβάσεων (όχι μόνο "ισορροπημένες"
# καταστάσεις) ώστε το world model να μάθει πλούσια δυναμική.
KP_THETA, KD_THETA = 12.0, 2.0  # κέρδη γωνίας / γωνιακής ταχύτητας
KP_X, KD_X = 0.0, 0.0           # κέρδη θέσης / ταχύτητας κάρου (0 = ανενεργά)
EPSILON = 0.25                  # πιθανότητα τυχαίας ενέργειας (exploration)

# Θόρυβος που προστίθεται στην ΚΑΤΑΣΤΑΣΗ (μόνο θέση & γωνία, όχι ταχύτητες).
# Τα labels (2/5/10) γίνονται κλειδιά noisy_states_2 / _5 / _10 στο .npz.
NOISE_LEVELS = [(0.025, 2), (0.05, 5), (0.10, 10)]
POSITION_REF = 4.8 * 2
ANGLE_REF = 0.84


def pd_action(obs, action_space):
    """ PD/linear controller με ε-greedy τυχαιότητα.
    obs = [x, x_dot, theta, theta_dot]. Επιστρέφει 0 (αριστερά) ή 1 (δεξιά). """
    if np.random.rand() < EPSILON:
        return action_space.sample()
    x, x_dot, theta, theta_dot = obs
    u = KP_THETA * theta + KD_THETA * theta_dot + KP_X * x + KD_X * x_dot
    return 1 if u > 0 else 0


def resize_frame(img):
    """ RGB render -> (IMG_H, IMG_W, 3) uint8. """
    return np.asarray(Image.fromarray(img).resize((IMG_W, IMG_H)), dtype=np.uint8)


def collect_episode(env, max_steps=MAX_STEPS):
    obs, _ = env.reset()
    imgs, acts, states = [], [], []
    for _ in range(max_steps):
        img = resize_frame(env.render())
        action = pd_action(obs, env.action_space)

        imgs.append(img)
        acts.append(action)
        states.append(obs)

        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    return (np.array(imgs),
            np.array(acts),
            np.array(states, dtype=np.float32))


def save_episode(save_dir, run_id, imgs, acts, states):
    if len(states) == 0:
        return

    noisy_versions = {}
    for level, label in NOISE_LEVELS:
        noise = np.zeros_like(states)
        noise[:, 0] = np.random.normal(0, level * POSITION_REF, size=states.shape[0])
        noise[:, 2] = np.random.normal(0, level * ANGLE_REF, size=states.shape[0])
        noisy_versions[f"noisy_states_{label}"] = (states + noise).astype(np.float32)

    save_path = os.path.join(save_dir, f"{run_id}.npz")
    np.savez_compressed(save_path, imgs=imgs, acts=acts, states=states, **noisy_versions)


if __name__ == '__main__':
    base_dir = "/kaggle/working/cartpole_data"
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    env = gym.make("CartPole-v1", render_mode="rgb_array")

    for i in range(NUM_EPISODES):
        if i % 100 == 0:
            print(f"Συλλογή επεισοδίου: {i}/{NUM_EPISODES}")

        imgs, acts, states = collect_episode(env)
        out_dir = test_dir if np.random.rand() < TEST_FRACTION else train_dir
        save_episode(out_dir, i, imgs, acts, states)

    env.close()
    print("Η συλλογή δεδομένων ολοκληρώθηκε επιτυχώς!")
