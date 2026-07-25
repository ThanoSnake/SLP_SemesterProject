"""
dataCollect.py — dataset collection for LunarLander (discrete), following cartpole's dataCollect.py.
Requires: pip install "gymnasium[box2d]". State=8, actions=4.
"""
import os

import gymnasium as gym
import numpy as np
from PIL import Image

from paths import DATA_ROOT

# ----------------------------------------------------------------------------
NUM_EPISODES = 4000             # 2x cartpole (2000) -> coverage of the more complex 8D state
MAX_STEPS = 400                 # landings take ~200-400 steps (longer than cartpole)
IMG_H, IMG_W = 80, 120          # SAME as cartpole -> the same VAE (encoder /8 -> 10x15)
SEED = 0

TRAIN_FRACTION = 0.8
VAL_FRACTION = 0.1
TEST_FRACTION = 0.1

EPSILON = 0.20                  # ε-greedy around the heuristic; lower than cartpole (0.5):
                                # LunarLander crashes under too much randomness

# STATE noise only on the position/angle dims -> noisy_states_2/_5/_10 (weak supervision).
# dims: 0=x, 1=y, 4=theta. The velocities (2,3,5) & legs (6,7) are NOT noised.
NOISE_LEVELS = [(0.025, 2), (0.05, 5), (0.10, 10)]
NOISE_REF = {0: 2.5, 1: 2.0, 4: 2.0}   # reference range per dim


def make_env():
    for env_id in ("LunarLander-v3", "LunarLander-v2"):
        try:
            return gym.make(env_id, render_mode="rgb_array")
        except Exception:
            continue
    raise RuntimeError("LunarLander not found (pip install 'gymnasium[box2d]').")


def heuristic_action(obs, action_space, rng):
    """ Standard LunarLander PD-like heuristic + ε-greedy -> action ∈ {0,1,2,3}. """
    if rng.random() < EPSILON:
        return int(action_space.sample())
    x, y, vx, vy, theta, omega, leg1, leg2 = obs
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


def resize_frame(img):
    return np.asarray(Image.fromarray(img).resize((IMG_W, IMG_H)), dtype=np.uint8)


def collect_episode(env, rng, ep_seed, max_steps=MAX_STEPS):
    obs, _ = env.reset(seed=ep_seed)
    imgs, acts, states = [], [], []
    for _ in range(max_steps):
        img = resize_frame(env.render())
        action = heuristic_action(obs, env.action_space, rng)
        imgs.append(img)
        acts.append(action)
        states.append(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    return np.array(imgs), np.array(acts), np.array(states, dtype=np.float32)


def save_episode(save_dir, run_id, imgs, acts, states, rng):
    if len(states) == 0:
        return 0
    noisy = {}
    for level, label in NOISE_LEVELS:
        noise = np.zeros_like(states)
        for dim, ref in NOISE_REF.items():
            noise[:, dim] = rng.normal(0, level * ref, size=states.shape[0])
        noisy[f"noisy_states_{label}"] = (states + noise).astype(np.float32)
    np.savez_compressed(os.path.join(save_dir, f"{run_id}.npz"),
                        imgs=imgs, acts=acts, states=states, **noisy)
    return len(states)


def choose_split(rng):
    r = rng.random()
    if r < TRAIN_FRACTION:
        return "train"
    if r < TRAIN_FRACTION + VAL_FRACTION:
        return "val"
    return "test"


def compute_and_save_norm_stats(train_dir, out_path):
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

    base_dir = DATA_ROOT   # where the training scripts read it back from
    dirs = {s: os.path.join(base_dir, s) for s in ("train", "val", "test")}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    rng = np.random.default_rng(SEED)
    env = make_env()
    env.action_space.seed(SEED)

    lengths = []
    for i in range(NUM_EPISODES):
        if i % 100 == 0:
            print(f"Collecting episode: {i}/{NUM_EPISODES}")
        imgs, acts, states = collect_episode(env, rng, ep_seed=SEED + i)
        split = choose_split(rng)
        lengths.append(save_episode(dirs[split], i, imgs, acts, states, rng))
    env.close()

    lengths = np.array(lengths)
    print(f"\nEpisodes: {len(lengths)} | mean={lengths.mean():.1f} "
          f"median={np.median(lengths):.0f} min={lengths.min()} max={lengths.max()} "
          f"| % reaching MAX_STEPS: {(lengths >= MAX_STEPS).mean() * 100:.1f}%")

    compute_and_save_norm_stats(dirs["train"], os.path.join(base_dir, "norm_stats.npz"))
    print("Data collection finished successfully!")