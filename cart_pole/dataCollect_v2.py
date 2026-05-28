"""Collect CartPole trajectories for world-model training.

Each episode is saved as an .npz file with keys:
    imgs   : (T, IMG_H, IMG_W, 3)  uint8     -- rendered frames, pre-resized
    acts   : (T,)                  int64     -- action taken at each step
    states : (T, 4)                float32   -- [cart_pos, cart_vel, pole_angle, pole_ang_vel]

Episodes are filtered to have at least --min_steps frames so they remain
usable as fixed-length LSTM windows.
"""
import argparse
import os

import gymnasium as gym
import numpy as np
from PIL import Image


# VAE expects 80x120 input (its fc_mu = 64 * 10 * 15 after 3 stride-2 convs).
IMG_H, IMG_W = 80, 120


def pd_action(state, p_random):
    """Simple PD controller for CartPole with epsilon-random exploration.

    CartPole action space: 0 = push cart left, 1 = push cart right.
    We push in the direction of `pole_angle + 0.5 * pole_ang_vel`, which keeps
    the pole upright most of the time. The p_random branch injects exploration
    so the trajectories don't all look like a perfect balance.
    """
    if np.random.random() < p_random:
        return int(np.random.randint(2))
    _, _, theta, theta_dot = state
    return 1 if (theta + 0.5 * theta_dot) > 0.0 else 0


def render_resized(env):
    """Render the current frame and resize to (IMG_H, IMG_W)."""
    frame = env.render()
    img = Image.fromarray(frame).resize((IMG_W, IMG_H), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def collect_episode(env, max_steps, p_random, seed=None):
    if seed is not None:
        obs, _ = env.reset(seed=seed)
    else:
        obs, _ = env.reset()

    imgs, acts, states = [], [], []
    for _ in range(max_steps):
        imgs.append(render_resized(env))
        action = pd_action(obs, p_random=p_random)
        acts.append(action)
        states.append(obs.copy())

        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    return (
        np.asarray(imgs,   dtype=np.uint8),
        np.asarray(acts,   dtype=np.int64),
        np.asarray(states, dtype=np.float32),
    )


def collect_split(out_dir, n_episodes, max_steps, min_steps, p_random, seed):
    os.makedirs(out_dir, exist_ok=True)
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    saved, attempt = 0, 0
    while saved < n_episodes:
        imgs, acts, states = collect_episode(
            env, max_steps=max_steps, p_random=p_random, seed=seed + attempt,
        )
        attempt += 1
        if len(imgs) < min_steps:
            continue
        path = os.path.join(out_dir, f"ep_{saved:05d}.npz")
        np.savez_compressed(path, imgs=imgs, acts=acts, states=states)
        saved += 1
        if saved % 50 == 0:
            print(f"  {out_dir}: {saved}/{n_episodes}  (attempts={attempt})")

    env.close()
    print(f"  finished: {saved} episodes in {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir",  type=str,   default="./cartpole_data")
    parser.add_argument("--n_train",   type=int,   default=1000)
    parser.add_argument("--n_test",    type=int,   default=200)
    parser.add_argument("--max_steps", type=int,   default=500)
    parser.add_argument("--min_steps", type=int,   default=33,
                        help="reject episodes shorter than this (LSTM window=32+1)")
    parser.add_argument("--p_random",  type=float, default=0.2,
                        help="probability of a random action at each step")
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    print(f"Collecting {args.n_train} training episodes -> {args.save_dir}/train")
    collect_split(
        out_dir   = os.path.join(args.save_dir, "train"),
        n_episodes= args.n_train,
        max_steps = args.max_steps,
        min_steps = args.min_steps,
        p_random  = args.p_random,
        seed      = args.seed,
    )

    print(f"Collecting {args.n_test} test episodes -> {args.save_dir}/test")
    collect_split(
        out_dir   = os.path.join(args.save_dir, "test"),
        n_episodes= args.n_test,
        max_steps = args.max_steps,
        min_steps = args.min_steps,
        p_random  = args.p_random,
        seed      = args.seed + 10_000,
    )

    print("Done.")


if __name__ == "__main__":
    main()
