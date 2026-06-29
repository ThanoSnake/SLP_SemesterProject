"""
lunar_data_collect_control.py

Control-aware LunarLander dataset collector.

Why this exists:
  The original dataset is mostly heuristic/PID with epsilon=0.20. That is good
  for training a pixel -> physical-state encoder, but too narrow for MPC: the
  learned dynamics is asked to predict counterfactual action sequences that are
  rare in the data. This collector keeps normal heuristic data, but adds more
  exploration, action bursts of length 5, perturbed PID gains, and many wind
  episodes.

Saved keys per episode:
  Old-compatible keys:
    imgs, acts, states, noisy_states_2, noisy_states_5, noisy_states_10

  Extra transition keys:
    next_imgs, next_states, rewards, dones, terminateds, truncateds

The old keys keep the same convention as dataCollect.py:
  imgs[t], states[t], acts[t] are the frame/state/action before env.step().
The extra keys add the exact result of that transition:
  next_imgs[t], next_states[t], rewards[t], dones[t].
"""
import os
from collections import Counter
from multiprocessing import Pool

import gymnasium as gym
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
BASE_DIR = os.environ.get("LUNARLANDER_DATA_DIR", os.path.expanduser("~/lunarlander_control_data"))
NUM_EPISODES = 4000
NUM_WORKERS = min(12, os.cpu_count() or 1)
CHUNKSIZE = 4
MAX_STEPS = 400
IMG_H, IMG_W = 80, 120
SEED = 0

TRAIN_FRACTION = 0.8
VAL_FRACTION = 0.1
TEST_FRACTION = 0.1

N_ACTIONS = 4
BURST_LEN = 5                 # Same horizon length as the current MPC.
BURST_START_PROB = 0.08       # Per-step probability when not already inside a burst.

# Wind-heavy mixture. If you want the earlier 25% wind setting, change wind_mixed
# to 0.25 and heuristic_eps020 to 0.35.
MODE_SPECS = [
    ("heuristic_eps020", 0.30),
    ("heuristic_eps050", 0.15),
    ("random_bursts", 0.15),
    ("perturbed_pid", 0.10),
    ("wind_mixed", 0.30),
]

# Inside wind episodes we still want mostly reasonable control, plus some
# counterfactual coverage.
WIND_SUBPOLICY_SPECS = [
    ("heuristic_eps020", 0.55),
    ("heuristic_eps050", 0.20),
    ("random_bursts", 0.15),
    ("perturbed_pid", 0.10),
]

WIND_POWER_RANGE = (5.0, 20.0)
TURBULENCE_POWER_RANGE = (0.5, 2.0)

NOISE_LEVELS = [(0.025, 2), (0.05, 5), (0.10, 10)]
NOISE_REF = {0: 2.5, 1: 2.0, 4: 2.0}   # dims: x, y, theta

DIRS = {s: os.path.join(BASE_DIR, s) for s in ("train", "val", "test")}

DEFAULT_GAINS = {
    "angle_x": 0.5,
    "angle_vx": 1.0,
    "angle_clip": 0.4,
    "angle_p": 0.5,
    "omega_d": 1.0,
    "hover_abs_x": 0.55,
    "hover_y": 0.5,
    "hover_vy": 0.5,
    "main_thr": 0.05,
    "side_thr": 0.05,
}


def _assert_probs(specs, name):
    total = sum(p for _, p in specs)
    assert abs(total - 1.0) < 1e-8, f"{name} probabilities sum to {total}, not 1.0"


def make_env(enable_wind=False, wind_power=15.0, turbulence_power=1.5):
    last_err = None
    for env_id in ("LunarLander-v3", "LunarLander-v2"):
        try:
            kw = dict(render_mode="rgb_array")
            if enable_wind:
                kw.update(enable_wind=True, wind_power=float(wind_power), turbulence_power=float(turbulence_power))
            return gym.make(env_id, **kw)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"LunarLander not found or wind kwargs unsupported. {last_err}")


def resize_frame(img):
    return np.asarray(Image.fromarray(img).resize((IMG_W, IMG_H)), dtype=np.uint8)


def choose_from_specs(rng, specs):
    r = rng.random()
    acc = 0.0
    for name, p in specs:
        acc += p
        if r < acc:
            return name
    return specs[-1][0]


def choose_split(rng):
    r = rng.random()
    if r < TRAIN_FRACTION:
        return "train"
    if r < TRAIN_FRACTION + VAL_FRACTION:
        return "val"
    return "test"


def sample_gains(rng, perturbed=False):
    gains = dict(DEFAULT_GAINS)
    if not perturbed:
        return gains

    # Mild multiplicative perturbations: diverse but still mostly lander-like.
    for key in ("angle_x", "angle_vx", "angle_p", "omega_d", "hover_abs_x", "hover_y", "hover_vy"):
        gains[key] *= rng.uniform(0.75, 1.25)
    gains["angle_clip"] *= rng.uniform(0.85, 1.15)
    gains["main_thr"] *= rng.uniform(0.75, 1.25)
    gains["side_thr"] *= rng.uniform(0.75, 1.25)
    return gains


def heuristic_action(obs, rng, epsilon=0.20, gains=None):
    if rng.random() < epsilon:
        return int(rng.integers(N_ACTIONS))

    g = DEFAULT_GAINS if gains is None else gains
    x, y, vx, vy, theta, omega, leg1, leg2 = obs
    angle_targ = float(np.clip(x * g["angle_x"] + vx * g["angle_vx"], -g["angle_clip"], g["angle_clip"]))
    hover_targ = g["hover_abs_x"] * abs(x)
    angle_todo = (angle_targ - theta) * g["angle_p"] - omega * g["omega_d"]
    hover_todo = (hover_targ - y) * g["hover_y"] - vy * g["hover_vy"]
    if leg1 or leg2:
        angle_todo, hover_todo = 0.0, -vy * g["hover_vy"]
    if hover_todo > abs(angle_todo) and hover_todo > g["main_thr"]:
        return 2
    if angle_todo < -g["side_thr"]:
        return 3
    if angle_todo > g["side_thr"]:
        return 1
    return 0


def make_episode_config(i):
    rng = np.random.default_rng(SEED + i)
    top_mode = choose_from_specs(rng, MODE_SPECS)

    enable_wind = top_mode == "wind_mixed"
    if enable_wind:
        subpolicy = choose_from_specs(rng, WIND_SUBPOLICY_SPECS)
        wind_power = rng.uniform(*WIND_POWER_RANGE)
        turbulence_power = rng.uniform(*TURBULENCE_POWER_RANGE)
        policy_mode = f"wind_{subpolicy}"
    else:
        subpolicy = top_mode
        wind_power = 0.0
        turbulence_power = 0.0
        policy_mode = top_mode

    epsilon = 0.50 if subpolicy == "heuristic_eps050" else 0.20
    use_bursts = subpolicy == "random_bursts"
    gains = sample_gains(rng, perturbed=(subpolicy == "perturbed_pid"))

    return {
        "top_mode": top_mode,
        "policy_mode": policy_mode,
        "subpolicy": subpolicy,
        "enable_wind": enable_wind,
        "wind_power": float(wind_power),
        "turbulence_power": float(turbulence_power),
        "epsilon": float(epsilon),
        "use_bursts": use_bursts,
        "gains": gains,
    }


def choose_action(obs, rng, cfg, burst_state):
    if cfg["use_bursts"]:
        if burst_state["left"] <= 0 and rng.random() < BURST_START_PROB:
            burst_state["left"] = BURST_LEN
        if burst_state["left"] > 0:
            burst_state["left"] -= 1
            burst_state["count"] += 1
            return int(rng.integers(N_ACTIONS))

    return heuristic_action(obs, rng, epsilon=cfg["epsilon"], gains=cfg["gains"])


def save_episode(save_dir, run_id, arrays, cfg, rng):
    states = arrays["states"]
    if len(states) == 0:
        return 0

    noisy = {}
    for level, label in NOISE_LEVELS:
        noise = np.zeros_like(states)
        for dim, ref in NOISE_REF.items():
            noise[:, dim] = rng.normal(0, level * ref, size=states.shape[0])
        noisy[f"noisy_states_{label}"] = (states + noise).astype(np.float32)

    np.savez_compressed(
        os.path.join(save_dir, f"{run_id}.npz"),
        imgs=arrays["imgs"],
        acts=arrays["acts"],
        states=arrays["states"],
        next_imgs=arrays["next_imgs"],
        next_states=arrays["next_states"],
        rewards=arrays["rewards"],
        dones=arrays["dones"],
        terminateds=arrays["terminateds"],
        truncateds=arrays["truncateds"],
        policy_mode=np.array(cfg["policy_mode"]),
        top_mode=np.array(cfg["top_mode"]),
        wind_enabled=np.array(cfg["enable_wind"]),
        wind_power=np.array(cfg["wind_power"], dtype=np.float32),
        turbulence_power=np.array(cfg["turbulence_power"], dtype=np.float32),
        epsilon=np.array(cfg["epsilon"], dtype=np.float32),
        burst_actions=np.array(arrays["burst_actions"], dtype=np.int32),
        **noisy,
    )
    return len(states)


def _collect_and_save(i):
    rng = np.random.default_rng(SEED + i)
    split_rng = np.random.default_rng(2_000_000_000 + i)
    cfg = make_episode_config(i)
    env = make_env(cfg["enable_wind"], cfg["wind_power"], cfg["turbulence_power"])

    obs, _ = env.reset(seed=SEED + i)
    burst_state = {"left": 0, "count": 0}
    imgs, next_imgs = [], []
    acts, states, next_states = [], [], []
    rewards, dones, terminateds, truncateds = [], [], [], []

    for _ in range(MAX_STEPS):
        img = resize_frame(env.render())
        action = choose_action(obs, rng, cfg, burst_state)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_img = resize_frame(env.render())
        done = terminated or truncated

        imgs.append(img)
        next_imgs.append(next_img)
        acts.append(action)
        states.append(obs)
        next_states.append(next_obs)
        rewards.append(reward)
        dones.append(done)
        terminateds.append(terminated)
        truncateds.append(truncated)

        obs = next_obs
        if done:
            break

    env.close()

    arrays = {
        "imgs": np.asarray(imgs, dtype=np.uint8),
        "next_imgs": np.asarray(next_imgs, dtype=np.uint8),
        "acts": np.asarray(acts, dtype=np.int64),
        "states": np.asarray(states, dtype=np.float32),
        "next_states": np.asarray(next_states, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "dones": np.asarray(dones, dtype=np.bool_),
        "terminateds": np.asarray(terminateds, dtype=np.bool_),
        "truncateds": np.asarray(truncateds, dtype=np.bool_),
        "burst_actions": burst_state["count"],
    }
    split = choose_split(split_rng)
    n = save_episode(DIRS[split], i, arrays, cfg, rng)
    return {
        "n": n,
        "split": split,
        "top_mode": cfg["top_mode"],
        "policy_mode": cfg["policy_mode"],
        "wind": cfg["enable_wind"],
        "burst_actions": burst_state["count"],
    }


def compute_and_save_norm_stats(train_dir, out_path):
    chunks = [np.load(os.path.join(train_dir, f))["states"] for f in os.listdir(train_dir) if f.endswith(".npz")]
    if not chunks:
        raise RuntimeError(f"No train episodes found in {train_dir}")
    all_states = np.concatenate(chunks, axis=0)
    mean = all_states.mean(axis=0)
    std = all_states.std(axis=0) + 1e-8
    np.savez(out_path, mean=mean, std=std)
    print("State mean:", mean)
    print("State std :", std)


def print_counts(title, values):
    counts = Counter(values)
    total = sum(counts.values())
    print(title)
    for k in sorted(counts):
        print(f"  {k:<22} {counts[k]:5d}  ({100.0 * counts[k] / max(total, 1):5.1f}%)")


if __name__ == "__main__":
    _assert_probs(MODE_SPECS, "MODE_SPECS")
    _assert_probs(WIND_SUBPOLICY_SPECS, "WIND_SUBPOLICY_SPECS")
    assert abs(TRAIN_FRACTION + VAL_FRACTION + TEST_FRACTION - 1.0) < 1e-6
    for d in DIRS.values():
        os.makedirs(d, exist_ok=True)

    print(f"Collecting {NUM_EPISODES} control-aware LunarLander episodes")
    print(f"Output: {BASE_DIR}")
    print(f"Workers: {NUM_WORKERS}")
    print("Mode mixture:")
    for name, p in MODE_SPECS:
        print(f"  {name:<22} {100.0 * p:5.1f}%")

    results = []
    with Pool(NUM_WORKERS) as pool:
        for done, res in enumerate(pool.imap_unordered(_collect_and_save, range(NUM_EPISODES), chunksize=CHUNKSIZE), 1):
            results.append(res)
            if done % 200 == 0:
                print(f"  {done}/{NUM_EPISODES}")

    lengths = np.asarray([r["n"] for r in results], dtype=np.int32)
    print(
        f"\nEpisodes: {len(lengths)} | mean={lengths.mean():.1f} "
        f"median={np.median(lengths):.0f} min={lengths.min()} max={lengths.max()} "
        f"| reached MAX_STEPS: {(lengths >= MAX_STEPS).mean() * 100:.1f}%"
    )
    print(f"Wind episodes: {np.mean([r['wind'] for r in results]) * 100:.1f}%")
    print(f"Total random burst actions: {sum(r['burst_actions'] for r in results)}")
    print_counts("\nSplits:", [r["split"] for r in results])
    print_counts("\nTop modes:", [r["top_mode"] for r in results])
    print_counts("\nPolicy modes:", [r["policy_mode"] for r in results])

    compute_and_save_norm_stats(DIRS["train"], os.path.join(BASE_DIR, "norm_stats.npz"))
    print("Control-aware dataset collection finished.")
