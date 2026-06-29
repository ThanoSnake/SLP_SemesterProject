"""
lunar_data_collect_elite.py

Collect an elite/recovery add-on dataset for the MPC control experiments.

Targets, by default:
  clean_elite: 1000 episodes with return >= 200
  clean_near : 1000 episodes with return >= 150
  wind_elite : 1000 episodes with return >= 200
  wind_near  : 1000 episodes with return >= 150

The saved episode format is compatible with lunar_data_collect_control.py:
  imgs, acts, states, next_imgs, next_states, rewards, dones,
  terminateds, truncateds, noisy_states_2/5/10

Extra metadata marks the target bucket, total return, candidate policy, and
whether the episode strictly passed the requested threshold.

Run:
  python lunarlander/lunar_data_collect_elite.py

Useful overrides:
  LUNARLANDER_ELITE_DATA_DIR=lunarlander/lunarlander_elite_recovery_4000
  CLEAN_ELITE_QUOTA=1000 CLEAN_NEAR_QUOTA=1000
  WIND_ELITE_QUOTA=1000 WIND_NEAR_QUOTA=1000
  NUM_WORKERS=12 MAX_ATTEMPTS_PER_ACCEPTED=300

Set NEAR_MAX_RETURN=200 if you explicitly want the near buckets to be the
narrow band 150 <= return < 200. The default is broader because successful
heuristic trajectories often jump just above 200, and rejecting those makes
collection unnecessarily brittle.
"""
import os
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np

try:
    from lunar_data_collect_control import (
        DEFAULT_GAINS,
        IMG_H,
        IMG_W,
        MAX_STEPS as CONTROL_MAX_STEPS,
        NOISE_LEVELS,
        NOISE_REF,
        compute_and_save_norm_stats,
        heuristic_action,
        make_env,
        resize_frame,
    )
except ImportError:
    from .lunar_data_collect_control import (
        DEFAULT_GAINS,
        IMG_H,
        IMG_W,
        MAX_STEPS as CONTROL_MAX_STEPS,
        NOISE_LEVELS,
        NOISE_REF,
        compute_and_save_norm_stats,
        heuristic_action,
        make_env,
        resize_frame,
    )


HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.environ.get(
    "LUNARLANDER_ELITE_DATA_DIR",
    os.path.join(HERE, "lunarlander_elite_recovery_4000"),
)

SEED = int(os.environ.get("SEED", "40_000_000"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", str(min(12, os.cpu_count() or 1))))
CHUNKSIZE = int(os.environ.get("CHUNKSIZE", "1"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", str(CONTROL_MAX_STEPS)))
MAX_ATTEMPTS_PER_ACCEPTED = int(os.environ.get("MAX_ATTEMPTS_PER_ACCEPTED", "300"))
SAVE_BEST_FALLBACK = os.environ.get("SAVE_BEST_FALLBACK", "0") == "1"
RESUME = os.environ.get("RESUME", "1") == "1"
COLLECTION_STRATEGY = os.environ.get("COLLECTION_STRATEGY", "routed").strip().lower()
BATCH_MULTIPLIER = int(os.environ.get("BATCH_MULTIPLIER", "4"))

TRAIN_FRACTION = 0.8
VAL_FRACTION = 0.1
TEST_FRACTION = 0.1

ELITE_THRESHOLD = float(os.environ.get("ELITE_THRESHOLD", "200.0"))
NEAR_THRESHOLD = float(os.environ.get("NEAR_THRESHOLD", "150.0"))
NEAR_MAX_RETURN_RAW = os.environ.get("NEAR_MAX_RETURN", "").strip().lower()
NEAR_MAX_RETURN = None if NEAR_MAX_RETURN_RAW in ("", "none", "inf", "infinity") else float(NEAR_MAX_RETURN_RAW)

CLEAN_ELITE_QUOTA = int(os.environ.get("CLEAN_ELITE_QUOTA", "1000"))
CLEAN_NEAR_QUOTA = int(os.environ.get("CLEAN_NEAR_QUOTA", "1000"))
WIND_ELITE_QUOTA = int(os.environ.get("WIND_ELITE_QUOTA", "1000"))
WIND_NEAR_QUOTA = int(os.environ.get("WIND_NEAR_QUOTA", "1000"))

WIND_POWER_RANGE = (
    float(os.environ.get("WIND_POWER_MIN", "5.0")),
    float(os.environ.get("WIND_POWER_MAX", "20.0")),
)
TURBULENCE_POWER_RANGE = (
    float(os.environ.get("TURBULENCE_POWER_MIN", "0.5")),
    float(os.environ.get("TURBULENCE_POWER_MAX", "2.0")),
)

DIRS = {s: os.path.join(BASE_DIR, s) for s in ("train", "val", "test")}

ELITE_POLICY_SPECS = [
    ("heuristic_eps000", 0.50),
    ("heuristic_eps002", 0.20),
    ("heuristic_eps005", 0.10),
    ("perturbed_pid_light", 0.15),
    ("recovery_kick", 0.05),
]

NEAR_POLICY_SPECS = [
    ("heuristic_eps000", 0.25),
    ("heuristic_eps002", 0.15),
    ("heuristic_eps005", 0.10),
    ("perturbed_pid_light", 0.30),
    ("recovery_kick", 0.20),
]

TARGETS = [
    {
        "bucket": "clean_elite",
        "domain": "clean",
        "quota": CLEAN_ELITE_QUOTA,
        "min_return": ELITE_THRESHOLD,
        "max_return": None,
    },
    {
        "bucket": "clean_near",
        "domain": "clean",
        "quota": CLEAN_NEAR_QUOTA,
        "min_return": NEAR_THRESHOLD,
        "max_return": NEAR_MAX_RETURN,
    },
    {
        "bucket": "wind_elite",
        "domain": "wind",
        "quota": WIND_ELITE_QUOTA,
        "min_return": ELITE_THRESHOLD,
        "max_return": None,
    },
    {
        "bucket": "wind_near",
        "domain": "wind",
        "quota": WIND_NEAR_QUOTA,
        "min_return": NEAR_THRESHOLD,
        "max_return": NEAR_MAX_RETURN,
    },
]
TARGET_BY_BUCKET = {t["bucket"]: t for t in TARGETS}


def _assert_probs(specs, name):
    total = sum(p for _, p in specs)
    assert abs(total - 1.0) < 1e-8, f"{name} probabilities sum to {total}, not 1.0"


def _choose_from_specs(rng, specs):
    r = rng.random()
    acc = 0.0
    for name, p in specs:
        acc += p
        if r < acc:
            return name
    return specs[-1][0]


def _choose_split(rng):
    r = rng.random()
    if r < TRAIN_FRACTION:
        return "train"
    if r < TRAIN_FRACTION + VAL_FRACTION:
        return "val"
    return "test"


def _sample_light_gains(rng):
    gains = dict(DEFAULT_GAINS)
    for key in ("angle_x", "angle_vx", "angle_p", "omega_d", "hover_abs_x", "hover_y", "hover_vy"):
        gains[key] *= rng.uniform(0.90, 1.10)
    gains["angle_clip"] *= rng.uniform(0.95, 1.05)
    gains["main_thr"] *= rng.uniform(0.90, 1.10)
    gains["side_thr"] *= rng.uniform(0.90, 1.10)
    return gains


def _make_candidate_config(rng, target):
    bucket = target["bucket"]
    domain = target["domain"]
    enable_wind = domain == "wind"
    specs = ELITE_POLICY_SPECS if bucket.endswith("elite") else NEAR_POLICY_SPECS
    candidate_policy = _choose_from_specs(rng, specs)

    epsilon_by_policy = {
        "heuristic_eps000": 0.00,
        "heuristic_eps002": 0.02,
        "heuristic_eps005": 0.05,
        "perturbed_pid_light": 0.02,
        "recovery_kick": 0.00,
    }
    gains = _sample_light_gains(rng) if candidate_policy in ("perturbed_pid_light", "recovery_kick") else dict(DEFAULT_GAINS)

    if enable_wind:
        wind_power = rng.uniform(*WIND_POWER_RANGE)
        turbulence_power = rng.uniform(*TURBULENCE_POWER_RANGE)
    else:
        wind_power = 0.0
        turbulence_power = 0.0

    kick_start = -1
    kick_actions = []
    if candidate_policy == "recovery_kick":
        kick_start = int(rng.integers(8, 80))
        kick_len = int(rng.integers(2, 6))
        kick_actions = [int(a) for a in rng.integers(0, 4, size=kick_len)]

    return {
        "bucket": bucket,
        "proposal_bucket": bucket,
        "top_mode": domain,
        "policy_mode": f"{bucket}_{candidate_policy}",
        "candidate_policy": candidate_policy,
        "enable_wind": enable_wind,
        "wind_power": float(wind_power),
        "turbulence_power": float(turbulence_power),
        "epsilon": float(epsilon_by_policy[candidate_policy]),
        "gains": gains,
        "kick_start": kick_start,
        "kick_actions": kick_actions,
    }


def _target_accepts_return(total_return, target):
    if total_return < target["min_return"]:
        return False
    max_return = target["max_return"]
    if max_return is not None and total_return >= max_return:
        return False
    return True


def _score_for_target(total_return, target):
    if target["max_return"] is None:
        return total_return
    center = 0.5 * (target["min_return"] + target["max_return"])
    return -abs(total_return - center)


def _choose_action(obs, step_idx, rng, cfg):
    kick_actions = cfg["kick_actions"]
    kick_start = cfg["kick_start"]
    if kick_actions and kick_start <= step_idx < kick_start + len(kick_actions):
        return int(kick_actions[step_idx - kick_start]), True
    action = heuristic_action(obs, rng, epsilon=cfg["epsilon"], gains=cfg["gains"])
    return int(action), False


def _rollout_candidate(candidate_seed, target):
    rng = np.random.default_rng(candidate_seed)
    cfg = _make_candidate_config(rng, target)
    env = make_env(cfg["enable_wind"], cfg["wind_power"], cfg["turbulence_power"])

    obs, _ = env.reset(seed=int(candidate_seed % (2**31 - 1)))
    imgs, next_imgs = [], []
    acts, states, next_states = [], [], []
    rewards, dones, terminateds, truncateds = [], [], [], []
    forced_actions = 0

    for step_idx in range(MAX_STEPS):
        img = resize_frame(env.render())
        action, forced = _choose_action(obs, step_idx, rng, cfg)
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
        forced_actions += int(forced)

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
        "burst_actions": forced_actions,
    }
    total_return = float(arrays["rewards"].sum())
    return arrays, cfg, total_return


def _collect_accepted_task(task):
    target, local_id = task
    base_seed = SEED + target["target_idx"] * 10_000_000 + local_id * MAX_ATTEMPTS_PER_ACCEPTED
    best = None
    best_score = -np.inf

    for attempt_idx in range(MAX_ATTEMPTS_PER_ACCEPTED):
        candidate_seed = base_seed + attempt_idx
        arrays, cfg, total_return = _rollout_candidate(candidate_seed, target)
        score = _score_for_target(total_return, target)
        if score > best_score:
            best = (arrays, cfg, total_return, candidate_seed, attempt_idx)
            best_score = score

        if _target_accepts_return(total_return, target):
            cfg.update(
                {
                    "target_idx": target["target_idx"],
                    "target_local_id": local_id,
                    "candidate_seed": candidate_seed,
                    "attempt_idx": attempt_idx,
                    "episode_return": total_return,
                    "min_return": target["min_return"],
                    "max_return": np.inf if target["max_return"] is None else target["max_return"],
                    "accepted_strict": True,
                }
            )
            return arrays, cfg

    if SAVE_BEST_FALLBACK and best is not None:
        arrays, cfg, total_return, candidate_seed, attempt_idx = best
        cfg.update(
            {
                "target_idx": target["target_idx"],
                "target_local_id": local_id,
                "candidate_seed": candidate_seed,
                "attempt_idx": attempt_idx,
                "episode_return": total_return,
                "min_return": target["min_return"],
                "max_return": np.inf if target["max_return"] is None else target["max_return"],
                "accepted_strict": False,
            }
        )
        return arrays, cfg

    best_return = float(best[2]) if best is not None else float("nan")
    raise RuntimeError(
        f"Could not find {target['bucket']} episode {local_id} after "
        f"{MAX_ATTEMPTS_PER_ACCEPTED} attempts. Best return={best_return:.2f}. "
        "Increase MAX_ATTEMPTS_PER_ACCEPTED or set SAVE_BEST_FALLBACK=1."
    )


def _save_episode(save_dir, file_stem, arrays, cfg):
    states = arrays["states"]
    if len(states) == 0:
        return 0

    noise_rng = np.random.default_rng(int(cfg["candidate_seed"]) + 999_983)
    noisy = {}
    for level, label in NOISE_LEVELS:
        noise = np.zeros_like(states)
        for dim, ref in NOISE_REF.items():
            noise[:, dim] = noise_rng.normal(0, level * ref, size=states.shape[0])
        noisy[f"noisy_states_{label}"] = (states + noise).astype(np.float32)

    np.savez_compressed(
        os.path.join(save_dir, f"{file_stem}.npz"),
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
        elite_bucket=np.array(cfg["bucket"]),
        proposal_bucket=np.array(cfg.get("proposal_bucket", cfg["bucket"])),
        candidate_policy=np.array(cfg["candidate_policy"]),
        episode_return=np.array(cfg["episode_return"], dtype=np.float32),
        min_return=np.array(cfg["min_return"], dtype=np.float32),
        max_return=np.array(cfg["max_return"], dtype=np.float32),
        accepted_strict=np.array(cfg["accepted_strict"]),
        candidate_seed=np.array(cfg["candidate_seed"], dtype=np.int64),
        attempt_idx=np.array(cfg["attempt_idx"], dtype=np.int32),
        target_local_id=np.array(cfg["target_local_id"], dtype=np.int32),
        **noisy,
    )
    return len(states)


def _scalar(v):
    a = np.asarray(v)
    return a.item() if a.shape == () else a


def _scalar_str(v):
    x = _scalar(v)
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


def _existing_task_ids():
    existing = set()
    counts = Counter()
    duplicate_counts = Counter()

    for split_dir in DIRS.values():
        if not os.path.isdir(split_dir):
            continue
        for name in os.listdir(split_dir):
            if not name.endswith(".npz"):
                continue
            path = os.path.join(split_dir, name)
            try:
                with np.load(path) as ep:
                    if "elite_bucket" not in ep or "target_local_id" not in ep:
                        continue
                    bucket = _scalar_str(ep["elite_bucket"])
                    local_id = int(_scalar(ep["target_local_id"]))
            except Exception:
                continue

            key = (bucket, local_id)
            if key in existing:
                duplicate_counts[bucket] += 1
            existing.add(key)
            counts[bucket] += 1

    return existing, counts, duplicate_counts


def _build_tasks(existing=None):
    existing = existing or set()
    tasks = []
    for target_idx, target in enumerate(TARGETS):
        target = dict(target)
        target["target_idx"] = target_idx
        for local_id in range(target["quota"]):
            if (target["bucket"], local_id) in existing:
                continue
            tasks.append((target, local_id))
    rng = np.random.default_rng(SEED + 12345)
    order = rng.permutation(len(tasks))
    return [tasks[int(i)] for i in order]


def _used_ids_by_bucket(existing):
    used = {target["bucket"]: set() for target in TARGETS}
    for bucket, local_id in existing:
        if bucket in used:
            used[bucket].add(local_id)
    return used


def _next_available_local_id(bucket, used_ids):
    quota = TARGET_BY_BUCKET[bucket]["quota"]
    for local_id in range(quota):
        if local_id not in used_ids[bucket]:
            return local_id
    return None


def _remaining_counts(used_ids):
    remaining = {}
    for target in TARGETS:
        bucket = target["bucket"]
        remaining[bucket] = max(target["quota"] - len(used_ids[bucket]), 0)
    return remaining


def _route_bucket(domain, total_return, remaining):
    elite_bucket = f"{domain}_elite"
    near_bucket = f"{domain}_near"

    # Prefer the stricter bucket when both buckets can accept the same return.
    if remaining.get(elite_bucket, 0) > 0 and _target_accepts_return(total_return, TARGET_BY_BUCKET[elite_bucket]):
        return elite_bucket
    if remaining.get(near_bucket, 0) > 0 and _target_accepts_return(total_return, TARGET_BY_BUCKET[near_bucket]):
        return near_bucket
    return None


def _choose_routed_target(rng, remaining):
    active_domains = []
    weights = []
    for domain in ("clean", "wind"):
        total = remaining.get(f"{domain}_elite", 0) + remaining.get(f"{domain}_near", 0)
        if total > 0:
            active_domains.append(domain)
            weights.append(total)
    if not active_domains:
        return None

    weights = np.asarray(weights, dtype=np.float64)
    domain = active_domains[int(rng.choice(len(active_domains), p=weights / weights.sum()))]
    profile_weights = np.asarray(
        [remaining.get(f"{domain}_elite", 0), remaining.get(f"{domain}_near", 0)],
        dtype=np.float64,
    )
    profile_names = ["elite", "near"]
    if profile_weights.sum() <= 0:
        return None
    profile = profile_names[int(rng.choice(2, p=profile_weights / profile_weights.sum()))]
    return dict(TARGET_BY_BUCKET[f"{domain}_{profile}"])


def _rollout_routed_candidate_task(task):
    candidate_seed, target = task
    arrays, cfg, total_return = _rollout_candidate(candidate_seed, target)
    cfg["candidate_seed"] = candidate_seed
    return arrays, cfg, total_return


def _fmt_target(target):
    max_ret = target["max_return"]
    if max_ret is None:
        return f"{target['bucket']}: n={target['quota']} return >= {target['min_return']:.0f}"
    return f"{target['bucket']}: n={target['quota']} {target['min_return']:.0f} <= return < {max_ret:.0f}"


def _print_counts(title, values):
    counts = Counter(values)
    total = sum(counts.values())
    print(title)
    for k in sorted(counts):
        print(f"  {k:<28} {counts[k]:5d}  ({100.0 * counts[k] / max(total, 1):5.1f}%)")


if __name__ == "__main__":
    _assert_probs(ELITE_POLICY_SPECS, "ELITE_POLICY_SPECS")
    _assert_probs(NEAR_POLICY_SPECS, "NEAR_POLICY_SPECS")
    assert abs(TRAIN_FRACTION + VAL_FRACTION + TEST_FRACTION - 1.0) < 1e-6
    for d in DIRS.values():
        os.makedirs(d, exist_ok=True)

    existing, existing_counts, duplicate_counts = _existing_task_ids() if RESUME else (set(), Counter(), Counter())
    used_ids = _used_ids_by_bucket(existing)
    initial_remaining = _remaining_counts(used_ids)
    tasks = _build_tasks(existing) if COLLECTION_STRATEGY == "fixed" else []
    total_remaining = sum(initial_remaining.values())
    total_requested = sum(t["quota"] for t in TARGETS)
    print(f"Collecting elite/recovery LunarLander episodes")
    print(f"Output: {BASE_DIR}")
    print(f"Workers: {NUM_WORKERS}")
    print(f"Collection strategy: {COLLECTION_STRATEGY}")
    print(f"Max attempts per accepted episode: {MAX_ATTEMPTS_PER_ACCEPTED}")
    print(f"Fallback below threshold: {SAVE_BEST_FALLBACK}")
    print(f"Resume existing episodes: {RESUME}")
    print("Targets:")
    for target in TARGETS:
        print(f"  {_fmt_target(target)}")
    if RESUME:
        print("Already present:")
        for target in TARGETS:
            bucket = target["bucket"]
            print(f"  {bucket:<24} {len(used_ids[bucket]):5d}/{target['quota']}")
        if duplicate_counts:
            print("Duplicate target ids detected:")
            for bucket in sorted(duplicate_counts):
                print(f"  {bucket:<24} {duplicate_counts[bucket]:5d}")
    print(f"Remaining to collect: {total_remaining}/{total_requested}")

    started = time.time()
    split_rng = np.random.default_rng(SEED + 777)
    results = []
    bucket_counts = Counter()
    evaluated_candidates = 0

    if COLLECTION_STRATEGY not in ("routed", "fixed"):
        raise ValueError("COLLECTION_STRATEGY must be 'routed' or 'fixed'")

    if total_remaining > 0 and COLLECTION_STRATEGY == "fixed":
        with Pool(NUM_WORKERS) as pool:
            for done, (arrays, cfg) in enumerate(pool.imap_unordered(_collect_accepted_task, tasks, chunksize=CHUNKSIZE), 1):
                split = _choose_split(split_rng)
                file_stem = f"{cfg['bucket']}_{cfg['target_local_id']:04d}_seed{cfg['candidate_seed']}"
                n_steps = _save_episode(DIRS[split], file_stem, arrays, cfg)
                result = {
                    "n": n_steps,
                    "split": split,
                    "bucket": cfg["bucket"],
                    "top_mode": cfg["top_mode"],
                    "policy_mode": cfg["policy_mode"],
                    "candidate_policy": cfg["candidate_policy"],
                    "wind": cfg["enable_wind"],
                    "episode_return": cfg["episode_return"],
                    "attempt_idx": cfg["attempt_idx"],
                    "accepted_strict": cfg["accepted_strict"],
                }
                results.append(result)
                bucket_counts[cfg["bucket"]] += 1

                if done % 25 == 0 or done == total_remaining:
                    elapsed = time.time() - started
                    mean_attempt = np.mean([r["attempt_idx"] + 1 for r in results])
                    print(
                        f"  {done}/{total_remaining} saved | "
                        f"clean_elite={bucket_counts['clean_elite']} "
                        f"clean_near={bucket_counts['clean_near']} "
                        f"wind_elite={bucket_counts['wind_elite']} "
                        f"wind_near={bucket_counts['wind_near']} | "
                        f"mean attempts={mean_attempt:.1f} | elapsed={elapsed / 60:.1f}m"
                    )

    if total_remaining > 0 and COLLECTION_STRATEGY == "routed":
        candidate_rng = np.random.default_rng(SEED + 424_242)
        candidate_counter = 0
        max_candidates = max(1, MAX_ATTEMPTS_PER_ACCEPTED * total_remaining)
        batch_size = max(NUM_WORKERS * BATCH_MULTIPLIER, 1)

        with Pool(NUM_WORKERS) as pool:
            while sum(_remaining_counts(used_ids).values()) > 0:
                remaining = _remaining_counts(used_ids)
                remaining_total = sum(remaining.values())
                n_batch = min(batch_size, max_candidates - candidate_counter)
                if n_batch <= 0:
                    raise RuntimeError(
                        f"Routed collection evaluated {evaluated_candidates} candidates but still needs "
                        f"{remaining_total} episodes. Increase MAX_ATTEMPTS_PER_ACCEPTED."
                    )

                batch = []
                for _ in range(n_batch):
                    target = _choose_routed_target(candidate_rng, remaining)
                    if target is None:
                        break
                    candidate_seed = SEED + 500_000_000 + candidate_counter
                    batch.append((candidate_seed, target))
                    candidate_counter += 1

                for arrays, cfg, total_return in pool.imap_unordered(_rollout_routed_candidate_task, batch, chunksize=CHUNKSIZE):
                    evaluated_candidates += 1
                    remaining = _remaining_counts(used_ids)
                    bucket = _route_bucket(cfg["top_mode"], total_return, remaining)
                    if bucket is None:
                        continue

                    local_id = _next_available_local_id(bucket, used_ids)
                    if local_id is None:
                        continue

                    target = TARGET_BY_BUCKET[bucket]
                    cfg["proposal_bucket"] = cfg.get("proposal_bucket", cfg["bucket"])
                    cfg["bucket"] = bucket
                    cfg["policy_mode"] = f"{bucket}_{cfg['candidate_policy']}"
                    cfg.update(
                        {
                            "target_local_id": local_id,
                            "attempt_idx": 0,
                            "episode_return": total_return,
                            "min_return": target["min_return"],
                            "max_return": np.inf if target["max_return"] is None else target["max_return"],
                            "accepted_strict": True,
                        }
                    )

                    split = _choose_split(split_rng)
                    file_stem = f"{cfg['bucket']}_{cfg['target_local_id']:04d}_seed{cfg['candidate_seed']}"
                    n_steps = _save_episode(DIRS[split], file_stem, arrays, cfg)
                    used_ids[bucket].add(local_id)
                    result = {
                        "n": n_steps,
                        "split": split,
                        "bucket": cfg["bucket"],
                        "top_mode": cfg["top_mode"],
                        "policy_mode": cfg["policy_mode"],
                        "candidate_policy": cfg["candidate_policy"],
                        "wind": cfg["enable_wind"],
                        "episode_return": cfg["episode_return"],
                        "attempt_idx": cfg["attempt_idx"],
                        "accepted_strict": cfg["accepted_strict"],
                    }
                    results.append(result)
                    bucket_counts[cfg["bucket"]] += 1

                    done = len(results)
                    if done % 25 == 0 or sum(_remaining_counts(used_ids).values()) == 0:
                        elapsed = time.time() - started
                        current_remaining = _remaining_counts(used_ids)
                        print(
                            f"  {done}/{total_remaining} new saved | "
                            f"clean_elite={bucket_counts['clean_elite']} "
                            f"clean_near={bucket_counts['clean_near']} "
                            f"wind_elite={bucket_counts['wind_elite']} "
                            f"wind_near={bucket_counts['wind_near']} | "
                            f"remaining={sum(current_remaining.values())} | "
                            f"accepted/evaluated={len(results)}/{evaluated_candidates} "
                            f"({100.0 * len(results) / max(evaluated_candidates, 1):.1f}%) | "
                            f"elapsed={elapsed / 60:.1f}m"
                        )

                    if sum(_remaining_counts(used_ids).values()) == 0:
                        break

    if results:
        lengths = np.asarray([r["n"] for r in results], dtype=np.int32)
        returns = np.asarray([r["episode_return"] for r in results], dtype=np.float32)

        print(
            f"\nNew episodes this run: {len(lengths)} | length mean={lengths.mean():.1f} "
            f"median={np.median(lengths):.0f} min={lengths.min()} max={lengths.max()} "
            f"| reached MAX_STEPS: {(lengths >= MAX_STEPS).mean() * 100:.1f}%"
        )
        print(
            f"Returns: mean={returns.mean():.1f} median={np.median(returns):.1f} "
            f"min={returns.min():.1f} max={returns.max():.1f}"
        )
        if COLLECTION_STRATEGY == "fixed":
            attempts = np.asarray([r["attempt_idx"] + 1 for r in results], dtype=np.int32)
            print(
                f"Attempts per saved episode: mean={attempts.mean():.1f} "
                f"median={np.median(attempts):.0f} max={attempts.max()}"
            )
        else:
            print(
                f"Candidates evaluated: {evaluated_candidates} | accepted rate: "
                f"{100.0 * len(results) / max(evaluated_candidates, 1):.1f}%"
            )
        print(f"Strictly accepted: {np.mean([r['accepted_strict'] for r in results]) * 100:.1f}%")
        _print_counts("\nNew splits:", [r["split"] for r in results])
        _print_counts("\nNew buckets:", [r["bucket"] for r in results])
        _print_counts("\nNew top modes:", [r["top_mode"] for r in results])
        _print_counts("\nNew candidate policies:", [r["candidate_policy"] for r in results])
    else:
        print("\nNo new episodes were needed.")

    compute_and_save_norm_stats(DIRS["train"], os.path.join(BASE_DIR, "norm_stats.npz"))
    print("Elite/recovery dataset collection finished.")
