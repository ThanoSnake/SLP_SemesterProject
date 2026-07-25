"""
lstm_baseline_control_kaggle.py

Kaggle entrypoint for lstm_baseline_control.py.

thanasis-notebook.ipynb does:
  !git clone ...
  %cd SLP_SemesterProject
  !python3 lunarlander/<script>.py

This wrapper fits that workflow:
  * sets Kaggle defaults for VAE_CKPT, NORM_STATS, OUT_ROOT
  * auto-discovers the control/elite dataset roots under /kaggle/input
  * then runs the normal lstm_baseline_control.py

If auto-discovery does not find the folders, pass them explicitly in the notebook:
  CONTROL_ROOT=/kaggle/input/.../lunarlander_control_data_8000 \
  ELITE_ROOT=/kaggle/input/.../lunarlander_elite_recovery_4000 \
  python3 lunarlander/lstm_baseline_control_kaggle.py
"""
import os
import runpy
from pathlib import Path


KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")

DEFAULT_ORIGINAL_DATA_ROOT = Path(
    "/kaggle/input/datasets/iliasbakos/lunarlander-dataset/lunarlander_data"
)
DEFAULT_MODEL_WEIGHTS = Path(
    "/kaggle/input/datasets/iliasbakos/lunarlander-modelweights/LunarLander_ModelWeights"
)


def _is_split_root(path):
    return all((path / split).is_dir() for split in ("train", "val", "test"))


def _find_split_root(env_name, tokens, candidates):
    explicit = os.environ.get(env_name)
    if explicit:
        p = Path(explicit)
        if not _is_split_root(p):
            raise FileNotFoundError(
                f"{env_name}={p} does not look like a dataset root with train/val/test."
            )
        return str(p)

    for raw in candidates:
        p = Path(raw)
        if _is_split_root(p):
            return str(p)

    if KAGGLE_INPUT.is_dir():
        token_l = [t.lower() for t in tokens]
        for root, dirs, _ in os.walk(KAGGLE_INPUT):
            p = Path(root)
            depth = len(p.relative_to(KAGGLE_INPUT).parts)
            if depth > 5:
                dirs[:] = []
                continue
            haystack = str(p).lower()
            if _is_split_root(p) and any(t in haystack for t in token_l):
                return str(p)

    top = []
    if KAGGLE_INPUT.is_dir():
        top = sorted(x.name for x in KAGGLE_INPUT.iterdir())[:40]
    raise FileNotFoundError(
        f"Could not auto-discover {env_name}. Looked for tokens={tokens}. "
        f"Top-level /kaggle/input entries: {top}. Set {env_name}=... explicitly."
    )


def _find_file(env_name, candidates, filename=None, prefer_tokens=()):
    explicit = os.environ.get(env_name)
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise FileNotFoundError(f"{env_name}={p} does not exist.")
        return str(p)

    for raw in candidates:
        p = Path(raw)
        if p.is_file():
            return str(p)

    if filename and KAGGLE_INPUT.is_dir():
        matches = []
        token_l = [t.lower() for t in prefer_tokens]
        for root, dirs, files in os.walk(KAGGLE_INPUT):
            p = Path(root)
            depth = len(p.relative_to(KAGGLE_INPUT).parts)
            if depth > 6:
                dirs[:] = []
                continue
            if filename in files:
                score = sum(t in str(p).lower() for t in token_l)
                matches.append((score, p / filename))
        if matches:
            matches.sort(key=lambda x: (-x[0], str(x[1])))
            return str(matches[0][1])

    raise FileNotFoundError(
        f"Could not find {env_name}. Set {env_name}=... explicitly in the notebook."
    )


def configure_env():
    control_root = _find_split_root(
        "CONTROL_ROOT",
        tokens=("lunarlander_control_data_8000", "control_data", "control-data"),
        candidates=(
            "/kaggle/input/lunarlander-control-data-8000/lunarlander_control_data_8000",
            "/kaggle/input/lunarlander-control-data-8000",
            "/kaggle/input/lunarlander-control-data/lunarlander_control_data_8000",
            "/kaggle/input/lunarlander-control-data",
        ),
    )
    elite_root = _find_split_root(
        "ELITE_ROOT",
        tokens=("lunarlander_elite_recovery_4000", "elite_recovery", "elite-recovery"),
        candidates=(
            "/kaggle/input/lunarlander-elite-recovery-4000/lunarlander_elite_recovery_4000",
            "/kaggle/input/lunarlander-elite-recovery-4000",
            "/kaggle/input/lunarlander-elite-recovery/lunarlander_elite_recovery_4000",
            "/kaggle/input/lunarlander-elite-recovery",
        ),
    )
    vae_ckpt = _find_file(
        "VAE_CKPT",
        candidates=(DEFAULT_MODEL_WEIGHTS / "lunarlander_baseline_vae.pth",),
        filename="lunarlander_baseline_vae.pth",
        prefer_tokens=("modelweights", "lunarlander"),
    )
    norm_stats = _find_file(
        "NORM_STATS",
        candidates=(DEFAULT_ORIGINAL_DATA_ROOT / "norm_stats.npz",),
        filename="norm_stats.npz",
        prefer_tokens=("lunarlander-dataset", "lunarlander_data"),
    )

    os.environ.setdefault("CONTROL_ROOT", control_root)
    os.environ.setdefault("ELITE_ROOT", elite_root)
    os.environ.setdefault("VAE_CKPT", vae_ckpt)
    os.environ.setdefault("NORM_STATS", norm_stats)
    os.environ.setdefault("OUT_ROOT", str(KAGGLE_WORKING))
    os.environ.setdefault("LATENT_ROOT", str(KAGGLE_WORKING / "lunarlander_baseline_control_latents"))
    os.environ.setdefault("SAVE_DIR", str(KAGGLE_WORKING / "lunarlander_baseline_control_lstm"))

    print("[kaggle baseline-control config]")
    for key in ("CONTROL_ROOT", "ELITE_ROOT", "VAE_CKPT", "NORM_STATS", "LATENT_ROOT", "SAVE_DIR"):
        print(f"  {key}={os.environ[key]}")


if __name__ == "__main__":
    configure_env()
    runpy.run_module("lstm_baseline_control", run_name="__main__")
