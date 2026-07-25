import os
from pathlib import Path


#
# file is at project root
#

PROJECT_ROOT = Path(__file__).resolve().parent


#
# run outputs (checkpoints, latent caches, figures)
#
# Everything a script writes goes under here. On Kaggle set OUTPUT_DIR=/kaggle/working.
#

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", PROJECT_ROOT / "outputs"))


def outputs(*parts) -> str:
    """Path under OUTPUT_DIR, e.g. outputs("cartpole_p1_vae") or outputs("sindy/out.png")."""
    p = OUTPUT_DIR.joinpath(*parts)
    return str(p)


#
# resolution function
#

def resolve_dir(env_var: str, candidates: list[Path], default: Path) -> Path:
    # set manually to override
    val = os.environ.get(env_var)
    if val:
        return Path(val)
    # first existing candidate
    for path in candidates:
        if path.exists():
            return path
    # default fallback
    return default


#
# kaggle input path
#

KAGGLE_INPUT = resolve_dir(
    env_var="KAGGLE_INPUT",
    candidates=[
        Path("/kaggle/input/datasets/iliasbakos"),
        Path("/kaggle/input/datasets/thanosnake"),
    ],
    default=Path("/kaggle/input/datasets/thanasisrigas"),
)

# set variable in kaggle
# os.environ["DATA_DIR"] = "/path/to/dataset/dir"

#
# data
#

DATA_DIR = resolve_dir(
    env_var="DATA_DIR",
    candidates=[
        KAGGLE_INPUT,
    ],
    default=PROJECT_ROOT / "data",
)

CARTPOLE_DATA = resolve_dir(
    env_var="CARTPOLE_DATA",
    candidates=[
        KAGGLE_INPUT / "cartpole-data" / "cartpole-data",
    ],
    default=DATA_DIR / "cartpole-data",
)

LUNARLANDER_DATA = resolve_dir(
    env_var="LUNARLANDER_DATA",
    candidates=[
        KAGGLE_INPUT / "lunarlander-data" / "lunarlander-data",
    ],
    default=DATA_DIR / "lunarlander-data",
)


#
# weights
#

WEIGHTS_DIR = resolve_dir(
    env_var="WEIGHTS_DIR",
    candidates=[],
    default=PROJECT_ROOT / "weights",
)

CARTPOLE_WEIGHTS = resolve_dir(
    env_var="CARTPOLE_WEIGHTS",
    candidates=[
        KAGGLE_INPUT / "cartpole-weights" / "cartpole-weights",
    ],
    default=WEIGHTS_DIR / "cartpole-weights",
)

LUNARLANDER_WEIGHTS = resolve_dir(
    env_var="LUNARLANDER_WEIGHTS",
    candidates=[
        KAGGLE_INPUT / "lunarlander-weights" / "lunarlander-weights",
    ],
    default=WEIGHTS_DIR / "lunarlander-weights",
)

CARTPOLE_BASELINE_VAE = resolve_dir(
    env_var="CARTPOLE_BASELINE_VAE",
    candidates=[],
    default=CARTPOLE_WEIGHTS / "cartpole_baseline_vae.pth",
)

CARTPOLE_BASELINE_LSTM = resolve_dir(
    env_var="CARTPOLE_BASELINE_LSTM",
    candidates=[],
    default=CARTPOLE_WEIGHTS / "cartpole_baseline_lstm.pth",
)

CARTPOLE_P1_VAE = resolve_dir(
    env_var="CARTPOLE_P1_VAE",
    candidates=[],
    default=CARTPOLE_WEIGHTS / "cartpole_p1_vae.pth",
)

CARTPOLE_P1_LSTM = resolve_dir(
    env_var="CARTPOLE_P1_LSTM",
    candidates=[],
    default=CARTPOLE_WEIGHTS / "cartpole_p1_lstm.pth",
)

CARTPOLE_P2_VAE = resolve_dir(
    env_var="CARTPOLE_P2_VAE",
    candidates=[],
    default=CARTPOLE_WEIGHTS / "cartpole_p2_vae.pth",
)

CARTPOLE_P2_LSTM = resolve_dir(
    env_var="CARTPOLE_P2_LSTM",
    candidates=[],
    default=CARTPOLE_WEIGHTS / "cartpole_p2_lstm.pth",
)

CARTPOLE_P3_SEMI_VAE = resolve_dir(
    env_var="CARTPOLE_P3_SEMI_VAE",
    candidates=[],
    default=CARTPOLE_WEIGHTS / "cartpole_p3_semi_vae.pth",
)

CARTPOLE_P3_SEMI_LSTM = resolve_dir(
    env_var="CARTPOLE_P3_SEMI_LSTM",
    candidates=[],
    default=CARTPOLE_WEIGHTS / "cartpole_p3_semi_lstm.pth",
)

CARTPOLE_P3_WEAK_VAE = resolve_dir(
    env_var="CARTPOLE_P3_WEAK_VAE",
    candidates=[],
    default=CARTPOLE_WEIGHTS / "cartpole_p3_weak_vae.pth",
)

CARTPOLE_P3_WEAK_LSTM = resolve_dir(
    env_var="CARTPOLE_P3_WEAK_LSTM",
    candidates=[],
    default=CARTPOLE_WEIGHTS / "cartpole_p3_weak_lstm.pth",
)

CARTPOLE_P4_VAE = resolve_dir(
    env_var="CARTPOLE_P4_VAE",
    candidates=[],
    default=CARTPOLE_WEIGHTS / "cartpole_p4_vae.pth",
)

LUNARLANDER_BASELINE_VAE = resolve_dir(
    env_var="LUNARLANDER_BASELINE_VAE",
    candidates=[],
    default=LUNARLANDER_WEIGHTS / "lunarlander_baseline_vae.pth",
)

LUNARLANDER_BASELINE_LSTM = resolve_dir(
    env_var="LUNARLANDER_BASELINE_LSTM",
    candidates=[],
    default=LUNARLANDER_WEIGHTS / "lunarlander_baseline_lstm.pth",
)

LUNARLANDER_P1_VAE = resolve_dir(
    env_var="LUNARLANDER_P1_VAE",
    candidates=[],
    default=LUNARLANDER_WEIGHTS / "lunarlander_p1_vae.pth",
)

LUNARLANDER_P1_LSTM = resolve_dir(
    env_var="LUNARLANDER_P1_LSTM",
    candidates=[],
    default=LUNARLANDER_WEIGHTS / "lunarlander_p1_lstm.pth",
)

LUNARLANDER_P2_VAE = resolve_dir(
    env_var="LUNARLANDER_P2_VAE",
    candidates=[],
    default=LUNARLANDER_WEIGHTS / "lunarlander_p2_vae.pth",
)

LUNARLANDER_P2_LSTM = resolve_dir(
    env_var="LUNARLANDER_P2_LSTM",
    candidates=[],
    default=LUNARLANDER_WEIGHTS / "lunarlander_p2_lstm.pth",
)

LUNARLANDER_P3_WEAK_VAE = resolve_dir(
    env_var="LUNARLANDER_P3_WEAK_VAE",
    candidates=[],
    default=LUNARLANDER_WEIGHTS / "lunarlander_p3_weak_vae.pth",
)

LUNARLANDER_P3_WEAK_LSTM = resolve_dir(
    env_var="LUNARLANDER_P3_WEAK_LSTM",
    candidates=[],
    default=LUNARLANDER_WEIGHTS / "lunarlander_p3_weak_lstm.pth",
)

LUNARLANDER_P3_SEMI_VAE = resolve_dir(
    env_var="LUNARLANDER_P3_SEMI_VAE",
    candidates=[],
    default=LUNARLANDER_WEIGHTS / "lunarlander_p3_semi_vae.pth",
)

LUNARLANDER_P3_SEMI_LSTM = resolve_dir(
    env_var="LUNARLANDER_P3_SEMI_LSTM",
    candidates=[],
    default=LUNARLANDER_WEIGHTS / "lunarlander_p3_semi_lstm.pth",
)

LUNARLANDER_P4_VAE = resolve_dir(
    env_var="LUNARLANDER_P4_VAE",
    candidates=[],
    default=LUNARLANDER_WEIGHTS / "lunarlander_p4_vae.pth",
)


#
# coverage-oriented control dataset + the encoder/LSTM retrained on it
# (produced by lunar_data_collect_control.py; not shipped in weights/)
#

LUNARLANDER_CONTROL_DATA = resolve_dir(
    env_var="LUNARLANDER_CONTROL_DATA",
    candidates=[
        KAGGLE_INPUT / "lunarlander-control-data" / "lunarlander_control_data_8000",
    ],
    default=DATA_DIR / "lunarlander-control-data",
)

LUNARLANDER_CONTROL_VAE = resolve_dir(
    env_var="LUNARLANDER_CONTROL_VAE",
    candidates=[],
    default=LUNARLANDER_WEIGHTS / "lunarlander_control_vae.pth",
)

LUNARLANDER_CONTROL_LSTM = resolve_dir(
    env_var="LUNARLANDER_CONTROL_LSTM",
    candidates=[],
    default=LUNARLANDER_WEIGHTS / "lunarlander_control_lstm.pth",
)


#
# artifacts produced by the extension scripts themselves (default under OUTPUT_DIR)
#

CARTPOLE_HNN = resolve_dir(
    env_var="CARTPOLE_HNN",
    candidates=[],
    default=OUTPUT_DIR / "cartpole_hnn" / "hnn_baseline.pth",
)

CARTPOLE_DROPOUT_VAE = resolve_dir(
    env_var="CARTPOLE_DROPOUT_VAE",
    candidates=[],
    default=OUTPUT_DIR / "cartpole_uncertainty" / "vae_dropout_best.pth",
)

CARTPOLE_DROPOUT_LSTM = resolve_dir(
    env_var="CARTPOLE_DROPOUT_LSTM",
    candidates=[],
    default=OUTPUT_DIR / "cartpole_uncertainty" / "lstm_dropout_best.pth",
)

CARTPOLE_HYBRID_BASELINE = resolve_dir(
    env_var="CARTPOLE_HYBRID_BASELINE",
    candidates=[],
    default=OUTPUT_DIR / "cartpole_sindy" / "hybrid_baseline.pth",
)

CARTPOLE_HYBRID_P1 = resolve_dir(
    env_var="CARTPOLE_HYBRID_P1",
    candidates=[],
    default=OUTPUT_DIR / "cartpole_sindy" / "hybrid_principle_1.pth",
)
