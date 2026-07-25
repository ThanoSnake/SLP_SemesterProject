"""
paths.py — makes the project-root config.py importable from this folder.

Scripts here run as `python lunarlander/<script>.py`, so sys.path[0] is lunarlander/ and
the project root is not importable by default. Importing this module puts the root on the
path and re-exports the LunarLander entries of config.py under short names.

Override any of them with the matching environment variable (see config.py), e.g.
    LUNARLANDER_DATA=/path/to/data OUTPUT_DIR=/kaggle/working python lunarlander/vae.py
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config  # noqa: E402

outputs = config.outputs

DATA_ROOT = str(config.LUNARLANDER_DATA)
NORM_STATS = os.path.join(DATA_ROOT, "norm_stats.npz")

BASELINE_VAE = str(config.LUNARLANDER_BASELINE_VAE)
BASELINE_LSTM = str(config.LUNARLANDER_BASELINE_LSTM)
P1_VAE = str(config.LUNARLANDER_P1_VAE)
P1_LSTM = str(config.LUNARLANDER_P1_LSTM)
P2_VAE = str(config.LUNARLANDER_P2_VAE)
P2_LSTM = str(config.LUNARLANDER_P2_LSTM)
P3_SEMI_VAE = str(config.LUNARLANDER_P3_SEMI_VAE)
P3_SEMI_LSTM = str(config.LUNARLANDER_P3_SEMI_LSTM)
P3_WEAK_VAE = str(config.LUNARLANDER_P3_WEAK_VAE)
P3_WEAK_LSTM = str(config.LUNARLANDER_P3_WEAK_LSTM)
P4_VAE = str(config.LUNARLANDER_P4_VAE)

# coverage-oriented control dataset + the encoder/LSTM retrained on it
CONTROL_DATA = str(config.LUNARLANDER_CONTROL_DATA)
CONTROL_VAE = str(config.LUNARLANDER_CONTROL_VAE)
CONTROL_LSTM = str(config.LUNARLANDER_CONTROL_LSTM)
