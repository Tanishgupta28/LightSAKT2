"""Persistent user hold, independent of immutable scientific run configuration."""
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
HOLD = PROJECT/'experiments/USER_TRAINING_HOLD.json'


def training_is_held(path=None):
    # Presence is enough: an incomplete or malformed marker must fail closed.
    return Path(path if path is not None else HOLD).exists()


def require_training_enabled(path=None):
    if training_is_held(path):
        raise SystemExit('LightSAKT2 is on explicit user hold for HiTSKT priority; '
                         'no automatic training/GPU analysis/recovery launch is permitted. '
                         'Preserve all states; resume requires a later explicit user decision.')
