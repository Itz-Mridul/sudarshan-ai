"""Small helpers for loading the checkpoint formats used by Sudarshana."""

from typing import Any


def extract_state_dict(checkpoint: Any):
    """Return a model state dict from either the legacy or metadata format.

    Training checkpoints are saved as a metadata dictionary containing a
    ``state_dict`` key, while older artifacts are bare state dictionaries.
    Keeping the compatibility rule in one place prevents evaluation and
    deployment code from accidentally passing the metadata wrapper to
    ``load_state_dict``.
    """
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict):
                return candidate
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(
        "Unsupported checkpoint format: expected a state dict or a metadata "
        f"dict, got {type(checkpoint).__name__}."
    )
