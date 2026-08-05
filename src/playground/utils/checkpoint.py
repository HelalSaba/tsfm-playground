from __future__ import annotations

import pathlib
from pathlib import Path
from typing import Any

import torch


def _normalize_checkpoint_value(value: Any) -> Any:
    if isinstance(value, pathlib.PurePath):
        return str(value).replace("\\", "/")
    if isinstance(value, dict):
        return {key: _normalize_checkpoint_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_checkpoint_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalize_checkpoint_value(item) for item in value)
    return value


def serialize_args(args: Any) -> dict[str, Any]:
    raw = vars(args) if not isinstance(args, dict) else args
    return _normalize_checkpoint_value(raw)


def load_checkpoint(
    path: str | Path,
    *,
    map_location: torch.device | str | None = None,
) -> dict[str, Any]:
    # Checkpoints saved on Windows may contain pathlib.WindowsPath objects.
    original_windows_path = pathlib.WindowsPath
    pathlib.WindowsPath = pathlib.PosixPath
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    finally:
        pathlib.WindowsPath = original_windows_path

    if isinstance(checkpoint.get("args"), dict):
        checkpoint["args"] = _normalize_checkpoint_value(checkpoint["args"])
    return checkpoint
