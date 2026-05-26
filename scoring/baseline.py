"""Calibration baseline profile I/O and construction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CHANNELS = ("acoustic", "linguistic", "gaze", "lip")


def _channel_stats(values: list[float]) -> dict[str, float]:
    import numpy as np

    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "std": 1.0}
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1) if arr.size > 1 else 0.0)}


def build_baseline_profile(
    acoustic: list[float],
    linguistic: list[float],
    gaze: list[float],
    lip: list[float],
    *,
    video_path: str | None = None,
) -> dict[str, Any]:
    """Aggregate per-frame/per-window features into channel baselines."""
    profile: dict[str, Any] = {
        "version": 1,
        "channels": {
            "acoustic": _channel_stats(acoustic),
            "linguistic": _channel_stats(linguistic),
            "gaze": _channel_stats(gaze),
            "lip": _channel_stats(lip),
        },
    }
    if video_path:
        profile["source_video"] = str(video_path)
    return profile


def save_baseline_profile(profile: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(profile, indent=2), encoding="utf-8")


def load_baseline_profile(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "acoustic_reading_profile" not in data and "channels" not in data:
        raise ValueError(f"Invalid calibration profile: {path}")
    return data
