"""
Windows fix for SpeechBrain LazyModule + inspect.py (k2_fsa import crash).

WhisperX VAD pulls in pyannote/speechbrain. On Windows, speechbrain's lazy
import guard uses `/inspect.py` and fails to block probes, then tries to load
k2_fsa (no Windows wheels) → ImportError during calibration/transcription.

Upstream fix: speechbrain PR #3052 — use os.path.basename() for inspect.py check.
"""

from __future__ import annotations

import importlib
import inspect
import os
import sys
import warnings
from types import ModuleType


_PATCHED = False


def apply_speechbrain_windows_patch() -> None:
    """Patch LazyModule.ensure_module on Windows (idempotent)."""
    global _PATCHED
    if _PATCHED or not sys.platform.startswith("win"):
        return

    try:
        from speechbrain.utils import importutils
    except ImportError:
        return

    if getattr(importutils.LazyModule, "_sentinel_windows_patch", False):
        _PATCHED = True
        return

    def _ensure_module_fixed(self: importutils.LazyModule, stacklevel: int) -> ModuleType:
        importer_frame = None
        try:
            importer_frame = inspect.getframeinfo(sys._getframe(stacklevel + 1))
        except AttributeError:
            warnings.warn(
                "Failed to inspect frame for SpeechBrain lazy import guard.",
                stacklevel=2,
            )

        if importer_frame is not None and os.path.basename(importer_frame.filename) == "inspect.py":
            raise AttributeError()

        if self.lazy_module is None:
            try:
                if self.package is None:
                    self.lazy_module = importlib.import_module(self.target)
                else:
                    self.lazy_module = importlib.import_module(
                        f".{self.target}", self.package
                    )
            except Exception as e:
                raise ImportError(f"Lazy import of {repr(self)} failed") from e

        return self.lazy_module

    importutils.LazyModule.ensure_module = _ensure_module_fixed  # type: ignore[method-assign]
    importutils.LazyModule._sentinel_windows_patch = True  # type: ignore[attr-defined]
    _PATCHED = True
