"""Audio byte helpers for remote GPU upload."""

from __future__ import annotations

import io
import wave


def pcm16_to_wav(pcm_bytes: bytes, sample_rate: int = 16_000) -> bytes:
    """Wrap raw int16 mono PCM in a WAV container (Kaggle ffmpeg expects this)."""
    if not pcm_bytes:
        return b""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()
