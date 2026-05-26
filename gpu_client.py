"""
HTTP client for the SentinEL Kaggle GPU server (kaggle_gpu_server.ipynb).

Setup:
  1. On Kaggle: enable GPU, upload/run kaggle_gpu_server.ipynb (all cells).
  2. Copy the printed ngrok URL into .env as KAGGLE_GPU_URL.
  3. Set KAGGLE_SECRET (or SENTINEL_SECRET) to match the notebook.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


def _build_windows(duration_sec: float, *, hop: float = 2.0, win: float = 4.0) -> list[dict[str, float]]:
    """4s windows with 2s hop (matches Kaggle notebook)."""
    duration_sec = max(float(duration_sec), 0.0)
    if duration_sec <= 0:
        return [{"start": 0.0}]
    windows: list[dict[str, float]] = []
    t = 0.0
    while t < duration_sec:
        windows.append({"start": round(t, 4)})
        t += hop
        if len(windows) > 500:
            break
    if not windows:
        windows = [{"start": 0.0}]
    return windows


class KaggleGPUClient:
    """Calls remote FastAPI server on Kaggle (WhisperX + Parselmouth GPU scoring)."""

    def __init__(
        self,
        base_url: str = "",
        secret: str = "",
        timeout: int = 180,
    ) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.secret = (secret or "").strip()
        self.timeout = int(timeout)
        self.enabled = bool(self.base_url)
        self._client: httpx.Client | None = None

        if self.enabled:
            logger.info("KaggleGPUClient: configured at %s", self.base_url)
            try:
                health = self.health()
                logger.info(
                    "KaggleGPUClient: health OK — device=%s model_loaded=%s",
                    health.get("device"),
                    health.get("model_loaded"),
                )
            except Exception as exc:
                logger.warning(
                    "KaggleGPUClient: health check failed (%s). "
                    "Is the Kaggle notebook running and ngrok URL current?",
                    exc,
                )
        else:
            logger.info(
                "KaggleGPUClient: no KAGGLE_GPU_URL — GPU channel disabled (local CPU mode)."
            )

    def _headers(self) -> dict[str, str]:
        h = {"ngrok-skip-browser-warning": "true"}
        if self.secret:
            h["X-Sentinel-Secret"] = self.secret
        return h

    def _client_or_raise(self) -> httpx.Client:
        if not self.enabled:
            raise RuntimeError("Kaggle GPU client is not configured (KAGGLE_GPU_URL empty).")
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout, connect=30.0),
                headers=self._headers(),
            )
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def health(self) -> dict[str, Any]:
        if not self.enabled:
            return {"status": "disabled"}
        client = self._client_or_raise()
        resp = client.get("/health")
        resp.raise_for_status()
        return resp.json()

    def calibrate(self, audio_bytes: bytes) -> dict[str, Any] | None:
        """
        Build GPU reading baseline from calibration audio (one answer segment).

        Returns dict with parselmouth_baseline, whisper_stats, linguistic_baseline.
        """
        if not self.enabled or not audio_bytes:
            return None

        client = self._client_or_raise()
        files = {
            "audio_file": ("calibration.wav", audio_bytes, "audio/wav"),
        }
        data = {}
        if self.secret:
            data["secret"] = self.secret

        try:
            resp = client.post("/calibrate", files=files, data=data)
            resp.raise_for_status()
            payload = resp.json()
            if isinstance(payload, dict) and payload.get("error"):
                logger.error("Kaggle /calibrate error: %s", payload["error"])
                return None
            if payload.get("status") == "ok" or "parselmouth_baseline" in payload:
                return payload
            return payload
        except httpx.HTTPError as exc:
            logger.error("Kaggle /calibrate failed: %s", exc)
            return None

    def analyze(
        self,
        audio_bytes: bytes,
        reading_profile: dict[str, Any],
        duration: float,
    ) -> dict[str, Any] | None:
        """
        Score one answer via /analyze_batch.

        reading_profile should be gpu_reading_profile from calibration
        (parselmouth_baseline + whisper_stats + linguistic_baseline).
        """
        if not self.enabled or not audio_bytes:
            return None

        baseline_payload = _normalize_gpu_baseline(reading_profile)
        if not baseline_payload:
            logger.warning(
                "KaggleGPUClient.analyze: missing gpu_reading_profile / parselmouth_baseline "
                "— re-run calibration with KAGGLE_GPU_URL set."
            )
            return None

        windows = _build_windows(duration)
        client = self._client_or_raise()
        files = {
            "audio_file": ("answer.wav", audio_bytes, "audio/wav"),
        }
        data = {
            "windows_json": json.dumps(windows),
            "parselmouth_baseline": json.dumps(baseline_payload),
        }
        if self.secret:
            data["secret"] = self.secret

        try:
            resp = client.post("/analyze_batch", files=files, data=data)
            resp.raise_for_status()
            payload = resp.json()
            if isinstance(payload, dict) and payload.get("error"):
                logger.error("Kaggle /analyze_batch error: %s", payload["error"])
                return None

            results = payload.get("results") or []
            if not results:
                return None

            raw_scores = [
                float(r["gpu_raw_score"])
                for r in results
                if r.get("gpu_raw_score") is not None
            ]
            score = max(raw_scores) if raw_scores else None
            script_sims = [
                float(r["script_similarity"])
                for r in results
                if r.get("script_similarity") is not None
            ]
            return {
                "score": score,
                "gpu_score": score,
                "gpu_raw_score": score,
                "script_similarity": max(script_sims) if script_sims else None,
                "windows": results,
                "processing_time_ms": payload.get("processing_time_ms"),
            }
        except httpx.HTTPError as exc:
            logger.error("Kaggle /analyze_batch failed: %s", exc)
            return None


def _normalize_gpu_baseline(profile: dict[str, Any]) -> dict[str, Any] | None:
    """Accept gpu_reading_profile or nested calibration export."""
    if not profile:
        return None
    if "parselmouth_baseline" in profile:
        return profile
    if "gpu_reading_profile" in profile:
        inner = profile["gpu_reading_profile"]
        return inner if isinstance(inner, dict) else None
    # Legacy: only openSMILE acoustic profile — cannot run Kaggle batch
    return None
