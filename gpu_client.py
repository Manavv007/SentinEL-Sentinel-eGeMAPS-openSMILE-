"""
HTTP client for the SentinEL Kaggle GPU server (kaggle_gpu_server.ipynb).

When KAGGLE_GPU_URL is set and KAGGLE_OFFLOAD=true, interview transcription
and GPU scoring run on Kaggle (Whisper large-v3 on GPU) instead of local CPU.
"""

from __future__ import annotations

import io
import json
import logging
import wave
from pathlib import Path
from typing import Any

import httpx

import config
from utils.audio_bytes import pcm16_to_wav

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
    return windows or [{"start": 0.0}]


def _normalize_gpu_baseline(profile: dict[str, Any]) -> dict[str, Any] | None:
    if not profile:
        return None
    if "parselmouth_baseline" in profile:
        return profile
    if "gpu_reading_profile" in profile:
        inner = profile["gpu_reading_profile"]
        return inner if isinstance(inner, dict) else None
    return None


def _transcript_payload(
    *,
    answer_id: int,
    transcript: str,
    words: list[dict[str, Any]],
    segments: list[dict[str, Any]] | None = None,
    duration_sec: float = 0.0,
    backend: str = "kaggle-whisperx-gpu",
) -> dict[str, Any]:
    return {
        "answer_id": answer_id,
        "transcript": transcript,
        "words": words,
        "segments": segments or [],
        "duration_sec": duration_sec,
        "transcription_backend": backend,
    }


class KaggleGPUClient:
    """Calls remote FastAPI server on Kaggle (WhisperX + Parselmouth GPU scoring)."""

    def __init__(
        self,
        base_url: str = "",
        secret: str = "",
        timeout: int = 180,
        calibrate_timeout: int | None = None,
    ) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.secret = (secret or "").strip()
        self.timeout = int(timeout)
        self.calibrate_timeout = int(
            calibrate_timeout or config.KAGGLE_CALIBRATE_TIMEOUT_SEC
        )
        self.segment_timeout = int(config.KAGGLE_SEGMENT_TIMEOUT_SEC)
        self.enabled = bool(self.base_url)
        self._client: httpx.Client | None = None
        self._calibrate_client: httpx.Client | None = None
        self._segment_client: httpx.Client | None = None

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

    @property
    def kaggle_offload_enabled(self) -> bool:
        return self.enabled and config.KAGGLE_OFFLOAD

    @property
    def offload_segmentation_active(self) -> bool:
        return self.kaggle_offload_enabled and config.KAGGLE_OFFLOAD_SEGMENTATION

    @property
    def offload_active(self) -> bool:
        return self.kaggle_offload_enabled and config.KAGGLE_OFFLOAD_TRANSCRIPTION

    def _headers(self) -> dict[str, str]:
        h = {"ngrok-skip-browser-warning": "true"}
        if self.secret:
            h["X-Sentinel-Secret"] = self.secret
        return h

    def _client_for(self, *, calibrate: bool = False) -> httpx.Client:
        if not self.enabled:
            raise RuntimeError("Kaggle GPU client is not configured (KAGGLE_GPU_URL empty).")
        timeout_sec = self.calibrate_timeout if calibrate else self.timeout
        if calibrate:
            if self._calibrate_client is None:
                self._calibrate_client = httpx.Client(
                    base_url=self.base_url,
                    timeout=httpx.Timeout(timeout_sec, connect=30.0),
                    headers=self._headers(),
                )
            return self._calibrate_client
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.base_url,
                timeout=httpx.Timeout(timeout_sec, connect=30.0),
                headers=self._headers(),
            )
        return self._client

    def _client_for_segment(self) -> httpx.Client:
        if not self.enabled:
            raise RuntimeError("Kaggle GPU client is not configured (KAGGLE_GPU_URL empty).")
        if self._segment_client is None:
            self._segment_client = httpx.Client(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.segment_timeout, connect=30.0),
                headers=self._headers(),
            )
        return self._segment_client

    def close(self) -> None:
        for client in (self._client, self._calibrate_client, self._segment_client):
            if client is not None:
                client.close()
        self._client = None
        self._calibrate_client = None
        self._segment_client = None

    def health(self) -> dict[str, Any]:
        if not self.enabled:
            return {"status": "disabled"}
        client = self._client_for()
        resp = client.get("/health")
        resp.raise_for_status()
        return resp.json()

    def _upload_audio(self, pcm_bytes: bytes, filename: str) -> tuple[str, bytes, str]:
        wav = pcm16_to_wav(pcm_bytes)
        return filename, wav, "audio/wav"

    def calibrate(self, audio_bytes: bytes) -> dict[str, Any] | None:
        """Build GPU reading baseline from calibration audio."""
        if not self.enabled or not audio_bytes:
            return None

        client = self._client_for(calibrate=True)
        fname, payload, mime = self._upload_audio(audio_bytes, "calibration.wav")
        files = {"audio_file": (fname, payload, mime)}
        data = {"secret": self.secret} if self.secret else {}

        try:
            resp = client.post("/calibrate", files=files, data=data)
            resp.raise_for_status()
            body = resp.json()
            if isinstance(body, dict) and body.get("error"):
                logger.error("Kaggle /calibrate error: %s", body["error"])
                return None
            if body.get("status") == "ok" or "parselmouth_baseline" in body:
                return body
            return body
        except httpx.HTTPError as exc:
            logger.error("Kaggle /calibrate failed: %s", exc)
            return None

    def segment_interview(
        self,
        video_path: str,
        *,
        wav_path: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Kaggle segmentation (fast VAD or pyannote) → candidate turn boundaries.
        Prefer wav_path (smaller upload, skips Kaggle ffmpeg).
        """
        if not self.enabled:
            return None

        upload_path = Path(wav_path) if wav_path else Path(video_path)
        if not upload_path.is_file():
            logger.error("segment_interview: file not found: %s", upload_path)
            return None

        client = self._client_for_segment()
        data: dict[str, str] = {
            "candidate_speaker": config.CANDIDATE_SPEAKER,
            "num_speakers": "2",
            "segment_mode": config.KAGGLE_SEGMENT_MODE,
            "min_candidate_sec": str(config.KAGGLE_FAST_MIN_CANDIDATE_SEC),
        }
        if config.HF_TOKEN:
            data["hf_token"] = config.HF_TOKEN
        if self.secret:
            data["secret"] = self.secret

        file_key = "audio_file" if wav_path else "video_file"
        mime = "audio/wav" if wav_path else "application/octet-stream"

        try:
            with upload_path.open("rb") as handle:
                files = {file_key: (upload_path.name, handle, mime)}
                resp = client.post("/segment_interview", files=files, data=data)
            body: dict[str, Any] = {}
            try:
                body = resp.json()
            except Exception:
                body = {"error": resp.text[:2000]}
            if resp.status_code >= 400:
                err = body.get("error") or resp.text[:500]
                logger.error(
                    "Kaggle /segment_interview HTTP %s: %s",
                    resp.status_code,
                    err,
                )
                if body.get("traceback"):
                    logger.error("Kaggle traceback:\\n%s", body["traceback"][-1500:])
                return None
            if isinstance(body, dict) and body.get("error"):
                logger.error("Kaggle /segment_interview error: %s", body["error"])
                return None
            if body.get("status") != "ok" and not body.get("answers"):
                logger.error("Kaggle /segment_interview unexpected payload: %s", body)
                return None
            return body
        except httpx.HTTPError as exc:
            logger.error("Kaggle /segment_interview failed: %s", exc)
            return None

    def transcribe_answer(
        self,
        audio_bytes: bytes,
        *,
        answer_id: int = 0,
        duration_sec: float = 0.0,
    ) -> dict[str, Any] | None:
        """Transcribe one answer on Kaggle GPU (WhisperX large-v3)."""
        if not self.enabled or not audio_bytes:
            return None

        client = self._client_for()
        fname, payload, mime = self._upload_audio(audio_bytes, "answer.wav")
        files = {"audio_file": (fname, payload, mime)}
        data: dict[str, str] = {}
        if self.secret:
            data["secret"] = self.secret
        if config.KAGGLE_SKIP_ALIGN_INTERVIEW:
            data["skip_align"] = "true"

        try:
            resp = client.post("/transcribe_answer", files=files, data=data)
            resp.raise_for_status()
            body = resp.json()
            if isinstance(body, dict) and body.get("error"):
                logger.error("Kaggle /transcribe_answer error: %s", body["error"])
                return None
            words = body.get("words") or []
            transcript = str(body.get("transcript") or " ".join(
                str(w.get("word", "")) for w in words
            )).strip()
            return _transcript_payload(
                answer_id=answer_id,
                transcript=transcript,
                words=words,
                segments=body.get("segments") or [],
                duration_sec=float(body.get("duration_sec") or duration_sec),
                backend=str(body.get("transcription_backend") or "kaggle-whisperx-gpu"),
            )
        except httpx.HTTPError as exc:
            logger.error("Kaggle /transcribe_answer failed: %s", exc)
            return None

    def analyze(
        self,
        audio_bytes: bytes,
        reading_profile: dict[str, Any],
        duration: float,
        *,
        answer_id: int = 0,
    ) -> dict[str, Any] | None:
        """
        Score one answer via /analyze_batch.
        Also returns transcript + words when the notebook includes them.
        """
        if not self.enabled or not audio_bytes:
            return None

        baseline_payload = _normalize_gpu_baseline(reading_profile)
        if not baseline_payload:
            logger.warning(
                "KaggleGPUClient.analyze: missing gpu_reading_profile — "
                "re-run calibration with KAGGLE_GPU_URL set."
            )
            return None

        windows = _build_windows(duration)
        client = self._client_for()
        fname, payload, mime = self._upload_audio(audio_bytes, "answer.wav")
        files = {"audio_file": (fname, payload, mime)}
        data = {
            "windows_json": json.dumps(windows),
            "parselmouth_baseline": json.dumps(baseline_payload),
        }
        if self.secret:
            data["secret"] = self.secret

        try:
            resp = client.post("/analyze_batch", files=files, data=data)
            resp.raise_for_status()
            body = resp.json()
            if isinstance(body, dict) and body.get("error"):
                logger.error("Kaggle /analyze_batch error: %s", body["error"])
                return None

            results = body.get("results") or []
            if not results:
                return None

            raw_scores = [
                float(r["gpu_raw_score"])
                for r in results
                if r.get("gpu_raw_score") is not None
            ]
            score = max(raw_scores) if raw_scores else None
            words = body.get("words") or []
            transcript = str(body.get("transcript") or "").strip()
            if not transcript and words:
                transcript = " ".join(str(w.get("word", "")) for w in words).strip()
            if not transcript:
                transcript = " ".join(
                    str(r.get("transcript", "")).strip() for r in results if r.get("transcript")
                ).strip()

            return {
                "score": score,
                "gpu_score": score,
                "gpu_raw_score": score,
                "windows": results,
                "processing_time_ms": body.get("processing_time_ms"),
                "transcript": _transcript_payload(
                    answer_id=answer_id,
                    transcript=transcript,
                    words=words,
                    segments=body.get("segments") or [],
                    duration_sec=float(duration),
                    backend="kaggle-analyze-batch-gpu",
                )
                if transcript or words
                else None,
            }
        except httpx.HTTPError as exc:
            logger.error("Kaggle /analyze_batch failed: %s", exc)
            return None

    def process_answer(
        self,
        audio_bytes: bytes,
        *,
        answer_id: int,
        duration_sec: float,
        gpu_reading_profile: dict[str, Any] | None,
    ) -> tuple[dict[str, Any] | None, float | None]:
        """
        One Kaggle round-trip: analyze_batch if profile exists (GPU + transcript),
        else transcribe_answer only.
        """
        if gpu_reading_profile and not config.KAGGLE_TRANSCRIBE_ONLY:
            batch = self.analyze(
                audio_bytes,
                gpu_reading_profile,
                duration_sec,
                answer_id=answer_id,
            )
            if batch:
                transcript = batch.get("transcript")
                return transcript, batch.get("score")

        transcript = self.transcribe_answer(
            audio_bytes,
            answer_id=answer_id,
            duration_sec=duration_sec,
        )
        return transcript, None
