"""Full audio pipeline: extract, diarize, segment answers, extract features."""

from __future__ import annotations

import logging
import tempfile
import threading
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

import config
from processors.speaker_selection import (
    build_candidate_turns,
    build_dual_track_boundaries,
    extract_segments_from_diarization,
    group_into_answers,
)

logger = logging.getLogger(__name__)

# Process-wide heavy models — avoid reloading pyannote/openSMILE on every web job.
_diarization_pipeline = None
_opensmile_instance = None
_model_init_lock = threading.Lock()

SAMPLE_RATE = 16_000
WINDOW_SEC = 4.0
HOP_SEC = 2.0
SILENCE_GAP_SEC = 3.0

OPENSMILE_FEATURES = (
    "F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
    "MeanVoicedSegmentLengthSec",
    "MeanUnvoicedSegmentLength",
)

PARSELMOUTH_FEATURES = (
    "jitter_local",
    "shimmer_local",
    "hnr",
    "pitch_range_hz",
)


class ProcessingMode(str, Enum):
    CALIBRATION = "calibration"
    INTERVIEW = "interview"


@dataclass
class SpeechSegment:
    start_sec: float
    end_sec: float


class AudioProcessor:
    """Extract audio from video, diarize, segment answers, and extract window features."""

    def __init__(self) -> None:
        self._hf_token = config.HF_TOKEN
        self.last_speaker_selection: dict[str, Any] | None = None

    @staticmethod
    def preload_heavy_models(*, diarization: bool = True, opensmile: bool = True) -> None:
        """Warm pyannote/openSMILE once per worker process (web startup)."""
        proc = AudioProcessor()
        if opensmile:
            proc._get_opensmile()
        if diarization:
            proc._get_diarization_pipeline()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_calibration(self, video_path: str) -> list[dict[str, Any]]:
        """Single-speaker calibration video → answers with feature windows."""
        return self._process(video_path, ProcessingMode.CALIBRATION)

    def process_interview(self, video_path: str) -> list[dict[str, Any]]:
        """Two-speaker interview → candidate answers with feature windows."""
        return self._process(video_path, ProcessingMode.INTERVIEW)

    def extract_audio(self, video_path: str) -> str:
        """Extract 16 kHz mono WAV; caller must delete the temp file."""
        return self._extract_audio(video_path)

    def segment_interview_dual_track(
        self, wav_path: str
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        """
        Run pyannote once and return primary + alternate speaker answer boundaries.
        Used when Kaggle primary track is mostly AI interviewer speech.
        """
        segments = self._diarize_raw_segments(wav_path)
        if not segments:
            duration = self._wav_duration(wav_path)
            fallback = [{"answer_id": 0, "start_sec": 0.0, "end_sec": duration}]
            return fallback, [], {"reason": "no_diarization_segments"}

        primary, alternate, selection = build_dual_track_boundaries(
            segments,
            config.CANDIDATE_SPEAKER,
            silence_gap_sec=SILENCE_GAP_SEC,
        )
        self.last_speaker_selection = selection
        primary_answers = [
            {"answer_id": i, "start_sec": float(s), "end_sec": float(e)}
            for i, (s, e) in enumerate(primary)
        ]
        alternate_answers = [
            {"answer_id": i, "start_sec": float(s), "end_sec": float(e)}
            for i, (s, e) in enumerate(alternate)
        ]
        return primary_answers, alternate_answers, selection

    def process_interview_from_segmentation(
        self,
        video_path: str,
        boundaries: list[dict[str, Any]],
        *,
        speaker_selection: dict[str, Any] | None = None,
        wav_path: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Build interview answers locally after Kaggle returns candidate turn boundaries.
        Skips local pyannote diarization (slow on CPU).
        """
        owns_wav = wav_path is None
        if wav_path is None:
            wav_path = self._extract_audio(video_path)
        try:
            waveform, _ = sf.read(wav_path, dtype="float32")
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)
            if speaker_selection is not None:
                self.last_speaker_selection = speaker_selection

            items = sorted(boundaries, key=lambda x: float(x.get("start_sec", 0)))
            workers = min(config.AUDIO_WINDOW_PARALLEL_WORKERS, max(len(items), 1))

            def _one(item: dict[str, Any]) -> dict[str, Any] | None:
                start_sec = float(item["start_sec"])
                end_sec = float(item["end_sec"])
                if end_sec <= start_sec:
                    return None
                answer_id = int(item.get("answer_id", 0))
                return {
                    "answer_id": answer_id,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "audio_bytes": self._slice_audio_bytes(waveform, start_sec, end_sec),
                    "windows": self._extract_windows(
                        waveform, start_sec, end_sec,
                        parallel=config.AUDIO_WINDOW_PARALLEL_WORKERS > 1,
                    ),
                }

            answers: list[dict[str, Any]] = []
            if workers <= 1 or len(items) <= 1:
                for item in items:
                    row = _one(item)
                    if row:
                        answers.append(row)
            else:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    rows = list(pool.map(_one, items))
                answers = [r for r in rows if r is not None]
            answers.sort(key=lambda a: a["start_sec"])
            for i, ans in enumerate(answers):
                ans["answer_id"] = i
            return answers
        finally:
            if owns_wav:
                Path(wav_path).unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def _process(self, video_path: str, mode: ProcessingMode) -> list[dict[str, Any]]:
        wav_path = self._extract_audio(video_path)
        try:
            waveform, _ = sf.read(wav_path, dtype="float32")
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)

            if mode == ProcessingMode.CALIBRATION and config.SKIP_DIARIZATION_CALIBRATION:
                duration = len(waveform) / float(SAMPLE_RATE)
                speech_segments = [SpeechSegment(0.0, duration)]
            else:
                speech_segments = self._diarize_speech(wav_path, mode)
            answers = self._group_into_answers(speech_segments)

            if mode == ProcessingMode.CALIBRATION:
                window_workers = config.CALIBRATION_WINDOW_PARALLEL_WORKERS
            else:
                window_workers = config.AUDIO_WINDOW_PARALLEL_WORKERS

            parallel_windows = window_workers > 1
            parallel_answers = window_workers > 1 and len(answers) > 1
            workers = min(window_workers, max(len(answers), 1))

            def _build_answer(answer_id: int, start_sec: float, end_sec: float) -> dict[str, Any]:
                return {
                    "answer_id": answer_id,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "audio_bytes": self._slice_audio_bytes(waveform, start_sec, end_sec),
                    "windows": self._extract_windows(
                        waveform,
                        start_sec,
                        end_sec,
                        parallel=parallel_windows,
                    ),
                }

            results: list[dict[str, Any]] = []
            if parallel_answers:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    rows = list(
                        pool.map(
                            lambda item: _build_answer(*item),
                            [(i, s, e) for i, (s, e) in enumerate(answers)],
                        )
                    )
                results = rows
            else:
                for answer_id, (start_sec, end_sec) in enumerate(answers):
                    results.append(_build_answer(answer_id, start_sec, end_sec))
            return results
        finally:
            Path(wav_path).unlink(missing_ok=True)

    def _extract_audio(self, video_path: str) -> str:
        import ffmpeg

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        audio_path = tmp.name
        (
            ffmpeg.input(video_path)
            .output(audio_path, ac=1, ar=SAMPLE_RATE)
            .overwrite_output()
            .run(quiet=True)
        )
        return audio_path

    def _get_diarization_pipeline(self):
        global _diarization_pipeline
        if _diarization_pipeline is not None:
            return _diarization_pipeline

        with _model_init_lock:
            if _diarization_pipeline is not None:
                return _diarization_pipeline

            from pyannote.audio import Pipeline

            kwargs: dict[str, Any] = {}
            try:
                pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    token=self._hf_token,
                    **kwargs,
                )
            except TypeError:
                pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=self._hf_token,
                )
            _diarization_pipeline = pipeline
            logger.info("Loaded pyannote diarization pipeline (cached for process lifetime)")
            return _diarization_pipeline

    def _diarize_raw_segments(
        self, wav_path: str, *, interview: bool = True
    ) -> list[tuple[float, float, str]]:
        import soundfile as sf
        import torch

        pipeline = self._get_diarization_pipeline()
        data, sr = sf.read(wav_path, dtype="float32")
        if data.ndim == 1:
            waveform = torch.from_numpy(data).unsqueeze(0)
        else:
            waveform = torch.from_numpy(data.T)

        audio_input: dict[str, Any] = {"waveform": waveform, "sample_rate": int(sr)}
        diarize_kwargs: dict[str, Any] = {}
        if interview:
            n_spk = int(config.DIARIZATION_NUM_SPEAKERS)
            diarize_kwargs = {
                "num_speakers": n_spk,
                "min_speakers": n_spk,
                "max_speakers": n_spk,
            }
        try:
            diarization_output = pipeline(audio_input, **diarize_kwargs)
        except TypeError:
            try:
                diarization_output = pipeline(
                    audio_input, num_speakers=config.DIARIZATION_NUM_SPEAKERS
                )
            except Exception:
                diarization_output = pipeline(audio_input)
        except Exception as exc:
            logger.warning(
                "Diarization kwargs failed; retrying with in-memory waveform only: %s",
                exc,
            )
            diarization_output = pipeline(audio_input)

        return extract_segments_from_diarization(diarization_output)

    def _diarize_speech(self, wav_path: str, mode: ProcessingMode) -> list[SpeechSegment]:
        segments = self._diarize_raw_segments(
            wav_path, interview=(mode == ProcessingMode.INTERVIEW)
        )

        if not segments:
            duration = self._wav_duration(wav_path)
            return [SpeechSegment(0.0, duration)]

        if mode == ProcessingMode.CALIBRATION:
            return [SpeechSegment(s, e) for s, e, _ in segments]

        candidate_turns, selection = build_candidate_turns(segments)
        self.last_speaker_selection = selection
        logger.info(
            "Interview speaker selection: strategy=%s candidate=%s debug=%s",
            config.CANDIDATE_SPEAKER,
            selection.get("chosen_speaker"),
            selection,
        )
        return [SpeechSegment(start, end) for start, end in candidate_turns]

    @staticmethod
    def _wav_duration(wav_path: str) -> float:
        with wave.open(wav_path, "rb") as wf:
            return wf.getnframes() / float(wf.getframerate())

    @staticmethod
    def _group_into_answers(segments: list[SpeechSegment]) -> list[tuple[float, float]]:
        if not segments:
            return []
        pairs = [(s.start_sec, s.end_sec) for s in sorted(segments, key=lambda x: x.start_sec)]
        return group_into_answers(pairs, silence_gap_sec=SILENCE_GAP_SEC)

    @staticmethod
    def _slice_audio_bytes(
        waveform: np.ndarray,
        start_sec: float,
        end_sec: float,
    ) -> bytes:
        pad = float(getattr(config, "CANDIDATE_SEGMENT_END_PAD_SEC", 0.0) or 0.0)
        max_end = len(waveform) / float(SAMPLE_RATE)
        end_sec = min(max_end, float(end_sec) + pad)
        start = int(start_sec * SAMPLE_RATE)
        end = int(end_sec * SAMPLE_RATE)
        chunk = waveform[start:end]
        if chunk.size == 0:
            return b""
        pcm = np.clip(chunk, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype(np.int16)
        return pcm.tobytes()

    def _extract_windows(
        self,
        waveform: np.ndarray,
        answer_start: float,
        answer_end: float,
        *,
        parallel: bool = False,
    ) -> list[dict[str, Any]]:
        win_samples = int(WINDOW_SEC * SAMPLE_RATE)
        hop_samples = int(HOP_SEC * SAMPLE_RATE)
        abs_start = int(answer_start * SAMPLE_RATE)
        abs_end = int(answer_end * SAMPLE_RATE)
        answer_audio = waveform[abs_start:abs_end]

        if answer_audio.size < win_samples // 2:
            return []

        chunks: list[tuple[int, np.ndarray]] = []
        offset = 0
        while offset + win_samples <= answer_audio.size:
            chunks.append((offset, answer_audio[offset : offset + win_samples]))
            offset += hop_samples

        def _one(item: tuple[int, np.ndarray]) -> dict[str, Any]:
            off, chunk = item
            window_start = answer_start + off / SAMPLE_RATE
            opensmile, parselmouth = self._extract_features_parallel(chunk)
            return {
                "window_start": round(window_start, 4),
                "opensmile": opensmile,
                "parselmouth": parselmouth,
            }

        if not parallel or len(chunks) <= 1:
            return [_one(c) for c in chunks]

        workers = min(
            max(config.AUDIO_WINDOW_PARALLEL_WORKERS, config.CALIBRATION_WINDOW_PARALLEL_WORKERS),
            len(chunks),
        )
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(_one, chunks))

    def _extract_features_parallel(
        self,
        chunk: np.ndarray,
    ) -> tuple[dict[str, float], dict[str, float]]:
        with ThreadPoolExecutor(max_workers=2) as pool:
            smile_future = pool.submit(self._extract_opensmile, chunk)
            praat_future = pool.submit(self._extract_parselmouth, chunk)
            return smile_future.result(), praat_future.result()

    def _get_opensmile(self):
        global _opensmile_instance
        if _opensmile_instance is not None:
            return _opensmile_instance

        with _model_init_lock:
            if _opensmile_instance is not None:
                return _opensmile_instance

            import opensmile

            _opensmile_instance = opensmile.Smile(
                feature_set=opensmile.FeatureSet.eGeMAPSv02,
                feature_level=opensmile.FeatureLevel.Functionals,
            )
            logger.info("Loaded openSMILE eGeMAPS (cached for process lifetime)")
            return _opensmile_instance

    def _extract_opensmile(self, chunk: np.ndarray) -> dict[str, float]:
        smile = self._get_opensmile()
        df = smile.process_signal(chunk, SAMPLE_RATE)
        row = df.iloc[0]
        out: dict[str, float] = {}
        for name in OPENSMILE_FEATURES:
            val = float(row[name]) if name in row.index else 0.0
            out[name] = val if np.isfinite(val) else 0.0
        return out

    @staticmethod
    def _extract_parselmouth(chunk: np.ndarray) -> dict[str, float]:
        import parselmouth
        from parselmouth.praat import call

        sound = parselmouth.Sound(chunk, sampling_frequency=SAMPLE_RATE)
        out: dict[str, float] = {}

        try:
            jitter = call(sound, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            out["jitter_local"] = float(jitter) if np.isfinite(jitter) else 0.0
        except Exception:
            out["jitter_local"] = 0.0

        try:
            shimmer = call(
                sound, "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
            )
            out["shimmer_local"] = float(shimmer) if np.isfinite(shimmer) else 0.0
        except Exception:
            out["shimmer_local"] = 0.0

        try:
            harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            hnr = call(harmonicity, "Get mean", 0, 0)
            out["hnr"] = float(hnr) if np.isfinite(hnr) else 0.0
        except Exception:
            out["hnr"] = 0.0

        try:
            pitch = sound.to_pitch()
            freqs = pitch.selected_array["frequency"]
            voiced = freqs[freqs > 0]
            if voiced.size > 0:
                out["pitch_range_hz"] = float(voiced.max() - voiced.min())
            else:
                out["pitch_range_hz"] = 0.0
        except Exception:
            out["pitch_range_hz"] = 0.0

        return out

    @staticmethod
    def collect_windows(answers: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Flatten all feature windows from a list of answer dicts."""
        windows: list[dict[str, Any]] = []
        for answer in answers:
            windows.extend(answer.get("windows", []))
        return windows
