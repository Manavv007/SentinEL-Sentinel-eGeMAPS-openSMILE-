"""Full audio pipeline: extract, diarize, segment answers, extract features."""

from __future__ import annotations

import logging
import statistics
import tempfile
import wave
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

import config

logger = logging.getLogger(__name__)

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
        self._diarization_pipeline = None
        self._opensmile = None
        self.last_speaker_selection: dict[str, Any] | None = None

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
                    "windows": self._extract_windows(waveform, start_sec, end_sec),
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

            results: list[dict[str, Any]] = []
            for answer_id, (start_sec, end_sec) in enumerate(answers):
                audio_bytes = self._slice_audio_bytes(waveform, start_sec, end_sec)
                windows = self._extract_windows(waveform, start_sec, end_sec)
                results.append(
                    {
                        "answer_id": answer_id,
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                        "audio_bytes": audio_bytes,
                        "windows": windows,
                    }
                )
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
        if self._diarization_pipeline is not None:
            return self._diarization_pipeline

        from pyannote.audio import Pipeline

        kwargs: dict[str, Any] = {}
        try:
            self._diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=self._hf_token,
                **kwargs,
            )
        except TypeError:
            self._diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self._hf_token,
            )
        return self._diarization_pipeline

    def _diarize_speech(self, wav_path: str, mode: ProcessingMode) -> list[SpeechSegment]:
        import soundfile as sf
        import torch

        pipeline = self._get_diarization_pipeline()
        data, sr = sf.read(wav_path, dtype="float32")
        if data.ndim == 1:
            waveform = torch.from_numpy(data).unsqueeze(0)
        else:
            waveform = torch.from_numpy(data.T)

        # Always pass in-memory audio — avoids pyannote 4.x AudioDecoder errors on Windows
        audio_input: dict[str, Any] = {"waveform": waveform, "sample_rate": int(sr)}
        try:
            diarization_output = pipeline(audio_input)
        except Exception:
            diarization_output = pipeline(wav_path)

        annotation = (
            diarization_output.speaker_diarization
            if hasattr(diarization_output, "speaker_diarization")
            else diarization_output
        )

        segments: list[tuple[float, float, str]] = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            segments.append((float(turn.start), float(turn.end), str(speaker)))

        if not segments:
            duration = self._wav_duration(wav_path)
            return [SpeechSegment(0.0, duration)]

        if mode == ProcessingMode.CALIBRATION:
            return [SpeechSegment(s, e) for s, e, _ in segments]

        candidate, selection = _select_candidate_speaker(segments)
        self.last_speaker_selection = selection
        logger.info(
            "Interview speaker selection: strategy=%s candidate=%s debug=%s",
            config.CANDIDATE_SPEAKER,
            candidate,
            selection,
        )
        return [
            SpeechSegment(start, end)
            for start, end, spk in segments
            if spk == candidate
        ]

    @staticmethod
    def _wav_duration(wav_path: str) -> float:
        with wave.open(wav_path, "rb") as wf:
            return wf.getnframes() / float(wf.getframerate())

    @staticmethod
    def _group_into_answers(segments: list[SpeechSegment]) -> list[tuple[float, float]]:
        if not segments:
            return []

        ordered = sorted(segments, key=lambda s: s.start_sec)
        groups: list[list[SpeechSegment]] = [[ordered[0]]]

        for seg in ordered[1:]:
            prev_end = groups[-1][-1].end_sec
            if seg.start_sec - prev_end > SILENCE_GAP_SEC:
                groups.append([seg])
            else:
                groups[-1].append(seg)

        return [(g[0].start_sec, g[-1].end_sec) for g in groups]

    @staticmethod
    def _slice_audio_bytes(
        waveform: np.ndarray,
        start_sec: float,
        end_sec: float,
    ) -> bytes:
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
    ) -> list[dict[str, Any]]:
        win_samples = int(WINDOW_SEC * SAMPLE_RATE)
        hop_samples = int(HOP_SEC * SAMPLE_RATE)
        abs_start = int(answer_start * SAMPLE_RATE)
        abs_end = int(answer_end * SAMPLE_RATE)
        answer_audio = waveform[abs_start:abs_end]

        if answer_audio.size < win_samples // 2:
            return []

        windows: list[dict[str, Any]] = []
        offset = 0
        while offset + win_samples <= answer_audio.size:
            chunk = answer_audio[offset : offset + win_samples]
            window_start = answer_start + offset / SAMPLE_RATE
            opensmile, parselmouth = self._extract_features_parallel(chunk)
            windows.append(
                {
                    "window_start": round(window_start, 4),
                    "opensmile": opensmile,
                    "parselmouth": parselmouth,
                }
            )
            offset += hop_samples

        return windows

    def _extract_features_parallel(
        self,
        chunk: np.ndarray,
    ) -> tuple[dict[str, float], dict[str, float]]:
        with ThreadPoolExecutor(max_workers=2) as pool:
            smile_future = pool.submit(self._extract_opensmile, chunk)
            praat_future = pool.submit(self._extract_parselmouth, chunk)
            return smile_future.result(), praat_future.result()

    def _get_opensmile(self):
        if self._opensmile is None:
            import opensmile

            self._opensmile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.eGeMAPSv02,
                feature_level=opensmile.FeatureLevel.Functionals,
            )
        return self._opensmile

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


def _speaker_total_durations(
    segments: list[tuple[float, float, str]],
) -> dict[str, float]:
    durations: dict[str, float] = defaultdict(float)
    for start, end, speaker in segments:
        durations[speaker] += max(0.0, end - start)
    return dict(durations)


def _select_candidate_speaker(
    segments: list[tuple[float, float, str]],
) -> tuple[str, dict[str, Any]]:
    """
    Pick which diarization label is the human candidate (not AI interviewer).

    Strategy from config.CANDIDATE_SPEAKER.
    """
    totals = _speaker_total_durations(segments)
    if not totals:
        return "SPEAKER_00", {"strategy": config.CANDIDATE_SPEAKER, "reason": "no_segments"}

    strategy = config.CANDIDATE_SPEAKER

    if strategy == "least_speech":
        candidate = min(totals, key=totals.get)
        return candidate, {
            "strategy": strategy,
            "speaker_total_sec": {k: round(v, 2) for k, v in totals.items()},
            "chosen_total_sec": round(totals[candidate], 2),
        }

    if strategy == "longest_turns":
        min_turn = config.CANDIDATE_TURN_MIN_SEC
        turn_lengths: dict[str, list[float]] = defaultdict(list)
        for start, end, speaker in segments:
            dur = max(0.0, end - start)
            if dur >= min_turn:
                turn_lengths[speaker].append(dur)

        scores: dict[str, float] = {}
        for speaker in totals:
            lengths = turn_lengths.get(speaker) or []
            if not lengths:
                # Fallback: all turns for this speaker
                lengths = [
                    max(0.0, end - start)
                    for start, end, spk in segments
                    if spk == speaker
                ]
            scores[speaker] = (
                float(statistics.median(lengths)) if lengths else 0.0
            )

        candidate = max(scores, key=scores.get)
        return candidate, {
            "strategy": strategy,
            "turn_min_sec": min_turn,
            "speaker_median_turn_sec": {k: round(v, 2) for k, v in scores.items()},
            "speaker_total_sec": {k: round(v, 2) for k, v in totals.items()},
            "long_turn_counts": {k: len(turn_lengths.get(k, [])) for k in totals},
        }

    # most_speech (default)
    candidate = max(totals, key=totals.get)
    return candidate, {
        "strategy": strategy,
        "speaker_total_sec": {k: round(v, 2) for k, v in totals.items()},
        "chosen_total_sec": round(totals[candidate], 2),
    }
