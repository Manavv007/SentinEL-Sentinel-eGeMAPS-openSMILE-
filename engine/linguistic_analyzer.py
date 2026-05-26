"""Linguistic feature analysis from WhisperX / whisper-timestamped transcripts."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

import config

LINGUISTIC_WEIGHTS: dict[str, float] = {
    "filler_score": 0.50,
    "wps_cv_score": 0.25,
    "gap_variance_score": 0.25,
}

DEFINITE_FILLERS = frozenset({"um", "uh", "er", "ah", "hmm"})

DISCOURSE_MARKERS = [
    "like",
    "you know",
    "i mean",
    "sort of",
    "kind of",
    "basically",
    "literally",
    "right",
    "okay so",
    "so basically",
]

LONG_ANSWER_SEC = 20.0


class LinguisticAnalyzer:
    """Score linguistic patterns against calibration reading baselines."""

    def calibrate(self, transcripts: list[dict[str, Any]]) -> dict[str, float]:
        """Build wps_cv and gap-variance baselines from calibration transcripts."""
        wps_values: list[float] = []
        gap_values: list[float] = []

        for t in transcripts:
            wps_cv = self._wps_cv(t)
            if wps_cv is not None and np.isfinite(wps_cv):
                wps_values.append(wps_cv)
            gap_var = self._inter_word_gap_variance(t)
            if gap_var is not None and np.isfinite(gap_var):
                gap_values.append(gap_var)

        wps_mean = float(np.mean(wps_values)) if wps_values else 0.0
        gap_mean = float(np.mean(gap_values)) if gap_values else 0.0

        return {
            "wps_cv_mean": wps_mean,
            "wps_cv_std": self._robust_std(wps_values, fallback=max(wps_mean * 0.25, config.STD_FLOOR)),
            "gap_variance_mean": gap_mean,
            "gap_variance_std": self._robust_std(
                gap_values, fallback=max(gap_mean * 0.25, config.STD_FLOOR)
            ),
        }

    def analyze(
        self,
        transcript: dict[str, Any],
        calibration: dict[str, float],
    ) -> tuple[float, dict[str, float]]:
        """Return (linguistic_score, per-feature breakdown)."""
        duration = float(
            transcript.get("duration_sec")
            or self._duration_from_words(transcript.get("words", []))
            or 0.0
        )

        filler_rate = self._filler_rate_per_30s(transcript, duration)
        filler_score = self._filler_score(filler_rate, duration)

        wps_cv = self._wps_cv(transcript)
        wps_cv_score = self._wps_cv_score(wps_cv, calibration)

        gap_var = self._inter_word_gap_variance(transcript)
        gap_variance_score = self._gap_variance_score(gap_var, calibration)

        components = {
            "filler_score": round(filler_score, 6),
            "wps_cv_score": round(wps_cv_score, 6),
            "gap_variance_score": round(gap_variance_score, 6),
            "filler_rate_per_30s": round(filler_rate, 6),
            "wps_cv": round(wps_cv, 6) if wps_cv is not None else 0.0,
            "gap_variance": round(gap_var, 6) if gap_var is not None else 0.0,
        }

        score = sum(components[k] * LINGUISTIC_WEIGHTS[k] for k in LINGUISTIC_WEIGHTS)
        return round(score, 6), components

    # ------------------------------------------------------------------
    # Feature 1: filler rate
    # ------------------------------------------------------------------

    def _filler_rate_per_30s(self, transcript: dict[str, Any], duration_sec: float) -> float:
        if duration_sec <= 0:
            return 0.0

        count = self._count_fillers(transcript)
        return count / (duration_sec / 30.0)

    @staticmethod
    def _filler_score(filler_rate: float, duration_sec: float) -> float:
        if duration_sec > LONG_ANSWER_SEC and filler_rate == 0.0:
            return 1.0
        return float(np.exp(-filler_rate / 3.0))

    def _count_fillers(self, transcript: dict[str, Any]) -> int:
        words = self._normalised_words(transcript)
        if not words:
            return 0

        count = 0
        text_lower = " ".join(words)

        for w in words:
            if w in DEFINITE_FILLERS:
                count += 1

        for marker in DISCOURSE_MARKERS:
            pattern = r"(?<!\w)" + re.escape(marker) + r"(?!\w)"
            count += len(re.findall(pattern, text_lower))

        for i in range(len(words) - 1):
            if words[i] == words[i + 1]:
                count += 1

        for i, w in enumerate(words):
            bare = w.rstrip("-")
            if len(bare) < 3 and (w.endswith("-") or (i + 1 < len(words) and words[i + 1] != bare)):
                count += 1

        return count

    # ------------------------------------------------------------------
    # Feature 2: words-per-second CV
    # ------------------------------------------------------------------

    @staticmethod
    def _wps_cv(transcript: dict[str, Any]) -> float | None:
        segments = transcript.get("segments", [])
        wps_list: list[float] = []

        for seg in segments:
            text = str(seg.get("text", "")).strip()
            start = seg.get("start")
            end = seg.get("end")
            if not text or start is None or end is None:
                continue
            dur = float(end) - float(start)
            if dur <= 0:
                continue
            word_count = len(text.split())
            if word_count > 0:
                wps_list.append(word_count / dur)

        if len(wps_list) < 2:
            words = transcript.get("words", [])
            dur = LinguisticAnalyzer._duration_from_words(words)
            if dur and dur > 0 and len(words) >= 2:
                wps_list = [len(words) / dur]
            else:
                return None

        if len(wps_list) < 2:
            return 0.0

        arr = np.asarray(wps_list, dtype=np.float64)
        mean = float(arr.mean())
        if mean < 1e-8:
            return 0.0
        return float(arr.std(ddof=0) / mean)

    @staticmethod
    def _wps_cv_score(wps_cv: float | None, calibration: dict[str, float]) -> float:
        if wps_cv is None:
            return 0.5
        cal_mean = float(calibration.get("wps_cv_mean", wps_cv))
        cal_std = max(float(calibration.get("wps_cv_std", config.STD_FLOOR)), config.STD_FLOOR)
        z = (wps_cv - cal_mean) / cal_std
        return float(np.exp(-0.5 * z * z))

    # ------------------------------------------------------------------
    # Feature 3: inter-word gap variance
    # ------------------------------------------------------------------

    @staticmethod
    def _inter_word_gap_variance(transcript: dict[str, Any]) -> float | None:
        words = transcript.get("words", [])
        if len(words) < 2:
            return None

        gaps: list[float] = []
        for i in range(len(words) - 1):
            end_i = words[i].get("end")
            start_next = words[i + 1].get("start")
            if end_i is None or start_next is None:
                continue
            gaps.append(float(start_next) - float(end_i))

        if len(gaps) < 2:
            return None
        return float(np.var(np.asarray(gaps, dtype=np.float64), ddof=1))

    @staticmethod
    def _gap_variance_score(gap_var: float | None, calibration: dict[str, float]) -> float:
        if gap_var is None:
            return 0.5
        cal_mean = float(calibration.get("gap_variance_mean", gap_var))
        cal_std = max(
            float(calibration.get("gap_variance_std", config.STD_FLOOR)),
            config.STD_FLOOR,
        )
        z = (gap_var - cal_mean) / cal_std
        return float(np.exp(-0.5 * z * z))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalised_words(transcript: dict[str, Any]) -> list[str]:
        raw = transcript.get("words", [])
        out: list[str] = []
        for w in raw:
            token = re.sub(r"[^\w'-]", "", str(w.get("word", "")).lower()).strip("'")
            if token:
                out.append(token)
        if out:
            return out
        text = str(transcript.get("transcript", "")).lower()
        return [t for t in re.findall(r"[a-z']+", text) if t]

    @staticmethod
    def _duration_from_words(words: list[dict[str, Any]]) -> float:
        if not words:
            return 0.0
        starts = [w.get("start") for w in words if w.get("start") is not None]
        ends = [w.get("end") for w in words if w.get("end") is not None]
        if not starts or not ends:
            return 0.0
        return float(max(ends)) - float(min(starts))

    @staticmethod
    def _robust_std(values: list[float], *, fallback: float) -> float:
        if len(values) < 2:
            return fallback
        arr = np.asarray(values, dtype=np.float64)
        q75, q25 = np.percentile(arr, [75, 25])
        return max(float((q75 - q25) / 1.349), config.STD_FLOOR)
