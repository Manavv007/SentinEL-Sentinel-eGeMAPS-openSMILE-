"""Unified per-window behavioral feature vectors for profile comparison."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from engine.linguistic_analyzer import LinguisticAnalyzer

FILLER_RE = re.compile(
    r"\b(um|uh|er|ah|hmm|like|you know|i mean|sort of|kind of|basically|literally)\b",
    re.I,
)
CORRECTION_RE = re.compile(r"\b(actually|sorry|wait|no i mean|let me rephrase)\b", re.I)
TECHNICAL_RE = re.compile(
    r"\b(api|apis|websocket|lambda|microservice|microservices|database|sql|nosql|"
    r"authentication|auth|oauth|jwt|kubernetes|docker|architecture|endpoint|rest|"
    r"graphql|serverless|cache|redis|postgres|mongodb|frontend|backend|deployment|"
    r"ci/cd|pipeline|scalability|latency|throughput|encryption|token|session)\b",
    re.I,
)


def extract_acoustic_features(window: dict[str, Any]) -> dict[str, float]:
    """Pull openSMILE + Parselmouth metrics from one audio window."""
    feats: dict[str, float] = {}
    for k, v in window.get("opensmile", {}).items():
        feats[f"acoustic_{k}"] = float(v)
    for k, v in window.get("parselmouth", {}).items():
        feats[f"acoustic_{k}"] = float(v)
    return feats


def extract_linguistic_features(
    transcript: dict[str, Any],
    *,
    start_sec: float,
    end_sec: float,
) -> dict[str, float]:
    """Lexical / timing features for words falling inside [start_sec, end_sec]."""
    words = [
        w
        for w in transcript.get("words", [])
        if w.get("start") is not None
        and w.get("end") is not None
        and float(w["end"]) > start_sec
        and float(w["start"]) < end_sec
    ]
    if not words:
        return {}

    text = " ".join(str(w.get("word", "")) for w in words).lower()
    duration = max(end_sec - start_sec, 1e-6)
    tokens = LinguisticAnalyzer._normalised_words({"words": words, "transcript": text})

    gaps: list[float] = []
    for i in range(len(words) - 1):
        g = float(words[i + 1]["start"]) - float(words[i]["end"])
        if g >= 0:
            gaps.append(g)

    wps_list: list[float] = []
    if len(words) >= 2:
        chunk_dur = max(float(words[-1]["end"]) - float(words[0]["start"]), 1e-6)
        wps_list.append(len(tokens) / chunk_dur)

    filler_count = len(FILLER_RE.findall(text))
    correction_count = len(CORRECTION_RE.findall(text))

    pause_entropy = _entropy(gaps) if len(gaps) >= 2 else 0.0
    retrieval_pause = max(gaps) if gaps else 0.0

    # Filler clustering: repeated fillers within 2s
    filler_times = [
        float(w["start"])
        for w in words
        if FILLER_RE.search(str(w.get("word", "")))
    ]
    clusters = 0
    for i in range(1, len(filler_times)):
        if filler_times[i] - filler_times[i - 1] < 2.0:
            clusters += 1

    tech_hits = len(TECHNICAL_RE.findall(text))
    token_n = max(len(tokens), 1)

    return {
        "ling_filler_rate_per_30s": filler_count / (duration / 30.0),
        "ling_wps": len(tokens) / duration,
        "ling_gap_variance": float(np.var(gaps, ddof=1)) if len(gaps) >= 2 else 0.0,
        "ling_pause_entropy": pause_entropy,
        "ling_retrieval_pause_max": retrieval_pause,
        "ling_self_corrections": float(correction_count),
        "ling_filler_clusters": float(clusters),
        "ling_repetition_rate": _repetition_rate(tokens),
        "ling_technical_density": tech_hits / token_n,
        "ling_has_words": 1.0,
    }


def extract_video_features(timeline: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate gaze/lip timeline frames into one feature row."""
    if not timeline:
        return {}

    gx = [float(f["gaze_x_ratio"]) for f in timeline if f.get("face_detected")]
    gy = [float(f["gaze_y_ratio"]) for f in timeline if f.get("face_detected")]
    ear = [float(f["ear"]) for f in timeline if f.get("face_detected")]
    lip_a = [float(f["lip_aperture"]) for f in timeline if f.get("face_detected")]

    feats: dict[str, float] = {}
    if gx:
        feats["video_gaze_x_std"] = float(np.std(gx, ddof=1) if len(gx) > 1 else 0.0)
        feats["video_gaze_x_mean"] = float(np.mean(gx))
    if gy:
        feats["video_gaze_down_frac"] = float(np.mean([1.0 if y > 0.6 else 0.0 for y in gy]))
    if ear:
        feats["video_blink_rate"] = float(np.mean([1.0 if e < 0.2 else 0.0 for e in ear]))
    if lip_a:
        feats["video_lip_aperture_std"] = float(np.std(lip_a, ddof=1) if len(lip_a) > 1 else 0.0)
        feats["video_lip_aperture_mean"] = float(np.mean(lip_a))
    return feats


def build_window_features(
    *,
    audio_window: dict[str, Any],
    transcript: dict[str, Any],
    timeline_slice: list[dict[str, Any]],
    pitch_delta: float | None = None,
    answer_start_sec: float | None = None,
    answer_end_sec: float | None = None,
) -> dict[str, float]:
    """Merge acoustic, linguistic, and video features for one temporal window."""
    feats: dict[str, float] = {}
    feats.update(extract_acoustic_features(audio_window))
    start = float(audio_window.get("window_start", 0))
    end = start + 4.0
    # Whisper word times are relative to each answer clip (0 = answer start), not interview clock.
    ling_start, ling_end = start, end
    if answer_start_sec is not None:
        base = float(answer_start_sec)
        ling_start = start - base
        ling_end = end - base
    ling = extract_linguistic_features(transcript, start_sec=ling_start, end_sec=ling_end)
    if not ling and answer_start_sec is not None and answer_end_sec is not None:
        ling = extract_linguistic_features(
            transcript,
            start_sec=0.0,
            end_sec=float(answer_end_sec) - float(answer_start_sec),
        )
        if ling:
            ling["ling_scope_fallback"] = 1.0
    feats.update(ling)
    feats.update(extract_video_features(timeline_slice))
    if pitch_delta is not None and np.isfinite(pitch_delta):
        feats["acoustic_pitch_delta"] = float(pitch_delta)
    return feats


def _entropy(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    hist, _ = np.histogram(arr, bins=min(8, len(arr)))
    total = hist.sum()
    if total == 0:
        return 0.0
    p = hist[hist > 0] / total
    return float(-np.sum(p * np.log2(p + 1e-12)))


def _repetition_rate(tokens: list[str]) -> float:
    if len(tokens) < 2:
        return 0.0
    reps = sum(1 for i in range(len(tokens) - 1) if tokens[i] == tokens[i + 1])
    return reps / max(len(tokens) - 1, 1)
