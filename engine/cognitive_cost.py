"""
Cognitive cost profile — question difficulty vs observed retrieval/hesitation cost.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import config
from engine.personal_baseline import PersonalBaselineModel


def _estimate_question_difficulty(transcript: dict[str, Any], duration: float) -> float:
    text = str(transcript.get("transcript", "")).lower()
    tokens = text.split()
    tech = sum(1 for t in tokens if len(t) > 8)
    complexity = min(1.0, len(tokens) / max(duration, 1.0) / 4.0)
    return float(min(1.0, 0.4 * complexity + 0.3 * (tech / max(len(tokens), 1)) + 0.3))


def cognitive_cost_profile(
    *,
    answer: dict[str, Any],
    transcript: dict[str, Any],
    window_rows: list[dict[str, float]],
    baseline: PersonalBaselineModel,
    prev_answer_end: float | None,
) -> dict[str, float]:
    duration = max(float(answer.get("end_sec", 0)) - float(answer.get("start_sec", 0)), 1e-6)
    words = transcript.get("words") or []

    pre_latency = 0.0
    if prev_answer_end is not None and words:
        first_start = float(words[0].get("start", 0))
        # words are answer-relative; gap from prior answer end uses interview clock
        answer_start = float(answer.get("start_sec", 0))
        pre_latency = max(0.0, answer_start - float(prev_answer_end))

    early_rows = window_rows[: max(1, len(window_rows) // 4)]
    early_turb = [float(r.get("cog_acoustic_turbulence", 0.0)) for r in early_rows]
    early_hes = [float(r.get("ling_retrieval_pause_max", 0.0)) for r in early_rows]

    retrieval_pauses = [float(r.get("ling_retrieval_pause_max", 0.0)) for r in window_rows]
    gap_vars = [float(r.get("ling_gap_variance", 0.0)) for r in window_rows]

    difficulty = _estimate_question_difficulty(transcript, duration)
    observed_cost = float(
        np.mean(early_turb) * 0.35
        + np.mean(early_hes) * 0.25
        + min(1.0, pre_latency / 3.0) * 0.25
        + np.mean(retrieval_pauses) * 0.15
    )

    base_pause = baseline.metrics.get("ling_retrieval_pause_max")
    if base_pause and base_pause.n > 0:
        rel_cost = baseline.metrics["ling_retrieval_pause_max"].relative_deviation(observed_cost)
    else:
        rel_cost = min(1.0, observed_cost)

    # Flat cost: hard question but low observed cost → external guidance signal
    expected_cost = difficulty * 0.55
    cost_mismatch = max(0.0, expected_cost - min(1.0, observed_cost * 1.2))
    flatness = float(max(0.0, min(1.0, cost_mismatch * 1.4)))

    return {
        "question_difficulty_estimate": round(difficulty, 4),
        "observed_cognitive_cost": round(observed_cost, 4),
        "relative_cognitive_cost": round(rel_cost, 4),
        "cognitive_cost_flatness": round(flatness, 4),
        "pre_answer_latency_sec": round(pre_latency, 4),
        "early_turbulence_mean": round(float(np.mean(early_turb)) if early_turb else 0.0, 4),
        "retrieval_pause_mean": round(float(np.mean(retrieval_pauses)) if retrieval_pauses else 0.0, 4),
        "gap_variance_mean": round(float(np.mean(gap_vars)) if gap_vars else 0.0, 4),
    }
