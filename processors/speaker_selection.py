"""
Interview speaker diarization helpers — pick human candidate vs AI interviewer.

Centralizes logic used by local AudioProcessor and (mirrored in) Kaggle notebook.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any

import config

# Turns shorter than this on the candidate track are usually mis-labeled AI prompts.
MIN_CANDIDATE_SEGMENT_SEC: float = getattr(config, "MIN_CANDIDATE_SEGMENT_SEC", 4.0)
AI_SHORT_TURN_SEC: float = getattr(config, "AI_SHORT_TURN_SEC", 2.5)


def extract_segments_from_diarization(diarization_output: Any) -> list[tuple[float, float, str]]:
    """Parse pyannote 3.x Annotation or 4.x DiarizeOutput."""
    annotation = diarization_output
    if hasattr(diarization_output, "speaker_diarization"):
        annotation = diarization_output.speaker_diarization
    elif hasattr(diarization_output, "exclusive_speaker_diarization"):
        annotation = diarization_output.exclusive_speaker_diarization

    segments: list[tuple[float, float, str]] = []
    if hasattr(annotation, "itertracks"):
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            segments.append((float(turn.start), float(turn.end), str(speaker)))
    return segments


def speaker_total_durations(
    segments: list[tuple[float, float, str]],
) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for start, end, speaker in segments:
        totals[speaker] += max(0.0, end - start)
    return dict(totals)


def turn_lengths_for_speaker(
    segments: list[tuple[float, float, str]],
    speaker: str,
    *,
    min_turn_sec: float = 0.0,
) -> list[float]:
    out: list[float] = []
    for start, end, spk in segments:
        if spk != speaker:
            continue
        dur = max(0.0, end - start)
        if dur >= min_turn_sec:
            out.append(dur)
    return out


def filter_candidate_segments(
    segments: list[tuple[float, float, str]],
    candidate: str,
    *,
    min_duration_sec: float | None = None,
) -> list[tuple[float, float]]:
    """Keep only candidate speech; drop short blips (AI questions mis-labeled)."""
    min_dur = float(min_duration_sec if min_duration_sec is not None else MIN_CANDIDATE_SEGMENT_SEC)
    kept: list[tuple[float, float]] = []
    dropped = 0
    for start, end, spk in segments:
        if spk != candidate:
            continue
        dur = max(0.0, end - start)
        if dur >= min_dur:
            kept.append((start, end))
        else:
            dropped += 1
    return kept


def select_candidate_speaker(
    segments: list[tuple[float, float, str]],
    strategy: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    Pick which diarization label is the human candidate (not AI interviewer).

    Strategies:
      most_speech   — legacy default; works when candidate talks more overall
      least_speech  — when AI interviewer dominates talk time
      longest_turns — when AI asks short prompts and candidate gives long answers
      auto          — vote across heuristics + composite (recommended for 2-speaker)
    """
    totals = speaker_total_durations(segments)
    strategy = (strategy or config.CANDIDATE_SPEAKER or "most_speech").lower()

    if not totals:
        return "SPEAKER_00", {"strategy": strategy, "reason": "no_segments"}

    if strategy == "auto":
        return _select_auto(segments, totals)

    if strategy == "least_speech":
        candidate = min(totals, key=totals.get)
        return candidate, _selection_meta(strategy, candidate, totals, segments)

    if strategy == "longest_turns":
        candidate = _pick_longest_turns(segments, totals)
        return candidate, _selection_meta(strategy, candidate, totals, segments)

    # most_speech
    candidate = max(totals, key=totals.get)
    return candidate, _selection_meta(strategy, candidate, totals, segments)


def _pick_longest_turns(
    segments: list[tuple[float, float, str]],
    totals: dict[str, float],
) -> str:
    min_turn = config.CANDIDATE_TURN_MIN_SEC
    scores: dict[str, float] = {}
    for speaker in totals:
        lengths = turn_lengths_for_speaker(segments, speaker, min_turn_sec=min_turn)
        if not lengths:
            lengths = turn_lengths_for_speaker(segments, speaker, min_turn_sec=0.0)
        scores[speaker] = float(statistics.median(lengths)) if lengths else 0.0
    return max(scores, key=scores.get)


def _select_auto(
    segments: list[tuple[float, float, str]],
    totals: dict[str, float],
) -> tuple[str, dict[str, Any]]:
    """Robust 2-speaker pick: vote + composite score."""
    speakers = list(totals.keys())
    if len(speakers) == 1:
        return speakers[0], _selection_meta("auto", speakers[0], totals, segments)

    votes: dict[str, int] = defaultdict(int)
    most = max(totals, key=totals.get)
    least = min(totals, key=totals.get)
    longest = _pick_longest_turns(segments, totals)
    votes[most] += 1
    votes[least] += 1
    votes[longest] += 1

    composite = _pick_composite(segments, totals)
    votes[composite] += 2  # stronger weight on structure-based score

    # Winner by votes; tie-break with composite
    max_votes = max(votes.values())
    winners = [spk for spk, v in votes.items() if v == max_votes]
    if len(winners) == 1:
        candidate = winners[0]
    else:
        candidate = composite

    meta = _selection_meta("auto", candidate, totals, segments)
    meta["votes"] = dict(votes)
    meta["composite_pick"] = composite
    meta["vote_winners"] = winners
    return candidate, meta


def _pick_composite(
    segments: list[tuple[float, float, str]],
    totals: dict[str, float],
) -> str:
    """
    Candidate likely has longer median turns, more long turns, fewer short AI blips.
    """
    metrics: dict[str, dict[str, float]] = {}
    for speaker in totals:
        lengths = turn_lengths_for_speaker(segments, speaker, min_turn_sec=0.0)
        if not lengths:
            lengths = [0.0]
        short_n = sum(1 for d in lengths if d < AI_SHORT_TURN_SEC)
        long_n = sum(1 for d in lengths if d >= 5.0)
        metrics[speaker] = {
            "median_turn": float(statistics.median(lengths)),
            "total_sec": totals[speaker],
            "short_frac": short_n / len(lengths),
            "long_count": float(long_n),
            "turn_count": float(len(lengths)),
        }

    max_median = max(m["median_turn"] for m in metrics.values()) or 1.0
    max_total = max(m["total_sec"] for m in metrics.values()) or 1.0
    max_long = max(m["long_count"] for m in metrics.values()) or 1.0

    def score(speaker: str) -> float:
        m = metrics[speaker]
        return (
            0.40 * (m["median_turn"] / max_median)
            + 0.25 * (m["total_sec"] / max_total)
            + 0.20 * (m["long_count"] / max_long)
            - 0.35 * m["short_frac"]
        )

    return max(metrics.keys(), key=score)


def _selection_meta(
    strategy: str,
    candidate: str,
    totals: dict[str, float],
    segments: list[tuple[float, float, str]],
) -> dict[str, Any]:
    min_turn = config.CANDIDATE_TURN_MIN_SEC
    turn_scores: dict[str, float] = {}
    for speaker in totals:
        lengths = turn_lengths_for_speaker(segments, speaker, min_turn_sec=min_turn)
        if not lengths:
            lengths = turn_lengths_for_speaker(segments, speaker, min_turn_sec=0.0)
        turn_scores[speaker] = round(
            float(statistics.median(lengths)) if lengths else 0.0, 2
        )
    return {
        "strategy": strategy,
        "speaker_total_sec": {k: round(v, 2) for k, v in totals.items()},
        "chosen_total_sec": round(totals.get(candidate, 0.0), 2),
        "speaker_median_turn_sec": turn_scores,
    }


def group_into_answers(
    candidate_segments: list[tuple[float, float]],
    *,
    silence_gap_sec: float = 3.0,
) -> list[tuple[float, float]]:
    if not candidate_segments:
        return []
    ordered = sorted(candidate_segments, key=lambda s: s[0])
    groups: list[list[tuple[float, float]]] = [[ordered[0]]]
    for start, end in ordered[1:]:
        prev_end = groups[-1][-1][1]
        if start - prev_end > silence_gap_sec:
            groups.append([(start, end)])
        else:
            groups[-1].append((start, end))
    return [(g[0][0], g[-1][1]) for g in groups]
