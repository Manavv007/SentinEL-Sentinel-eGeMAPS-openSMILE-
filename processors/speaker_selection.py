"""
Interview speaker diarization helpers — pick human candidate vs AI interviewer.

Centralizes logic used by local AudioProcessor and (mirrored in) Kaggle notebook.

AI-interviewer interviews (e.g. Saren / PostHog-style bots):
  - AI opens with a long intro (first speaker)
  - AI asks questions; human gives longer answers
  - ``most_speech`` and total-duration heuristics often pick the AI — avoid them in ``auto``.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any

import config

MIN_CANDIDATE_SEGMENT_SEC: float = getattr(config, "MIN_CANDIDATE_SEGMENT_SEC", 4.0)
AI_SHORT_TURN_SEC: float = getattr(config, "AI_SHORT_TURN_SEC", 2.5)
# Gaps under this between same-speaker segments are merged into one turn
TURN_MERGE_GAP_SEC: float = getattr(config, "TURN_MERGE_GAP_SEC", 0.35)


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


def merge_consecutive_turns(
    segments: list[tuple[float, float, str]],
    *,
    merge_gap_sec: float | None = None,
) -> list[tuple[float, float, str]]:
    """Merge adjacent same-speaker segments into dialogue turns."""
    gap = float(merge_gap_sec if merge_gap_sec is not None else TURN_MERGE_GAP_SEC)
    if not segments:
        return []
    ordered = sorted(segments, key=lambda x: x[0])
    merged: list[tuple[float, float, str]] = [ordered[0]]
    for start, end, spk in ordered[1:]:
        ps, pe, pspk = merged[-1]
        if spk == pspk and start - pe <= gap:
            merged[-1] = (ps, max(pe, end), spk)
        else:
            merged.append((start, end, spk))
    return merged


def filter_candidate_segments(
    segments: list[tuple[float, float, str]],
    candidate: str,
    *,
    min_duration_sec: float | None = None,
) -> list[tuple[float, float]]:
    """Keep only candidate speech; drop short blips (AI questions mis-labeled)."""
    min_dur = float(min_duration_sec if min_duration_sec is not None else MIN_CANDIDATE_SEGMENT_SEC)
    kept: list[tuple[float, float]] = []
    for start, end, spk in segments:
        if spk != candidate:
            continue
        dur = max(0.0, end - start)
        if dur >= min_dur:
            kept.append((start, end))
    return kept


def _pick_responder_speaker(segments: list[tuple[float, float, str]]) -> str | None:
    """
    In Q&A, the candidate usually speaks *after* the other speaker (answers questions).
    """
    turns = merge_consecutive_turns(segments)
    follow_after_other: dict[str, int] = defaultdict(int)
    for i in range(1, len(turns)):
        _, _, prev = turns[i - 1]
        _, _, curr = turns[i]
        if prev != curr:
            follow_after_other[curr] += 1
    if not follow_after_other:
        return None
    return max(follow_after_other, key=follow_after_other.get)


def _pick_not_opener_speaker(segments: list[tuple[float, float, str]]) -> str | None:
    """AI interviewers typically open the session (intro + first question)."""
    turns = merge_consecutive_turns(segments)
    if not turns:
        return None
    opener = turns[0][2]
    speakers = {spk for _, _, spk in segments}
    if len(speakers) != 2:
        return None
    others = [s for s in speakers if s != opener]
    return others[0] if others else None


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return float(ordered[idx])


def select_candidate_speaker(
    segments: list[tuple[float, float, str]],
    strategy: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    Pick which diarization label is the human candidate (not AI interviewer).

    Strategies:
      most_speech   — legacy; often wrong when AI intro + many questions dominate time
      least_speech  — when AI interviewer dominates talk time
      longest_turns — when AI asks short prompts and candidate gives long answers
      responder     — speaker who most often talks after the other (Q&A pattern)
      auto          — AI-interview-aware voting (recommended for 2-speaker bots)
    """
    totals = speaker_total_durations(segments)
    strategy = (strategy or config.CANDIDATE_SPEAKER or "auto").lower()

    if not totals:
        return "SPEAKER_00", {"strategy": strategy, "reason": "no_segments"}

    if strategy == "auto":
        return _select_auto(segments, totals)

    if strategy == "responder":
        candidate = _pick_responder_speaker(segments) or _pick_longest_turns(segments, totals)
        return candidate, _selection_meta(strategy, candidate, totals, segments)

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
        merged = [
            max(0.0, e - s)
            for s, e, spk in merge_consecutive_turns(segments)
            if spk == speaker
        ]
        if not merged:
            merged = turn_lengths_for_speaker(segments, speaker, min_turn_sec=0.0)
        lengths = [d for d in merged if d >= min_turn] or merged
        # Prefer p90 — sustained answers vs short AI question bursts
        scores[speaker] = max(
            float(statistics.median(lengths)) if lengths else 0.0,
            _percentile(lengths, 90),
        )
    return max(scores, key=scores.get)


def _select_auto(
    segments: list[tuple[float, float, str]],
    totals: dict[str, float],
) -> tuple[str, dict[str, Any]]:
    """
    AI-interview-aware 2-speaker pick.

    Does NOT vote for ``most_speech`` — that label frequently selects the bot.
    """
    speakers = list(totals.keys())
    if len(speakers) == 1:
        return speakers[0], _selection_meta("auto", speakers[0], totals, segments)

    votes: dict[str, int] = defaultdict(int)
    responder = _pick_responder_speaker(segments)
    not_opener = _pick_not_opener_speaker(segments)
    longest = _pick_longest_turns(segments, totals)
    least = min(totals, key=totals.get)
    composite = _pick_composite(segments, totals)

    if responder:
        votes[responder] += 4
    if not_opener:
        votes[not_opener] += 3
    votes[longest] += 3
    votes[least] += 1
    votes[composite] += 2

    max_votes = max(votes.values())
    winners = [spk for spk, v in votes.items() if v == max_votes]
    if len(winners) == 1:
        candidate = winners[0]
    elif responder and responder in winners:
        candidate = responder
    elif longest in winners:
        candidate = longest
    else:
        candidate = composite

    meta = _selection_meta("auto", candidate, totals, segments)
    meta["votes"] = dict(votes)
    meta["composite_pick"] = composite
    meta["responder_pick"] = responder
    meta["not_opener_pick"] = not_opener
    meta["vote_winners"] = winners
    return candidate, meta


def _pick_composite(
    segments: list[tuple[float, float, str]],
    totals: dict[str, float],
) -> str:
    """
    Candidate likely has longer sustained turns and fewer short AI-style blips.
    Total talk time is de-emphasized (AI often wins on duration alone).
    """
    metrics: dict[str, dict[str, float]] = {}
    for speaker in totals:
        lengths = [
            max(0.0, e - s)
            for s, e, spk in merge_consecutive_turns(segments)
            if spk == speaker
        ]
        if not lengths:
            lengths = turn_lengths_for_speaker(segments, speaker, min_turn_sec=0.0)
        if not lengths:
            lengths = [0.0]
        short_n = sum(1 for d in lengths if d < AI_SHORT_TURN_SEC)
        long_n = sum(1 for d in lengths if d >= 8.0)
        metrics[speaker] = {
            "median_turn": float(statistics.median(lengths)),
            "p90_turn": _percentile(lengths, 90),
            "total_sec": totals[speaker],
            "short_frac": short_n / len(lengths),
            "long_count": float(long_n),
            "turn_count": float(len(lengths)),
        }

    max_median = max(m["median_turn"] for m in metrics.values()) or 1.0
    max_p90 = max(m["p90_turn"] for m in metrics.values()) or 1.0
    max_long = max(m["long_count"] for m in metrics.values()) or 1.0

    def score(speaker: str) -> float:
        m = metrics[speaker]
        return (
            0.35 * (m["median_turn"] / max_median)
            + 0.35 * (m["p90_turn"] / max_p90)
            + 0.25 * (m["long_count"] / max_long)
            - 0.40 * m["short_frac"]
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
        lengths = [
            max(0.0, e - s)
            for s, e, spk in merge_consecutive_turns(segments)
            if spk == speaker
        ]
        if not lengths:
            lengths = turn_lengths_for_speaker(segments, speaker, min_turn_sec=min_turn)
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
