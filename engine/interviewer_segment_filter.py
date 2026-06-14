"""Detect AI-interviewer prompts mis-segmented as candidate answers."""

from __future__ import annotations

import re
from typing import Any

import config

_DEFAULT_MIN_ANSWER_WORDS = max(
    1, int(getattr(config, "DIARIZATION_MIN_ANSWER_WORDS", 5) or 5)
)

_INTERVIEWER_OPENERS = re.compile(
    r"^\s*("
    r"hello and welcome|hi,?\s+welcome|"
    r"we(?:'|')?ll be going|let(?:'|')?s begin|to start|feel free to|"
    r"moving on|shift focus|further detail|redirecting|noted\.|indeed\.|understood\.|"
    r"could you|can you|would you|how would you|what would you|which .{0,40} would you|"
    r"tell me about|walk me through|share what|go through a series|"
    r"let(?:'|')?s begin with the first question|considering scalability"
    r")",
    re.IGNORECASE,
)

_AI_BOT_MARKERS = re.compile(
    r"\b("
    r"I(?:'|')?m Saren|I am Saren|"
    r"I(?:'|')?ll be conducting your interview|conducting your interview today|"
    r"go through a series of questions|"
    r"explain your reasoning as you go|"
    r"think through your responses|"
    r"introduce yourself and share what|"
    r"attracted you to this role|"
    r"focused on .{0,40} developer|"
    r"backend developer with some follow-ups"
    r")\b",
    re.IGNORECASE,
)

_CANDIDATE_OPENER = re.compile(
    r"^\s*("
    r"yeah|yes|sure|so|well|okay|ok|"
    r"I(?:'|')?m\b|I am\b|my name|at my|in my|we use|our team|our project"
    r")\b",
    re.IGNORECASE,
)

_QUESTION_LEAD = re.compile(
    r"\b(how|what|which|why|when|where|could|would|can|tell me|describe|explain)\b",
    re.IGNORECASE,
)

_AI_ACK_THEN_QUESTION = re.compile(
    r"^\s*(understood|noted|indeed|okay|ok|right|sure|great)\b.{0,40}\?",
    re.IGNORECASE | re.DOTALL,
)

# Segments that are clearly not from this interview's Q&A (wrong diarization slice).
_WRONG_SESSION_CONTENT = re.compile(
    r"\b("
    r"seven habits|translate the page|particle gas|"
    r"fuels are mixtures"
    r")\b",
    re.IGNORECASE,
)


def is_interviewer_transcript(text: str) -> bool:
    """
    True when transcript reads like an interviewer prompt, not a candidate answer.
    Used after ASR to drop mis-labeled diarization segments.
    """
    t = " ".join(str(text or "").split()).strip()
    if not t:
        return True

    sample = t[:240]
    if _INTERVIEWER_OPENERS.search(sample):
        return True
    if _AI_BOT_MARKERS.search(t):
        return True
    if _AI_ACK_THEN_QUESTION.search(sample):
        return True

    words = t.split()
    n_words = len(words)
    q_marks = t.count("?")

    if re.search(r"\bhow would you\b", t, re.IGNORECASE):
        if not _CANDIDATE_OPENER.search(t[:140]):
            return True

    if q_marks >= 1 and n_words <= 60:
        leads = len(_QUESTION_LEAD.findall(t))
        if leads >= 1 and not _CANDIDATE_OPENER.search(t[:100]):
            if t.rstrip().endswith("?") or leads >= 2:
                return True

    if q_marks >= 2 and n_words <= 80:
        return True

    # Long AI intro monologue without candidate self-introduction opener
    if n_words >= 25 and not _CANDIDATE_OPENER.search(t[:160]):
        if _AI_BOT_MARKERS.search(t) or (
            q_marks >= 1 and re.search(r"\b(you|your)\b", t, re.IGNORECASE)
        ):
            if re.search(
                r"\b(introduce yourself|interview today|first question|this role)\b",
                t,
                re.IGNORECASE,
            ):
                return True

    return False


def is_off_topic_or_wrong_slice(text: str) -> bool:
    """True when transcript is clearly from the wrong segment of the recording."""
    t = " ".join(str(text or "").split()).strip()
    if not t:
        return False
    return bool(_WRONG_SESSION_CONTENT.search(t))


def is_too_short_answer(text: str, *, min_words: int | None = None) -> bool:
    """Drop filler acknowledgments ('Sure.', 'Okay.') that are not scorable answers."""
    threshold = int(min_words if min_words is not None else _DEFAULT_MIN_ANSWER_WORDS)
    words = [w for w in str(text or "").split() if w.strip()]
    return len(words) < threshold


def filter_segments_by_transcript(
    answers: list[dict[str, Any]],
    transcripts: list[dict[str, Any]],
    *,
    min_answer_words: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[bool]]:
    """
    Drop answer segments whose transcript is interviewer speech or too short.
    Returns (kept_answers, excluded_log_entries, keep_mask aligned to inputs).
    """
    kept: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    keep_mask: list[bool] = []

    for idx, (answer, transcript) in enumerate(zip(answers, transcripts)):
        text = str(transcript.get("transcript", ""))
        if is_interviewer_transcript(text):
            excluded.append(
                {
                    "index": idx,
                    "answer_id": answer.get("answer_id"),
                    "start_sec": answer.get("start_sec"),
                    "end_sec": answer.get("end_sec"),
                    "transcript_preview": text[:160],
                    "reason": "interviewer_prompt",
                }
            )
            keep_mask.append(False)
            continue
        if is_off_topic_or_wrong_slice(text):
            excluded.append(
                {
                    "index": idx,
                    "answer_id": answer.get("answer_id"),
                    "start_sec": answer.get("start_sec"),
                    "end_sec": answer.get("end_sec"),
                    "transcript_preview": text[:160],
                    "reason": "off_topic_or_wrong_slice",
                }
            )
            keep_mask.append(False)
            continue
        if is_too_short_answer(text, min_words=min_answer_words):
            excluded.append(
                {
                    "index": idx,
                    "answer_id": answer.get("answer_id"),
                    "start_sec": answer.get("start_sec"),
                    "end_sec": answer.get("end_sec"),
                    "transcript_preview": text[:160],
                    "reason": "too_short",
                }
            )
            keep_mask.append(False)
            continue
        kept.append(answer)
        keep_mask.append(True)

    for i, answer in enumerate(kept):
        answer["answer_id"] = i

    return kept, excluded, keep_mask


def assess_kept_segment_quality(
    transcripts: list[dict[str, Any]],
) -> tuple[bool, dict[str, Any]]:
    """
    False when kept segments still contain interviewer speech, off-topic slices,
    or too few usable answers for a typical interview.
    """
    texts = [str(t.get("transcript", "")).strip() for t in transcripts]
    texts = [t for t in texts if t]
    if not texts:
        return False, {"reason": "no_transcripts"}

    interviewer = sum(1 for t in texts if is_interviewer_transcript(t))
    off_topic = sum(1 for t in texts if is_off_topic_or_wrong_slice(t))
    too_short = sum(1 for t in texts if is_too_short_answer(t))

    meta: dict[str, Any] = {
        "kept": len(texts),
        "interviewer_in_kept": interviewer,
        "off_topic_in_kept": off_topic,
        "too_short_in_kept": too_short,
    }

    if interviewer > 0:
        meta["reason"] = "interviewer_in_kept"
        return False, meta
    if off_topic > 0:
        meta["reason"] = "off_topic_in_kept"
        return False, meta
    if len(texts) < 3:
        meta["reason"] = "too_few_segments"
        return False, meta
    if off_topic >= max(1, len(texts) // 2):
        meta["reason"] = "mostly_off_topic"
        return False, meta

    return True, meta


def infer_alternate_speaker(speaker_selection: dict[str, Any] | None) -> str | None:
    """Pick the diarization label that was not chosen as primary candidate."""
    if not speaker_selection:
        return None
    chosen = speaker_selection.get("chosen_speaker")
    if not chosen:
        # Older Kaggle notebooks returned debug votes but not the final field.
        # Infer the primary track from the strongest available selection hint.
        for key in (
            "original_candidate",
            "forced_speaker",
            "responder_pick",
            "vote_winners",
            "composite_pick",
        ):
            value = speaker_selection.get(key)
            if isinstance(value, list) and value:
                chosen = value[0]
                break
            if isinstance(value, str) and value:
                chosen = value
                break
    if not chosen:
        winners = speaker_selection.get("vote_winners") or []
        chosen = winners[0] if winners else None
    if not chosen:
        return None
    totals = speaker_selection.get("speaker_total_sec") or {}
    others = [spk for spk in totals if spk != chosen]
    if others:
        return others[0]
    alt = speaker_selection.get("alternate_speaker")
    return str(alt) if alt else None


def needs_speaker_track_recovery(
    excluded_count: int,
    total_count: int,
    kept_count: int,
) -> bool:
    """True when most primary-track segments look like interviewer prompts."""
    if total_count <= 0:
        return False
    if kept_count < 3:
        return True
    return (excluded_count / total_count) >= 0.45
