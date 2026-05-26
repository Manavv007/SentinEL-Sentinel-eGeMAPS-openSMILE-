"""
Cognitive spontaneity vs guided-explanation discrimination.

Distinguishes fluent natural cognition (retrieval friction, semantic wobble, repairs)
from preconstructed / externally guided delivery (smooth retrieval, high compression).
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np

import config
from engine.linguistic_analyzer import LinguisticAnalyzer

SEMANTIC_REPAIR_RE = re.compile(
    r"\b("
    r"what i mean|i mean|sorry|wait|no i mean|let me rephrase|actually|"
    r"or rather|in other words|to clarify|what i'm trying to say|"
    r"hold on|scratch that|not quite|sort of like|kind of like"
    r")\b",
    re.I,
)
CORRECTION_RE = re.compile(
    r"\b(actually|sorry|wait|no i mean|let me rephrase)\b",
    re.I,
)
BACKTRACK_RE = re.compile(
    r"\b(again|back to|as i said|like i said|so again|anyway|"
    r"going back|to go back|let me go back)\b",
    re.I,
)
TRANSITION_RE = re.compile(
    r"\b(so|now|okay|ok|well|right|anyway|next|then|because|"
    r"the thing is|the reason is|first|second|finally)\b",
    re.I,
)
FILLER_RE = re.compile(
    r"\b(um|uh|er|ah|hmm|like|you know|i mean)\b",
    re.I,
)
TECHNICAL_RE = re.compile(
    r"\b(api|apis|websocket|lambda|microservice|database|sql|"
    r"authentication|kubernetes|docker|architecture|endpoint|"
    r"serverless|cache|redis|postgres|frontend|backend|"
    r"deployment|scalability|latency|encryption|token|session)\b",
    re.I,
)
HEDGE_RE = re.compile(
    r"\b(try to|i think|i guess|maybe|probably|sort of|kind of|"
    r"that actually|which part|in terms of|for me)\b",
    re.I,
)
ABANDON_RE = re.compile(r"\b(\.{2,}|—|–|-{2,})\b")


def _words_in_range(
    transcript: dict[str, Any],
    start_sec: float,
    end_sec: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for w in transcript.get("words", []) or []:
        if w.get("start") is None or w.get("end") is None:
            continue
        ws, we = float(w["start"]), float(w["end"])
        if we > start_sec and ws < end_sec:
            out.append(w)
    return out


def _gaps_before_tokens(
    words: list[dict[str, Any]],
    token_re: re.Pattern[str],
) -> list[float]:
    """Pause (sec) immediately before matched tokens."""
    pauses: list[float] = []
    for i, w in enumerate(words):
        token = str(w.get("word", "")).lower()
        if not token_re.search(token):
            continue
        if i == 0:
            pauses.append(float(w["start"]))
            continue
        gap = float(w["start"]) - float(words[i - 1]["end"])
        if gap >= 0:
            pauses.append(gap)
    return pauses


def _sentence_lengths(text: str) -> list[int]:
    parts = re.split(r"[.!?]+", text)
    return [len(p.split()) for p in parts if p.strip()]


def compute_answer_cognitive_profile(transcript: dict[str, Any]) -> dict[str, float]:
    """Full-answer cognitive signature (used to stabilize per-window estimates)."""
    words = list(transcript.get("words", []) or [])
    duration = float(transcript.get("duration_sec") or 0.0)
    if not words:
        text = str(transcript.get("transcript", "")).strip()
        if not text:
            return _empty_profile()
        tokens = text.split()
        duration = duration or max(len(tokens) / 2.5, 1.0)
        step = duration / max(len(tokens), 1)
        words = [
            {"word": tok, "start": i * step, "end": (i + 1) * step}
            for i, tok in enumerate(tokens)
        ]
    else:
        duration = max(
            float(words[-1]["end"]) - float(words[0]["start"]),
            duration,
            1e-6,
        )
    return _compute_cognitive_metrics(words, duration, full_text=True)


def extract_window_cognitive_features(
    transcript: dict[str, Any],
    *,
    start_sec: float,
    end_sec: float,
    answer_profile: dict[str, float] | None = None,
) -> dict[str, float]:
    words = _words_in_range(transcript, start_sec, end_sec)
    if not words:
        return {}
    duration = max(end_sec - start_sec, 1e-6)
    metrics = _compute_cognitive_metrics(words, duration, full_text=False)
    if answer_profile:
        metrics["cog_answer_spontaneity"] = float(
            answer_profile.get("cog_spontaneity_index", 0.0)
        )
        metrics["cog_answer_guided"] = float(
            answer_profile.get("cog_guided_explanation_index", 0.0)
        )
    return metrics


def _compute_cognitive_metrics(
    words: list[dict[str, Any]],
    duration: float,
    *,
    full_text: bool,
) -> dict[str, float]:
    text = " ".join(str(w.get("word", "")) for w in words).lower()
    tokens = LinguisticAnalyzer._normalised_words({"words": words, "transcript": text})
    token_n = max(len(tokens), 1)

    gaps: list[float] = []
    for i in range(len(words) - 1):
        g = float(words[i + 1]["start"]) - float(words[i]["end"])
        if g >= 0:
            gaps.append(g)

    tech_pauses = _gaps_before_tokens(words, TECHNICAL_RE)
    trans_pauses = _gaps_before_tokens(words, TRANSITION_RE)

    filler_before_tech = 0
    tech_positions = [i for i, w in enumerate(words) if TECHNICAL_RE.search(str(w.get("word", "")))]
    for idx in tech_positions:
        if idx > 0 and FILLER_RE.search(str(words[idx - 1].get("word", ""))):
            filler_before_tech += 1

    repair_count = len(SEMANTIC_REPAIR_RE.findall(text))
    backtrack_count = len(BACKTRACK_RE.findall(text))
    tech_density = len(TECHNICAL_RE.findall(text)) / token_n
    hedge_count = len(HEDGE_RE.findall(text))

    # Retrieval friction: uneven pauses + hesitation before content words
    pause_before_content = tech_pauses + trans_pauses
    friction_pause = (
        float(np.mean(pause_before_content)) if pause_before_content else 0.0
    )
    gap_var = float(np.var(gaps, ddof=1)) if len(gaps) >= 2 else 0.0
    retrieval_friction = min(
        1.0,
        0.45 * min(1.0, friction_pause / 0.85)
        + 0.30 * min(1.0, math.sqrt(gap_var / 0.012))
        + 0.25 * min(1.0, filler_before_tech / 2.0),
    )

    # Semantic drift / wobble
    bigram_repeat = _bigram_repetition(tokens)
    type_token_ratio = len(set(tokens)) / token_n
    semantic_drift = min(
        1.0,
        0.40 * min(1.0, backtrack_count / 2.0)
        + 0.35 * min(1.0, bigram_repeat * 3.0)
        + 0.25 * min(1.0, abs(type_token_ratio - 0.72) / 0.2),
    )

    # Concept compression: dense *deep* technical content + smooth delivery (not everyday fluency)
    filler_rate = len(FILLER_RE.findall(text)) / (duration / 30.0)
    smoothness = 1.0 - min(1.0, gap_var / 0.015)
    concept_compression = min(
        1.0,
        tech_density * (0.55 + 0.45 * smoothness) * (1.0 + min(1.0, 2.0 / (filler_rate + 0.5))),
    )
    if tech_density < 0.08 and hedge_count >= 1:
        concept_compression *= 0.45

    # Sentence assembly variance
    sent_lens = _sentence_lengths(text)
    if len(sent_lens) >= 2:
        assembly_var = min(1.0, float(np.std(sent_lens, ddof=1)) / (float(np.mean(sent_lens)) + 1e-6))
    elif len(tokens) >= 6:
        mid = len(tokens) // 2
        assembly_var = min(1.0, abs(len(tokens[:mid]) - len(tokens[mid:])) / len(tokens))
    else:
        assembly_var = 0.35 if full_text else 0.2

      # Semantic repair (reduces suspicion when elevated)
    repair_density = repair_count / (duration / 30.0)
    semantic_repair = min(
        1.0,
        repair_density / 2.5
        + len(CORRECTION_RE.findall(text)) * 0.15
        + min(1.0, hedge_count / 2.0) * 0.25,
    )

    # Cognitive wobble composite
    cognitive_wobble = min(
        1.0,
        0.35 * semantic_drift
        + 0.30 * retrieval_friction
        + 0.20 * assembly_var
        + 0.15 * semantic_repair,
    )

    spontaneity_index = min(
        1.0,
        0.28 * retrieval_friction
        + 0.22 * semantic_drift
        + 0.18 * assembly_var
        + 0.32 * semantic_repair
        + min(0.15, hedge_count * 0.08),
    )

    guided_index = min(
        1.0,
        0.40 * concept_compression
        + 0.30 * smoothness
        + 0.20 * max(0.0, 1.0 - retrieval_friction)
        + 0.10 * max(0.0, 1.0 - semantic_repair),
    )
    if semantic_repair >= 0.25 and (semantic_drift >= 0.15 or hedge_count >= 1):
        guided_index *= 0.55
    if spontaneity_index >= 0.40 and tech_density < 0.10:
        guided_index *= 0.70

    # Fluent trap: smooth + compressed + low friction (false positive driver)
    fluency_trap = min(
        1.0,
        smoothness * concept_compression * (1.0 - retrieval_friction * 0.7),
    )

    return {
        "cog_retrieval_friction": round(retrieval_friction, 6),
        "cog_semantic_drift": round(semantic_drift, 6),
        "cog_concept_compression": round(concept_compression, 6),
        "cog_assembly_variance": round(assembly_var, 6),
        "cog_semantic_repair": round(semantic_repair, 6),
        "cog_cognitive_wobble": round(cognitive_wobble, 6),
        "cog_spontaneity_index": round(spontaneity_index, 6),
        "cog_guided_explanation_index": round(guided_index, 6),
        "cog_fluency_trap": round(fluency_trap, 6),
    }


def score_cognitive_dimensions(
    features: dict[str, float],
    answer_profile: dict[str, float] | None = None,
) -> tuple[float, float, dict[str, float]]:
    """
    Returns (spontaneity_index, guided_index, breakdown).
    Blends window + answer-level when available.
    """
    profile = answer_profile or features
    spont = float(
        features.get("cog_spontaneity_index", profile.get("cog_spontaneity_index", 0.0))
    )
    guided = float(
        features.get(
            "cog_guided_explanation_index",
            profile.get("cog_guided_explanation_index", 0.0),
        )
    )
    if answer_profile:
        spont = 0.55 * spont + 0.45 * float(answer_profile.get("cog_spontaneity_index", spont))
        guided = 0.55 * guided + 0.45 * float(
            answer_profile.get("cog_guided_explanation_index", guided)
        )

    breakdown = {
        "retrieval_friction": float(features.get("cog_retrieval_friction", 0.0)),
        "semantic_drift": float(features.get("cog_semantic_drift", 0.0)),
        "concept_compression": float(features.get("cog_concept_compression", 0.0)),
        "assembly_variance": float(features.get("cog_assembly_variance", 0.0)),
        "semantic_repair": float(features.get("cog_semantic_repair", 0.0)),
        "cognitive_wobble": float(features.get("cog_cognitive_wobble", 0.0)),
        "fluency_trap": float(features.get("cog_fluency_trap", 0.0)),
    }
    return round(min(1.0, spont), 6), round(min(1.0, guided), 6), breakdown


def cognitive_spontaneity_suppression(
    spontaneity_index: float,
    guided_index: float,
    features: dict[str, float],
) -> float:
    """Extra suppression for fluent natural cognition (reduces contrastive score)."""
    repair = float(features.get("cog_semantic_repair", 0.0))
    wobble = float(features.get("cog_cognitive_wobble", 0.0))
    raw = (
        spontaneity_index * config.COGNITIVE_SPONTANEITY_SUPPRESSION_WEIGHT
        + repair * config.COGNITIVE_REPAIR_SUPPRESSION_WEIGHT
        + wobble * 0.08
    )
    if guided_index >= config.COGNITIVE_GUIDED_HIGH_THRESHOLD:
        raw *= max(0.25, 1.0 - guided_index)
    return round(min(config.COGNITIVE_MAX_EXTRA_SUPPRESSION, raw), 6)


def guided_explanation_boost(guided_index: float, fluency_trap: float) -> float:
    """Positive boost when delivery looks preconstructed."""
    if guided_index < config.COGNITIVE_GUIDED_BOOST_MIN:
        return 0.0
    trap = max(guided_index, fluency_trap)
    return round(
        min(
            config.COGNITIVE_GUIDED_MAX_BOOST,
            trap * config.COGNITIVE_GUIDED_BOOST_SCALE,
        ),
        6,
    )


def dampen_linguistic_fluency_score(
    linguistic_score: float,
    transcript: dict[str, Any],
    duration_sec: float,
) -> float:
    """Reduce fused-path over-penalty for fluent spontaneous answers."""
    profile = compute_answer_cognitive_profile(transcript)
    spont = float(profile.get("cog_spontaneity_index", 0.0))
    guided = float(profile.get("cog_guided_explanation_index", 0.0))
    if spont >= config.FLUENT_NATURAL_SPONTANEITY_FLOOR and guided < 0.50:
        reduction = (spont - config.FLUENT_NATURAL_SPONTANEITY_FLOOR) * config.FLUENT_LINGUISTIC_DAMPEN
        return round(max(0.0, linguistic_score - reduction), 6)
    return linguistic_score


def _bigram_repetition(tokens: list[str]) -> float:
    if len(tokens) < 4:
        return 0.0
    bigrams = [tuple(tokens[i : i + 2]) for i in range(len(tokens) - 1)]
    if not bigrams:
        return 0.0
    unique = len(set(bigrams))
    return 1.0 - (unique / len(bigrams))


def _empty_profile() -> dict[str, float]:
    return {
        "cog_retrieval_friction": 0.0,
        "cog_semantic_drift": 0.0,
        "cog_concept_compression": 0.0,
        "cog_assembly_variance": 0.0,
        "cog_semantic_repair": 0.0,
        "cog_cognitive_wobble": 0.0,
        "cog_spontaneity_index": 0.0,
        "cog_guided_explanation_index": 0.0,
        "cog_fluency_trap": 0.0,
    }
