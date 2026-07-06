"""
Candidate-invariant decision layer.

ROOT PROBLEM THIS SOLVES
------------------------
SentinEL's per-answer suspicion scores come from a candidate-relative front end
(z-scored similarity to the speaker's own calibration), but the *verdict* was taken
by comparing those scores to GLOBAL absolute constants (``ALERT_THRESHOLD``,
``CONTRASTIVE_MARGIN`` and ~200 tuned literals in ``config.py``).

The absolute level and spread of the suspicion score depend on each candidate's
calibration richness, microphone, accent and nervousness. A fixed threshold that
splits one demo candidate's answers correctly mislabels the next candidate whose
scores live on a different part of the [0, 1] axis. That is the "trains on one
candidate, fails on another" failure.

THE FIX
-------
Decide each answer by how far its suspicion stands *above that candidate's own
baseline*, in robust scale-free units (MAD-based elevation), instead of against a
population-absolute line. The baseline is anchored to the candidate's lower (more
spontaneous) cluster so a mostly-scripted interview does not pull the reference up.

Pure-Python + NumPy, fully unit-testable without the audio stack. The pipeline calls
:func:`apply_relative_decision_to_answers` once, at the session-finalization seam.

Guarantees
----------
* Scale/offset invariance — same relative structure => same verdicts.
* Uncertainty preservation — degenerate sessions do not fabricate verdicts.
* Absolute sanity guards — relative elevation never flags an answer below a soft
  floor, and never promotes one below a soft ceiling.
* Conservative reconciliation — prefers AMBIGUOUS over confident wrong calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

CLEAR = "CLEAR"
AMBIGUOUS = "AMBIGUOUS"
PROBABLE = "PROBABLE_SCRIPT_READING"

_RANK = {CLEAR: 0, AMBIGUOUS: 1, PROBABLE: 2}


@dataclass(frozen=True)
class RelativeDecisionConfig:
    """Unitless decision parameters — candidate-invariant by construction."""

    z_high: float = 1.25
    z_low: float = 0.5
    baseline_quantile: float = 0.35
    scale_floor: float = 0.04
    soft_abs_floor: float = 0.45
    soft_abs_ceiling: float = 0.40
    min_answers: int = 3
    min_session_spread: float = 0.05

    @classmethod
    def from_config(cls, cfg_module: Any) -> "RelativeDecisionConfig":
        def g(name: str, default: float) -> float:
            return float(getattr(cfg_module, name, default))

        def gi(name: str, default: int) -> int:
            return int(getattr(cfg_module, name, default))

        return cls(
            z_high=g("RELATIVE_DECISION_Z_HIGH", cls.z_high),
            z_low=g("RELATIVE_DECISION_Z_LOW", cls.z_low),
            baseline_quantile=g("RELATIVE_DECISION_BASELINE_QUANTILE", cls.baseline_quantile),
            scale_floor=g("RELATIVE_DECISION_SCALE_FLOOR", cls.scale_floor),
            soft_abs_floor=g("RELATIVE_DECISION_SOFT_ABS_FLOOR", cls.soft_abs_floor),
            soft_abs_ceiling=g("RELATIVE_DECISION_SOFT_ABS_CEILING", cls.soft_abs_ceiling),
            min_answers=gi("RELATIVE_DECISION_MIN_ANSWERS", cls.min_answers),
            min_session_spread=g("RELATIVE_DECISION_MIN_SPREAD", cls.min_session_spread),
        )


@dataclass
class CandidateBaseline:
    center: float
    scale: float
    spread: float
    n: int
    degenerate: bool


@dataclass
class RelativeDecision:
    index: int
    score: float
    elevation_z: float
    rel_status: str
    confidence: str
    reason: str


def robust_baseline(scores: Sequence[float], cfg: RelativeDecisionConfig) -> CandidateBaseline:
    """Estimate the candidate's own suspicion baseline (low-quantile anchor) and MAD scale."""
    arr = np.asarray([float(s) for s in scores if np.isfinite(s)], dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        return CandidateBaseline(0.0, cfg.scale_floor, 0.0, 0, True)

    spread = float(arr.max() - arr.min())
    center = float(np.quantile(arr, cfg.baseline_quantile))
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    scale = max(1.4826 * mad, cfg.scale_floor)  # 1.4826 => MAD consistent with std
    degenerate = n < cfg.min_answers or spread < cfg.min_session_spread
    return CandidateBaseline(center, scale, spread, n, degenerate)


def elevation_z(score: float, baseline: CandidateBaseline) -> float:
    """Scale-free elevation of a score above the candidate baseline (signed)."""
    return float((float(score) - baseline.center) / max(baseline.scale, 1e-9))


def _confidence(z: float, baseline: CandidateBaseline, cfg: RelativeDecisionConfig) -> str:
    if baseline.degenerate:
        return "LOW"
    if z >= cfg.z_high + 0.75 or z <= cfg.z_low - 0.75:
        return "HIGH"
    if z >= cfg.z_high or z <= cfg.z_low:
        return "MEDIUM"
    return "LOW"


def relative_decision(
    scores: Sequence[float], cfg: RelativeDecisionConfig | None = None
) -> list[RelativeDecision]:
    """Decide each answer from candidate-relative elevation, not an absolute threshold."""
    cfg = cfg or RelativeDecisionConfig()
    baseline = robust_baseline(scores, cfg)
    out: list[RelativeDecision] = []

    for i, raw in enumerate(scores):
        score = float(raw) if np.isfinite(raw) else baseline.center
        z = elevation_z(score, baseline)

        if baseline.degenerate:
            out.append(
                RelativeDecision(
                    i, score, round(z, 4), AMBIGUOUS, "LOW",
                    "degenerate session (n=%d, spread=%.3f) — relative decision withheld, "
                    "uncertainty preserved" % (baseline.n, baseline.spread),
                )
            )
            continue

        if z >= cfg.z_high and score >= cfg.soft_abs_floor:
            status = PROBABLE
            reason = (
                "elevation z=%.2f >= %.2f above candidate baseline %.3f "
                "(score %.3f >= soft floor %.2f)"
                % (z, cfg.z_high, baseline.center, score, cfg.soft_abs_floor)
            )
        elif z <= cfg.z_low or score <= cfg.soft_abs_ceiling:
            status = CLEAR
            reason = (
                "elevation z=%.2f near candidate baseline %.3f (<= z_low %.2f or "
                "score %.3f <= ceiling %.2f)"
                % (z, baseline.center, cfg.z_low, score, cfg.soft_abs_ceiling)
            )
        else:
            status = AMBIGUOUS
            reason = (
                "elevation z=%.2f between z_low %.2f and z_high %.2f — mixed evidence"
                % (z, cfg.z_low, cfg.z_high)
            )

        out.append(
            RelativeDecision(i, score, round(z, 4), status, _confidence(z, baseline, cfg), reason)
        )

    return out


# --------------------------------------------------------------------------- #
# Score extraction + reconciliation for SentinEL answer payloads
# --------------------------------------------------------------------------- #

def default_score_getter(answer: dict[str, Any]) -> float | None:
    """Per-answer suspicion scalar; prefers raw_score over cross-answer EWMA."""
    for key in ("raw_score", "fused_score", "smoothed_score", "ewma_score"):
        v = answer.get(key)
        if isinstance(v, (int, float)) and np.isfinite(v):
            return float(v)
    contrastive = answer.get("contrastive") or {}
    for key in ("composite_score", "ewma_score", "answer_score"):
        v = contrastive.get(key)
        if isinstance(v, (int, float)) and np.isfinite(v):
            return float(v)
    return None


def _is_personal_protected(answer: dict[str, Any]) -> bool:
    """True when transcript reads as genuine personal narrative (protect CLEAR)."""
    spec = answer.get("semantic_specificity")
    if not spec:
        return False
    try:
        from engine.semantic_specificity import is_personal_natural_answer

        return bool(is_personal_natural_answer(spec))
    except Exception:
        return False


def _is_semantically_scripted(answer: dict[str, Any]) -> bool:
    """
    True when the transcript independently reads as scripted content (generic essay /
    platitudes or memorized-technical definition prose).

    Used as a guard so the candidate-relative pass never *clears* an answer whose words
    look scripted -- which protects against a uniformly-scripted session where the
    per-candidate baseline is itself contaminated (an answer can sit at/below that
    contaminated baseline yet still be a memorized script).
    """
    spec = answer.get("semantic_specificity")
    if not spec:
        return False
    g = float(spec.get("generic_script_likelihood", 0.0) or 0.0)
    m = float(spec.get("memorized_technical_script_score", 0.0) or 0.0)
    generic_min, mem_min = 0.58, 0.42  # config defaults (SPECIFICITY_GENERIC_PROBABLE_MIN, MEMORIZED_TECHNICAL_PROBABLE_MIN)
    try:
        import config

        generic_min = float(getattr(config, "SPECIFICITY_GENERIC_PROBABLE_MIN", generic_min))
        mem_min = float(getattr(config, "MEMORIZED_TECHNICAL_PROBABLE_MIN", mem_min))
    except Exception:
        pass
    return g >= generic_min or m >= mem_min


def reconcile(
    current_status: str,
    rel: RelativeDecision,
    *,
    personal_protected: bool,
    semantically_scripted: bool = False,
    preserve_uncertainty: bool = True,
) -> tuple[str, str | None]:
    """Combine existing status with the candidate-relative verdict (uncertainty-preserving)."""
    if rel.rel_status == PROBABLE:
        if personal_protected:
            if current_status == PROBABLE and preserve_uncertainty:
                return AMBIGUOUS, (
                    "candidate-relative elevation high but personal narrative protected "
                    "— softened to AMBIGUOUS"
                )
            return current_status, None
        if _RANK[current_status] < _RANK[PROBABLE]:
            return PROBABLE, "candidate-relative: " + rel.reason
        return current_status, None

    if rel.rel_status == CLEAR and current_status in (PROBABLE, AMBIGUOUS):
        # Never clear content that independently reads as scripted: guards a uniformly
        # scripted session whose own baseline is contaminated (e.g. a memorized essay
        # scoring at/below baseline must still stay flagged).
        if semantically_scripted:
            return current_status, None
        if personal_protected:
            return CLEAR, (
                "candidate-relative: not elevated for this candidate + personal narrative "
                "— absolute-threshold false positive removed (" + rel.reason + ")"
            )
        # Confident "at/below this candidate's own baseline" with no scripted-content
        # markers is positive CLEAR evidence (the person-relative thesis) -- clear it.
        if rel.confidence in ("MEDIUM", "HIGH"):
            return CLEAR, (
                "candidate-relative: at/below this candidate's baseline with no "
                "scripted-content markers — cleared (" + rel.reason + ")"
            )
        if current_status == PROBABLE and preserve_uncertainty:
            return AMBIGUOUS, (
                "candidate-relative: not elevated for this candidate — absolute-threshold "
                "call softened to AMBIGUOUS (" + rel.reason + ")"
            )
        return current_status, None

    return current_status, None


def apply_relative_decision_to_answers(
    answers: list[dict[str, Any]],
    cfg: RelativeDecisionConfig | None = None,
    *,
    score_getter: Callable[[dict[str, Any]], float | None] = default_score_getter,
    preserve_uncertainty: bool = True,
) -> dict[str, Any]:
    """
    Re-decide per-answer statuses on a candidate-relative basis (in place).

    Single entry point the pipeline calls at the finalization seam. Mutates each
    answer's ``status`` (and nested ``contrastive`` block if present), records the
    relative evidence, and returns a session-level summary. No-op when the session is
    degenerate (too few answers / no spread).
    """
    cfg = cfg or RelativeDecisionConfig()

    indexed: list[tuple[int, float]] = []
    for idx, ans in enumerate(answers):
        s = score_getter(ans)
        if s is not None:
            indexed.append((idx, s))

    scores = [s for _, s in indexed]
    baseline = robust_baseline(scores, cfg)
    decisions = relative_decision(scores, cfg)

    changes: list[dict[str, Any]] = []
    applied = 0

    if not baseline.degenerate:
        for (idx, _score), rel in zip(indexed, decisions):
            ans = answers[idx]
            current = str(ans.get("status", CLEAR))
            protected = _is_personal_protected(ans)
            scripted = _is_semantically_scripted(ans)
            new_status, reason = reconcile(
                current, rel, personal_protected=protected,
                semantically_scripted=scripted,
                preserve_uncertainty=preserve_uncertainty,
            )
            ans["relative_decision"] = {
                "elevation_z": rel.elevation_z,
                "rel_status": rel.rel_status,
                "confidence": rel.confidence,
                "candidate_baseline": round(baseline.center, 4),
                "candidate_scale": round(baseline.scale, 4),
                "reason": rel.reason,
            }
            if new_status != current and reason:
                ans["status"] = new_status
                ans.setdefault("status_history", []).append(
                    {"from": current, "to": new_status, "by": "relative_decision"}
                )
                contrastive = ans.get("contrastive")
                if isinstance(contrastive, dict):
                    contrastive["status"] = new_status
                    expl = contrastive.get("decision_explanation") or []
                    if not isinstance(expl, list):
                        expl = [str(expl)]
                    if reason not in expl:
                        expl.append(reason)
                    contrastive["decision_explanation"] = expl
                changes.append(
                    {"answer_index": idx, "from": current, "to": new_status, "reason": reason}
                )
                applied += 1

    return {
        "enabled": True,
        "applied": applied,
        "degenerate_session": baseline.degenerate,
        "candidate_baseline": round(baseline.center, 4),
        "candidate_scale": round(baseline.scale, 4),
        "session_spread": round(baseline.spread, 4),
        "n_scored_answers": baseline.n,
        "config": {
            "z_high": cfg.z_high,
            "z_low": cfg.z_low,
            "baseline_quantile": cfg.baseline_quantile,
            "soft_abs_floor": cfg.soft_abs_floor,
            "soft_abs_ceiling": cfg.soft_abs_ceiling,
            "min_answers": cfg.min_answers,
            "min_session_spread": cfg.min_session_spread,
        },
        "changes": changes,
    }
