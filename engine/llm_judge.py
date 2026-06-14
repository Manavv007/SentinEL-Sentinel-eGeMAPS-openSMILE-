"""Optional LLM tie-breaker for AMBIGUOUS answers only."""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

import config

logger = logging.getLogger(__name__)

VALID_VERDICTS = frozenset({"CLEAR", "AMBIGUOUS", "PROBABLE_SCRIPT_READING"})
VALID_CONFIDENCE = frozenset({"LOW", "MEDIUM", "HIGH"})
_CONFIDENCE_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}

JUDGE_SYSTEM_PROMPT = """You are a tie-breaker integrity judge for SentinEL, a technical interview analysis system. You receive candidate answer transcripts and supporting background signals. Your task: determine whether the answer shows signs of scripted/memorized delivery or spontaneous speech.

You will ONLY be called when all prior automated layers have returned AMBIGUOUS. You do not override CLEAR or PROBABLE_SCRIPT_READING verdicts set elsewhere.

## OUTPUT FORMAT

Respond with ONLY valid JSON. No preamble, no explanation outside the JSON.

{
  "script_reading_likelihood": <float 0.0–1.0>,
  "confidence": "LOW" | "MEDIUM" | "HIGH",
  "verdict": "CLEAR" | "AMBIGUOUS" | "PROBABLE_SCRIPT_READING",
  "is_interviewer_speech": <boolean>,
  "reasons": [<2 or 3 short strings, plain English, max ~15 words each>]
}


## STEP 1 — CHECK FOR INTERVIEWER SPEECH

Before analyzing for scripting, determine: is this segment an interviewer prompt, not a candidate answer?

Mark is_interviewer_speech=true if the text is clearly soliciting information:
- "Could you walk me through...", "Tell me about...", "How would you approach...", "Describe a time when..."
- No substantive answer content; question/prompt form only

If is_interviewer_speech=true, set verdict=CLEAR, likelihood=0.05, confidence=HIGH. Stop.

---

## STEP 2 — ASSESS FOR SCRIPTED DELIVERY

Think step by step internally. Output only the final JSON.

**Strong signals of scripted/memorized content (raise likelihood):**
- Textbook or encyclopedia-style definitions: "X is a technique that enables Y by doing Z"
- Documentation prose: reads like it was copied from official docs, a blog post, or study notes
- Generic buzzword sequences (microservices, distributed systems, CI/CD, scalability) with no situational anchoring — no specific project, constraint, failure, or outcome mentioned
- Perfectly structured prose: clear intro → body → conclusion rhythm with no messiness, hedging, or pivoting
- Fully explained technical concepts at tutorial depth with zero personalization
- Memorized lists of best practices, steps, or tradeoffs without any "in our case" or "when this broke" context

**Strong signals of spontaneous delivery (lower likelihood — require substance):**
- Concrete situational detail: a specific system that broke, a metric that changed, a deadline, a teammate, a particular config choice and why
- Non-generic project narrative: constraints, tradeoffs made under real conditions, outcomes (positive or negative)
- Answers that reference partial knowledge or frame uncertainty: "I haven't used X in this exact way, but..."

**Signals that are UNRELIABLE ALONE — do not use as evidence of spontaneity:**
- First-person pronouns (I, my, we): Scripted interview prep routinely uses first person. "I use Redis for caching" can be memorized. First person is NOT evidence of naturalness.
- Fluent grammar or eloquent phrasing: a memorized answer can be perfectly grammatical
- Confidence or tone
- Naming a technology without project-specific context

---

## STEP 3 — CALIBRATE VERDICT AND CONFIDENCE

Use the background signals (acoustic_score, linguistic_score, generic_script_likelihood, etc.) to inform your overall judgment, but NEVER quote field names, numeric values, or math in the reasons field.

Conservative calibration rules:
- Only output PROBABLE_SCRIPT_READING if you have genuine multi-signal evidence of scripted delivery. When in doubt, output AMBIGUOUS.
- Only output CLEAR if the answer has clear situational specificity and no strong scripting patterns.
- LOW confidence = mixed or weak evidence; prefer AMBIGUOUS
- HIGH confidence = you have strong, multiple independent indicators pointing the same direction
- Do not output likelihood > 0.85 or HIGH confidence on a single weak cue

When evidence is mixed or thin → verdict=AMBIGUOUS, confidence=LOW, likelihood in [0.35–0.65].

---

## STEP 4 — WRITE REASONS

Write exactly 2 or 3 reasons. Each must:
- Be plain English (no metric names, no scores, no field references)
- Be ~15 words or fewer
- Read like a brief analyst observation
- Describe the text's delivery style, specificity, or structure — not correctness or eloquence

Good examples:
- "Definition-heavy explanation with no project constraints, failures, or outcomes mentioned"
- "Generic best-practice list without any situational anchoring to a real system"
- "Names multiple tools but gives no tradeoff, failure, or team context"
- "Specific incident cited with concrete metrics and a resolution path"
- "Answer pivots mid-explanation, suggesting real-time reasoning rather than recitation"

Bad examples (do not write these):
- "Uses first-person pronouns indicating personal experience" ← FORBIDDEN
- "Answer is fluent and well-structured" ← not evidence of anything
- "generic_script_likelihood is high" ← never quote field names
- "Likelihood 0.81 suggests scripting" ← never quote numbers
- "Candidate appears confident" ← irrelevant

---

## CRITICAL REMINDERS

1. First-person (I/my/we) is NOT evidence of spontaneity. Scripted answers use first person constantly. Never cite it as a reason to lower likelihood.
2. A correct, polished, thorough answer is not necessarily spontaneous. Judge delivery style and specificity, not quality.
3. You are an integrity signal, not a hiring decision. Prefer staying AMBIGUOUS over aggressive false positives.
4. Reasons are read by end users. Write them as analyst notes, not model introspection.
5. Output ONLY the JSON object. Nothing before it, nothing after it.

"""

_MAX_PUBLIC_REASONS = 3

# Reasons that wrongly treat first-person phrasing as proof of natural speech.
_DISALLOWED_REASON_PATTERNS = (
    re.compile(r"first[\s-]?person", re.I),
    re.compile(r"personal (narrative|experience|story)", re.I),
    re.compile(r"\bI use\b", re.I),
    re.compile(r"indicating personal", re.I),
    re.compile(r"uses? (the )?(word|pronoun) ['\"]?(I|my|we)", re.I),
    re.compile(r"\b(likelihood|confidence|score|acoustic|linguistic|ewma|evidence)\b", re.I),
)


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))


def _confidence_meets_minimum(confidence: str, minimum: str) -> bool:
    return _CONFIDENCE_RANK.get(confidence, 0) >= _CONFIDENCE_RANK.get(minimum, 1)


def build_signal_summary(answer: dict[str, Any]) -> str:
    """Compact read-only signal bundle for the LLM (no raw audio)."""
    signals = answer.get("signals") or {}
    spec = answer.get("semantic_specificity") or {}
    contrastive = answer.get("contrastive") or {}
    intra = answer.get("intra_individual") or {}
    lines = [
        f"answer_id: {answer.get('answer_id')}",
        f"duration_sec: {float(answer.get('end_sec', 0)) - float(answer.get('start_sec', 0)):.1f}",
        f"current_status: {answer.get('status', 'AMBIGUOUS')}",
        f"confidence: {answer.get('confidence', 'LOW')}",
        f"acoustic_score: {float(signals.get('acoustic', 0) or 0):.3f}",
        f"linguistic_score: {float(signals.get('linguistic', 0) or 0):.3f}",
        f"generic_script_likelihood: {float(spec.get('generic_script_likelihood', 0) or 0):.3f}",
        (
            "memorized_technical_script_score: "
            f"{float(spec.get('memorized_technical_script_score', 0) or 0):.3f}"
        ),
        f"p_external_guidance: {float(intra.get('p_external_guidance', 0) or 0):.3f}",
        f"contrastive_status: {contrastive.get('status', '')}",
        f"weighted_evidence: {float(contrastive.get('weighted_evidence', 0) or 0):.3f}",
    ]
    return "\n".join(lines)


def build_user_message(answer: dict[str, Any]) -> str:
    transcript = str(answer.get("transcript", "")).strip()
    return (
        "TRANSCRIPT:\n"
        f"{transcript}\n\n"
        "BACKGROUND SIGNALS (for your internal reasoning only — never quote these numbers in reasons):\n"
        f"{build_signal_summary(answer)}"
    )


def sanitize_public_reasons(reasons: list[str], *, max_reasons: int = _MAX_PUBLIC_REASONS) -> list[str]:
    """Drop weak or overly technical reasons; keep 2-3 plain-language bullets."""
    cleaned: list[str] = []
    for raw in reasons:
        text = re.sub(r"\b\d+\.\d+\b", "", str(raw))
        text = " ".join(text.split()).strip(" -•")
        if len(text) < 8:
            continue
        if any(p.search(text) for p in _DISALLOWED_REASON_PATTERNS):
            continue
        if text not in cleaned:
            cleaned.append(text)
        if len(cleaned) >= max_reasons:
            break
    if len(cleaned) >= 2:
        return cleaned[:max_reasons]
    if cleaned:
        return cleaned
    return ["Mixed cues — no single dominant pattern in the transcript"]


def normalize_llm_response(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate and coerce LLM JSON into expected shape."""
    likelihood = float(raw.get("script_reading_likelihood", 0.5))
    likelihood = max(0.0, min(1.0, likelihood))

    confidence = str(raw.get("confidence", "LOW")).upper()
    if confidence not in VALID_CONFIDENCE:
        confidence = "LOW"

    verdict = str(raw.get("verdict", "AMBIGUOUS")).upper()
    if verdict == "PROBABLE":
        verdict = "PROBABLE_SCRIPT_READING"
    if verdict not in VALID_VERDICTS:
        verdict = "AMBIGUOUS"

    is_interviewer = bool(raw.get("is_interviewer_speech", False))

    reasons_raw = raw.get("reasons") or []
    if isinstance(reasons_raw, str):
        reasons_raw = [reasons_raw]
    reasons = sanitize_public_reasons(
        [str(r).strip() for r in reasons_raw if str(r).strip()]
    )

    return {
        "script_reading_likelihood": round(likelihood, 4),
        "confidence": confidence,
        "verdict": verdict,
        "is_interviewer_speech": is_interviewer,
        "reasons": reasons,
    }


def format_public_explanation(status_after: str, reasons: list[str]) -> list[str]:
    """Verdict line plus 2-3 compact reasons for UI and decision_explanation."""
    label = status_after
    if status_after == "PROBABLE_SCRIPT_READING":
        label = "PROBABLE script reading"
    elif status_after == "AMBIGUOUS":
        label = "AMBIGUOUS"
    elif status_after == "CLEAR":
        label = "CLEAR"
    return [f"LLM judge: {label}", *sanitize_public_reasons(reasons)]


def map_llm_verdict_to_status(
    llm: dict[str, Any],
    *,
    status_before: str = "AMBIGUOUS",
    promote_min: float | None = None,
    clear_max: float | None = None,
    min_confidence: str | None = None,
) -> tuple[str, list[str]]:
    """
    Conservative tie-breaker mapping. Only promotes/demotes from AMBIGUOUS.
    Returns (status_after, explanation_lines).
    """
    if status_before != "AMBIGUOUS":
        return status_before, []

    promote_min = promote_min if promote_min is not None else config.LLM_JUDGE_PROMOTE_MIN
    clear_max = clear_max if clear_max is not None else config.LLM_JUDGE_CLEAR_MAX
    min_confidence = min_confidence or config.LLM_JUDGE_MIN_CONFIDENCE

    likelihood = float(llm.get("script_reading_likelihood", 0.5))
    confidence = str(llm.get("confidence", "LOW")).upper()
    verdict = str(llm.get("verdict", "AMBIGUOUS")).upper()
    reasons = list(llm.get("reasons") or [])

    if llm.get("is_interviewer_speech"):
        return "CLEAR", format_public_explanation(
            "CLEAR",
            ["Segment reads like an interviewer question, not a candidate answer", *reasons],
        )

    if (
        verdict == "PROBABLE_SCRIPT_READING"
        and _confidence_meets_minimum(confidence, min_confidence)
        and likelihood >= promote_min
    ):
        return "PROBABLE_SCRIPT_READING", format_public_explanation(
            "PROBABLE_SCRIPT_READING", reasons
        )

    if (
        verdict == "CLEAR"
        and confidence == "HIGH"
        and likelihood <= clear_max
    ):
        return "CLEAR", format_public_explanation("CLEAR", reasons)

    return "AMBIGUOUS", format_public_explanation("AMBIGUOUS", reasons)


def _append_decision_explanation(answer: dict[str, Any], lines: list[str]) -> None:
    contrastive = answer.get("contrastive")
    if not isinstance(contrastive, dict):
        contrastive = {}
        answer["contrastive"] = contrastive
    existing = contrastive.get("decision_explanation") or []
    if not isinstance(existing, list):
        existing = [str(existing)]
    for line in lines:
        if line and line not in existing:
            existing.append(line)
    contrastive["decision_explanation"] = existing


def apply_llm_judge_to_answers(
    answers: list[dict[str, Any]],
    *,
    log: Any | None = None,
    provider: Any | None = None,
) -> list[dict[str, Any]]:
    """
    Run LLM judge on AMBIGUOUS answers only. Fail open on errors.
    Mutates answers in place and returns the same list.
    """
    if not config.ENABLE_LLM_JUDGE:
        return answers

    if provider is None:
        try:
            from engine.llm_providers.factory import get_llm_provider

            provider = get_llm_provider()
        except Exception as exc:
            logger.warning("LLM judge disabled: %s", exc)
            if log is not None:
                log.log(
                    "llm_judge",
                    f"LLM judge skipped: {exc}",
                    phase="analyze",
                    level="warning",
                )
            return answers

    ambiguous = [
        a for a in answers if str(a.get("status", "")).upper() == "AMBIGUOUS"
    ]
    max_answers = int(config.LLM_JUDGE_MAX_ANSWERS or 0)
    if max_answers > 0:
        ambiguous = ambiguous[:max_answers]

    for answer in ambiguous:
        aid = answer.get("answer_id", "?")
        transcript = str(answer.get("transcript", "")).strip()
        min_words = int(config.LLM_JUDGE_MIN_WORDS)
        if not transcript or _word_count(transcript) < min_words:
            answer["llm_judge"] = {
                "ran": False,
                "skipped": True,
                "reason": f"transcript too short (< {min_words} words)",
            }
            continue

        status_before = str(answer.get("status", "AMBIGUOUS"))
        user_msg = build_user_message(answer)
        prompt_hash = hashlib.sha256(
            (JUDGE_SYSTEM_PROMPT + user_msg).encode("utf-8")
        ).hexdigest()[:16]

        try:
            raw = provider.complete_json(system=JUDGE_SYSTEM_PROMPT, user=user_msg)
            llm = normalize_llm_response(raw)
        except Exception as exc:
            logger.warning("LLM judge failed for answer %s: %s", aid, exc)
            answer["llm_judge"] = {
                "ran": False,
                "error": str(exc),
                "status_before": status_before,
                "status_after": status_before,
                "prompt_hash": prompt_hash,
            }
            if log is not None:
                log.log(
                    "llm_judge",
                    f"Answer {aid}: LLM judge failed — kept {status_before}",
                    phase="analyze",
                    level="warning",
                    metrics={"answer_id": aid, "error": str(exc)},
                    decision=status_before,
                )
            continue

        status_after, explain_lines = map_llm_verdict_to_status(
            llm, status_before=status_before
        )
        if status_after != status_before:
            answer["status"] = status_after
            contrastive = answer.get("contrastive")
            if isinstance(contrastive, dict):
                contrastive["status"] = status_after

        _append_decision_explanation(answer, explain_lines)

        public_reasons = sanitize_public_reasons(llm["reasons"])
        judge_block = {
            "ran": True,
            "verdict": status_after,
            "reasons": public_reasons,
        }
        answer["llm_judge"] = judge_block

        if log is not None:
            reason_summary = "; ".join(public_reasons[:3]) if public_reasons else "no reasons"
            log.log(
                "llm_judge",
                f"Answer {aid}: {status_after} — {reason_summary}",
                phase="analyze",
                metrics={
                    "answer_id": aid,
                    "verdict": status_after,
                    "reasons": public_reasons,
                    "status_before": status_before,
                    "script_reading_likelihood": llm["script_reading_likelihood"],
                    "confidence": llm["confidence"],
                    "is_interviewer_speech": llm["is_interviewer_speech"],
                    "prompt_hash": prompt_hash,
                },
                decision=status_after,
            )

    return answers
