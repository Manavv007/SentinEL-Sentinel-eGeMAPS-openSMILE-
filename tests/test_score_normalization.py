"""
Tests for the candidate-invariant decision layer (engine/score_normalization.py).

These prove the property the project actually needs: the SAME relative structure
yields the SAME verdicts regardless of each candidate's absolute score scale/offset —
which is exactly what a global absolute threshold fails to do.
"""

from __future__ import annotations

from engine.score_normalization import (
    AMBIGUOUS,
    CLEAR,
    PROBABLE,
    RelativeDecision,
    RelativeDecisionConfig,
    apply_relative_decision_to_answers,
    elevation_z,
    reconcile,
    relative_decision,
    robust_baseline,
)

CFG = RelativeDecisionConfig()


def _statuses(scores):
    return [d.rel_status for d in relative_decision(scores, CFG)]


def test_offset_invariance_holds_under_default_guards():
    """A pure level shift (e.g. different mic gain) must not change verdicts,
    as long as scores stay within the absolute guard band."""
    base = [0.46, 0.48, 0.50, 0.51, 0.52, 0.62, 0.65, 0.68]
    shifted = [s - 0.16 for s in base]  # quieter mic; scripted 0.62->0.46 still >= floor
    assert _statuses(base) == _statuses(shifted)
    verdicts = _statuses(base)
    assert verdicts[-3:] == [PROBABLE, PROBABLE, PROBABLE]
    assert verdicts[0] == CLEAR


def test_scale_and_offset_invariance_in_elevation():
    """The elevation logic itself is scale+offset invariant. With the absolute guards
    relaxed, a shifted+compressed copy yields identical verdicts (the guards exist only
    to clamp pathological absolute levels, see test_soft_abs_floor_blocks_low_score_flag)."""
    relaxed = RelativeDecisionConfig(soft_abs_floor=0.0, soft_abs_ceiling=0.0)
    base = [0.46, 0.48, 0.50, 0.51, 0.52, 0.62, 0.65, 0.68]
    center = 0.5
    compressed = [center + (s - center) * 0.8 for s in base]      # narrower dynamic range
    shifted = [s - 0.2 for s in base]                              # lower level

    sb = [d.rel_status for d in relative_decision(base, relaxed)]
    sc = [d.rel_status for d in relative_decision(compressed, relaxed)]
    ss = [d.rel_status for d in relative_decision(shifted, relaxed)]
    assert sb == sc == ss


def test_absolute_threshold_would_fail_where_relative_succeeds():
    """The low-scoring candidate: absolute 0.55 misses every scripted answer."""
    # Quiet-mic candidate: scripted answers sit at 0.48-0.52, below the global 0.55.
    scores = [0.30, 0.32, 0.34, 0.36, 0.38, 0.48, 0.50, 0.52]
    abs_pred = [PROBABLE if s >= 0.55 else CLEAR for s in scores]
    rel_pred = _statuses(scores)

    # Absolute calls everything CLEAR -> misses all 3 scripted answers.
    assert abs_pred.count(PROBABLE) == 0
    # Relative flags the elevated cluster.
    assert rel_pred[-3:] == [PROBABLE, PROBABLE, PROBABLE]
    assert rel_pred[:5] == [CLEAR] * 5


def test_high_baseline_candidate_no_false_positives():
    """Anxious/high-baseline candidate: absolute 0.55 flags honest answers; relative does not."""
    scores = [0.57, 0.58, 0.59, 0.60, 0.61, 0.73, 0.76, 0.78]
    abs_pred = [PROBABLE if s >= 0.55 else CLEAR for s in scores]
    rel_pred = _statuses(scores)

    # Absolute flags all 5 honest answers as PROBABLE (false positives).
    assert abs_pred[:5] == [PROBABLE] * 5
    # Relative keeps the honest cluster CLEAR and flags only the elevated cluster.
    assert rel_pred[:5] == [CLEAR] * 5
    assert rel_pred[-3:] == [PROBABLE, PROBABLE, PROBABLE]


def test_degenerate_session_preserves_uncertainty():
    """No real spread => no fabricated verdicts; everything AMBIGUOUS/LOW."""
    flat = [0.62, 0.61, 0.62, 0.63, 0.62, 0.62, 0.61, 0.62]  # spread 0.02 < min_spread
    decisions = relative_decision(flat, CFG)
    assert all(d.rel_status == AMBIGUOUS for d in decisions)
    assert all(d.confidence == "LOW" for d in decisions)


def test_too_few_answers_is_degenerate():
    decisions = relative_decision([0.3, 0.7], CFG)  # n < min_answers
    assert all(d.rel_status == AMBIGUOUS for d in decisions)


def test_soft_abs_floor_blocks_low_score_flag():
    """A relatively-elevated but absolutely-low answer is not flagged PROBABLE."""
    # Honest calm candidate; top answer is elevated vs baseline but well below floor.
    scores = [0.10, 0.12, 0.14, 0.16, 0.18, 0.34]
    decisions = relative_decision(scores, CFG)
    # The 0.34 answer is below soft_abs_floor (0.45) -> never PROBABLE.
    assert decisions[-1].rel_status != PROBABLE


def test_robust_baseline_uses_low_cluster_anchor():
    """Baseline anchors near the low (spontaneous) cluster, not the contaminated mean.

    Holds when the spontaneous answers are at least ~the baseline quantile fraction of
    the session (the supported regime; a candidate who scripts almost everything is a
    known hard case handled by the absolute guards + degenerate checks)."""
    # 4 spontaneous + 2 scripted: mean ~0.457, but the low-quantile anchor stays low.
    scores = [0.30, 0.32, 0.34, 0.36, 0.70, 0.72]
    bl = robust_baseline(scores, CFG)
    assert bl.center < 0.45          # below the gap, near the spontaneous cluster
    assert not bl.degenerate


def test_reconcile_demotes_false_positive_to_ambiguous():
    rel = RelativeDecision(0, 0.5, 0.1, CLEAR, "LOW", "near baseline")
    new_status, reason = reconcile(PROBABLE, rel, personal_protected=False)
    assert new_status == AMBIGUOUS
    assert reason is not None


def test_reconcile_personal_protected_demotes_to_clear():
    rel = RelativeDecision(0, 0.5, 0.1, CLEAR, "LOW", "near baseline")
    new_status, reason = reconcile(PROBABLE, rel, personal_protected=True)
    assert new_status == CLEAR


def test_reconcile_promotes_elevated_when_not_protected():
    rel = RelativeDecision(0, 0.7, 2.0, PROBABLE, "HIGH", "elevated")
    new_status, _ = reconcile(CLEAR, rel, personal_protected=False)
    assert new_status == PROBABLE


def test_reconcile_protects_personal_narrative_from_promotion():
    rel = RelativeDecision(0, 0.7, 2.0, PROBABLE, "HIGH", "elevated")
    # Personal-protected CLEAR is not promoted to PROBABLE.
    new_status, _ = reconcile(CLEAR, rel, personal_protected=True)
    assert new_status == CLEAR


def test_apply_to_answers_demotes_canva_style_false_positive():
    """
    Mirrors the real results.json failure: a personal workflow answer scored in the
    same band as the rest but flagged PROBABLE by the absolute threshold. With real
    per-candidate structure it should be demoted out of PROBABLE.
    """
    answers = [
        {"raw_score": 0.30, "status": CLEAR},
        {"raw_score": 0.32, "status": CLEAR},
        {"raw_score": 0.34, "status": CLEAR},
        {"raw_score": 0.36, "status": CLEAR},
        # Personal workflow answer, NOT elevated for this candidate, but previously PROBABLE.
        {
            "raw_score": 0.37,
            "status": PROBABLE,
            "semantic_specificity": {
                "specificity_score": 0.6,
                "generic_script_likelihood": 0.2,
                "memorized_technical_script_score": 0.1,
                "personal_narrative_score": 0.7,
            },
        },
        {"raw_score": 0.62, "status": PROBABLE},  # genuinely elevated
        {"raw_score": 0.65, "status": PROBABLE},
    ]
    summary = apply_relative_decision_to_answers(answers, CFG)

    assert summary["enabled"] is True
    assert summary["degenerate_session"] is False
    # The personal answer is no longer a confident PROBABLE.
    assert answers[4]["status"] in (CLEAR, AMBIGUOUS)
    # The genuinely elevated answers remain PROBABLE.
    assert answers[5]["status"] == PROBABLE
    assert answers[6]["status"] == PROBABLE
    # Every answer carries the relative-decision evidence for explainability.
    assert all("relative_decision" in a for a in answers)


def test_apply_to_answers_noop_on_degenerate_session():
    """The real demo session (all ~0.5-0.64, no separation) must not be re-decided."""
    answers = [
        {"raw_score": 0.503, "status": CLEAR},
        {"raw_score": 0.500, "status": CLEAR},
        {"raw_score": 0.554, "status": PROBABLE},
        {"raw_score": 0.629, "status": PROBABLE},
        {"raw_score": 0.624, "status": PROBABLE},
    ]
    before = [a["status"] for a in answers]
    summary = apply_relative_decision_to_answers(answers, RelativeDecisionConfig(min_session_spread=0.2))
    after = [a["status"] for a in answers]
    assert summary["degenerate_session"] is True
    assert summary["applied"] == 0
    assert before == after  # nothing changed


def test_elevation_z_sign():
    bl = robust_baseline([0.2, 0.3, 0.4, 0.5, 0.6], CFG)
    assert elevation_z(0.6, bl) > 0
    assert elevation_z(0.2, bl) < 0


def test_real_harsh_interview_ground_truth():
    """
    Regression for the real ML-candidate ("Harsh") interview.

    Ground truth: [CLEAR, CLEAR, PROBABLE, PROBABLE]
      a0 genuine self-introduction (acoustically suspicious but at the candidate's
         own baseline, no scripted-content markers) -> must be CLEAR
      a1 genuine project answer (personal narrative)                 -> CLEAR
      a2 scripted essay ("technology has changed the way we...")      -> PROBABLE
      a3 scripted essay ("spending too much time on screens...") sits BELOW the
         contaminated baseline, but the semantic-scripted guard keeps it PROBABLE.
    """
    answers = [
        {
            "raw_score": 0.433,            # z=-0.06, at baseline -> rel CLEAR (MEDIUM)
            "status": PROBABLE,            # pushed up by acoustic + session uniformity
            "semantic_specificity": {
                "specificity_score": 0.30,
                "generic_script_likelihood": 0.45,   # below 0.58 -> not scripted
                "memorized_technical_script_score": 0.10,
                "personal_narrative_score": 0.20,    # intro verbs don't match narrow regex
            },
        },
        {
            "raw_score": 0.314,
            "status": CLEAR,
            "semantic_specificity": {
                "specificity_score": 0.76,
                "generic_script_likelihood": 0.20,
                "memorized_technical_script_score": 0.10,
                "personal_narrative_score": 1.00,
            },
        },
        {
            "raw_score": 0.483,
            "status": PROBABLE,
            "semantic_specificity": {
                "specificity_score": 0.10,
                "generic_script_likelihood": 0.85,   # generic essay -> scripted
                "memorized_technical_script_score": 0.10,
                "personal_narrative_score": 0.00,
            },
        },
        {
            "raw_score": 0.431,            # below baseline -> rel CLEAR, but guarded
            "status": PROBABLE,
            "semantic_specificity": {
                "specificity_score": 0.10,
                "generic_script_likelihood": 0.85,   # generic essay -> scripted
                "memorized_technical_script_score": 0.10,
                "personal_narrative_score": 0.00,
            },
        },
    ]
    apply_relative_decision_to_answers(answers, CFG)
    statuses = [a["status"] for a in answers]
    assert statuses == [CLEAR, CLEAR, PROBABLE, PROBABLE], statuses


def test_semantic_scripted_guard_blocks_clearing_below_baseline_script():
    """A scripted answer scoring below the candidate baseline must NOT be cleared."""
    rel = RelativeDecision(0, 0.43, -0.3, CLEAR, "MEDIUM", "below baseline")
    # Without the guard this confident-CLEAR would clear it; the guard keeps it.
    kept, reason = reconcile(PROBABLE, rel, personal_protected=False, semantically_scripted=True)
    assert kept == PROBABLE
    assert reason is None


def test_confident_not_elevated_clears_without_scripted_markers():
    """At/below baseline + no scripted markers -> CLEAR (fixes genuine-intro AMBIGUOUS)."""
    rel = RelativeDecision(0, 0.43, -0.06, CLEAR, "MEDIUM", "at baseline")
    cleared, reason = reconcile(PROBABLE, rel, personal_protected=False, semantically_scripted=False)
    assert cleared == CLEAR
    assert reason is not None
