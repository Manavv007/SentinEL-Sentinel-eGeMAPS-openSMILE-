"""
Tests for the cross-candidate evaluation harness (eval/harness.py).

The harness is the agentic-eval evaluator that makes "works on every candidate"
measurable. These tests assert that, on a dataset where candidates share relative
structure but differ in absolute scale, the candidate-relative decision strategy
beats the legacy absolute threshold on the generalization metrics.
"""

from __future__ import annotations

from eval.harness import (
    absolute_threshold_strategy,
    evaluate_strategy,
    leave_one_candidate_out,
    relative_decision_strategy,
    run_comparison,
)
from eval.synthetic_data import build_synthetic_candidates


def test_relative_lower_cross_candidate_variance():
    candidates = build_synthetic_candidates()
    result = run_comparison(candidates)
    abs_var = result["absolute_threshold"]["eval"]["cross_candidate_accuracy_variance"]
    rel_var = result["relative_decision"]["eval"]["cross_candidate_accuracy_variance"]
    # Lower cross-candidate variance == generalizes more uniformly.
    assert rel_var < abs_var


def test_relative_higher_min_accuracy():
    candidates = build_synthetic_candidates()
    result = run_comparison(candidates)
    abs_min = result["absolute_threshold"]["eval"]["min_accuracy"]
    rel_min = result["relative_decision"]["eval"]["min_accuracy"]
    # Worst-candidate accuracy is the honest "does it work for everyone" number.
    assert rel_min > abs_min
    assert rel_min >= 0.95


def test_relative_lower_false_positive_rate():
    candidates = build_synthetic_candidates()
    result = run_comparison(candidates)
    abs_fpr = result["absolute_threshold"]["eval"]["mean_false_positive_rate"]
    rel_fpr = result["relative_decision"]["eval"]["mean_false_positive_rate"]
    assert rel_fpr <= abs_fpr


def test_relative_higher_rubric_score():
    candidates = build_synthetic_candidates()
    result = run_comparison(candidates)
    assert (
        result["relative_decision"]["rubric_score"]
        > result["absolute_threshold"]["rubric_score"]
    )


def test_leave_one_candidate_out_relative_generalizes():
    candidates = build_synthetic_candidates()
    abs_loco = leave_one_candidate_out(
        candidates, lambda _t: absolute_threshold_strategy(0.55)
    )
    rel_loco = leave_one_candidate_out(
        candidates, lambda _t: relative_decision_strategy()
    )
    assert rel_loco["loco_mean_accuracy"] > abs_loco["loco_mean_accuracy"]
    assert rel_loco["loco_accuracy_variance"] <= abs_loco["loco_accuracy_variance"]


def test_absolute_overfits_to_one_candidate():
    """The absolute threshold is perfect on 'alpha' (the tuned demo) but not others."""
    candidates = build_synthetic_candidates()
    abs_eval = evaluate_strategy(candidates, absolute_threshold_strategy(0.55))
    by_id = {m["candidate_id"]: m["accuracy"] for m in abs_eval["per_candidate"]}
    assert by_id["alpha"] == 1.0          # the candidate it was tuned on
    assert by_id["beta"] < 1.0 or by_id["gamma"] < 1.0  # fails on at least one other
