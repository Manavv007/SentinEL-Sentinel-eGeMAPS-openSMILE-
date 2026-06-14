"""Unit tests for scoring_v3 fixes (spontaneity modulation, composite blend, dominance)."""

from __future__ import annotations

import config
from engine.scoring_v3 import (
    compute_answer_composite_score,
    modulated_suspicion_score,
    spontaneity_modulation_factor,
    weighted_percentile,
    window_temporal_weight,
)


def test_spontaneity_modulation_uses_computed_reduction():
    high_nat = spontaneity_modulation_factor(
        naturality_score=0.9,
        script_similarity=0.55,
        profile_confidence=0.8,
        fake_natularity=0.0,
    )
    low_nat = spontaneity_modulation_factor(
        naturality_score=0.1,
        script_similarity=0.55,
        profile_confidence=0.8,
        fake_natularity=0.0,
    )
    assert high_nat < low_nat
    assert low_nat > config.SPONTANEITY_MODULATION_FLOOR
    assert high_nat <= 1.0


def test_spontaneity_not_always_max_discount_for_strong_script():
    """Broken code returned the same floor for all naturality at strong script tier."""
    strong_script = config.STRONG_SCRIPT_THRESHOLD + 0.02
    natural = spontaneity_modulation_factor(
        naturality_score=0.85,
        script_similarity=strong_script,
        profile_confidence=0.7,
        fake_natularity=0.0,
    )
    robotic = spontaneity_modulation_factor(
        naturality_score=0.05,
        script_similarity=strong_script,
        profile_confidence=0.7,
        fake_natularity=0.0,
    )
    assert natural < robotic


def test_window_temporal_weight_respects_strong_threshold_config(monkeypatch):
    monkeypatch.setattr(config, "STRONG_SCRIPT_THRESHOLD", 0.70)
    w_at = window_temporal_weight(0.70, 0.0)
    w_above = window_temporal_weight(0.86, 0.0)
    assert w_above > w_at


def test_weighted_percentile_uses_weights_not_multipliers():
  values = [0.2, 0.5, 0.9]
  weights = [1.0, 1.0, 1.0]
  p90 = weighted_percentile(values, weights, 90.0)
  assert 0.5 <= p90 <= 0.9
  assert p90 < 0.9 * 2.2  # not score*weight inflation


def test_composite_blend_allows_benign_ewma_to_dominate_single_spike():
    windows = [
        {"script_similarity": 0.40, "contrastive_score": 0.05, "suspicious_flag": False},
        {"script_similarity": 0.42, "contrastive_score": 0.06, "suspicious_flag": False},
        {"script_similarity": 0.41, "contrastive_score": 0.04, "suspicious_flag": False},
        {"script_similarity": 0.90, "contrastive_score": 0.55, "suspicious_flag": True},
    ]
    ewma = 0.08
    peak_ewma = 0.55
    composite, meta = compute_answer_composite_score(
        windows, ewma=ewma, peak_ewma=peak_ewma, margin=config.CONTRASTIVE_MARGIN
    )
    assert composite < peak_ewma * config.PEAK_EWMA_BLEND
    assert meta["peak_credibility_alpha"] < 1.0


def test_dominance_boost_gated_on_profile_confidence():
    cold = modulated_suspicion_score(
        script_similarity=0.72,
        natural_similarity=0.05,
        naturality_score=0.4,
        profile_confidence=0.1,
        suppression=0.0,
        technical_density=0.0,
        cognitive_spontaneity=0.2,
    )
    mature = modulated_suspicion_score(
        script_similarity=0.72,
        natural_similarity=0.05,
        naturality_score=0.4,
        profile_confidence=0.8,
        suppression=0.0,
        technical_density=0.0,
        cognitive_spontaneity=0.2,
    )
    assert mature > cold
