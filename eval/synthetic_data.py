"""
Synthetic multi-candidate dataset for the cross-candidate evaluation harness.

Each candidate has the SAME relative structure — a cluster of spontaneous (CLEAR)
answers plus a few scripted (PROBABLE) answers that are clearly elevated *relative to
that candidate's own baseline* — but the candidates live at DIFFERENT absolute score
levels (different mic / accent / nervousness). This is the exact situation that
breaks a global absolute threshold and that a candidate-relative decision survives.

    alpha : the demo the 0.55 threshold was tuned on   (clear ~0.5, scripted ~0.65)
    beta  : quieter mic / lower scores                 (clear ~0.34, scripted ~0.50)
    gamma : anxious / high baseline                     (clear ~0.59, scripted ~0.76)
    delta : fully honest, no scripted answers           (all low, all CLEAR)

A fixed 0.55 line can be "right" for at most one of alpha/beta/gamma:
  * beta's scripted answers fall below 0.55 -> all missed (false negatives)
  * gamma's honest answers sit above 0.55  -> all flagged (false positives)
The candidate-relative layer classifies all four correctly because it measures
elevation above each candidate's own baseline.
"""

from __future__ import annotations

from engine.score_normalization import CLEAR, PROBABLE
from eval.harness import Candidate


def _answers(scores: list[float], labels: list[str]) -> list[dict]:
    return [{"score": s, "label": l, "personal_protected": False} for s, l in zip(scores, labels)]


def build_synthetic_candidates() -> list[Candidate]:
    """Deterministic dataset (no RNG) so harness output is reproducible."""
    specs = {
        # candidate_id: (clear_scores, scripted_scores)
        "alpha": ([0.46, 0.48, 0.50, 0.51, 0.52], [0.62, 0.65, 0.68]),
        "beta":  ([0.30, 0.32, 0.34, 0.36, 0.38], [0.48, 0.50, 0.52]),
        "gamma": ([0.57, 0.58, 0.59, 0.60, 0.61], [0.73, 0.76, 0.78]),
        "delta": ([0.26, 0.29, 0.31, 0.33, 0.35, 0.37, 0.39], []),
    }
    candidates: list[Candidate] = []
    for cid, (clear, scripted) in specs.items():
        scores = list(clear) + list(scripted)
        labels = [CLEAR] * len(clear) + [PROBABLE] * len(scripted)
        ans = _answers(scores, labels)
        candidates.append(
            Candidate(
                candidate_id=cid,
                scores=[a["score"] for a in ans],
                labels=[a["label"] for a in ans],
                personal_protected=[a["personal_protected"] for a in ans],
            )
        )
    return candidates
