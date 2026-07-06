"""
Cross-candidate evaluation harness (agentic-eval skill).

WHY THIS EXISTS
---------------
SentinEL's generalization failure was *caused* by tuning ~200 global constants
against a single demo interview with no held-out candidates. You cannot build "a
model that works on every candidate" without a metric that measures exactly that.

This harness implements the agentic-eval evaluator pattern with the one change that
matters: it scores a decision strategy with **leave-one-candidate-out** folds and a
weighted rubric whose headline metric is *cross-candidate variance of accuracy* —
the direct measure of "does it work on everyone, not just the demo".

It is dependency-light (stdlib + NumPy) and operates on already-produced per-answer
suspicion scores + ground-truth labels, so it runs without the audio/ML stack:

    {
      "candidate_id": "c1",
      "answers": [{"score": 0.62, "label": "PROBABLE_SCRIPT_READING",
                   "personal_protected": false}, ...]
    }

Decision strategies compared:
  * ``absolute_threshold`` — the legacy verdict (score >= ALERT_THRESHOLD).
  * ``relative_decision``  — the candidate-invariant layer in
    ``engine.score_normalization``.

Run:
    python -m eval.harness                 # uses the built-in synthetic dataset
    python -m eval.harness <data_dir>      # one *.json per candidate (schema above)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from engine.score_normalization import (
    PROBABLE,
    CLEAR,
    AMBIGUOUS,
    RelativeDecisionConfig,
    relative_decision,
)

# Weighted rubric (agentic-eval rubric-based pattern). cross_candidate_variance is the
# generalization metric: low variance across candidates == "works on everyone".
RUBRIC: dict[str, dict[str, float]] = {
    "per_answer_accuracy": {"weight": 0.30},
    "false_positive_rate": {"weight": 0.25},   # honest answers flagged — penalize hard
    "cross_candidate_variance": {"weight": 0.20},
    "class_separation": {"weight": 0.15},
    "calibration_robustness": {"weight": 0.10},
}


@dataclass
class Candidate:
    candidate_id: str
    scores: list[float]
    labels: list[str]            # ground-truth: CLEAR / AMBIGUOUS / PROBABLE_SCRIPT_READING
    personal_protected: list[bool]


# --------------------------------------------------------------------------- #
# Dataset loading
# --------------------------------------------------------------------------- #

def load_candidates(data_dir: str | Path) -> list[Candidate]:
    data_dir = Path(data_dir)
    out: list[Candidate] = []
    for path in sorted(data_dir.glob("*.json")):
        obj = json.loads(path.read_text(encoding="utf-8"))
        answers = obj.get("answers", [])
        out.append(
            Candidate(
                candidate_id=str(obj.get("candidate_id", path.stem)),
                scores=[float(a["score"]) for a in answers],
                labels=[str(a["label"]) for a in answers],
                personal_protected=[bool(a.get("personal_protected", False)) for a in answers],
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Decision strategies  (scores -> predicted statuses)
# --------------------------------------------------------------------------- #

def absolute_threshold_strategy(threshold: float = 0.55) -> Callable[[Candidate], list[str]]:
    """Legacy verdict: predict PROBABLE iff score >= global threshold, else CLEAR."""

    def predict(c: Candidate) -> list[str]:
        return [PROBABLE if s >= threshold else CLEAR for s in c.scores]

    return predict


def relative_decision_strategy(
    cfg: RelativeDecisionConfig | None = None,
) -> Callable[[Candidate], list[str]]:
    """Candidate-invariant verdict from per-candidate elevation."""
    cfg = cfg or RelativeDecisionConfig()

    def predict(c: Candidate) -> list[str]:
        return [d.rel_status for d in relative_decision(c.scores, cfg)]

    return predict


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def _accuracy(pred: Sequence[str], gold: Sequence[str]) -> float:
    if not gold:
        return 0.0
    return float(np.mean([p == g for p, g in zip(pred, gold)]))


def _false_positive_rate(
    pred: Sequence[str], gold: Sequence[str], protected: Sequence[bool]
) -> float:
    """Fraction of genuinely-CLEAR answers wrongly flagged PROBABLE (the costly error)."""
    neg = [i for i, g in enumerate(gold) if g == CLEAR]
    if not neg:
        return 0.0
    fp = sum(1 for i in neg if pred[i] == PROBABLE)
    return float(fp / len(neg))


def _class_separation(scores: Sequence[float], gold: Sequence[str]) -> float:
    """mean(score | scripted) - mean(score | clear). Larger => more separable."""
    pos = [s for s, g in zip(scores, gold) if g == PROBABLE]
    neg = [s for s, g in zip(scores, gold) if g == CLEAR]
    if not pos or not neg:
        return 0.0
    return float(np.mean(pos) - np.mean(neg))


def evaluate_strategy(
    candidates: list[Candidate],
    predict: Callable[[Candidate], list[str]],
) -> dict[str, Any]:
    """Per-candidate metrics + aggregates, including the cross-candidate variance."""
    per_candidate: list[dict[str, Any]] = []
    for c in candidates:
        pred = predict(c)
        per_candidate.append(
            {
                "candidate_id": c.candidate_id,
                "accuracy": round(_accuracy(pred, c.labels), 4),
                "false_positive_rate": round(
                    _false_positive_rate(pred, c.labels, c.personal_protected), 4
                ),
                "class_separation": round(_class_separation(c.scores, c.labels), 4),
                "n_answers": len(c.labels),
            }
        )

    accs = np.array([m["accuracy"] for m in per_candidate], dtype=np.float64)
    fprs = np.array([m["false_positive_rate"] for m in per_candidate], dtype=np.float64)
    seps = np.array([m["class_separation"] for m in per_candidate], dtype=np.float64)

    return {
        "per_candidate": per_candidate,
        "mean_accuracy": round(float(accs.mean()), 4) if accs.size else 0.0,
        "min_accuracy": round(float(accs.min()), 4) if accs.size else 0.0,
        "mean_false_positive_rate": round(float(fprs.mean()), 4) if fprs.size else 0.0,
        "cross_candidate_accuracy_variance": round(float(accs.var()), 6) if accs.size else 0.0,
        "cross_candidate_accuracy_std": round(float(accs.std()), 4) if accs.size else 0.0,
        "mean_class_separation": round(float(seps.mean()), 4) if seps.size else 0.0,
    }


def leave_one_candidate_out(
    candidates: list[Candidate],
    strategy_factory: Callable[[list[Candidate]], Callable[[Candidate], list[str]]],
) -> dict[str, Any]:
    """
    Evaluate a strategy with leave-one-candidate-out folds.

    ``strategy_factory`` receives the TRAINING candidates (all but the held-out one)
    and returns a predictor. Strategies that need no fitting simply ignore the
    training split. The held-out candidate is never seen while building the predictor
    — this is what exposes overfitting that a single-interview test cannot.
    """
    fold_metrics: list[dict[str, Any]] = []
    for i, held_out in enumerate(candidates):
        train = [c for j, c in enumerate(candidates) if j != i]
        predict = strategy_factory(train)
        pred = predict(held_out)
        fold_metrics.append(
            {
                "held_out": held_out.candidate_id,
                "accuracy": round(_accuracy(pred, held_out.labels), 4),
                "false_positive_rate": round(
                    _false_positive_rate(pred, held_out.labels, held_out.personal_protected), 4
                ),
            }
        )

    accs = np.array([m["accuracy"] for m in fold_metrics], dtype=np.float64)
    fprs = np.array([m["false_positive_rate"] for m in fold_metrics], dtype=np.float64)
    return {
        "folds": fold_metrics,
        "loco_mean_accuracy": round(float(accs.mean()), 4) if accs.size else 0.0,
        "loco_min_accuracy": round(float(accs.min()), 4) if accs.size else 0.0,
        "loco_accuracy_variance": round(float(accs.var()), 6) if accs.size else 0.0,
        "loco_mean_false_positive_rate": round(float(fprs.mean()), 4) if fprs.size else 0.0,
    }


def rubric_score(strategy_eval: dict[str, Any]) -> float:
    """
    Single 0..1 quality number from the weighted rubric (agentic-eval rubric pattern).

    Higher is better. cross_candidate_variance and false_positive_rate are inverted so
    that lower raw values contribute higher rubric points.
    """
    acc = strategy_eval["mean_accuracy"]
    fpr = strategy_eval["mean_false_positive_rate"]
    var = strategy_eval["cross_candidate_accuracy_variance"]
    sep = strategy_eval["mean_class_separation"]
    robustness = strategy_eval.get("calibration_robustness", strategy_eval["min_accuracy"])

    dims = {
        "per_answer_accuracy": acc,
        "false_positive_rate": 1.0 - min(1.0, fpr),
        # variance in [0, 0.25]-ish for accuracy; map to [0,1] via 1 - 4*var (clamped).
        "cross_candidate_variance": max(0.0, 1.0 - 4.0 * var),
        "class_separation": min(1.0, max(0.0, sep / 0.4)),
        "calibration_robustness": robustness,
    }
    return round(sum(dims[d] * RUBRIC[d]["weight"] for d in RUBRIC), 4)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def run_comparison(
    candidates: list[Candidate],
    *,
    absolute_threshold: float = 0.55,
    rel_cfg: RelativeDecisionConfig | None = None,
) -> dict[str, Any]:
    """Compare absolute-threshold vs candidate-relative decision across candidates."""
    rel_cfg = rel_cfg or RelativeDecisionConfig()

    abs_eval = evaluate_strategy(candidates, absolute_threshold_strategy(absolute_threshold))
    rel_eval = evaluate_strategy(candidates, relative_decision_strategy(rel_cfg))

    abs_loco = leave_one_candidate_out(
        candidates, lambda _train: absolute_threshold_strategy(absolute_threshold)
    )
    rel_loco = leave_one_candidate_out(
        candidates, lambda _train: relative_decision_strategy(rel_cfg)
    )

    return {
        "n_candidates": len(candidates),
        "absolute_threshold": {
            "threshold": absolute_threshold,
            "eval": abs_eval,
            "loco": abs_loco,
            "rubric_score": rubric_score(abs_eval),
        },
        "relative_decision": {
            "eval": rel_eval,
            "loco": rel_loco,
            "rubric_score": rubric_score(rel_eval),
        },
    }


def format_report(result: dict[str, Any]) -> str:
    a = result["absolute_threshold"]
    r = result["relative_decision"]
    lines = [
        "=" * 72,
        "SentinEL cross-candidate evaluation  (candidates=%d)" % result["n_candidates"],
        "=" * 72,
        "%-26s %14s %14s" % ("metric", "absolute(0.55)", "relative"),
        "-" * 72,
        "%-26s %14.4f %14.4f" % ("mean accuracy", a["eval"]["mean_accuracy"], r["eval"]["mean_accuracy"]),
        "%-26s %14.4f %14.4f" % ("min accuracy (worst cand)", a["eval"]["min_accuracy"], r["eval"]["min_accuracy"]),
        "%-26s %14.4f %14.4f" % ("mean false-positive rate", a["eval"]["mean_false_positive_rate"], r["eval"]["mean_false_positive_rate"]),
        "%-26s %14.6f %14.6f" % ("cross-cand acc variance", a["eval"]["cross_candidate_accuracy_variance"], r["eval"]["cross_candidate_accuracy_variance"]),
        "%-26s %14.4f %14.4f" % ("LOCO mean accuracy", a["loco"]["loco_mean_accuracy"], r["loco"]["loco_mean_accuracy"]),
        "%-26s %14.4f %14.4f" % ("LOCO acc variance", a["loco"]["loco_accuracy_variance"], r["loco"]["loco_accuracy_variance"]),
        "%-26s %14.4f %14.4f" % ("RUBRIC SCORE (0-1)", a["rubric_score"], r["rubric_score"]),
        "=" * 72,
        "per-candidate accuracy:",
    ]
    for am, rm in zip(a["eval"]["per_candidate"], r["eval"]["per_candidate"]):
        lines.append(
            "  %-10s  absolute=%.2f  relative=%.2f  (n=%d)"
            % (am["candidate_id"], am["accuracy"], rm["accuracy"], am["n_answers"])
        )
    verdict = (
        "RELATIVE generalizes better (lower cross-candidate variance, higher min accuracy)"
        if (
            r["eval"]["cross_candidate_accuracy_variance"]
            <= a["eval"]["cross_candidate_accuracy_variance"]
            and r["eval"]["min_accuracy"] >= a["eval"]["min_accuracy"]
        )
        else "no improvement — inspect data / config"
    )
    lines += ["-" * 72, "VERDICT: " + verdict, "=" * 72]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if argv:
        candidates = load_candidates(argv[0])
    else:
        from eval.synthetic_data import build_synthetic_candidates

        candidates = build_synthetic_candidates()
        print("(no data dir given - using built-in synthetic multi-candidate dataset)\n")
    result = run_comparison(candidates)
    print(format_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
