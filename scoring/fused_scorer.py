"""Backward-compatible re-export — implementation lives in engine.fused_scorer."""

from engine.fused_scorer import BASE_WEIGHTS, FuseResult, FusedScorer

__all__ = ["BASE_WEIGHTS", "FuseResult", "FusedScorer"]

# Legacy alias
AnswerScore = FuseResult
