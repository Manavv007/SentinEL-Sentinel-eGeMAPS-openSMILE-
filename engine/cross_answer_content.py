"""
Cross-answer transcript content diversity — session-level person-independent signal.

Natural candidates shift vocabulary/topic per question; script readers often
maintain similar essay-style phrasing across answers.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import config


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", (text or "").lower())


_STOP = frozenset(
    "a an the and or but in on at to for of is are was were be been being i you he she it we they".split()
)


class CrossAnswerContentTracker:
    """Incremental TF-IDF cosine similarity across answer transcripts."""

    def __init__(self) -> None:
        self._texts: list[str] = []
        self._answer_ids: list[int] = []

    def record(self, answer_id: int, transcript: dict[str, Any]) -> None:
        text = str(transcript.get("transcript", "") or "").strip()
        if not text:
            return
        self._answer_ids.append(int(answer_id))
        self._texts.append(text)

    def _tfidf_vectors(self) -> tuple[list[Counter[str]], dict[str, float]]:
        docs = [_tokenize(t) for t in self._texts]
        df: Counter[str] = Counter()
        for doc in docs:
            for term in set(doc):
                if term not in _STOP and len(term) > 2:
                    df[term] += 1
        n_docs = max(len(docs), 1)
        idf = {term: math.log((1 + n_docs) / (1 + count)) + 1.0 for term, count in df.items()}
        vectors: list[Counter[str]] = []
        for doc in docs:
            tf = Counter(t for t in doc if t not in _STOP and len(t) > 2)
            total = sum(tf.values()) or 1
            vec: Counter[str] = Counter()
            for term, cnt in tf.items():
                if term in idf:
                    vec[term] = (cnt / total) * idf[term]
            vectors.append(vec)
        return vectors, idf

    @staticmethod
    def _cosine(a: Counter[str], b: Counter[str]) -> float:
        if not a or not b:
            return 0.0
        keys = set(a) | set(b)
        dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        if na <= 0 or nb <= 0:
            return 0.0
        return dot / (na * nb)

    def session_profile(self) -> dict[str, Any]:
        if len(self._texts) < 2:
            return {
                "answer_count": len(self._texts),
                "mean_pairwise_similarity": 0.0,
                "content_uniformity": 0.0,
                "content_diversity": 1.0,
                "ready": False,
            }

        vectors, _ = self._tfidf_vectors()
        sims: list[float] = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                sims.append(self._cosine(vectors[i], vectors[j]))

        mean_sim = float(sum(sims) / len(sims)) if sims else 0.0
        content_uniformity = max(0.0, min(1.0, mean_sim))
        content_diversity = 1.0 - content_uniformity

        return {
            "answer_count": len(self._texts),
            "mean_pairwise_similarity": round(mean_sim, 4),
            "content_uniformity": round(content_uniformity, 4),
            "content_diversity": round(content_diversity, 4),
            "ready": len(self._texts) >= config.CONTENT_DIVERSITY_MIN_ANSWERS,
        }


class SessionEvidenceAccumulator:
    """
    Incremental session external-evidence tracker for feedforward into later answers.
    Separate from intra-individual SessionProbabilityState (do not modify that module).
    """

    def __init__(self) -> None:
        self.content = CrossAnswerContentTracker()
        self._external_scores: list[float] = []
        self._internal_scores: list[float] = []
        self.session_external_prior: float = config.SESSION_P_PRIOR

    def after_answer(
        self,
        *,
        answer_id: int,
        transcript: dict[str, Any],
        generic_script_likelihood: float,
        contrastive_external: float = 0.0,
        contrastive_internal: float = 0.0,
    ) -> dict[str, Any]:
        self.content.record(answer_id, transcript)
        content_prof = self.content.session_profile()

        ext = max(
            float(generic_script_likelihood),
            float(contrastive_external),
            content_prof.get("content_uniformity", 0.0) * 0.85,
        )
        int_sig = max(
            float(contrastive_internal),
            1.0 - float(generic_script_likelihood),
            content_prof.get("content_diversity", 0.5),
        )
        self._external_scores.append(ext)
        self._internal_scores.append(int_sig)

        mean_ext = sum(self._external_scores) / len(self._external_scores)
        mean_int = sum(self._internal_scores) / len(self._internal_scores)
        uniformity = float(content_prof.get("content_uniformity", 0.0))

        lr = (0.5 + mean_ext) / max(0.2, 0.4 + mean_int)
        lr = max(0.7, min(1.4, lr))
        strength = config.SESSION_FEEDFORWARD_UPDATE_STRENGTH

        import math

        p = max(1e-6, min(1.0 - 1e-6, self.session_external_prior))
        logit = math.log(p / (1.0 - p)) + strength * (lr - 1.0)
        self.session_external_prior = 1.0 / (1.0 + math.exp(-logit))
        self.session_external_prior = max(0.0, min(1.0, self.session_external_prior))

        return {
            "answer_id": answer_id,
            "session_external_prior": round(self.session_external_prior, 4),
            "content_profile": content_prof,
            "mean_external_signal": round(mean_ext, 4),
        }

    def feedforward_active(self, answer_index: int) -> bool:
        return (
            answer_index >= config.SESSION_FEEDFORWARD_MIN_ANSWERS
            and self.session_external_prior >= config.SESSION_FEEDFORWARD_P_MIN
        )
