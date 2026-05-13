from __future__ import annotations

from app.schemas.models import ExtractedObservation, FieldCandidate


class ConfidenceScoringService:
    """Combine scores modèle / heuristique et retrieval."""

    @staticmethod
    def blend_with_retrieval(
        observation: ExtractedObservation,
        retrieval_score: float | None,
    ) -> float:
        base = observation.confidence
        if retrieval_score is None:
            return base
        return max(0.0, min(1.0, 0.5 * base + 0.5 * retrieval_score))

    def score_candidate(
        self,
        candidate: FieldCandidate,
        retrieval_score: float | None,
        autofill_threshold: float,
    ) -> tuple[float, bool]:
        blended = self.blend_with_retrieval(candidate.observation, retrieval_score)
        return blended, blended >= autofill_threshold
