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
        rs_from_obs: float | None = None
        if observation.extra:
            raw_rs = observation.extra.get("retrieval_score")
            if raw_rs is not None:
                try:
                    rs_from_obs = float(raw_rs)
                except (TypeError, ValueError):
                    rs_from_obs = None
        final_rs = rs_from_obs if rs_from_obs is not None else retrieval_score
        if final_rs is None:
            return base
        return max(0.0, min(1.0, 0.5 * base + 0.5 * final_rs))

    def score_candidate(
        self,
        candidate: FieldCandidate,
        retrieval_score: float | None,
        autofill_threshold: float,
    ) -> tuple[float, bool]:
        blended = self.blend_with_retrieval(candidate.observation, retrieval_score)
        return blended, blended >= autofill_threshold
