from abc import ABC, abstractmethod


class BaseMetric(ABC):
    """Abstract base class for custom metric implementations.

    Implement this when an evaluation task requires a metric not natively
    supported by lm-eval or lmms-eval (e.g. custom safety scores for T4.4,
    or domain-specific metrics for T4.3).

    The ``compute`` method must return a scalar in [0, 1] by convention,
    though higher-range metrics (e.g. OCRBench score /1000) are allowed when
    the metric name makes the range unambiguous.

    Example::

        class ExactMatchMetric(BaseMetric):
            @property
            def name(self) -> str:
                return "exact_match"

            def compute(
                self,
                predictions: list[str],
                references: list[str],
            ) -> float:
                if not predictions:
                    return 0.0
                correct = sum(p == r for p, r in zip(predictions, references))
                return correct / len(predictions)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique metric identifier, e.g. ``"vqa_score"`` or ``"anls"``."""

    @abstractmethod
    def compute(
        self,
        predictions: list[str],
        references: list[str],
    ) -> float:
        """Compute the metric score.

        Args:
            predictions: Model-generated answers (one per sample).
            references:  Ground-truth answers (one per sample).

        Returns:
            Scalar score. Conventionally in [0, 1]; higher is better.
        """
