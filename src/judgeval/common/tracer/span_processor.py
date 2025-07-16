"""
Base class for span processors with default no-op implementations.
"""

from judgeval.data import TraceSpan
from judgeval.evaluation_run import EvaluationRun


class SpanProcessorBase:
    """
    Base class for Judgment span processors with default no-op implementations.

    This eliminates the need for optional typing and null checks.
    When monitoring is disabled, we use this base class directly.
    When monitoring is enabled, we use JudgmentSpanProcessor which overrides the methods.
    """

    def queue_span_update(self, span: TraceSpan, span_state: str = "input") -> None:
        """
        Queue a span update for processing. Default no-op implementation.

        Args:
            span: The TraceSpan object to update
            span_state: State of the span ("input", "output", "completed", etc.)
        """
        pass

    def queue_evaluation_run(
        self, evaluation_run: EvaluationRun, span_id: str, span_data: TraceSpan
    ) -> None:
        """
        Queue an evaluation run for processing. Default no-op implementation.

        Args:
            evaluation_run: The EvaluationRun object to queue
            span_id: The span ID associated with this evaluation run
            span_data: The span data at the time of evaluation
        """
        pass

    def flush_pending_spans(self) -> None:
        """
        Flush all pending span updates. Default no-op implementation.
        """
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """
        Force flush all pending spans with a timeout. Default no-op implementation.

        Args:
            timeout_millis: Timeout in milliseconds

        Returns:
            True (always successful for no-op)
        """
        return True

    def shutdown(self) -> None:
        """
        Shutdown the processor and clean up resources. Default no-op implementation.
        """
        pass
