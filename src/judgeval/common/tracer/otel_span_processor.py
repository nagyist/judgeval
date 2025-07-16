"""
Custom OpenTelemetry span processor for Judgment API.

This processor uses BatchSpanProcessor to handle batching and export
of TraceSpan objects converted to OpenTelemetry format.
"""

from __future__ import annotations

from typing import Any, Optional

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanProcessor
from opentelemetry.trace import Span, Status, StatusCode, SpanContext, TraceFlags
from opentelemetry.trace.span import TraceState, INVALID_SPAN_CONTEXT
from opentelemetry.util.types import Attributes

from judgeval.common.logger import judgeval_logger
from judgeval.common.tracer.otel_exporter import JudgmentAPISpanExporter
from judgeval.common.tracer.span_transformer import SpanTransformer
from judgeval.data import TraceSpan
from judgeval.evaluation_run import EvaluationRun


class SimpleReadableSpan(ReadableSpan):
    """
    Simple ReadableSpan implementation that wraps TraceSpan data.
    """

    def __init__(self, trace_span: TraceSpan, span_state: str = "completed"):
        self._name = trace_span.function
        self._span_id = trace_span.span_id
        self._trace_id = trace_span.trace_id

        # Convert timestamps to nanoseconds for OpenTelemetry
        self._start_time = (
            int(trace_span.created_at * 1_000_000_000)
            if trace_span.created_at
            else None
        )
        self._end_time: Optional[int] = None

        # Calculate end time if completed and has duration
        if (
            span_state == "completed"
            and trace_span.duration is not None
            and self._start_time is not None
        ):
            self._end_time = self._start_time + int(trace_span.duration * 1_000_000_000)

        # Set status based on errors
        self._status = (
            Status(StatusCode.ERROR) if trace_span.error else Status(StatusCode.OK)
        )

        # Convert TraceSpan to OpenTelemetry attributes
        self._attributes = SpanTransformer.trace_span_to_otel_attributes(
            trace_span, span_state
        )

        # Create proper OpenTelemetry SpanContext
        try:
            # Convert string IDs to integers for OpenTelemetry
            trace_id_int = (
                int(trace_span.trace_id.replace("-", ""), 16)
                if trace_span.trace_id
                else 0
            )
            span_id_int = (
                int(trace_span.span_id.replace("-", ""), 16)
                if trace_span.span_id
                else 0
            )

            self._context = SpanContext(
                trace_id=trace_id_int,
                span_id=span_id_int,
                is_remote=False,
                trace_flags=TraceFlags(0x01),  # SAMPLED
                trace_state=TraceState(),
            )
        except (ValueError, TypeError) as e:
            judgeval_logger.warning(f"Failed to create proper SpanContext: {e}")
            self._context = INVALID_SPAN_CONTEXT

        # Empty collections for compatibility
        self._parent: Optional[SpanContext] = None
        self._events: list[Any] = []
        self._links: list[Any] = []
        self._resource: Optional[Any] = None
        self._instrumentation_info: Optional[Any] = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def context(self) -> SpanContext:
        return self._context

    @property
    def parent(self) -> Optional[SpanContext]:
        return self._parent

    @property
    def start_time(self) -> Optional[int]:
        return self._start_time

    @property
    def end_time(self) -> Optional[int]:
        return self._end_time

    @property
    def status(self) -> Status:
        return self._status

    @property
    def attributes(self) -> Optional[Attributes]:
        return self._attributes

    @property
    def events(self):
        return self._events

    @property
    def links(self):
        return self._links

    @property
    def resource(self) -> Optional[Any]:
        return self._resource

    @property
    def instrumentation_info(self) -> Optional[Any]:
        return self._instrumentation_info


class JudgmentSpanProcessor(SpanProcessor):
    """
    Span processor that converts TraceSpan objects to OpenTelemetry format
    and uses BatchSpanProcessor for export.
    """

    def __init__(
        self,
        judgment_api_key: str,
        organization_id: str,
        batch_size: int = 50,
        flush_interval: float = 1.0,
        max_queue_size: int = 2048,
        export_timeout: int = 30000,
    ):
        self.judgment_api_key = judgment_api_key
        self.organization_id = organization_id

        # Use BatchSpanProcessor for all the heavy lifting
        self.batch_processor = BatchSpanProcessor(
            JudgmentAPISpanExporter(
                judgment_api_key=judgment_api_key,
                organization_id=organization_id,
            ),
            max_queue_size=max_queue_size,
            schedule_delay_millis=int(flush_interval * 1000),
            max_export_batch_size=batch_size,
            export_timeout_millis=export_timeout,
        )

    def on_start(self, span: Span, parent_context: Optional[Context] = None) -> None:
        """Called when a span starts - delegate to batch processor."""
        self.batch_processor.on_start(span, parent_context)

    def on_end(self, span: ReadableSpan) -> None:
        """Called when a span ends - delegate to batch processor."""
        self.batch_processor.on_end(span)

    def queue_span_update(self, span: TraceSpan, span_state: str = "input") -> None:
        """
        Queue a span update for export.

        Args:
            span: The TraceSpan object to update
            span_state: State of the span ("input", "output", "completed", etc.)
        """
        # Handle update_id logic based on span state
        if span_state == "completed":
            span.set_update_id_to_ending_number()
        else:
            span.increment_update_id()

        # Create ReadableSpan and send to BatchSpanProcessor
        readable_span = SimpleReadableSpan(span, span_state)
        self.batch_processor.on_end(readable_span)

    def queue_evaluation_run(
        self, evaluation_run: EvaluationRun, span_id: str, span_data: TraceSpan
    ) -> None:
        """
        Queue an evaluation run for export.

        Args:
            evaluation_run: The EvaluationRun object to queue
            span_id: The span ID associated with this evaluation run
            span_data: The span data at the time of evaluation
        """
        # Convert evaluation run to span attributes
        attributes = SpanTransformer.evaluation_run_to_otel_attributes(
            evaluation_run, span_id, span_data
        )

        # Create ReadableSpan with evaluation run data
        readable_span = SimpleReadableSpan(span_data, "evaluation_run")
        readable_span._attributes.update(attributes)

        judgeval_logger.debug(f"Queuing evaluation run for span {span_id}")

        # Send to BatchSpanProcessor
        self.batch_processor.on_end(readable_span)

    def shutdown(self) -> None:
        """Shutdown the processor."""
        self.batch_processor.shutdown()
        judgeval_logger.debug("JudgmentSpanProcessor shutdown complete")

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush all pending spans."""
        success = self.batch_processor.force_flush(timeout_millis)
        judgeval_logger.debug(
            f"JudgmentSpanProcessor force_flush {'succeeded' if success else 'failed'}"
        )
        return success
