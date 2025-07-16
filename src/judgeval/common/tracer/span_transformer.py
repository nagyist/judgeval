"""
Span transformer utilities for converting between TraceSpan and OpenTelemetry formats.

This module provides utilities for transforming TraceSpan objects to OpenTelemetry
format and vice versa, maintaining compatibility with existing data structures.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Union

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import Status, StatusCode

from judgeval.data import TraceSpan, TraceUsage
from judgeval.data.tool import Tool
from judgeval.evaluation_run import EvaluationRun


class SpanTransformer:
    """
    Utility class for transforming between TraceSpan and OpenTelemetry formats.
    """

    # Fields that need JSON serialization when converting to OTEL attributes
    JSON_FIELDS = {
        "inputs",
        "output",
        "error",
        "usage",
        "expected_tools",
        "additional_metadata",
        "state_before",
        "state_after",
        "span_data",
        "examples",
        "scorers",
        "traces",
        "rules",
        "tools",
    }

    @staticmethod
    def _safe_json_handle(obj: Any, serialize: bool = True) -> Any:
        """
        Safely handle JSON serialization/deserialization.

        Args:
            obj: Object to process
            serialize: If True, serialize to JSON string. If False, deserialize from JSON string.

        Returns:
            Processed object
        """
        if serialize:
            if obj is None:
                return None
            try:
                return json.dumps(obj, default=str)
            except Exception:
                return json.dumps(str(obj))
        else:
            if not isinstance(obj, str):
                return obj
            try:
                return json.loads(obj)
            except (json.JSONDecodeError, TypeError):
                return obj

    @staticmethod
    def _format_timestamp(timestamp: Optional[Union[float, int, str]]) -> str:
        """Format timestamp to ISO format with timezone."""
        if timestamp is None:
            return datetime.now(timezone.utc).isoformat()

        if isinstance(timestamp, str):
            return timestamp

        try:
            dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            return dt.isoformat()
        except (ValueError, OSError):
            return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def trace_span_to_otel_attributes(
        trace_span: TraceSpan, span_state: str = "completed"
    ) -> Dict[str, Any]:
        """Convert a TraceSpan object to OpenTelemetry span attributes."""
        # Get serialized data from model_dump to handle complex objects properly
        serialized_data = trace_span.model_dump()
        attributes: Dict[str, Any] = {}

        # Add all fields with judgment prefix
        for field_name, value in serialized_data.items():
            if value is None:
                continue

            attr_name = f"judgment.{field_name}"

            # Handle special cases
            if field_name == "created_at":
                attributes[attr_name] = SpanTransformer._format_timestamp(value)
            elif field_name == "expected_tools" and value:
                attributes[attr_name] = SpanTransformer._safe_json_handle(
                    [tool.model_dump() for tool in trace_span.expected_tools]
                )
            elif field_name == "usage" and value:
                attributes[attr_name] = SpanTransformer._safe_json_handle(
                    trace_span.usage.model_dump()
                )
            elif field_name in SpanTransformer.JSON_FIELDS:
                attributes[attr_name] = SpanTransformer._safe_json_handle(value)
            else:
                attributes[attr_name] = value

        # Add computed fields
        attributes["judgment.span_state"] = span_state
        if not attributes.get("judgment.span_type"):
            attributes["judgment.span_type"] = "span"

        return attributes

    @staticmethod
    def otel_attributes_to_judgment_data(attributes: Dict[str, Any]) -> Dict[str, Any]:
        """Convert OpenTelemetry span attributes to Judgment data format."""
        judgment_data: Dict[str, Any] = {}

        for key, value in attributes.items():
            if not key.startswith("judgment."):
                continue

            field_name = key[9:]  # Remove "judgment." prefix

            if field_name in SpanTransformer.JSON_FIELDS:
                judgment_data[field_name] = SpanTransformer._safe_json_handle(
                    value, serialize=False
                )
            else:
                judgment_data[field_name] = value

        return judgment_data

    @staticmethod
    def otel_span_to_judgment_format(span: ReadableSpan) -> Dict[str, Any]:
        """Convert OpenTelemetry ReadableSpan to Judgment API format."""
        attributes = span.attributes or {}
        judgment_data = SpanTransformer.otel_attributes_to_judgment_data(attributes)

        # Calculate duration
        duration = judgment_data.get("duration")
        if duration is None and span.end_time and span.start_time:
            duration = (span.end_time - span.start_time) / 1_000_000_000

        # Get or generate IDs
        span_id = judgment_data.get("span_id") or str(uuid.uuid4())
        trace_id = judgment_data.get("trace_id") or str(uuid.uuid4())

        # Calculate created_at
        created_at = judgment_data.get("created_at")
        if not created_at:
            created_at = (
                span.start_time / 1_000_000_000 if span.start_time else time.time()
            )

        return {
            "type": "span",
            "data": {
                "span_id": span_id,
                "trace_id": trace_id,
                "function": span.name,
                "depth": judgment_data.get("depth", 0),
                "created_at": SpanTransformer._format_timestamp(created_at),
                "parent_span_id": judgment_data.get("parent_span_id"),
                "span_type": judgment_data.get("span_type", "span"),
                "inputs": judgment_data.get("inputs"),
                "error": judgment_data.get("error"),
                "output": judgment_data.get("output"),
                "usage": judgment_data.get("usage"),
                "duration": duration,
                "expected_tools": judgment_data.get("expected_tools"),
                "additional_metadata": judgment_data.get("additional_metadata"),
                "has_evaluation": judgment_data.get("has_evaluation", False),
                "agent_name": judgment_data.get("agent_name"),
                "state_before": judgment_data.get("state_before"),
                "state_after": judgment_data.get("state_after"),
                "update_id": judgment_data.get("update_id", 1),
                "span_state": judgment_data.get("span_state", "completed"),
                "queued_at": time.time(),
            },
        }

    @staticmethod
    def evaluation_run_to_otel_attributes(
        evaluation_run: EvaluationRun, span_id: str, span_data: TraceSpan
    ) -> Dict[str, Any]:
        """Convert an EvaluationRun to OpenTelemetry span attributes."""
        attributes = {
            "judgment.evaluation_run": True,
            "judgment.associated_span_id": span_id,
            "judgment.span_data": SpanTransformer._safe_json_handle(
                span_data.model_dump()
            ),
        }

        # Add evaluation run data
        eval_data = evaluation_run.model_dump()
        for key, value in eval_data.items():
            if value is None:
                continue

            attr_name = f"judgment.{key}"
            if key in SpanTransformer.JSON_FIELDS:
                attributes[attr_name] = SpanTransformer._safe_json_handle(value)
            else:
                attributes[attr_name] = value

        return attributes

    @staticmethod
    def otel_span_to_evaluation_run_format(span: ReadableSpan) -> Dict[str, Any]:
        """Convert OpenTelemetry ReadableSpan to evaluation run format."""
        attributes = span.attributes or {}
        judgment_data = SpanTransformer.otel_attributes_to_judgment_data(attributes)

        associated_span_id = judgment_data.get("associated_span_id") or str(
            uuid.uuid4()
        )

        return {
            "type": "evaluation_run",
            "data": {
                **{
                    key: value
                    for key, value in judgment_data.items()
                    if key not in ["associated_span_id", "span_data", "evaluation_run"]
                },
                "associated_span_id": associated_span_id,
                "span_data": judgment_data.get("span_data"),
                "queued_at": time.time(),
            },
        }

    @staticmethod
    def create_trace_span_from_otel_attributes(
        attributes: Dict[str, Any], span_name: str
    ) -> TraceSpan:
        """Create a TraceSpan object from OpenTelemetry attributes."""
        judgment_data = SpanTransformer.otel_attributes_to_judgment_data(attributes)

        # Create TraceUsage if usage data exists
        usage = None
        if judgment_data.get("usage"):
            usage_data = judgment_data["usage"]
            if isinstance(usage_data, dict):
                usage = TraceUsage(**usage_data)

        # Create expected tools if they exist
        expected_tools = []
        if judgment_data.get("expected_tools"):
            tools_data = judgment_data["expected_tools"]
            if isinstance(tools_data, list):
                for tool_data in tools_data:
                    if isinstance(tool_data, dict):
                        expected_tools.append(Tool(**tool_data))

        # Parse created_at timestamp
        created_at = judgment_data.get("created_at", time.time())
        if isinstance(created_at, str):
            try:
                dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                created_at = dt.timestamp()
            except ValueError:
                created_at = time.time()

        return TraceSpan(
            span_id=judgment_data.get("span_id", ""),
            trace_id=judgment_data.get("trace_id", ""),
            function=span_name,
            depth=judgment_data.get("depth", 0),
            created_at=created_at,
            parent_span_id=judgment_data.get("parent_span_id"),
            span_type=judgment_data.get("span_type", "span"),
            inputs=judgment_data.get("inputs"),
            error=judgment_data.get("error"),
            output=judgment_data.get("output"),
            usage=usage,
            duration=judgment_data.get("duration"),
            expected_tools=expected_tools,
            additional_metadata=judgment_data.get("additional_metadata"),
            has_evaluation=judgment_data.get("has_evaluation", False),
            agent_name=judgment_data.get("agent_name"),
            state_before=judgment_data.get("state_before"),
            state_after=judgment_data.get("state_after"),
            update_id=judgment_data.get("update_id", 1),
        )

    @staticmethod
    def get_span_status_from_error(error: Optional[Dict[str, Any]]) -> Status:
        """Get OpenTelemetry span status from error information."""
        if error:
            return Status(StatusCode.ERROR, description=str(error))
        return Status(StatusCode.OK)
