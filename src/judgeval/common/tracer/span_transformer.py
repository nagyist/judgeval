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

    This class provides methods to convert TraceSpan objects to OpenTelemetry
    attributes and vice versa, handling serialization and deserialization of
    complex data structures.
    """

    @staticmethod
    def _format_timestamp(timestamp: Optional[Union[float, int, str]]) -> str:
        """
        Format timestamp to ISO format with timezone.

        Args:
            timestamp: Unix timestamp (seconds since epoch) or existing formatted timestamp

        Returns:
            ISO formatted timestamp string with timezone
        """
        if timestamp is None:
            return datetime.now(timezone.utc).isoformat()

        if isinstance(timestamp, str):
            return timestamp

        try:
            # Convert Unix timestamp to datetime with UTC timezone
            dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            return dt.isoformat()
        except (ValueError, OSError):
            return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def trace_span_to_otel_attributes(
        trace_span: TraceSpan, span_state: str = "completed"
    ) -> Dict[str, Any]:
        """
        Convert a TraceSpan object to OpenTelemetry span attributes.

        Args:
            trace_span: The TraceSpan object to convert
            span_state: Current state of the span

        Returns:
            Dictionary of OpenTelemetry span attributes
        """
        attributes: Dict[str, Any] = {}

        # Basic span information
        attributes["judgment.span_id"] = trace_span.span_id
        attributes["judgment.trace_id"] = trace_span.trace_id
        attributes["judgment.depth"] = trace_span.depth
        attributes["judgment.created_at"] = SpanTransformer._format_timestamp(
            trace_span.created_at
        )
        attributes["judgment.span_type"] = trace_span.span_type or "span"
        attributes["judgment.span_state"] = span_state
        attributes["judgment.update_id"] = trace_span.update_id

        # Optional fields
        if trace_span.parent_span_id:
            attributes["judgment.parent_span_id"] = trace_span.parent_span_id
        if trace_span.duration is not None:
            attributes["judgment.duration"] = trace_span.duration
        if trace_span.has_evaluation is not None:
            attributes["judgment.has_evaluation"] = trace_span.has_evaluation
        if trace_span.agent_name:
            attributes["judgment.agent_name"] = trace_span.agent_name

        # Use the serialized data from model_dump() to avoid double serialization
        # model_dump() already properly handles complex objects like LangChain messages
        serialized_data = trace_span.model_dump()

        # Complex fields (use pre-serialized data from model_dump())
        if serialized_data.get("inputs") is not None:
            attributes["judgment.inputs"] = json.dumps(serialized_data["inputs"])
        if serialized_data.get("output") is not None:
            attributes["judgment.output"] = json.dumps(serialized_data["output"])
        if serialized_data.get("error") is not None:
            attributes["judgment.error"] = json.dumps(serialized_data["error"])
        if trace_span.usage is not None:
            attributes["judgment.usage"] = json.dumps(trace_span.usage.model_dump())
        if trace_span.expected_tools is not None:
            attributes["judgment.expected_tools"] = json.dumps(
                [tool.model_dump() for tool in trace_span.expected_tools]
            )
        if serialized_data.get("additional_metadata") is not None:
            attributes["judgment.additional_metadata"] = json.dumps(
                serialized_data["additional_metadata"]
            )
        if serialized_data.get("state_before") is not None:
            attributes["judgment.state_before"] = json.dumps(
                serialized_data["state_before"]
            )
        if serialized_data.get("state_after") is not None:
            attributes["judgment.state_after"] = json.dumps(
                serialized_data["state_after"]
            )

        return attributes

    @staticmethod
    def otel_attributes_to_judgment_data(attributes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert OpenTelemetry span attributes to Judgment API format.

        Args:
            attributes: Dictionary of OpenTelemetry span attributes

        Returns:
            Dictionary in Judgment API format
        """
        judgment_data: Dict[str, Any] = {}

        for key, value in attributes.items():
            if key.startswith("judgment."):
                field_name = key[9:]  # Remove "judgment." prefix

                # Handle JSON-serialized fields (these are the ones that get serialized in trace_span_to_otel_attributes and evaluation_run_to_otel_attributes)
                if field_name in [
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
                ]:
                    judgment_data[field_name] = SpanTransformer._safe_json_deserialize(
                        value
                    )
                else:
                    judgment_data[field_name] = value

        return judgment_data

    @staticmethod
    def otel_span_to_judgment_format(span: ReadableSpan) -> Dict[str, Any]:
        """
        Convert OpenTelemetry ReadableSpan to Judgment API format.

        Args:
            span: OpenTelemetry ReadableSpan object

        Returns:
            Dictionary in Judgment API format
        """
        attributes = span.attributes or {}
        judgment_data = SpanTransformer.otel_attributes_to_judgment_data(attributes)

        # Calculate duration from multiple sources
        duration = None

        # First try to get duration from attributes (preferred for intermediate states)
        if judgment_data.get("duration") is not None:
            duration = judgment_data.get("duration")
        # Fall back to calculating from start/end times
        elif span.end_time and span.start_time:
            duration = (
                span.end_time - span.start_time
            ) / 1_000_000_000  # Convert nanoseconds to seconds

        # Build the span data in existing format
        # Use existing span/trace IDs if available, otherwise generate new UUIDs
        span_id = judgment_data.get("span_id")
        if not span_id:
            span_id = str(uuid.uuid4())

        trace_id = judgment_data.get("trace_id")
        if not trace_id:
            trace_id = str(uuid.uuid4())

        # Calculate created_at timestamp
        created_at = judgment_data.get("created_at")
        if not created_at:
            created_at = (
                span.start_time / 1_000_000_000 if span.start_time else time.time()
            )

        span_data = {
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

        return span_data

    @staticmethod
    def evaluation_run_to_otel_attributes(
        evaluation_run: EvaluationRun, span_id: str, span_data: TraceSpan
    ) -> Dict[str, Any]:
        """
        Convert an EvaluationRun to OpenTelemetry span attributes.

        Args:
            evaluation_run: The EvaluationRun object
            span_id: The span ID associated with this evaluation run
            span_data: The span data at the time of evaluation

        Returns:
            Dictionary of OpenTelemetry span attributes
        """
        attributes: Dict[str, Any] = {}

        # Mark as evaluation run
        attributes["judgment.evaluation_run"] = True
        attributes["judgment.associated_span_id"] = span_id
        attributes["judgment.span_data"] = SpanTransformer._safe_json_serialize(
            span_data.model_dump()
        )

        # Add evaluation run data
        eval_data = evaluation_run.model_dump()
        for key, value in eval_data.items():
            if isinstance(value, (dict, list)):
                attributes[f"judgment.{key}"] = SpanTransformer._safe_json_serialize(
                    value
                )
            else:
                attributes[f"judgment.{key}"] = value

        return attributes

    @staticmethod
    def otel_span_to_evaluation_run_format(span: ReadableSpan) -> Dict[str, Any]:
        """
        Convert OpenTelemetry ReadableSpan to evaluation run format.

        Args:
            span: OpenTelemetry ReadableSpan object

        Returns:
            Dictionary in evaluation run format
        """
        attributes = span.attributes or {}
        judgment_data = SpanTransformer.otel_attributes_to_judgment_data(attributes)

        # Structure evaluation run data
        associated_span_id = judgment_data.get("associated_span_id")
        if not associated_span_id:
            associated_span_id = str(uuid.uuid4())

        eval_data = {
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

        return eval_data

    @staticmethod
    def create_trace_span_from_otel_attributes(
        attributes: Dict[str, Any], span_name: str
    ) -> TraceSpan:
        """
        Create a TraceSpan object from OpenTelemetry attributes.

        Args:
            attributes: Dictionary of OpenTelemetry span attributes
            span_name: Name of the span

        Returns:
            TraceSpan object
        """
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
            # If it's already a formatted timestamp string, try to parse it back to Unix timestamp
            try:
                dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                created_at = dt.timestamp()
            except ValueError:
                # If parsing fails, use current time
                created_at = time.time()

        # Create TraceSpan
        trace_span = TraceSpan(
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

        return trace_span

    @staticmethod
    def _safe_json_serialize(obj: Any) -> str:
        """
        Safely serialize an object to JSON string.

        Args:
            obj: Object to serialize

        Returns:
            JSON string representation
        """
        try:
            return json.dumps(obj, default=SpanTransformer._fallback_encoder)
        except Exception:
            return json.dumps(str(obj))

    @staticmethod
    def _safe_json_deserialize(json_str: str) -> Any:
        """
        Safely deserialize a JSON string to an object.

        Args:
            json_str: JSON string to deserialize

        Returns:
            Deserialized object
        """
        try:
            return json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            return json_str

    @staticmethod
    def _fallback_encoder(obj: Any) -> str:
        """
        Fallback encoder for JSON serialization.

        Args:
            obj: Object to encode

        Returns:
            String representation of the object
        """
        try:
            return str(obj)
        except Exception:
            return f"<{type(obj).__name__}>"

    @staticmethod
    def get_span_status_from_error(error: Optional[Dict[str, Any]]) -> Status:
        """
        Get OpenTelemetry span status from error information.

        Args:
            error: Error information dictionary

        Returns:
            OpenTelemetry Status object
        """
        if error:
            return Status(StatusCode.ERROR, description=str(error))
        return Status(StatusCode.OK)

    @staticmethod
    def validate_span_data(span_data: Dict[str, Any]) -> bool:
        """
        Validate span data contains required fields.

        Args:
            span_data: Span data dictionary

        Returns:
            True if valid, False otherwise
        """
        required_fields = ["span_id", "trace_id", "function"]
        return all(field in span_data for field in required_fields)

    @staticmethod
    def sanitize_span_data(span_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize span data by removing or replacing invalid values.

        Args:
            span_data: Span data dictionary

        Returns:
            Sanitized span data dictionary
        """
        sanitized: Dict[str, Any] = {}

        for key, value in span_data.items():
            if value is None:
                continue

            # Handle specific data types
            if key in ["created_at", "duration"]:
                if isinstance(value, (int, float)) and value >= 0:
                    sanitized[key] = value
            elif key in ["depth", "update_id"]:
                if isinstance(value, int) and value >= 0:
                    sanitized[key] = value
            elif key in ["span_id", "trace_id", "function", "span_type", "agent_name"]:
                if isinstance(value, str) and value:
                    sanitized[key] = value
            else:
                sanitized[key] = value

        return sanitized
