"""
Custom OpenTelemetry exporter for Judgment API.

This exporter sends spans to the Judgment API using the existing format.
The BatchSpanProcessor handles all batching, threading, and retry logic.
"""

from __future__ import annotations

import json
from http import HTTPStatus
from typing import Any, Dict, List, Sequence

from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.trace import ReadableSpan

from judgeval.common.tracer.span_transformer import SpanTransformer
from judgeval.constants import (
    JUDGMENT_TRACES_EVALUATION_RUNS_BATCH_API_URL,
    JUDGMENT_TRACES_SPANS_BATCH_API_URL,
)
from judgeval.utils.requests import requests


class JudgmentAPISpanExporter(SpanExporter):
    """
    Custom OpenTelemetry exporter that sends spans to Judgment API.

    This exporter is used by BatchSpanProcessor which handles all the
    batching, threading, and retry logic for us.
    """

    def __init__(
        self,
        judgment_api_key: str,
        organization_id: str,
        spans_endpoint: str = JUDGMENT_TRACES_SPANS_BATCH_API_URL,
        eval_runs_endpoint: str = JUDGMENT_TRACES_EVALUATION_RUNS_BATCH_API_URL,
    ):
        """
        Initialize the Judgment API exporter.

        Args:
            judgment_api_key: API key for Judgment service
            organization_id: Organization ID
            spans_endpoint: API endpoint for spans batch
            eval_runs_endpoint: API endpoint for evaluation runs batch
        """
        self.judgment_api_key = judgment_api_key
        self.organization_id = organization_id
        self.spans_endpoint = spans_endpoint
        self.eval_runs_endpoint = eval_runs_endpoint
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {judgment_api_key}",
            "X-Organization-Id": organization_id,
        }

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """
        Export spans to Judgment API.

        This method is called by BatchSpanProcessor with a batch of spans.
        We send them synchronously since BatchSpanProcessor handles threading.

        Args:
            spans: Sequence of OpenTelemetry ReadableSpan objects

        Returns:
            SpanExportResult indicating success or failure
        """
        if not spans:
            return SpanExportResult.SUCCESS

        try:
            # Separate spans and evaluation runs
            spans_data = []
            eval_runs_data = []

            for span in spans:
                span_data = self._convert_span_to_judgment_format(span)

                # Check if this span contains evaluation run data
                if span.attributes.get("judgment.evaluation_run"):
                    eval_runs_data.append(span_data)
                else:
                    spans_data.append(span_data)

            # Send spans if we have any
            if spans_data:
                self._send_spans_batch(spans_data)

            # Send evaluation runs if we have any
            if eval_runs_data:
                self._send_evaluation_runs_batch(eval_runs_data)

            return SpanExportResult.SUCCESS

        except Exception:
            return SpanExportResult.FAILURE

    def _convert_span_to_judgment_format(self, span: ReadableSpan) -> Dict[str, Any]:
        """
        Convert OpenTelemetry span to existing Judgment API format.

        Args:
            span: OpenTelemetry ReadableSpan object

        Returns:
            Dictionary in existing Judgment API format
        """
        # Check if this is an evaluation run span
        if span.attributes and span.attributes.get("judgment.evaluation_run"):
            # Use the evaluation run formatter
            return SpanTransformer.otel_span_to_evaluation_run_format(span)
        else:
            # Use the regular span formatter
            return SpanTransformer.otel_span_to_judgment_format(span)

    def _send_spans_batch(self, spans: List[Dict[str, Any]]):
        """
        Send a batch of spans to the spans endpoint.

        Args:
            spans: List of span dictionaries to send
        """
        payload = {
            "spans": [span["data"] for span in spans],
            "organization_id": self.organization_id,
        }

        serialized_data = json.dumps(payload, default=self._fallback_encoder)

        response = requests.post(
            self.spans_endpoint,
            data=serialized_data,
            headers=self.headers,
            verify=True,
            timeout=30,
        )

        if response.status_code != HTTPStatus.OK:
            raise Exception(f"HTTP {response.status_code} - {response.text}")

    def _send_evaluation_runs_batch(self, evaluation_runs: List[Dict[str, Any]]):
        """
        Send a batch of evaluation runs to the evaluation runs endpoint.

        Args:
            evaluation_runs: List of evaluation run dictionaries to send
        """
        # Structure payload to match existing API format
        evaluation_entries = []
        for eval_run in evaluation_runs:
            eval_data = eval_run["data"]
            # Structure entry to match existing API format
            entry = {
                "evaluation_run": {
                    # Extract evaluation run fields (excluding span-specific fields)
                    key: value
                    for key, value in eval_data.items()
                    if key not in ["associated_span_id", "span_data", "queued_at"]
                },
                "associated_span": {
                    "span_id": eval_data.get("associated_span_id"),
                    "span_data": eval_data.get("span_data"),
                },
                "queued_at": eval_data.get("queued_at"),
            }
            evaluation_entries.append(entry)

        payload = {
            "organization_id": self.organization_id,
            "evaluation_entries": evaluation_entries,
        }

        serialized_data = json.dumps(payload, default=self._fallback_encoder)

        response = requests.post(
            self.eval_runs_endpoint,
            data=serialized_data,
            headers=self.headers,
            verify=True,
            timeout=30,
        )

        if response.status_code != HTTPStatus.OK:
            raise Exception(f"HTTP {response.status_code} - {response.text}")

    def _fallback_encoder(self, obj: Any) -> str:
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

    def shutdown(self, timeout_millis: int = 30000) -> None:
        """
        Shutdown the exporter.

        Since this exporter sends data synchronously and doesn't have any
        background threads or resources, there's nothing to clean up.

        Args:
            timeout_millis: Timeout in milliseconds (ignored)
        """
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """
        Force flush any pending requests.

        Since this exporter sends data synchronously in the export() method
        and doesn't have any internal buffering, there's nothing to flush.

        Args:
            timeout_millis: Timeout in milliseconds (ignored)

        Returns:
            True indicating success (no-op)
        """
        return True
