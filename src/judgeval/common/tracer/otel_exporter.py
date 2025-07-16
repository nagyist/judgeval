"""
Custom OpenTelemetry exporter for Judgment API.

This exporter sends spans to the Judgment API using the existing format,
replacing the manual BackgroundSpanService with OpenTelemetry's robust
batching, retry, and error handling mechanisms.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from http import HTTPStatus
from typing import Any, Dict, List, Sequence

from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.trace import ReadableSpan
from requests import RequestException

from judgeval.common.tracer.span_transformer import SpanTransformer
from judgeval.constants import (
    JUDGMENT_TRACES_EVALUATION_RUNS_BATCH_API_URL,
    JUDGMENT_TRACES_SPANS_BATCH_API_URL,
)
from judgeval.utils.requests import requests


class JudgmentAPISpanExporter(SpanExporter):
    """
    Custom OpenTelemetry exporter that sends spans to Judgment API in existing format.

    This exporter maintains compatibility with the existing API format while leveraging
    OpenTelemetry's robust processing pipeline for batching, retries, and error handling.
    """

    def __init__(
        self,
        judgment_api_key: str,
        organization_id: str,
        spans_endpoint: str = JUDGMENT_TRACES_SPANS_BATCH_API_URL,
        eval_runs_endpoint: str = JUDGMENT_TRACES_EVALUATION_RUNS_BATCH_API_URL,
        max_workers: int = 4,
    ):
        """
        Initialize the Judgment API exporter.

        Args:
            judgment_api_key: API key for Judgment service
            organization_id: Organization ID
            spans_endpoint: API endpoint for spans batch
            eval_runs_endpoint: API endpoint for evaluation runs batch
            max_workers: Maximum number of concurrent HTTP requests
        """
        self.judgment_api_key = judgment_api_key
        self.organization_id = organization_id
        self.spans_endpoint = spans_endpoint
        self.eval_runs_endpoint = eval_runs_endpoint
        self.max_workers = max_workers
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {judgment_api_key}",
            "X-Organization-Id": organization_id,
        }

        # Thread pool for handling HTTP requests
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._pending_futures: List[Future[Any]] = []
        self._lock = threading.Lock()
        self._shutdown = False

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """
        Export spans using existing Judgment API format.

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

            # Submit HTTP requests to thread pool
            with self._lock:
                try:
                    if spans_data:
                        future = self._executor.submit(
                            self._send_spans_batch, spans_data
                        )
                        self._pending_futures.append(future)

                    if eval_runs_data:
                        future = self._executor.submit(
                            self._send_evaluation_runs_batch, eval_runs_data
                        )
                        self._pending_futures.append(future)

                except RuntimeError as e:
                    if "cannot schedule new futures after interpreter shutdown" in str(
                        e
                    ) or "cannot schedule new futures after shutdown" in str(e):
                        return SpanExportResult.SUCCESS
                    else:
                        raise

                # Clean up completed futures
                self._pending_futures = [
                    f for f in self._pending_futures if not f.done()
                ]

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

        try:
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

        except RequestException:
            raise
        except Exception:
            raise

    def _send_evaluation_runs_batch(self, evaluation_runs: List[Dict[str, Any]]):
        """
        Send a batch of evaluation runs to the evaluation runs endpoint.

        Args:
            evaluation_runs: List of evaluation run dictionaries to send
        """
        payload = {
            "evaluation_runs": [eval_run["data"] for eval_run in evaluation_runs],
            "organization_id": self.organization_id,
        }

        try:
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

        except RequestException:
            raise
        except Exception:
            raise

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
        Shutdown the exporter and wait for all pending requests to complete.

        Args:
            timeout_millis: Timeout in milliseconds
        """
        with self._lock:
            self._shutdown = True

            # Wait for all pending futures to complete
            if self._pending_futures:
                try:
                    # Use as_completed with timeout
                    timeout_seconds = timeout_millis / 1000.0
                    for future in as_completed(
                        self._pending_futures, timeout=timeout_seconds
                    ):
                        try:
                            future.result()  # This will raise any exceptions
                        except Exception:
                            pass  # Ignore exceptions during shutdown
                except Exception:
                    pass  # Ignore timeout exceptions during shutdown

            # Shutdown the executor
            self._executor.shutdown(wait=True)

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """
        Force flush all pending requests.

        Args:
            timeout_millis: Timeout in milliseconds

        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            if not self._pending_futures:
                return True

            try:
                # Wait for all pending futures to complete
                timeout_seconds = timeout_millis / 1000.0
                completed_futures = []

                for future in as_completed(
                    self._pending_futures, timeout=timeout_seconds
                ):
                    try:
                        future.result()
                        completed_futures.append(future)
                    except Exception:
                        return False

                # Remove completed futures
                self._pending_futures = [
                    f for f in self._pending_futures if f not in completed_futures
                ]
                return True

            except Exception:
                return False
