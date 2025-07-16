from __future__ import annotations

from judgeval.common.logger import judgeval_logger
from judgeval.common.api import JudgmentApiClient
from judgeval.common.api.constants import EvaluationEntryResponse
from judgeval.data import TraceSpan
from judgeval.evaluation_run import EvaluationRun


import atexit
import queue
import threading
import time
from typing import Any, Dict, List, cast


class BackgroundSpanService:
    """
    Background service for queueing and batching trace spans for efficient saving.

    This service:
    - Queues spans as they complete
    - Batches them for efficient network usage
    - Sends spans periodically or when batches reach a certain size
    - Handles automatic flushing when the main event terminates
    """

    def __init__(
        self,
        judgment_api_key: str,
        organization_id: str,
        batch_size: int = 10,
        flush_interval: float = 5.0,
        num_workers: int = 1,
    ):
        """
        Initialize the background span service.

        Args:
            judgment_api_key: API key for Judgment service
            organization_id: Organization ID
            batch_size: Number of spans to batch before sending (default: 10)
            flush_interval: Time in seconds between automatic flushes (default: 5.0)
            num_workers: Number of worker threads to process the queue (default: 1)
        """
        self.api_client = JudgmentApiClient(judgment_api_key, organization_id)
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.num_workers = max(1, num_workers)

        self._span_queue: queue.Queue[Dict[str, Any]] = queue.Queue()

        self._worker_threads: List[threading.Thread] = []
        self._shutdown_event = threading.Event()

        atexit.register(self.shutdown)

        self._start_workers()

    def _start_workers(self):
        """Start the background worker threads."""
        for i in range(self.num_workers):
            if len(self._worker_threads) < self.num_workers:
                worker_thread = threading.Thread(
                    target=self._worker_loop, daemon=True, name=f"SpanWorker-{i + 1}"
                )
                worker_thread.start()
                self._worker_threads.append(worker_thread)

    def _worker_loop(self):
        """Main worker loop that processes spans in batches."""
        batch = []
        last_flush_time = time.time()
        pending_task_count = 0

        while not self._shutdown_event.is_set() or self._span_queue.qsize() > 0:
            try:
                # First, do a blocking get to wait for at least one item
                if not batch:  # Only block if we don't have items already
                    try:
                        span_data = self._span_queue.get(timeout=1.0)
                        batch.append(span_data)
                        pending_task_count += 1
                    except queue.Empty:
                        # No new spans, continue to check for flush conditions
                        pass

                # Then, do non-blocking gets to drain any additional available items
                # up to our batch size limit
                while len(batch) < self.batch_size:
                    try:
                        span_data = self._span_queue.get_nowait()  # Non-blocking
                        batch.append(span_data)
                        pending_task_count += 1
                    except queue.Empty:
                        # No more items immediately available
                        break

                current_time = time.time()
                should_flush = len(batch) >= self.batch_size or (
                    batch and (current_time - last_flush_time) >= self.flush_interval
                )

                if should_flush and batch:
                    self._send_batch(batch)

                    # Only mark tasks as done after successful sending
                    for _ in range(pending_task_count):
                        self._span_queue.task_done()
                    pending_task_count = 0

                    batch.clear()
                    last_flush_time = current_time

            except Exception as e:
                judgeval_logger.warning(f"Error in span service worker loop: {e}")
                # On error, still need to mark tasks as done to prevent hanging
                for _ in range(pending_task_count):
                    self._span_queue.task_done()
                pending_task_count = 0
                batch.clear()

        # Final flush on shutdown
        if batch:
            self._send_batch(batch)
            # Mark remaining tasks as done
            for _ in range(pending_task_count):
                self._span_queue.task_done()

    def _send_batch(self, batch: List[Dict[str, Any]]):
        """
        Send a batch of spans to the server.

        Args:
            batch: List of span dictionaries to send
        """
        if not batch:
            return

        try:
            spans_to_send = []
            evaluation_runs_to_send = []

            for item in batch:
                if item["type"] == "span":
                    spans_to_send.append(item["data"])
                elif item["type"] == "evaluation_run":
                    evaluation_runs_to_send.append(item["data"])

            if spans_to_send:
                self._send_spans_batch(spans_to_send)

            if evaluation_runs_to_send:
                self._send_evaluation_runs_batch(evaluation_runs_to_send)

        except Exception as e:
            judgeval_logger.warning(f"Failed to send batch: {e}")

    def _send_spans_batch(self, spans: List[Dict[str, Any]]):
        """Send a batch of spans to the spans endpoint."""
        try:
            self.api_client.send_spans_batch(spans)
        except Exception as e:
            judgeval_logger.warning(f"Failed to send spans batch: {e}")

    def _send_evaluation_runs_batch(self, evaluation_runs: List[Dict[str, Any]]):
        """Send a batch of evaluation runs with their associated span data to the endpoint."""
        evaluation_entries: List[EvaluationEntryResponse] = []
        for eval_data in evaluation_runs:
            entry: EvaluationEntryResponse = {
                "evaluation_run": {
                    key: value
                    for key, value in eval_data.items()
                    if key not in ["associated_span_id", "span_data", "queued_at"]
                },
                "associated_span": {
                    "span_id": eval_data.get("associated_span_id"),
                    "span_data": eval_data.get("span_data"),
                },
                "queued_at": cast(float, eval_data.get("queued_at")),
            }
            evaluation_entries.append(entry)

        try:
            self.api_client.send_evaluation_runs_batch(evaluation_entries)
        except Exception as e:
            judgeval_logger.warning(f"Failed to send evaluation runs batch: {e}")

    def queue_span(self, span: TraceSpan, span_state: str = "input"):
        """
        Queue a span for background sending.

        Args:
            span: The TraceSpan object to queue
            span_state: State of the span ("input", "output", "completed")
        """
        if not self._shutdown_event.is_set():
            # Set update_id to ending number when span is completed, otherwise increment
            if span_state == "completed":
                span.set_update_id_to_ending_number()
            else:
                span.increment_update_id()

            span_data = {
                "type": "span",
                "data": {
                    **span.model_dump(),
                    "span_state": span_state,
                    "queued_at": time.time(),
                },
            }
            self._span_queue.put(span_data)

    def queue_evaluation_run(
        self, evaluation_run: EvaluationRun, span_id: str, span_data: TraceSpan
    ):
        """
        Queue an evaluation run for background sending.

        Args:
            evaluation_run: The EvaluationRun object to queue
            span_id: The span ID associated with this evaluation run
            span_data: The span data at the time of evaluation (to avoid race conditions)
        """
        if not self._shutdown_event.is_set():
            eval_data = {
                "type": "evaluation_run",
                "data": {
                    **evaluation_run.model_dump(),
                    "associated_span_id": span_id,
                    "span_data": span_data.model_dump(),  # Include span data to avoid race conditions
                    "queued_at": time.time(),
                },
            }
            self._span_queue.put(eval_data)

    def flush(self):
        """Force immediate sending of all queued spans."""
        try:
            # Wait for the queue to be processed
            self._span_queue.join()
        except Exception as e:
            judgeval_logger.warning(f"Error during flush: {e}")

    def shutdown(self):
        """Shutdown the background service and flush remaining spans."""
        if self._shutdown_event.is_set():
            return

        try:
            # Signal shutdown to stop new items from being queued
            self._shutdown_event.set()

            # Try to flush any remaining spans
            try:
                self.flush()
            except Exception as e:
                judgeval_logger.warning(f"Error during final flush: {e}")
        except Exception as e:
            judgeval_logger.warning(f"Error during BackgroundSpanService shutdown: {e}")
        finally:
            # Clear the worker threads list (daemon threads will be killed automatically)
            self._worker_threads.clear()

    def get_queue_size(self) -> int:
        """Get the current size of the span queue."""
        return self._span_queue.qsize()
