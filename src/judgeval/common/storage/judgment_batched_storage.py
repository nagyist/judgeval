from datetime import datetime
from http import HTTPStatus
import os
import time
from typing import List
from judgeval.common.api.client import APIClient
from judgeval.common.storage.storage import ABCStorage
from judgeval.common.tracer.model import TraceSave, SpansBatchRequest, SpanBatchItem
from rich import print as rprint

from judgeval.common.tracer.utils import fallback_encoder


class JudgmentBatchedStorage(ABCStorage):
    """
    Batched storage implementation for judgment data using the spans batch API.
    """

    client: APIClient

    def __init__(self):
        super().__init__()

        JUDGMENT_API_KEY = os.environ.get("JUDGMENT_API_KEY")
        JUDGMENT_ORG_ID = os.environ.get("JUDGMENT_ORG_ID")

        if not JUDGMENT_API_KEY or not JUDGMENT_ORG_ID:
            raise ValueError(
                "Environment variables JUDGMENT_API_KEY and JUDGMENT_ORG_ID must be set."
            )

        self.client = APIClient(api_key=JUDGMENT_API_KEY, org_id=JUDGMENT_ORG_ID)

    def save_trace(self, trace_data: TraceSave, trace_id: str, project_name: str):
        """
        Save trace data using the spans batch API endpoint.

        Args:
            trace_data (TraceSave): The trace data to be saved.
            trace_id (str): Unique identifier for the trace.
            project_name (str): Name of the project associated with the trace.

        Returns:
            dict: The server response.
        """
        # Convert TraceSpan objects to SpanBatchItem objects
        batch_spans: List[SpanBatchItem] = []

        for trace_span in trace_data.trace_spans:
            span_batch_item = SpanBatchItem(
                span_id=trace_span.span_id,
                trace_id=trace_span.trace_id,
                function=trace_span.function,
                depth=trace_span.depth,
                created_at=datetime.utcfromtimestamp(time.time()).isoformat(),
                parent_span_id=trace_span.parent_span_id,
                span_type=trace_span.span_type,
                inputs=trace_span.inputs,
                output=trace_span.output,
                error=trace_span.error,
                usage=trace_span.usage.model_dump() if trace_span.usage else None,
                duration=trace_span.duration,
                annotation=trace_span.annotation,
                expected_tools=(
                    [tool.model_dump() for tool in trace_span.expected_tools]
                    if trace_span.expected_tools
                    else None
                ),
                additional_metadata=trace_span.additional_metadata,
                has_evaluation=trace_span.has_evaluation,
                agent_name=trace_span.agent_name,
                state_before=trace_span.state_before,
                state_after=trace_span.state_after,
                span_state="completed",
                queued_at=0.0,
            )
            batch_spans.append(span_batch_item)

        spans_batch_request = SpansBatchRequest(
            spans=batch_spans, organization_id=self.client.org_id
        )

        trace_json = trace_data.model_dump_json(fallback=fallback_encoder)
        batch_json = spans_batch_request.model_dump_json(fallback=fallback_encoder)
        response = self.client.do_post("/traces/upsert/", trace_json)

        server_response = response.json()

        if "ui_results_url" in server_response:
            pretty_str = f"\n🔍 You can view your trace data here: [rgb(106,0,255)][link={server_response['ui_results_url']}]View Trace[/link]\n"
            rprint(pretty_str)

        response = self.client.do_post("/traces/spans/batch/", batch_json)

        if response.status_code == HTTPStatus.BAD_REQUEST:
            raise ValueError(
                f"Failed to save batch spans data: Check your span data for conflicts: {response.text}"
            )
        elif response.status_code != HTTPStatus.OK:
            raise ValueError(f"Failed to save batch spans data: {response.text}")

        return server_response
