import litellm
from litellm.integrations.custom_logger import CustomLogger
from judgeval.common.tracer import TraceClient, TraceSpan, current_span_var
from typing import Optional
import time
import uuid


class JudgevalLitellmCallbackHandler(CustomLogger):

    def __init__(self, tracer):
        self.tracer = tracer
        self._current_span_id: Optional[str] = None
        self._current_trace_client: Optional[TraceClient] = None

    def log_pre_api_call(self, model, messages, kwargs):
        """Start span for LLM call"""
        print(f"[LiteLLM] log_pre_api_call called for model: {model}")

        trace_client = self.tracer.get_current_trace()
        if not trace_client:
            print(f"[LiteLLM] No trace client found, exiting")
            return

        # Store trace client reference so it's available even if context is cleared
        self._current_trace_client = trace_client

        # Signal that a LiteLLM operation is starting
        self.tracer.litellm_operation_started()
        print(f"[LiteLLM] litellm_operation_started() called")

        span_id = str(uuid.uuid4())
        self._current_span_id = span_id
        print(f"[LiteLLM] Created span ID: {span_id}")

        parent_span_id = current_span_var.get()
        depth = 0
        if parent_span_id and parent_span_id in trace_client.span_id_to_span:
            depth = trace_client.span_id_to_span[parent_span_id].depth + 1
        print(f"[LiteLLM] Parent span: {parent_span_id}, depth: {depth}")

        span = TraceSpan(
            span_id=span_id,
            trace_id=trace_client.trace_id,
            parent_span_id=parent_span_id,
            function=f"LiteLLM-{model}",
            depth=depth,
            created_at=time.time(),
            span_type="llm"
        )
        span.inputs = {"model": model, "messages": messages}
        print(
            f"[LiteLLM] Created span object: {span.span_id}, function: {span.function}")

        trace_client.add_span(span)
        print(f"[LiteLLM] Added span to trace client")

        # Verify span was added
        stored_span = trace_client.span_id_to_span.get(span_id)
        if stored_span:
            print(
                f"[LiteLLM] Verified span added to trace - ID: {stored_span.span_id}")
        else:
            print(f"[LiteLLM] WARNING: Span not found in trace after adding!")

    def log_post_api_call(self, kwargs, response_obj, start_time, end_time):
        """Post-processing - no action needed"""
        pass

    def _finish_span(self, response_obj, start_time, end_time, error=None):
        """Complete span with results"""
        print(f"[LiteLLM] _finish_span called with error={error is not None}")

        # Use stored trace client instead of context variable
        trace_client = self._current_trace_client
        if not trace_client or not self._current_span_id:
            print(
                f"[LiteLLM] Early exit - trace_client={trace_client is not None}, span_id={self._current_span_id}")
            self.tracer.litellm_operation_completed()
            return

        span = trace_client.span_id_to_span.get(self._current_span_id)
        if not span:
            print(f"[LiteLLM] Span not found for ID: {self._current_span_id}")
            self.tracer.litellm_operation_completed()
            return

        print(
            f"[LiteLLM] Found span: {span.span_id}, function: {span.function}")
        print(
            f"[LiteLLM] Before update - duration: {getattr(span, 'duration', 'None')}, output: {getattr(span, 'output', 'None')}")

        duration = None
        if start_time and end_time:
            if hasattr(start_time, 'timestamp'):
                duration = (end_time - start_time).total_seconds()
            else:
                duration = end_time - start_time
            print(f"[LiteLLM] Calculated duration: {duration} seconds")

        output = error
        if not error and response_obj and hasattr(response_obj, 'choices') and response_obj.choices:
            try:
                output = response_obj.choices[0].message.content
                print(
                    f"[LiteLLM] Extracted output: {output[:100] if output else 'None'}...")
            except:
                output = str(response_obj)
                print(f"[LiteLLM] Fallback output: {str(output)[:100]}...")
        else:
            print(f"[LiteLLM] Using error as output: {output}")

        # Update span
        span.duration = duration
        span.output = output
        print(
            f"[LiteLLM] After update - duration: {span.duration}, output: {span.output[:100] if span.output else 'None'}...")

        # Verify the span is in the trace client's dictionary
        stored_span = trace_client.span_id_to_span.get(self._current_span_id)
        if stored_span:
            print(
                f"[LiteLLM] Verified span in trace - duration: {stored_span.duration}, output: {stored_span.output[:100] if stored_span.output else 'None'}...")
        else:
            print(f"[LiteLLM] WARNING: Span not found in trace client after update!")

        # Check if span object is the same reference
        print(
            f"[LiteLLM] Span object identity check: span is stored_span = {span is stored_span}")

        # Clean up references
        self._current_span_id = None
        self._current_trace_client = None
        print(f"[LiteLLM] Span update completed, calling litellm_operation_completed()")

        self.tracer.litellm_operation_completed()

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Handle successful LLM call"""
        print(f"[LiteLLM] log_success_event called")
        self._finish_span(response_obj, start_time, end_time)

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """Handle failed LLM call"""
        print(f"[LiteLLM] log_failure_event called")
        error = kwargs.get("exception", response_obj)
        self._finish_span(response_obj, start_time, end_time, error=error)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Handle successful async LLM call"""
        print(f"[LiteLLM] async_log_success_event called")
        self.log_success_event(kwargs, response_obj, start_time, end_time)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """Handle failed async LLM call"""
        print(f"[LiteLLM] async_log_failure_event called")
        self.log_failure_event(kwargs, response_obj, start_time, end_time)

    def get_trace_client(self) -> Optional[TraceClient]:
        """Get current trace client for manual operations"""
        return self.tracer.get_current_trace()
