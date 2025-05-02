from typing import Any, Dict, List, Optional, Sequence
from uuid import UUID
import time
import uuid
import traceback # For detailed error logging
import contextvars # <--- Import contextvars

from judgeval.common.tracer import TraceClient, TraceEntry, Tracer, SpanType

from langchain_core.language_models import BaseChatModel
from langchain_huggingface import ChatHuggingFace
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.callbacks.base import AsyncCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.outputs import LLMResult
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.tool import ToolMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.documents import Document

# --- Get context vars from tracer module ---
# Assuming tracer.py defines these and they are accessible
# If not, redefine them here or adjust import
from judgeval.common.tracer import current_span_var, current_trace_var # <-- Import current_trace_var

# --- Constants for Logging ---
HANDLER_LOG_PREFIX = "[JudgevalHandlerLog]"

class JudgevalCallbackHandler(BaseCallbackHandler):
    """
    LangChain Callback Handler using run_id/parent_run_id for hierarchy.
    Manages its own internal TraceClient instance created upon first use.
    Includes verbose logging and defensive checks.
    """
    # Make all properties ignored by LangChain's callback system
    # to prevent unexpected serialization issues.
    lc_serializable = False
    lc_kwargs = {}

    def __init__(self, tracer: Tracer):
        # --- Enhanced Logging ---
        instance_id = id(self)
        print(f"{HANDLER_LOG_PREFIX} *** Handler instance {instance_id} __init__ called. ***")
        # --- End Enhanced Logging ---
        self.tracer = tracer
        self._trace_client: Optional[TraceClient] = None
        self._run_id_to_span_id: Dict[UUID, str] = {}
        self._span_id_to_start_time: Dict[str, float] = {}
        self._span_id_to_depth: Dict[str, int] = {}
        self._root_run_id: Optional[UUID] = None
        self._trace_context_token: Optional[contextvars.Token] = None # NEW: Store trace context token

        self.executed_nodes: List[str] = []
        self.executed_tools: List[str] = []
        self.executed_node_tools: List[str] = []

    def _ensure_trace_client(self, run_id: UUID, event_name: str) -> Optional[TraceClient]:
        """Ensures the internal trace client is initialized. Returns client or None."""
        # --- Enhanced Logging ---
        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Handler {handler_instance_id}]"
        # --- End Enhanced Logging ---
        if self._trace_client is None:
            print(f"{log_prefix} Trace client is None. Attempting initialization triggered by {event_name} (run_id: {run_id})...")
            trace_id = str(uuid.uuid4())
            project = self.tracer.project_name
            # Use the event name as the initial trace name
            try:
                # --- Enhanced Logging ---
                print(f"{log_prefix} BEFORE TraceClient creation attempt.")
                # --- End Enhanced Logging ---
                client_instance = TraceClient(
                    self.tracer, trace_id, event_name, project_name=project,
                    overwrite=False, rules=self.tracer.rules,
                    enable_monitoring=self.tracer.enable_monitoring,
                    enable_evaluations=self.tracer.enable_evaluations
                )
                # --- Enhanced Logging ---
                print(f"{log_prefix} AFTER TraceClient creation attempt. Success? {'Yes' if client_instance else 'No'}")
                self._trace_client = client_instance # Assign the created instance
                # --- End Enhanced Logging ---

                if self._trace_client: # Check if assignment was successful
                     print(f"{log_prefix} Initialized TraceClient: ID={self._trace_client.trace_id}, Name='{event_name}', Instance ID={id(self._trace_client)}")
                     # Tentatively set root if not already set
                     if self._root_run_id is None:
                         self._root_run_id = run_id
                         print(f"{log_prefix} Tentatively set root run ID: {self._root_run_id}")
                else:
                    # This case should ideally not be reached if TraceClient creation succeeded
                    # but handles potential edge cases or if TraceClient init itself returned None implicitly.
                    print(f"{log_prefix} FATAL: TraceClient creation appears to have succeeded but self._trace_client is still None after assignment.")
                    return None
            except Exception as e:
                print(f"{log_prefix} FATAL: Failed to initialize TraceClient: {e}")
                print(traceback.format_exc()) # Print stack trace
                self._trace_client = None # Ensure it's None on failure
                return None # Failed to create client
        # else:
             # --- Enhanced Logging ---
             # print(f"{log_prefix} Trace client already exists (ID: {self._trace_client.trace_id}, Instance ID: {id(self._trace_client)}) for event {event_name} (run_id: {run_id}).")
             # --- End Enhanced Logging ---
        return self._trace_client

    def _log(self, message: str):
        """Helper for consistent logging format."""
        print(f"{HANDLER_LOG_PREFIX} {message}")

    def _start_span_tracking(
        self,
        trace_client: TraceClient, # Expect a valid client
        run_id: UUID,
        parent_run_id: Optional[UUID],
        name: str,
        span_type: SpanType = "span",
        inputs: Optional[Dict[str, Any]] = None
    ):
        self._log(f"_start_span_tracking called for: name='{name}', run_id={run_id}, parent_run_id={parent_run_id}, span_type={span_type}")

        # --- Add explicit check for trace_client ---
        if not trace_client:
            # --- Enhanced Logging ---
            handler_instance_id = id(self)
            log_prefix = f"{HANDLER_LOG_PREFIX} [Handler {handler_instance_id}]"
            # --- End Enhanced Logging ---
            self._log(f"{log_prefix} FATAL ERROR in _start_span_tracking: trace_client argument is None for name='{name}', run_id={run_id}. Aborting span start.")
            return
        # --- End check ---
        # --- Enhanced Logging ---
        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Handler {handler_instance_id}]"
        trace_client_instance_id = id(trace_client) if trace_client else 'None'
        print(f"{log_prefix} _start_span_tracking: Using TraceClient ID: {trace_client_instance_id}")
        # --- End Enhanced Logging ---

        start_time = time.time()
        span_id = str(uuid.uuid4())
        parent_span_id: Optional[str] = None
        current_depth = 0

        if parent_run_id and parent_run_id in self._run_id_to_span_id:
            parent_span_id = self._run_id_to_span_id[parent_run_id]
            if parent_span_id in self._span_id_to_depth:
                parent_depth = self._span_id_to_depth[parent_span_id]
                current_depth = parent_depth + 1
                self._log(f"  Found parent span_id={parent_span_id} with depth={parent_depth}. New depth={current_depth}.")
            else:
                self._log(f"  WARNING: Parent span depth not found for parent_span_id: {parent_span_id}. Setting depth to 0.")
                current_depth = 0
        elif parent_run_id:
            self._log(f"  WARNING: parent_run_id {parent_run_id} provided for '{name}' ({run_id}) but parent span not tracked. Treating as depth 0.")
        else:
            self._log(f"  No parent_run_id provided. Treating '{name}' as depth 0.")

        self._run_id_to_span_id[run_id] = span_id
        self._span_id_to_start_time[span_id] = start_time
        self._span_id_to_depth[span_id] = current_depth
        self._log(f"  Tracking new span: span_id={span_id}, depth={current_depth}")

        try:
            trace_client.add_entry(TraceEntry(
                type="enter", span_id=span_id, trace_id=trace_client.trace_id,
                parent_span_id=parent_span_id, function=name, depth=current_depth,
                message=name, created_at=start_time, span_type=span_type
            ))
            self._log(f"  Added 'enter' entry for span_id={span_id}")
        except Exception as e:
            self._log(f"  ERROR adding 'enter' entry for span_id {span_id}: {e}")
            print(traceback.format_exc())

        if inputs:
            # Pass the already validated trace_client
            self._record_input_data(trace_client, run_id, inputs)

        # --- Set context variable ONLY for chain (node) spans ---
        if span_type == "chain":
            try:
                # Set current_span_var, but don't store the token as we won't reset it here
                current_span_var.set(span_id)
                self._log(f"  Set current_span_var to {span_id} for run_id {run_id} (type: {span_type})")
            except Exception as e:
                self._log(f"  ERROR setting current_span_var for run_id {run_id}: {e}")
        # else: # For LLM, tool spans etc handled by callbacks, DO NOT set current_span_var
            # self._log(f"  Skipping current_span_var set for run_id {run_id} (type: {span_type})")
        # --- END Context Var Logic ---

        try:
            # TODO: Check if trace_client.add_entry needs await if TraceClient becomes async
            trace_client.add_entry(TraceEntry(
                type="enter", span_id=span_id, trace_id=trace_client.trace_id,
                parent_span_id=parent_span_id, function=name, depth=current_depth,
                message=name, created_at=start_time, span_type=span_type
            ))
            self._log(f"  Added 'enter' entry for span_id={span_id}")
        except Exception as e:
            self._log(f"  ERROR adding 'enter' entry for span_id {span_id}: {e}")
            print(traceback.format_exc())

        if inputs:
            # _record_input_data is also sync for now
            self._record_input_data(trace_client, run_id, inputs)

    def _end_span_tracking(
        self,
        trace_client: TraceClient, # Expect a valid client
        run_id: UUID,
        span_type: SpanType = "span",
        outputs: Optional[Any] = None,
        error: Optional[BaseException] = None
    ):
        self._log(f"_end_span_tracking called for: run_id={run_id}, span_type={span_type}")

        # --- Add explicit check for trace_client ---
        if not trace_client:
             # --- Enhanced Logging ---
            handler_instance_id = id(self)
            log_prefix = f"{HANDLER_LOG_PREFIX} [Handler {handler_instance_id}]"
            # --- End Enhanced Logging ---
            self._log(f"{log_prefix} FATAL ERROR in _end_span_tracking: trace_client argument is None for run_id={run_id}. Aborting span end.")
            return
        # --- End check ---
        # --- Enhanced Logging ---
        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Handler {handler_instance_id}]"
        trace_client_instance_id = id(trace_client) if trace_client else 'None'
        print(f"{log_prefix} _end_span_tracking: Using TraceClient ID: {trace_client_instance_id}")
        # --- End Enhanced Logging ---

        if run_id not in self._run_id_to_span_id:
            self._log(f"  WARNING: Attempting to end span for untracked run_id: {run_id}")
            return

        span_id = self._run_id_to_span_id[run_id]
        start_time = self._span_id_to_start_time.get(span_id)
        depth = self._span_id_to_depth.get(span_id, 0)
        duration = time.time() - start_time if start_time is not None else None
        self._log(f"  Ending span_id={span_id}. Start time={start_time}, Duration={duration}, Depth={depth}")

        # Record output/error first
        if error:
            self._log(f"  Recording error for span_id={span_id}: {error}")
            self._record_output_data(trace_client, run_id, error)
        elif outputs is not None:
            # Log output carefully, might be large
            output_repr = repr(outputs)
            log_output = (output_repr[:100] + '...') if len(output_repr) > 103 else output_repr
            self._log(f"  Recording output for span_id={span_id}: {log_output}")
            self._record_output_data(trace_client, run_id, outputs)

        # Add exit entry
        entry_function_name = "unknown"
        try:
            # Ensure entries list is accessible
            if hasattr(trace_client, 'entries') and trace_client.entries:
                entry_function_name = next((e.function for e in reversed(trace_client.entries) if e.span_id == span_id and e.type == "enter"), "unknown")
            else:
                self._log(f"  WARNING: Cannot determine function name for exit span_id {span_id}, trace_client.entries missing or empty.")
        except Exception as e:
            self._log(f"  ERROR finding function name for exit entry span_id {span_id}: {e}")
            print(traceback.format_exc())

        try:
            trace_client.add_entry(TraceEntry(
                type="exit", span_id=span_id, trace_id=trace_client.trace_id,
                depth=depth, created_at=time.time(), duration=duration,
                span_type=span_type, function=entry_function_name
            ))
            self._log(f"  Added 'exit' entry for span_id={span_id}, function='{entry_function_name}'")
        except Exception as e:
            self._log(f"  ERROR adding 'exit' entry for span_id {span_id}: {e}")
            print(traceback.format_exc())

        # Clean up dictionaries
        if span_id in self._span_id_to_start_time: del self._span_id_to_start_time[span_id]
        if span_id in self._span_id_to_depth: del self._span_id_to_depth[span_id]
        # Keep run_id_to_span_id for parent lookups until trace ends

        # --- NEW: Reset context variable --- 
        token = self._run_id_to_context_token.pop(run_id, None)
        if token:
            try:
                current_span_var.reset(token)
                self._log(f"  Reset current_span_var for run_id {run_id} (was {span_id})")
            except LookupError:
                # This error occurs if the token was created in a different context
                # Log a warning instead of erroring out, as the main goal was setting the context during the span
                self._log(f"  [WARN] Could not reset current_span_var for run_id {run_id} (token from different context).")
            except Exception as e:
                 # Log other potential errors during reset
                 self._log(f"  ERROR resetting current_span_var for run_id {run_id}: {e}")
        # --- END NEW --- 

        # Check if this is the root run ending
        if run_id == self._root_run_id:
            self._log(f"Root run {run_id} finished. Attempting to save trace...")
            if self._trace_client:
                try:
                    trace_id, _ = self._trace_client.save(overwrite=True)
                    self._log(f"Trace {trace_id} successfully saved.")
                except Exception as e:
                    self._log(f"ERROR saving trace {self._trace_client.trace_id}: {e}")
                    print(traceback.format_exc())
                finally:
                    # Reset state AFTER attempting save
                    self._log(f"Resetting handler state after root run end.")
                    self._trace_client = None
                    self._run_id_to_span_id = {}
                    self._span_id_to_start_time = {}
                    self._span_id_to_depth = {}
                    self._root_run_id = None
            else:
                self._log(f"  WARNING: Root run {run_id} ended, but trace client was already None. Cannot save trace.")
        # else:
        #     self._log(f"  Run {run_id} ended, but it was not the root run ({self._root_run_id}).")

    def _record_input_data(self,
                           trace_client: TraceClient,
                           run_id: UUID,
                           inputs: Dict[str, Any]):
        self._log(f"_record_input_data called for run_id={run_id}")
        if run_id not in self._run_id_to_span_id:
            self._log(f"  WARNING: Attempting to record input for untracked run_id: {run_id}")
            return
        if not trace_client:
             self._log(f"  ERROR: TraceClient is None when trying to record input for run_id={run_id}")
             return

        span_id = self._run_id_to_span_id[run_id]
        depth = self._span_id_to_depth.get(span_id, 0)
        self._log(f"  Found span_id={span_id}, depth={depth} for run_id={run_id}")

        function_name = "unknown"
        span_type: SpanType = "span"
        try:
            # Find the corresponding 'enter' entry to get the function name and span type
            enter_entry = next((e for e in reversed(trace_client.entries) if e.span_id == span_id and e.type == "enter"), None)
            if enter_entry:
                function_name = enter_entry.function
                span_type = enter_entry.span_type
                self._log(f"  Found function='{function_name}', span_type='{span_type}' for input span_id={span_id}")
            else:
                self._log(f"  WARNING: Could not find 'enter' entry for input span_id={span_id}")
        except Exception as e:
            self._log(f"  ERROR finding enter entry for input span_id {span_id}: {e}")
            print(traceback.format_exc())

        try:
            input_entry = TraceEntry(
                type="input",
                span_id=span_id,
                trace_id=trace_client.trace_id,
                parent_span_id=next((e.parent_span_id for e in reversed(trace_client.entries) if e.span_id == span_id and e.type == "enter"), None), # Get parent from enter entry
                function=function_name,
                depth=depth,
                message=f"Input to {function_name}",
                created_at=time.time(),
                inputs=inputs,
                span_type=span_type
            )
            trace_client.add_entry(input_entry)
            self._log(f"  Added 'input' entry directly for span_id={span_id}")
        except Exception as e:
            self._log(f"  ERROR adding 'input' entry directly for span_id {span_id}: {e}")
            print(traceback.format_exc())


    def _record_output_data(self,
                            trace_client: TraceClient,
                            run_id: UUID,
                            output: Any):
        self._log(f"_record_output_data called for run_id={run_id}")
        if run_id not in self._run_id_to_span_id:
            self._log(f"  WARNING: Attempting to record output for untracked run_id: {run_id}")
            return
        if not trace_client:
            self._log(f"  ERROR: TraceClient is None when trying to record output for run_id={run_id}")
            return

        span_id = self._run_id_to_span_id[run_id]
        depth = self._span_id_to_depth.get(span_id, 0)
        self._log(f"  Found span_id={span_id}, depth={depth} for run_id={run_id}")

        function_name = "unknown"
        span_type: SpanType = "span"
        try:
            # Find the corresponding 'enter' entry to get the function name and span type
            enter_entry = next((e for e in reversed(trace_client.entries) if e.span_id == span_id and e.type == "enter"), None)
            if enter_entry:
                function_name = enter_entry.function
                span_type = enter_entry.span_type
                self._log(f"  Found function='{function_name}', span_type='{span_type}' for output span_id={span_id}")
            else:
                 self._log(f"  WARNING: Could not find 'enter' entry for output span_id={span_id}")
        except Exception as e:
            self._log(f"  ERROR finding enter entry for output span_id {span_id}: {e}")
            print(traceback.format_exc())

        try:
            output_entry = TraceEntry(
                type="output",
                span_id=span_id,
                trace_id=trace_client.trace_id,
                parent_span_id=next((e.parent_span_id for e in reversed(trace_client.entries) if e.span_id == span_id and e.type == "enter"), None), # Get parent from enter entry
                function=function_name,
                depth=depth,
                message=f"Output from {function_name}",
                created_at=time.time(),
                output=output, # Langchain outputs are typically serializable directly
                span_type=span_type
            )
            trace_client.add_entry(output_entry)
            self._log(f"  Added 'output' entry directly for span_id={span_id}")
        except Exception as e:
            self._log(f"  ERROR adding 'output' entry directly for span_id {span_id}: {e}")
            print(traceback.format_exc())

    # --- Callback Methods ---
    # Each method now ensures the trace client exists before proceeding

    def on_retriever_start(self, serialized: Dict[str, Any], query: str, *, run_id: UUID, parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Handler {handler_instance_id}]"
        tc_id_on_entry = id(self._trace_client) if self._trace_client else 'None'
        print(f"{log_prefix} ENTERING on_retriever_start: name='{serialized.get('name')}', run_id={run_id}. Current TraceClient ID: {tc_id_on_entry}")

        try:
            name = f"RETRIEVER_{serialized.get('name', 'Generic').upper()}"
            trace_client = self._ensure_trace_client(run_id, name)
            if not trace_client:
                return # Error logged in _ensure_trace_client

            inputs = {
                'query': query, 'tags': tags, 'metadata': metadata,
                'kwargs': kwargs, 'serialized': serialized,
            }
            self._start_span_tracking(trace_client, run_id, parent_run_id, name, span_type="retriever", inputs=inputs)
        except Exception as e:
            tc_id_on_error = id(self._trace_client) if self._trace_client else 'None'
            self._log(f"{log_prefix} UNCAUGHT EXCEPTION in on_retriever_start for run_id={run_id} (TraceClient ID: {tc_id_on_error}): {e}")
            print(traceback.format_exc())

    def on_retriever_end(self, documents: Sequence[Document], *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> Any:
        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Handler {handler_instance_id}]"
        tc_id_on_entry = id(self._trace_client) if self._trace_client else 'None'
        print(f"{log_prefix} ENTERING on_retriever_end: run_id={run_id}. Current TraceClient ID: {tc_id_on_entry}")

        try:
            trace_client = self._ensure_trace_client(run_id, "RetrieverEnd")
            if not trace_client:
                return

            doc_summary = []
            for i, doc in enumerate(documents):
                doc_data = {"index": i, "page_content": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content, "metadata": doc.metadata}
                doc_summary.append(doc_data)
            outputs = {"document_count": len(documents), "documents": doc_summary, "kwargs": kwargs}
            self._end_span_tracking(trace_client, run_id, span_type="retriever", outputs=outputs)
        except Exception as e:
            tc_id_on_error = id(self._trace_client) if self._trace_client else 'None'
            self._log(f"{log_prefix} UNCAUGHT EXCEPTION in on_retriever_end for run_id={run_id} (TraceClient ID: {tc_id_on_error}): {e}")
            print(traceback.format_exc())

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], *, run_id: UUID, parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        # Log entry point immediately
        # --- Enhanced Logging ---
        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Handler {handler_instance_id}]"
        tc_id_on_entry = id(self._trace_client) if self._trace_client else 'None'
        print(f"{log_prefix} ENTERING on_chain_start: name='{serialized.get('name')}', run_id={run_id}. Current TraceClient ID: {tc_id_on_entry}")
        # --- End Enhanced Logging ---
        # self._log(f"on_chain_start: name='{serialized.get('name')}', run_id={run_id}, parent_run_id={parent_run_id}")

        try: # Wrap main logic
            span_type: SpanType = "chain"
            name = serialized.get("name", "Unnamed Chain")

            trace_client = self._ensure_trace_client(run_id, name)
            if not trace_client:
                # _ensure_trace_client already logs errors if creation fails
                # self._log(f"  ERROR: Failed to get/create trace client in on_chain_start for '{name}'")
                return

            # Detect the main LangGraph execution start more reliably
            is_langgraph_root = kwargs.get('name') == 'LangGraph' and parent_run_id is None
            if is_langgraph_root:
                name = "LangGraph"
                span_type = "chain"
                self._log(f"  LangGraph Root Start Detected: run_id={run_id}")
                # Explicitly set root ID if this is the true root event
                if self._root_run_id is None or self._root_run_id != run_id:
                    self._log(f"  Setting root run ID to {run_id} for trace {trace_client.trace_id}")
                    self._root_run_id = run_id
                # Update trace name if it was initialized with a different name (optional)
                if trace_client.name != name:
                    self._log(f"  Updating trace name from '{trace_client.name}' to '{name}'")
                    trace_client.name = name

            node_name = metadata.get("langgraph_node") if metadata else None
            if node_name:
                name = node_name
                span_type = "chain"
                self._log(f"  LangGraph Node Start: '{name}', run_id={run_id}, parent_run_id={parent_run_id}")
                if name not in self.executed_nodes: self.executed_nodes.append(name)

            combined_inputs = {
                'inputs': inputs, 'tags': tags, 'metadata': metadata,
                'kwargs': kwargs, 'serialized': serialized,
            }
            # Pass the validated trace_client
            self._start_span_tracking(trace_client, run_id, parent_run_id, name, span_type=span_type, inputs=combined_inputs)
        except Exception as e:
            # --- Enhanced Logging ---
            handler_instance_id = id(self)
            log_prefix = f"{HANDLER_LOG_PREFIX} [Handler {handler_instance_id}]"
            tc_id_on_error = id(self._trace_client) if self._trace_client else 'None'
            # --- End Enhanced Logging ---
            self._log(f"{log_prefix} UNCAUGHT EXCEPTION in on_chain_start for run_id={run_id} (TraceClient ID: {tc_id_on_error}): {e}")
            print(traceback.format_exc()) # Print full traceback for debugging
            # Optionally re-raise or handle differently
            # raise e

    def on_chain_end(self, outputs: Dict[str, Any], *, run_id: UUID, parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None, **kwargs: Any) -> Any:
        # Log entry point immediately
        # --- Enhanced Logging ---
        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Handler {handler_instance_id}]"
        tc_id_on_entry = id(self._trace_client) if self._trace_client else 'None'
        print(f"{log_prefix} ENTERING on_chain_end: run_id={run_id}. Current TraceClient ID: {tc_id_on_entry}")
        # --- End Enhanced Logging ---
        # self._log(f"on_chain_end: run_id={run_id}")

        try: # Wrap main logic
            trace_client = self._ensure_trace_client(run_id, "ChainEnd")
            if not trace_client:
                # _ensure_trace_client already logs errors
                # self._log(f"  ERROR: Failed to get/create trace client in on_chain_end")
                return

            span_id = self._run_id_to_span_id.get(run_id)
            span_type: SpanType = "chain"
            if span_id:
                try:
                    # Safely access entries only if trace_client is valid and has entries
                    if hasattr(trace_client, 'entries') and trace_client.entries:
                        enter_entry = next((e for e in reversed(trace_client.entries) if e.span_id == span_id and e.type == "enter"), None)
                        if enter_entry:
                            span_type = enter_entry.span_type
                    else:
                         self._log(f"  WARNING: trace_client.entries not available when determining span_type for on_chain_end span_id={span_id}")
                except Exception as e:
                    self._log(f"  ERROR finding enter entry for span_id {span_id} in on_chain_end: {e}")

            combined_outputs = {"outputs": outputs, "tags": tags, "kwargs": kwargs}
            # Pass the validated trace_client
            self._end_span_tracking(trace_client, run_id, span_type=span_type, outputs=combined_outputs)
        except Exception as e:
            handler_instance_id = id(self)
            log_prefix = f"{HANDLER_LOG_PREFIX} [Handler {handler_instance_id}]"
            tc_id_on_error = id(self._trace_client) if self._trace_client else 'None'
            self._log(f"{log_prefix} UNCAUGHT EXCEPTION in on_chain_end for run_id={run_id} (TraceClient ID: {tc_id_on_error}): {e}")
            print(traceback.format_exc()) # Print full traceback for debugging
            # Optionally re-raise or handle differently
            # raise e

    def on_chain_error(self, error: BaseException, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> Any:
        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Handler {handler_instance_id}]"
        tc_id_on_entry = id(self._trace_client) if self._trace_client else 'None'
        print(f"{log_prefix} ENTERING on_chain_error: run_id={run_id}. Current TraceClient ID: {tc_id_on_entry}")

        try:
            trace_client = self._ensure_trace_client(run_id, "ChainError")
            if not trace_client:
                return

            span_id = self._run_id_to_span_id.get(run_id)
            span_type: SpanType = "chain"
            if span_id:
                try:
                     if hasattr(trace_client, 'entries') and trace_client.entries:
                        enter_entry = next((e for e in reversed(trace_client.entries) if e.span_id == span_id and e.type == "enter"), None)
                        if enter_entry:
                            span_type = enter_entry.span_type
                     else:
                         self._log(f"  WARNING: trace_client.entries not available when determining span_type for on_chain_error span_id={span_id}")
                except Exception as e:
                    self._log(f"  ERROR finding enter entry for span_id {span_id} in on_chain_error: {e}")

            self._end_span_tracking(trace_client, run_id, span_type=span_type, error=error)
        except Exception as e:
            tc_id_on_error = id(self._trace_client) if self._trace_client else 'None'
            self._log(f"{log_prefix} UNCAUGHT EXCEPTION in on_chain_error for run_id={run_id} (TraceClient ID: {tc_id_on_error}): {e}")
            print(traceback.format_exc())

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, *, run_id: UUID, parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None, inputs: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Handler {handler_instance_id}]"
        tc_id_on_entry = id(self._trace_client) if self._trace_client else 'None'
        name = serialized.get("name", "Unnamed Tool")
        print(f"{log_prefix} ENTERING on_tool_start: name='{name}', run_id={run_id}. Current TraceClient ID: {tc_id_on_entry}")

        try:
            trace_client = self._ensure_trace_client(run_id, name)
            if not trace_client:
                return

            combined_inputs = {
                'input_str': input_str, 'inputs': inputs, 'tags': tags,
                'metadata': metadata, 'kwargs': kwargs, 'serialized': serialized,
            }
            self._start_span_tracking(trace_client, run_id, parent_run_id, name, span_type="tool", inputs=combined_inputs)

            if name not in self.executed_tools: self.executed_tools.append(name)
            parent_node_name = None
            if parent_run_id and parent_run_id in self._run_id_to_span_id:
                parent_span_id = self._run_id_to_span_id[parent_run_id]
                try:
                    if hasattr(trace_client, 'entries') and trace_client.entries:
                        parent_enter_entry = next((e for e in reversed(trace_client.entries) if e.span_id == parent_span_id and e.type == "enter" and e.span_type == "chain"), None)
                        if parent_enter_entry:
                            parent_node_name = parent_enter_entry.function
                    else:
                        self._log(f"  WARNING: trace_client.entries not available when finding parent node for tool start {parent_span_id}")
                except Exception as e:
                    self._log(f"  ERROR finding parent node name for tool start span_id {parent_span_id}: {e}")
                    print(traceback.format_exc())

            node_tool = f"{parent_node_name}:{name}" if parent_node_name else name
            if node_tool not in self.executed_node_tools: self.executed_node_tools.append(node_tool)
            self._log(f"  Tracked node_tool: {node_tool}")
        except Exception as e:
            tc_id_on_error = id(self._trace_client) if self._trace_client else 'None'
            self._log(f"{log_prefix} UNCAUGHT EXCEPTION in on_tool_start for run_id={run_id} (TraceClient ID: {tc_id_on_error}): {e}")
            print(traceback.format_exc())

    def on_tool_end(self, output: Any, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> Any:
        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Handler {handler_instance_id}]"
        tc_id_on_entry = id(self._trace_client) if self._trace_client else 'None'
        print(f"{log_prefix} ENTERING on_tool_end: run_id={run_id}. Current TraceClient ID: {tc_id_on_entry}")

        try:
            trace_client = self._ensure_trace_client(run_id, "ToolEnd")
            if not trace_client:
                return
            outputs = {"output": output, "kwargs": kwargs}
            self._end_span_tracking(trace_client, run_id, span_type="tool", outputs=outputs)
        except Exception as e:
            tc_id_on_error = id(self._trace_client) if self._trace_client else 'None'
            self._log(f"{log_prefix} UNCAUGHT EXCEPTION in on_tool_end for run_id={run_id} (TraceClient ID: {tc_id_on_error}): {e}")
            print(traceback.format_exc())

    def on_tool_error(self, error: BaseException, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> Any:
        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Handler {handler_instance_id}]"
        tc_id_on_entry = id(self._trace_client) if self._trace_client else 'None'
        print(f"{log_prefix} ENTERING on_tool_error: run_id={run_id}. Current TraceClient ID: {tc_id_on_entry}")

        try:
            trace_client = self._ensure_trace_client(run_id, "ToolError")
            if not trace_client:
                return
            self._end_span_tracking(trace_client, run_id, span_type="tool", error=error)
        except Exception as e:
            tc_id_on_error = id(self._trace_client) if self._trace_client else 'None'
            self._log(f"{log_prefix} UNCAUGHT EXCEPTION in on_tool_error for run_id={run_id} (TraceClient ID: {tc_id_on_error}): {e}")
            print(traceback.format_exc())

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], *, run_id: UUID, parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None, invocation_params: Optional[Dict[str, Any]] = None, options: Optional[Dict[str, Any]] = None, name: Optional[str] = None, **kwargs: Any) -> Any:
        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Handler {handler_instance_id}]"
        tc_id_on_entry = id(self._trace_client) if self._trace_client else 'None'
        llm_name = name or serialized.get("name", "LLM Call")
        print(f"{log_prefix} ENTERING on_llm_start: name='{llm_name}', run_id={run_id}. Current TraceClient ID: {tc_id_on_entry}")

        try:
            trace_client = self._ensure_trace_client(run_id, llm_name)
            if not trace_client:
                return
            inputs = {
                'prompts': prompts, 'invocation_params': invocation_params or kwargs,
                'options': options, 'tags': tags, 'metadata': metadata, 'serialized': serialized,
            }
            self._start_span_tracking(trace_client, run_id, parent_run_id, llm_name, span_type="llm", inputs=inputs)
        except Exception as e:
            tc_id_on_error = id(self._trace_client) if self._trace_client else 'None'
            self._log(f"{log_prefix} UNCAUGHT EXCEPTION in on_llm_start for run_id={run_id} (TraceClient ID: {tc_id_on_error}): {e}")
            print(traceback.format_exc())

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> Any:
        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Handler {handler_instance_id}]"
        tc_id_on_entry = id(self._trace_client) if self._trace_client else 'None'
        print(f"{log_prefix} ENTERING on_llm_end: run_id={run_id}. Current TraceClient ID: {tc_id_on_entry}")

        # --- DEBUGGING: Inspect the response object --- 
        print(f"{log_prefix} [DEBUG on_llm_end] Received response object for run_id={run_id}:")
        try:
            # Use rich print for better formatting if available, fallback to standard print
            from rich import print as rprint
            rprint(response)
        except ImportError:
            print(response)
        print(f"{log_prefix} [DEBUG on_llm_end] response.llm_output type: {type(response.llm_output)}")
        print(f"{log_prefix} [DEBUG on_llm_end] response.llm_output content:")
        try:
            from rich import print as rprint
            rprint(response.llm_output)
        except ImportError:
            print(response.llm_output)
        # --- END DEBUGGING ---

        try:
            trace_client = self._ensure_trace_client(run_id, "LLMEnd")
            if not trace_client: return
            outputs = {"response": response, "kwargs": kwargs}
            self._end_span_tracking(trace_client, run_id, span_type="llm", outputs=outputs)
        except Exception as e:
            tc_id_on_error = id(self._trace_client) if self._trace_client else 'None'
            self._log(f"{log_prefix} UNCAUGHT EXCEPTION in on_llm_end for run_id={run_id} (TraceClient ID: {tc_id_on_error}): {e}")
            print(traceback.format_exc())

    def on_llm_error(self, error: BaseException, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> Any:
        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Handler {handler_instance_id}]"
        tc_id_on_entry = id(self._trace_client) if self._trace_client else 'None'
        print(f"{log_prefix} ENTERING on_llm_error: run_id={run_id}. Current TraceClient ID: {tc_id_on_entry}")

        try:
            trace_client = self._ensure_trace_client(run_id, "LLMError")
            if not trace_client:
                return
            self._end_span_tracking(trace_client, run_id, span_type="llm", error=error)
        except Exception as e:
            tc_id_on_error = id(self._trace_client) if self._trace_client else 'None'
            self._log(f"{log_prefix} UNCAUGHT EXCEPTION in on_llm_error for run_id={run_id} (TraceClient ID: {tc_id_on_error}): {e}")
            print(traceback.format_exc())

    def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], *, run_id: UUID, parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None, invocation_params: Optional[Dict[str, Any]] = None, options: Optional[Dict[str, Any]] = None, name: Optional[str] = None, **kwargs: Any) -> Any:
        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Handler {handler_instance_id}]"
        tc_id_on_entry = id(self._trace_client) if self._trace_client else 'None'
        model_id = serialized.get("id", [])
        model_name_suffix = "CHAT_MODEL_CALL"
        if any("openai" in str(id_part).lower() for id_part in model_id): model_name_suffix = "OPENAI_API_CALL"
        elif any("anthropic" in str(id_part).lower() for id_part in model_id): model_name_suffix = "ANTHROPIC_API_CALL"
        elif any("together" in str(id_part).lower() for id_part in model_id): model_name_suffix = "TOGETHER_API_CALL"
        elif any("google" in str(id_part).lower() for id_part in model_id): model_name_suffix = "GOOGLE_API_CALL"
        chat_model_name = name or f"{serialized.get('name', 'UnknownChatModel')} {model_name_suffix}"
        print(f"{log_prefix} ENTERING on_chat_model_start: name='{chat_model_name}', run_id={run_id}. Current TraceClient ID: {tc_id_on_entry}")

        try:
            trace_client = self._ensure_trace_client(run_id, chat_model_name)
            if not trace_client:
                return

            inputs = {
                'messages': messages, 'invocation_params': invocation_params or kwargs.get("invocation_params", kwargs),
                'options': options or kwargs.get("options", {}), 'tags': tags, 'metadata': metadata, 'serialized': serialized,
            }
            self._start_span_tracking(trace_client, run_id, parent_run_id, chat_model_name, span_type="llm", inputs=inputs)
        except Exception as e:
            tc_id_on_error = id(self._trace_client) if self._trace_client else 'None'
            self._log(f"{log_prefix} UNCAUGHT EXCEPTION in on_chat_model_start for run_id={run_id} (TraceClient ID: {tc_id_on_error}): {e}")
            print(traceback.format_exc())

    # --- Agent Methods ---
    def on_agent_action(self, action: AgentAction, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> Any:
        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Handler {handler_instance_id}]"
        tc_id_on_entry = id(self._trace_client) if self._trace_client else 'None'
        print(f"{log_prefix} ENTERING on_agent_action: tool={action.tool}, run_id={run_id}. Current TraceClient ID: {tc_id_on_entry}")

        try:
            # Optional: Implement detailed tracing if needed
            pass
        except Exception as e:
            tc_id_on_error = id(self._trace_client) if self._trace_client else 'None'
            self._log(f"{log_prefix} UNCAUGHT EXCEPTION in on_agent_action for run_id={run_id} (TraceClient ID: {tc_id_on_error}): {e}")
            print(traceback.format_exc())

    def on_agent_finish(self, finish: AgentFinish, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> Any:
        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Handler {handler_instance_id}]"
        tc_id_on_entry = id(self._trace_client) if self._trace_client else 'None'
        print(f"{log_prefix} ENTERING on_agent_finish: run_id={run_id}. Current TraceClient ID: {tc_id_on_entry}")

        try:
            # Optional: Implement detailed tracing if needed
            pass
        except Exception as e:
            tc_id_on_error = id(self._trace_client) if self._trace_client else 'None'
            self._log(f"{log_prefix} UNCAUGHT EXCEPTION in on_agent_finish for run_id={run_id} (TraceClient ID: {tc_id_on_error}): {e}")
            print(traceback.format_exc())

# --- Async Handler --- 

# --- NEW Fully Functional Async Handler ---
class AsyncJudgevalCallbackHandler(AsyncCallbackHandler):
    """
    Async LangChain Callback Handler using run_id/parent_run_id for hierarchy.
    Manages its own internal TraceClient instance created upon first use.
    Includes verbose logging and defensive checks.
    """
    lc_serializable = False
    lc_kwargs = {}

    def __init__(self, tracer: Tracer):
        instance_id = id(self)
        print(f"{HANDLER_LOG_PREFIX} *** Async Handler instance {instance_id} __init__ called. ***")
        self.tracer = tracer
        self._trace_client: Optional[TraceClient] = None
        self._run_id_to_span_id: Dict[UUID, str] = {}
        self._span_id_to_start_time: Dict[str, float] = {}
        self._span_id_to_depth: Dict[str, int] = {}
        self._root_run_id: Optional[UUID] = None
        self._trace_context_token: Optional[contextvars.Token] = None # NEW: Store trace context token

        self.executed_nodes: List[str] = []
        self.executed_tools: List[str] = []
        self.executed_node_tools: List[str] = []

    # NOTE: _ensure_trace_client remains synchronous as it doesn't involve async I/O
    def _ensure_trace_client(self, run_id: UUID, event_name: str) -> Optional[TraceClient]:
        """Ensures the internal trace client is initialized. Returns client or None."""
        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Async Handler {handler_instance_id}]"
        if self._trace_client is None:
            print(f"{log_prefix} Trace client is None. Attempting initialization triggered by {event_name} (run_id: {run_id})...")
            trace_id = str(uuid.uuid4())
            project = self.tracer.project_name
            try:
                print(f"{log_prefix} BEFORE TraceClient creation attempt.")
                client_instance = TraceClient(
                    self.tracer, trace_id, event_name, project_name=project,
                    overwrite=False, rules=self.tracer.rules,
                    enable_monitoring=self.tracer.enable_monitoring,
                    enable_evaluations=self.tracer.enable_evaluations
                )
                print(f"{log_prefix} AFTER TraceClient creation attempt. Success? {'Yes' if client_instance else 'No'}")
                self._trace_client = client_instance
                if self._trace_client:
                     print(f"{log_prefix} Initialized TraceClient: ID={self._trace_client.trace_id}, Name='{event_name}', Instance ID={id(self._trace_client)}")
                     if self._root_run_id is None:
                         self._root_run_id = run_id
                         print(f"{log_prefix} Tentatively set root run ID: {self._root_run_id}")
                else:
                    print(f"{log_prefix} FATAL: TraceClient creation appears to have succeeded but self._trace_client is still None after assignment.")
                    return None
            except Exception as e:
                print(f"{log_prefix} FATAL: Failed to initialize TraceClient: {e}")
                print(traceback.format_exc())
                self._trace_client = None
                return None
        return self._trace_client

    def _log(self, message: str):
        """Helper for consistent logging format."""
        print(f"{HANDLER_LOG_PREFIX} {message}")

    # NOTE: _start_span_tracking remains mostly synchronous, TraceClient.add_entry might become async later
    def _start_span_tracking(
        self,
        trace_client: TraceClient,
        run_id: UUID,
        parent_run_id: Optional[UUID],
        name: str,
        span_type: SpanType = "span",
        inputs: Optional[Dict[str, Any]] = None
    ):
        self._log(f"_start_span_tracking called for: name='{name}', run_id={run_id}, parent_run_id={parent_run_id}, span_type={span_type}")
        if not trace_client:
            handler_instance_id = id(self)
            log_prefix = f"{HANDLER_LOG_PREFIX} [Async Handler {handler_instance_id}]"
            self._log(f"{log_prefix} FATAL ERROR in _start_span_tracking: trace_client argument is None for name='{name}', run_id={run_id}. Aborting span start.")
            return

        # --- NEW: Set trace context variable if not already set for this trace --- 
        if self._trace_context_token is None:
            try:
                self._trace_context_token = current_trace_var.set(trace_client)
                self._log(f"  Set current_trace_var for trace_id {trace_client.trace_id}")
            except Exception as e:
                self._log(f"  ERROR setting current_trace_var for trace_id {trace_client.trace_id}: {e}")
        # --- END NEW --- 

        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Async Handler {handler_instance_id}]"
        trace_client_instance_id = id(trace_client) if trace_client else 'None'
        print(f"{log_prefix} _start_span_tracking: Using TraceClient ID: {trace_client_instance_id}")

        start_time = time.time()
        span_id = str(uuid.uuid4())
        parent_span_id: Optional[str] = None
        current_depth = 0

        if parent_run_id and parent_run_id in self._run_id_to_span_id:
            parent_span_id = self._run_id_to_span_id[parent_run_id]
            if parent_span_id in self._span_id_to_depth:
                current_depth = self._span_id_to_depth[parent_span_id] + 1
            else:
                self._log(f"  WARNING: Parent span depth not found for parent_span_id: {parent_span_id}. Setting depth to 0.")
        elif parent_run_id:
            self._log(f"  WARNING: parent_run_id {parent_run_id} provided for '{name}' ({run_id}) but parent span not tracked. Treating as depth 0.")
        else:
            self._log(f"  No parent_run_id provided. Treating '{name}' as depth 0.")

        self._run_id_to_span_id[run_id] = span_id
        self._span_id_to_start_time[span_id] = start_time
        self._span_id_to_depth[span_id] = current_depth
        self._log(f"  Tracking new span: span_id={span_id}, depth={current_depth}")

        # --- NEW: Set context variable --- 
        try:
            token = current_span_var.set(span_id)
            self._run_id_to_context_token[run_id] = token
            self._log(f"  Set current_span_var to {span_id} for run_id {run_id}")
        except Exception as e:
            self._log(f"  ERROR setting current_span_var for run_id {run_id}: {e}")
        # --- END NEW --- 

        try:
            # TODO: Check if trace_client.add_entry needs await if TraceClient becomes async
            trace_client.add_entry(TraceEntry(
                type="enter", span_id=span_id, trace_id=trace_client.trace_id,
                parent_span_id=parent_span_id, function=name, depth=current_depth,
                message=name, created_at=start_time, span_type=span_type
            ))
            self._log(f"  Added 'enter' entry for span_id={span_id}")
        except Exception as e:
            self._log(f"  ERROR adding 'enter' entry for span_id {span_id}: {e}")
            print(traceback.format_exc())

        if inputs:
            # _record_input_data is also sync for now
            self._record_input_data(trace_client, run_id, inputs)

    # NOTE: _end_span_tracking remains mostly synchronous, TraceClient.save might become async later
    def _end_span_tracking(
        self,
        trace_client: TraceClient,
        run_id: UUID,
        span_type: SpanType = "span",
        outputs: Optional[Any] = None,
        error: Optional[BaseException] = None
    ):
        self._log(f"_end_span_tracking called for: run_id={run_id}, span_type={span_type}")
        if not trace_client:
            handler_instance_id = id(self)
            log_prefix = f"{HANDLER_LOG_PREFIX} [Async Handler {handler_instance_id}]"
            self._log(f"{log_prefix} FATAL ERROR in _end_span_tracking: trace_client argument is None for run_id={run_id}. Aborting span end.")
            return

        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Async Handler {handler_instance_id}]"
        trace_client_instance_id = id(trace_client) if trace_client else 'None'
        print(f"{log_prefix} _end_span_tracking: Using TraceClient ID: {trace_client_instance_id}")

        if run_id not in self._run_id_to_span_id:
            self._log(f"  WARNING: Attempting to end span for untracked run_id: {run_id}")
            return

        span_id = self._run_id_to_span_id[run_id]
        start_time = self._span_id_to_start_time.get(span_id)
        depth = self._span_id_to_depth.get(span_id, 0)
        duration = time.time() - start_time if start_time is not None else None
        self._log(f"  Ending span_id={span_id}. Start time={start_time}, Duration={duration}, Depth={depth}")

        # Record output/error (outputs dict now correctly contains token_usage for LLM spans)
        if error:
            self._log(f"  Recording error for span_id={span_id}: {error}")
            self._record_output_data(trace_client, run_id, error)
        elif outputs is not None:
            output_repr = repr(outputs)
            log_output = (output_repr[:100] + '...') if len(output_repr) > 103 else output_repr
            self._log(f"  Recording output for span_id={span_id}: {log_output}")
            self._record_output_data(trace_client, run_id, outputs)

        # Determine function name for the exit entry
        entry_function_name = "unknown"
        try:
            if hasattr(trace_client, 'entries') and trace_client.entries:
                entry_function_name = next((e.function for e in reversed(trace_client.entries) if e.span_id == span_id and e.type == "enter"), "unknown")
            else:
                self._log(f"  WARNING: Cannot determine function name for exit span_id {span_id}, trace_client.entries missing or empty.")
        except Exception as e:
            self._log(f"  ERROR finding function name for exit entry span_id {span_id}: {e}")
            print(traceback.format_exc())

        # Add exit entry (without metadata)
        try:
            # TODO: Check if trace_client.add_entry needs await if TraceClient becomes async
            exit_entry = TraceEntry(
                type="exit", span_id=span_id, trace_id=trace_client.trace_id,
                depth=depth, created_at=time.time(), duration=duration,
                span_type=span_type, function=entry_function_name
            )
            trace_client.add_entry(exit_entry)
            self._log(f"  Added 'exit' entry for span_id={span_id}, function='{entry_function_name}'")
        except Exception as e:
            self._log(f"  ERROR adding 'exit' entry for span_id {span_id}: {e}")
            print(traceback.format_exc())

        # Clean up state
        if span_id in self._span_id_to_start_time: del self._span_id_to_start_time[span_id]
        if span_id in self._span_id_to_depth: del self._span_id_to_depth[span_id]

        # --- NEW: Reset context variable --- 
        token = self._run_id_to_context_token.pop(run_id, None)
        if token:
            try:
                current_span_var.reset(token)
                self._log(f"  Reset current_span_var for run_id {run_id} (was {span_id})")
            except LookupError:
                # This error occurs if the token was created in a different context
                # Log a warning instead of erroring out, as the main goal was setting the context during the span
                self._log(f"  [WARN] Could not reset current_span_var for run_id {run_id} (token from different context).")
            except Exception as e:
                 # Log other potential errors during reset
                 self._log(f"  ERROR resetting current_span_var for run_id {run_id}: {e}")
        # --- END NEW --- 

        # --- Reset Trace Context Var ONLY when root finishes ---
        if run_id == self._root_run_id:
            if self._trace_context_token is not None:
                try:
                    current_trace_var.reset(self._trace_context_token)
                    self._log(f"  Reset current_trace_var for trace_id {self._trace_client.trace_id if self._trace_client else 'UNKNOWN'}")
                except LookupError:
                     self._log(f"  [WARN] Could not reset current_trace_var for trace_id {self._trace_client.trace_id if self._trace_client else 'UNKNOWN'} (token from different context).")
                except Exception as e:
                     self._log(f"  ERROR resetting current_trace_var: {e}")
                finally:
                     self._trace_context_token = None # Ensure it's reset
            # --- END Trace Context Reset ---

            self._log(f"Root run {run_id} finished. Attempting to save trace...")
            if self._trace_client:
                try:
                    # TODO: Check if trace_client.save needs await if TraceClient becomes async
                    trace_id, _ = self._trace_client.save(overwrite=True)
                    self._log(f"Trace {trace_id} successfully saved.")
                except Exception as e:
                    self._log(f"ERROR saving trace {self._trace_client.trace_id}: {e}")
                    print(traceback.format_exc())
                finally:
                    self._log(f"Resetting handler state after root run end.")
                    self._trace_client = None
                    self._run_id_to_span_id = {}
                    self._span_id_to_start_time = {}
                    self._span_id_to_depth = {}
                    self._root_run_id = None
            else:
                self._log(f"  WARNING: Root run {run_id} ended, but trace client was already None. Cannot save trace.")

    # NOTE: _record_input_data remains synchronous for now
    def _record_input_data(self,
                           trace_client: TraceClient,
                           run_id: UUID,
                           inputs: Dict[str, Any]):
        self._log(f"_record_input_data called for run_id={run_id}")
        if run_id not in self._run_id_to_span_id:
            self._log(f"  WARNING: Attempting to record input for untracked run_id: {run_id}")
            return
        if not trace_client:
             self._log(f"  ERROR: TraceClient is None when trying to record input for run_id={run_id}")
             return

        span_id = self._run_id_to_span_id[run_id]
        depth = self._span_id_to_depth.get(span_id, 0)
        self._log(f"  Found span_id={span_id}, depth={depth} for run_id={run_id}")

        function_name = "unknown"
        span_type: SpanType = "span"
        try:
            enter_entry = next((e for e in reversed(trace_client.entries) if e.span_id == span_id and e.type == "enter"), None)
            if enter_entry:
                function_name = enter_entry.function
                span_type = enter_entry.span_type
                self._log(f"  Found function='{function_name}', span_type='{span_type}' for input span_id={span_id}")
            else:
                self._log(f"  WARNING: Could not find 'enter' entry for input span_id={span_id}")
        except Exception as e:
            self._log(f"  ERROR finding enter entry for input span_id {span_id}: {e}")
            print(traceback.format_exc())

        try:
            # TODO: Check if trace_client.add_entry needs await
            input_entry = TraceEntry(
                type="input", span_id=span_id, trace_id=trace_client.trace_id,
                parent_span_id=next((e.parent_span_id for e in reversed(trace_client.entries) if e.span_id == span_id and e.type == "enter"), None),
                function=function_name, depth=depth, message=f"Input to {function_name}",
                created_at=time.time(), inputs=inputs, span_type=span_type
            )
            trace_client.add_entry(input_entry)
            self._log(f"  Added 'input' entry directly for span_id={span_id}")
        except Exception as e:
            self._log(f"  ERROR adding 'input' entry directly for span_id {span_id}: {e}")
            print(traceback.format_exc())

    # NOTE: _record_output_data remains synchronous for now
    def _record_output_data(self,
                            trace_client: TraceClient,
                            run_id: UUID,
                            output: Any):
        self._log(f"_record_output_data called for run_id={run_id}")
        if run_id not in self._run_id_to_span_id:
            self._log(f"  WARNING: Attempting to record output for untracked run_id: {run_id}")
            return
        if not trace_client:
            self._log(f"  ERROR: TraceClient is None when trying to record output for run_id={run_id}")
            return

        span_id = self._run_id_to_span_id[run_id]
        depth = self._span_id_to_depth.get(span_id, 0)
        self._log(f"  Found span_id={span_id}, depth={depth} for run_id={run_id}")

        function_name = "unknown"
        span_type: SpanType = "span"
        try:
            enter_entry = next((e for e in reversed(trace_client.entries) if e.span_id == span_id and e.type == "enter"), None)
            if enter_entry:
                function_name = enter_entry.function
                span_type = enter_entry.span_type
                self._log(f"  Found function='{function_name}', span_type='{span_type}' for output span_id={span_id}")
            else:
                 self._log(f"  WARNING: Could not find 'enter' entry for output span_id={span_id}")
        except Exception as e:
            self._log(f"  ERROR finding enter entry for output span_id {span_id}: {e}")
            print(traceback.format_exc())

        try:
            # TODO: Check if trace_client.add_entry needs await
            output_entry = TraceEntry(
                type="output", span_id=span_id, trace_id=trace_client.trace_id,
                parent_span_id=next((e.parent_span_id for e in reversed(trace_client.entries) if e.span_id == span_id and e.type == "enter"), None),
                function=function_name, depth=depth, message=f"Output from {function_name}",
                created_at=time.time(), output=output, span_type=span_type
            )
            trace_client.add_entry(output_entry)
            self._log(f"  Added 'output' entry directly for span_id={span_id}")
        except Exception as e:
            self._log(f"  ERROR adding 'output' entry directly for span_id {span_id}: {e}")
            print(traceback.format_exc())

    # --- Async Callback Methods ---

    async def on_retriever_start(self, serialized: Dict[str, Any], query: str, *, run_id: UUID, parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Async Handler {handler_instance_id}]"
        serialized_name = serialized.get('name', 'Unknown') if serialized else "Unknown (Serialized=None)"
        tc_id_on_entry = id(self._trace_client) if self._trace_client else 'None'
        print(f"{log_prefix} ENTERING on_retriever_start: name='{serialized_name}', run_id={run_id}. Current TraceClient ID: {tc_id_on_entry}")
        try:
            name = f"RETRIEVER_{(serialized_name).upper()}"
            trace_client = self._ensure_trace_client(run_id, name)
            if not trace_client: return
            inputs = {'query': query, 'tags': tags, 'metadata': metadata, 'kwargs': kwargs, 'serialized': serialized}
            self._start_span_tracking(trace_client, run_id, parent_run_id, name, span_type="retriever", inputs=inputs)
        except Exception as e:
            tc_id_on_error = id(self._trace_client) if self._trace_client else 'None'
            self._log(f"{log_prefix} UNCAUGHT EXCEPTION in on_retriever_start for run_id={run_id} (TraceClient ID: {tc_id_on_error}): {e}")
            print(traceback.format_exc())

    async def on_retriever_end(self, documents: Sequence[Document], *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> Any:
        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Async Handler {handler_instance_id}]"
        tc_id_on_entry = id(self._trace_client) if self._trace_client else 'None'
        print(f"{log_prefix} ENTERING on_retriever_end: run_id={run_id}. Current TraceClient ID: {tc_id_on_entry}")
        try:
            trace_client = self._ensure_trace_client(run_id, "RetrieverEnd")
            if not trace_client: return
            doc_summary = [{"index": i, "page_content": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content, "metadata": doc.metadata} for i, doc in enumerate(documents)]
            outputs = {"document_count": len(documents), "documents": doc_summary, "kwargs": kwargs}
            self._end_span_tracking(trace_client, run_id, span_type="retriever", outputs=outputs)
        except Exception as e:
            tc_id_on_error = id(self._trace_client) if self._trace_client else 'None'
            self._log(f"{log_prefix} UNCAUGHT EXCEPTION in on_retriever_end for run_id={run_id} (TraceClient ID: {tc_id_on_error}): {e}")
            print(traceback.format_exc())

    async def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], *, run_id: UUID, parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Async Handler {handler_instance_id}]"
        # Handle potential None for serialized safely
        serialized_name = serialized.get('name') if serialized else None
        tc_id_on_entry = id(self._trace_client) if self._trace_client else 'None'
        # Log the potentially generic or specific name found in serialized
        log_name = serialized_name if serialized_name else "Unknown (Serialized=None)"
        print(f"{HANDLER_LOG_PREFIX} [Async Handler {handler_instance_id}] ENTERING on_chain_start: serialized_name='{log_name}', run_id={run_id}. Current TraceClient ID: {tc_id_on_entry}")

        try:
            # Determine the best name and span type
            name = "Unknown Chain" # Default
            span_type: SpanType = "chain"
            node_name = metadata.get("langgraph_node") if metadata else None
            is_langgraph_root = kwargs.get('name') == 'LangGraph' and parent_run_id is None

            # Define generic names to ignore if node_name is not present
            GENERIC_NAMES = ["RunnableSequence", "RunnableParallel", "RunnableLambda", "LangGraph", "__start__", "__end__"]

            if node_name:
                name = node_name
                # span_type remains "chain"
                self._log(f"  LangGraph Node Start Detected: '{name}', run_id={run_id}, parent_run_id={parent_run_id}")
                if name not in self.executed_nodes: self.executed_nodes.append(name)
            elif serialized_name and serialized_name not in GENERIC_NAMES:
                # If no node_name, but serialized_name exists and isn't generic,
                # assume it's a meaningful function/step name (like a router)
                name = serialized_name
                # span_type remains "chain"
                self._log(f"  LangGraph Functional Step (Router?): '{name}', run_id={run_id}, parent_run_id={parent_run_id}")
            elif is_langgraph_root:
                name = "LangGraph"
                # span_type remains "chain"
                self._log(f"  LangGraph Root Start Detected: run_id={run_id}")
                if self._root_run_id is None or self._root_run_id != run_id:
                    self._log(f"  Setting root run ID to {run_id} for trace {self._trace_client.trace_id if self._trace_client else 'N/A'}")
                    self._root_run_id = run_id
                if self._trace_client and self._trace_client.name != name:
                    self._log(f"  Updating trace name from '{self._trace_client.name}' to '{name}'")
                    self._trace_client.name = name
            elif serialized_name: # Fallback if node_name missing and serialized_name was generic or root wasn't detected
                name = serialized_name
                # span_type remains "chain"
                self._log(f"  Fallback to serialized_name: '{name}', run_id={run_id}")
            # else: name remains "Unknown Chain"

            # Ensure trace client exists (using the determined name for initialization if needed)
            trace_client = self._ensure_trace_client(run_id, name)
            if not trace_client: return

            # Start span tracking using the determined name and span_type
            combined_inputs = {'inputs': inputs, 'tags': tags, 'metadata': metadata, 'kwargs': kwargs, 'serialized': serialized}
            self._start_span_tracking(trace_client, run_id, parent_run_id, name, span_type=span_type, inputs=combined_inputs)

        except Exception as e:
            tc_id_on_error = id(self._trace_client) if self._trace_client else 'None'
            self._log(f"{log_prefix} UNCAUGHT EXCEPTION in on_chain_start for run_id={run_id} (TraceClient ID: {tc_id_on_error}): {e}")
            print(traceback.format_exc())

    async def on_chain_end(self, outputs: Dict[str, Any], *, run_id: UUID, parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None, **kwargs: Any) -> Any:
        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Async Handler {handler_instance_id}]"
        tc_id_on_entry = id(self._trace_client) if self._trace_client else 'None'
        print(f"{log_prefix} ENTERING on_chain_end: run_id={run_id}. Current TraceClient ID: {tc_id_on_entry}")
        try:
            trace_client = self._ensure_trace_client(run_id, "ChainEnd")
            if not trace_client: return

            span_id = self._run_id_to_span_id.get(run_id)
            span_type: SpanType = "chain"
            if span_id:
                try:
                    if hasattr(trace_client, 'entries') and trace_client.entries:
                        enter_entry = next((e for e in reversed(trace_client.entries) if e.span_id == span_id and e.type == "enter"), None)
                        if enter_entry: span_type = enter_entry.span_type
                    else: self._log(f"  WARNING: trace_client.entries not available for on_chain_end span_id={span_id}")
                except Exception as e:
                    self._log(f"  ERROR finding enter entry for span_id {span_id} in on_chain_end: {e}")

            combined_outputs = {"outputs": outputs, "tags": tags, "kwargs": kwargs}
            self._end_span_tracking(trace_client, run_id, span_type=span_type, outputs=combined_outputs)
        except Exception as e:
            tc_id_on_error = id(self._trace_client) if self._trace_client else 'None'
            self._log(f"{log_prefix} UNCAUGHT EXCEPTION in on_chain_end for run_id={run_id} (TraceClient ID: {tc_id_on_error}): {e}")
            print(traceback.format_exc())

    async def on_chain_error(self, error: BaseException, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> Any:
        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Async Handler {handler_instance_id}]"
        tc_id_on_entry = id(self._trace_client) if self._trace_client else 'None'
        print(f"{log_prefix} ENTERING on_chain_error: run_id={run_id}. Current TraceClient ID: {tc_id_on_entry}")
        try:
            trace_client = self._ensure_trace_client(run_id, "ChainError")
            if not trace_client: return

            span_id = self._run_id_to_span_id.get(run_id)
            span_type: SpanType = "chain"
            if span_id:
                try:
                     if hasattr(trace_client, 'entries') and trace_client.entries:
                        enter_entry = next((e for e in reversed(trace_client.entries) if e.span_id == span_id and e.type == "enter"), None)
                        if enter_entry: span_type = enter_entry.span_type
                     else: self._log(f"  WARNING: trace_client.entries not available for on_chain_error span_id={span_id}")
                except Exception as e:
                    self._log(f"  ERROR finding enter entry for span_id {span_id} in on_chain_error: {e}")

            self._end_span_tracking(trace_client, run_id, span_type=span_type, error=error)
        except Exception as e:
            tc_id_on_error = id(self._trace_client) if self._trace_client else 'None'
            self._log(f"{log_prefix} UNCAUGHT EXCEPTION in on_chain_error for run_id={run_id} (TraceClient ID: {tc_id_on_error}): {e}")
            print(traceback.format_exc())

    async def on_tool_start(self, serialized: Dict[str, Any], input_str: str, *, run_id: UUID, parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None, inputs: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Async Handler {handler_instance_id}]"
        # Handle potential None for serialized
        name = serialized.get("name", "Unnamed Tool") if serialized else "Unknown Tool (Serialized=None)"
        tc_id_on_entry = id(self._trace_client) if self._trace_client else 'None'
        print(f"{log_prefix} ENTERING on_tool_start: name='{name}', run_id={run_id}. Current TraceClient ID: {tc_id_on_entry}")
        try:
            trace_client = self._ensure_trace_client(run_id, name)
            if not trace_client: return

            combined_inputs = {'input_str': input_str, 'inputs': inputs, 'tags': tags, 'metadata': metadata, 'kwargs': kwargs, 'serialized': serialized}
            self._start_span_tracking(trace_client, run_id, parent_run_id, name, span_type="tool", inputs=combined_inputs)

            if name not in self.executed_tools: self.executed_tools.append(name)
            parent_node_name = None
            if parent_run_id and parent_run_id in self._run_id_to_span_id:
                parent_span_id = self._run_id_to_span_id[parent_run_id]
                try:
                    if hasattr(trace_client, 'entries') and trace_client.entries:
                        parent_enter_entry = next((e for e in reversed(trace_client.entries) if e.span_id == parent_span_id and e.type == "enter" and e.span_type == "chain"), None)
                        if parent_enter_entry:
                            parent_node_name = parent_enter_entry.function
                    else:
                        self._log(f"  WARNING: trace_client.entries not available for parent node {parent_span_id}")
                except Exception as e:
                    self._log(f"  ERROR finding parent node name for tool start span_id {parent_span_id}: {e}")

            node_tool = f"{parent_node_name}:{name}" if parent_node_name else name
            if node_tool not in self.executed_node_tools: self.executed_node_tools.append(node_tool)
            self._log(f"  Tracked node_tool: {node_tool}")
        except Exception as e:
            tc_id_on_error = id(self._trace_client) if self._trace_client else 'None'
            self._log(f"{log_prefix} UNCAUGHT EXCEPTION in on_tool_start for run_id={run_id} (TraceClient ID: {tc_id_on_error}): {e}")
            print(traceback.format_exc())

    async def on_tool_end(self, output: Any, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> Any:
        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Async Handler {handler_instance_id}]"
        tc_id_on_entry = id(self._trace_client) if self._trace_client else 'None'
        print(f"{log_prefix} ENTERING on_tool_end: run_id={run_id}. Current TraceClient ID: {tc_id_on_entry}")
        try:
            trace_client = self._ensure_trace_client(run_id, "ToolEnd")
            if not trace_client: return
            outputs = {"output": output, "kwargs": kwargs}
            self._end_span_tracking(trace_client, run_id, span_type="tool", outputs=outputs)
        except Exception as e:
            tc_id_on_error = id(self._trace_client) if self._trace_client else 'None'
            self._log(f"{log_prefix} UNCAUGHT EXCEPTION in on_tool_end for run_id={run_id} (TraceClient ID: {tc_id_on_error}): {e}")
            print(traceback.format_exc())

    async def on_tool_error(self, error: BaseException, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> Any:
        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Async Handler {handler_instance_id}]"
        tc_id_on_entry = id(self._trace_client) if self._trace_client else 'None'
        print(f"{log_prefix} ENTERING on_tool_error: run_id={run_id}. Current TraceClient ID: {tc_id_on_entry}")
        try:
            trace_client = self._ensure_trace_client(run_id, "ToolError")
            if not trace_client: return
            self._end_span_tracking(trace_client, run_id, span_type="tool", error=error)
        except Exception as e:
            tc_id_on_error = id(self._trace_client) if self._trace_client else 'None'
            self._log(f"{log_prefix} UNCAUGHT EXCEPTION in on_tool_error for run_id={run_id} (TraceClient ID: {tc_id_on_error}): {e}")
            print(traceback.format_exc())

    async def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], *, run_id: UUID, parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None, invocation_params: Optional[Dict[str, Any]] = None, options: Optional[Dict[str, Any]] = None, name: Optional[str] = None, **kwargs: Any) -> Any:
        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Async Handler {handler_instance_id}]"
        tc_id_on_entry = id(self._trace_client) if self._trace_client else 'None'
        llm_name = name or serialized.get("name", "LLM Call")
        print(f"{log_prefix} ENTERING on_llm_start: name='{llm_name}', run_id={run_id}. Current TraceClient ID: {tc_id_on_entry}")
        try:
            trace_client = self._ensure_trace_client(run_id, llm_name)
            if not trace_client: return
            inputs = {
                'prompts': prompts, 'invocation_params': invocation_params or kwargs,
                'options': options, 'tags': tags, 'metadata': metadata, 'serialized': serialized,
            }
            self._start_span_tracking(trace_client, run_id, parent_run_id, llm_name, span_type="llm", inputs=inputs)
        except Exception as e:
            tc_id_on_error = id(self._trace_client) if self._trace_client else 'None'
            self._log(f"{log_prefix} UNCAUGHT EXCEPTION in on_llm_start for run_id={run_id} (TraceClient ID: {tc_id_on_error}): {e}")
            print(traceback.format_exc())

    async def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> Any:
        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Async Handler {handler_instance_id}]"
        tc_id_on_entry = id(self._trace_client) if self._trace_client else 'None'
        print(f"{log_prefix} ENTERING on_llm_end: run_id={run_id}. Current TraceClient ID: {tc_id_on_entry}")

        # --- DEBUGGING: Inspect the response object --- 
        print(f"{log_prefix} [DEBUG on_llm_end] Received response object for run_id={run_id}:")
        try:
            # Use rich print for better formatting if available, fallback to standard print
            from rich import print as rprint
            rprint(response)
        except ImportError:
            print(response)
        print(f"{log_prefix} [DEBUG on_llm_end] response.llm_output type: {type(response.llm_output)}")
        print(f"{log_prefix} [DEBUG on_llm_end] response.llm_output content:")
        try:
            from rich import print as rprint
            rprint(response.llm_output)
        except ImportError:
            print(response.llm_output)
        # --- END DEBUGGING ---

        try:
            trace_client = self._ensure_trace_client(run_id, "LLMEnd")
            if not trace_client: return

            # Prepare base outputs
            outputs = {"response": response, "kwargs": kwargs}

            # Attempt to extract token usage
            token_usage = None
            try:
                if response.llm_output and isinstance(response.llm_output, dict):
                    token_usage = response.llm_output.get('token_usage')
                    if token_usage:
                        self._log(f"  Extracted token usage for run_id={run_id}: {token_usage}")
                        # Add extracted usage to the outputs payload UNDER THE KEY 'usage'
                        outputs['usage'] = {
                            'prompt_tokens': token_usage.get('prompt_tokens'),
                            'completion_tokens': token_usage.get('completion_tokens'),
                            'total_tokens': token_usage.get('total_tokens')
                        }
                    else:
                         self._log(f"  'token_usage' key not found in llm_output for run_id={run_id}. llm_output: {response.llm_output}")
                else:
                     self._log(f"  llm_output not available or not a dict for run_id={run_id}. response: {response}")
            except Exception as e:
                self._log(f"  ERROR extracting token usage for run_id={run_id}: {e}")
                print(traceback.format_exc())

            # End the span, passing the potentially augmented outputs dictionary
            self._end_span_tracking(trace_client, run_id, span_type="llm", outputs=outputs)
        except Exception as e:
            tc_id_on_error = id(self._trace_client) if self._trace_client else 'None'
            self._log(f"{log_prefix} UNCAUGHT EXCEPTION in on_llm_end for run_id={run_id} (TraceClient ID: {tc_id_on_error}): {e}")
            print(traceback.format_exc())

    async def on_llm_error(self, error: BaseException, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> Any:
        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Async Handler {handler_instance_id}]"
        tc_id_on_entry = id(self._trace_client) if self._trace_client else 'None'
        print(f"{log_prefix} ENTERING on_llm_error: run_id={run_id}. Current TraceClient ID: {tc_id_on_entry}")
        try:
            trace_client = self._ensure_trace_client(run_id, "LLMError")
            if not trace_client: return
            self._end_span_tracking(trace_client, run_id, span_type="llm", error=error)
        except Exception as e:
            tc_id_on_error = id(self._trace_client) if self._trace_client else 'None'
            self._log(f"{log_prefix} UNCAUGHT EXCEPTION in on_llm_error for run_id={run_id} (TraceClient ID: {tc_id_on_error}): {e}")
            print(traceback.format_exc())

    async def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], *, run_id: UUID, parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None, invocation_params: Optional[Dict[str, Any]] = None, options: Optional[Dict[str, Any]] = None, name: Optional[str] = None, **kwargs: Any) -> Any:
        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Async Handler {handler_instance_id}]"
        tc_id_on_entry = id(self._trace_client) if self._trace_client else 'None'
        model_id = serialized.get("id", [])
        model_name_suffix = "CHAT_MODEL_CALL"
        if any("openai" in str(id_part).lower() for id_part in model_id): model_name_suffix = "OPENAI_API_CALL"
        elif any("anthropic" in str(id_part).lower() for id_part in model_id): model_name_suffix = "ANTHROPIC_API_CALL"
        elif any("together" in str(id_part).lower() for id_part in model_id): model_name_suffix = "TOGETHER_API_CALL"
        elif any("google" in str(id_part).lower() for id_part in model_id): model_name_suffix = "GOOGLE_API_CALL"
        chat_model_name = name or f"{serialized.get('name', 'UnknownChatModel')} {model_name_suffix}"
        print(f"{log_prefix} ENTERING on_chat_model_start: name='{chat_model_name}', run_id={run_id}. Current TraceClient ID: {tc_id_on_entry}")

        try:
            trace_client = self._ensure_trace_client(run_id, chat_model_name)
            if not trace_client: return

            inputs = {
                'messages': messages, 'invocation_params': invocation_params or kwargs.get("invocation_params", kwargs),
                'options': options or kwargs.get("options", {}), 'tags': tags, 'metadata': metadata, 'serialized': serialized,
            }
            self._start_span_tracking(trace_client, run_id, parent_run_id, chat_model_name, span_type="llm", inputs=inputs)
        except Exception as e:
            tc_id_on_error = id(self._trace_client) if self._trace_client else 'None'
            self._log(f"{log_prefix} UNCAUGHT EXCEPTION in on_chat_model_start for run_id={run_id} (TraceClient ID: {tc_id_on_error}): {e}")
            print(traceback.format_exc())

    # --- Agent Methods (Async versions) ---
    async def on_agent_action(self, action: AgentAction, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> Any:
        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Async Handler {handler_instance_id}]"
        tc_id_on_entry = id(self._trace_client) if self._trace_client else 'None'
        print(f"{log_prefix} ENTERING on_agent_action: tool={action.tool}, run_id={run_id}. Current TraceClient ID: {tc_id_on_entry}")
        try:
            # Optional: Add detailed async tracing if needed
            pass
        except Exception as e:
            tc_id_on_error = id(self._trace_client) if self._trace_client else 'None'
            self._log(f"{log_prefix} UNCAUGHT EXCEPTION in on_agent_action for run_id={run_id} (TraceClient ID: {tc_id_on_error}): {e}")
            print(traceback.format_exc())

    async def on_agent_finish(self, finish: AgentFinish, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> Any:
        handler_instance_id = id(self)
        log_prefix = f"{HANDLER_LOG_PREFIX} [Async Handler {handler_instance_id}]"
        tc_id_on_entry = id(self._trace_client) if self._trace_client else 'None'
        print(f"{log_prefix} ENTERING on_agent_finish: run_id={run_id}. Current TraceClient ID: {tc_id_on_entry}")
        try:
            # Optional: Add detailed async tracing if needed
            pass
        except Exception as e:
            tc_id_on_error = id(self._trace_client) if self._trace_client else 'None'
            self._log(f"{log_prefix} UNCAUGHT EXCEPTION in on_agent_finish for run_id={run_id} (TraceClient ID: {tc_id_on_error}): {e}")
            print(traceback.format_exc())