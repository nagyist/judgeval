"""
Object flow tracer for judgeval that tracks data relationships between functions.

This module provides utilities to track how objects flow between different 
function calls, making it possible to visualize data dependencies in a program.
"""
import functools
import uuid
import inspect
import threading
from typing import Dict, List, Set, Any, Optional, Tuple, Callable, Deque
from collections import defaultdict, deque
import wrapt
from pydantic import BaseModel, Field
from datetime import datetime

from judgeval.common.tracer import TraceClient, current_trace_var, current_span_var, Tracer
from judgeval.data.trace import TraceSpan


class ObjectInfo(BaseModel):
    """Information about a tracked object"""
    object_id: str
    creator_span_id: str
    creator_name: str
    value_repr: str
    type_name: str
    usage_spans: List[str] = Field(default_factory=list)
    
    def model_dump(self):
        """Convert to dict for storage"""
        return {
            "object_id": self.object_id,
            "creator_span_id": self.creator_span_id,
            "creator_name": self.creator_name,
            "value_repr": self.value_repr,
            "type_name": self.type_name,
            "usage_spans": self.usage_spans
        }


class FunctionCall(BaseModel):
    """Information about a function call"""
    span_id: str
    function_name: str
    parent_span_id: Optional[str] = None
    child_spans: List[str] = Field(default_factory=list)
    input_objects: List[str] = Field(default_factory=list)
    output_object: Optional[str] = None
    start_time: float = 0
    duration: float = 0


class ObjectTraceState:
    """Manages state for object flow tracing"""
    
    def __init__(self):
        self.next_object_id = 0
        self.object_info: Dict[str, ObjectInfo] = {}
        self.span_to_objects: Dict[str, List[str]] = defaultdict(list)
        self.current_span_id: Optional[str] = None
        self.object_usage: Dict[str, Set[str]] = defaultdict(set)
        self.span_names: Dict[str, str] = {}
        
        # Call stack and call hierarchy tracking
        self._call_stack: Dict[int, Deque[str]] = defaultdict(deque)  # thread_id -> stack of span_ids
        self.function_calls: Dict[str, FunctionCall] = {}  # span_id -> FunctionCall
        self.object_producers: Dict[str, str] = {}  # obj_id -> span_id of creator function
        self.object_consumers: Dict[str, List[str]] = defaultdict(list)  # obj_id -> list of span_ids using it
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)  # caller_span_id -> set of called_span_ids
    
    def get_next_object_id(self) -> int:
        """Get next sequential object ID"""
        obj_id = self.next_object_id
        self.next_object_id += 1
        return obj_id
    
    def register_object(self, obj: Any, span_id: str, span_name: str) -> str:
        """Register a new object created in a span"""
        obj_id = f"obj_{self.get_next_object_id()}"
        
        try:
            obj_repr = repr(obj)
            if len(obj_repr) > 100:
                obj_repr = obj_repr[:97] + "..."
        except Exception:
            obj_repr = "<<unrepresentable>>"
            
        self.object_info[obj_id] = ObjectInfo(
            object_id=obj_id,
            creator_span_id=span_id,
            creator_name=span_name,
            value_repr=obj_repr,
            type_name=type(obj).__name__
        )
        
        self.span_to_objects[span_id].append(obj_id)
        
        # Add to our producers mapping
        self.object_producers[obj_id] = span_id
        
        # Add to function output
        if span_id in self.function_calls:
            self.function_calls[span_id].output_object = obj_id
        
        return obj_id
    
    def record_object_usage(self, obj_id: str, span_id: str) -> None:
        """Record that an object was used in a span"""
        if obj_id in self.object_info:
            self.object_usage[obj_id].add(span_id)
            if span_id not in self.object_info[obj_id].usage_spans:
                self.object_info[obj_id].usage_spans.append(span_id)
            
            # Add to our consumers mapping
            if span_id not in self.object_consumers[obj_id]:
                self.object_consumers[obj_id].append(span_id)
            
            # Add to function input list
            if span_id in self.function_calls and obj_id not in self.function_calls[span_id].input_objects:
                self.function_calls[span_id].input_objects.append(obj_id)
                
            # Update dependency graph - connect the producer of this object with the consumer
            if obj_id in self.object_producers:
                producer_span_id = self.object_producers[obj_id]
                if producer_span_id != span_id:  # Don't add self-dependencies
                    self.dependency_graph[span_id].add(producer_span_id)
    
    def push_call_stack(self, span_id: str) -> None:
        """Push a span_id onto the call stack for the current thread"""
        thread_id = threading.get_ident()
        self._call_stack[thread_id].append(span_id)
    
    def pop_call_stack(self) -> Optional[str]:
        """Pop the top span_id from the call stack for the current thread"""
        thread_id = threading.get_ident()
        if not self._call_stack[thread_id]:
            return None
        return self._call_stack[thread_id].pop()
    
    def peek_call_stack(self) -> Optional[str]:
        """Get the current parent span_id (top of call stack) without popping"""
        thread_id = threading.get_ident()
        if not self._call_stack[thread_id]:
            return None
        return self._call_stack[thread_id][-1]
    
    def register_function_call(self, span_id: str, function_name: str) -> None:
        """Register a new function call"""
        # Get the parent span ID from the call stack (if any)
        parent_span_id = self.peek_call_stack()
        
        # Create the function call record
        self.function_calls[span_id] = FunctionCall(
            span_id=span_id,
            function_name=function_name,
            parent_span_id=parent_span_id
        )
        
        # Update the parent's child list if applicable
        if parent_span_id and parent_span_id in self.function_calls:
            self.function_calls[parent_span_id].child_spans.append(span_id)
            
            # Also update dependency graph
            self.dependency_graph[parent_span_id].add(span_id)


class ObjectProxy(wrapt.ObjectProxy):
    """A proxy wrapper for tracked objects"""
    
    def __init__(self, wrapped: Any, obj_id: str, state: ObjectTraceState):
        super().__init__(wrapped)
        self._self_obj_id = obj_id
        self._self_state = state
    
    def __call__(self, *args, **kwargs):
        """Record usage when the object is called"""
        if self._self_state.current_span_id:
            self._self_state.record_object_usage(self._self_obj_id, self._self_state.current_span_id)
        
        # Unwrap any proxy arguments
        new_args = []
        for arg in args:
            if isinstance(arg, ObjectProxy):
                if self._self_state.current_span_id:
                    self._self_state.record_object_usage(arg._self_obj_id, self._self_state.current_span_id)
                new_args.append(arg.__wrapped__)
            else:
                new_args.append(arg)
        
        new_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, ObjectProxy):
                if self._self_state.current_span_id:
                    self._self_state.record_object_usage(v._self_obj_id, self._self_state.current_span_id)
                new_kwargs[k] = v.__wrapped__
            else:
                new_kwargs[k] = v
        
        result = self.__wrapped__(*new_args, **new_kwargs)
        return result
    
    def __getattr__(self, name):
        """Record usage when an attribute is accessed"""
        if self._self_state.current_span_id and name not in ('_self_obj_id', '_self_state'):
            self._self_state.record_object_usage(self._self_obj_id, self._self_state.current_span_id)
        return super().__getattr__(name)


class ObjectTracer:
    """Traces object flow between functions"""
    
    def __init__(self, tracer: Optional[Tracer] = None):
        self.state = ObjectTraceState()
        self.tracer = tracer or Tracer()
    
    def observe(self, func: Callable) -> Callable:
        """Decorator to track object flow through a function"""
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get current trace and span from Judgment tracer context
            current_trace = current_trace_var.get()
            prev_span_id = current_span_var.get()
            
            if not current_trace:
                # Create a new trace if none exists
                with self.tracer.trace(func.__name__) as trace:
                    current_trace = trace
            
            # Create a new span for this function
            with current_trace.span(func.__name__) as span_context:
                # Get the current span ID from the context var after entering the span
                span_id = current_span_var.get()
                
                if not span_id:
                    # Fallback if span_id is not available via context var
                    span_id = str(uuid.uuid4())
                
                # Set this as the current span for object tracking
                self.state.current_span_id = span_id
                self.state.span_names[span_id] = func.__name__
                
                # Register the function call in our hierarchy
                self.state.register_function_call(span_id, func.__name__)
                
                # Push this span onto the call stack
                self.state.push_call_stack(span_id)
                
                try:
                    # Process args/kwargs, unwrapping proxies for the actual function call
                    new_args = []
                    for arg in args:
                        if isinstance(arg, ObjectProxy):
                            self.state.record_object_usage(arg._self_obj_id, span_id)
                            new_args.append(arg.__wrapped__)
                        else:
                            new_args.append(arg)
                    
                    new_kwargs = {}
                    for k, v in kwargs.items():
                        if isinstance(v, ObjectProxy):
                            self.state.record_object_usage(v._self_obj_id, span_id)
                            new_kwargs[k] = v.__wrapped__
                        else:
                            new_kwargs[k] = v
                    
                    # Add input metadata to the trace span if possible
                    spans = getattr(current_trace, 'trace_spans', [])
                    current_span = None
                    
                    # Find the current span in the trace
                    for s in spans:
                        if s.span_id == span_id:
                            current_span = s
                            break
                    
                    if current_span:
                        # Add input metadata to span
                        input_data = {
                            "args": [self._safe_repr(arg) for arg in args],
                            "kwargs": {k: self._safe_repr(v) for k, v in kwargs.items()}
                        }
                        
                        if hasattr(current_span, 'inputs') and current_span.inputs is not None:
                            if isinstance(current_span.inputs, dict):
                                current_span.inputs.update({"object_flow_inputs": input_data})
                        else:
                            current_span.inputs = {"object_flow_inputs": input_data}
                    
                    # Call the original function
                    result = func(*new_args, **new_kwargs)
                    
                    # Register the result as a new tracked object
                    obj_id = self.state.register_object(result, span_id, func.__name__)
                    
                    # Add output metadata to the span if possible
                    if current_span:
                        # Add output metadata to span
                        output_data = {
                            "object_id": obj_id,
                            "type": type(result).__name__,
                            "repr": self._safe_repr(result)
                        }
                        
                        if hasattr(current_span, 'output') and current_span.output is not None:
                            if isinstance(current_span.output, dict):
                                current_span.output.update({"object_flow_output": output_data})
                            else:
                                current_span.output = {"original": current_span.output, "object_flow_output": output_data}
                        else:
                            current_span.output = {"object_flow_output": output_data}
                    
                    # Wrap the result in a proxy
                    return ObjectProxy(result, obj_id, self.state)
                finally:
                    # Pop from the call stack
                    self.state.pop_call_stack()
                    
                    # Restore previous span
                    if prev_span_id:
                        self.state.current_span_id = prev_span_id
                    else:
                        self.state.current_span_id = None
        
        return wrapper
    
    def _safe_repr(self, obj: Any) -> str:
        """Safely get a string representation of an object"""
        try:
            repr_val = repr(obj)
            if len(repr_val) > 100:
                return repr_val[:97] + "..."
            return repr_val
        except Exception:
            return f"<{type(obj).__name__}>"
    
    def visualize_object_flow(self):
        """Generate a visualization of object flow between spans"""
        output = ["\n===== Object Flow Between Spans ====="]
        output.append("Format: Object ID (type) created in [Function] is used in: [Functions]")
        output.append("-" * 70)
        
        for obj_id, info in self.state.object_info.items():
            used_in_spans = [
                self.state.span_names.get(span_id, "unknown") 
                for span_id in info.usage_spans 
                if span_id != info.creator_span_id
            ]
            
            if used_in_spans:
                output.append(
                    f"Object {obj_id} ({info.type_name}: {info.value_repr}) created in "
                    f"[{info.creator_name}] is used in: {', '.join(used_in_spans)}"
                )
        
        output.append("\n===== Function Call Hierarchy =====")
        output.append("Format: Function (span_id) → Called Functions")
        output.append("-" * 70)
        
        # Find root functions (no parent)
        root_functions = [
            call for call in self.state.function_calls.values()
            if not call.parent_span_id
        ]
        
        # Display call hierarchy
        for root in root_functions:
            self._print_call_hierarchy(root, "", output)
        
        output.append("\n===== All Spans =====")
        output.append("Format: Span ID: Function Name")
        output.append("-" * 40)
        
        for span_id, name in self.state.span_names.items():
            output.append(f"{span_id[:8]}...: {name}")
        
        return "\n".join(output)
    
    def _print_call_hierarchy(self, call: FunctionCall, indent: str, output: List[str]):
        """Recursively print the call hierarchy for visualization"""
        output_object = ""
        if call.output_object and call.output_object in self.state.object_info:
            info = self.state.object_info[call.output_object]
            output_object = f" → returns {call.output_object} ({info.type_name})"
        
        output.append(f"{indent}{call.function_name} ({call.span_id[:8]}...){output_object}")
        
        for child_id in call.child_spans:
            if child_id in self.state.function_calls:
                child = self.state.function_calls[child_id]
                self._print_call_hierarchy(child, indent + "  ", output)

    def get_trace_metadata(self) -> Dict[str, Any]:
        """Get object flow data in a format suitable for trace metadata"""
        return {
            "object_flow": {
                "objects": {
                    obj_id: info.model_dump()
                    for obj_id, info in self.state.object_info.items()
                },
                "span_objects": dict(self.state.span_to_objects),
                "span_names": self.state.span_names,
                "function_calls": {
                    span_id: call.dict()
                    for span_id, call in self.state.function_calls.items()
                },
                "dependency_graph": {
                    caller: list(callees)
                    for caller, callees in self.state.dependency_graph.items()
                }
            }
        }


# Singleton instance for easy usage
_default_tracer = ObjectTracer()

def observe(func):
    """Decorator to track object flow through a function using the default tracer"""
    return _default_tracer.observe(func)

def visualize_object_flow():
    """Visualize object flow using the default tracer"""
    return _default_tracer.visualize_object_flow() 