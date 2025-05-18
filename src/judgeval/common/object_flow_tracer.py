"""
Object flow tracing for tracking data flow between functions.
This module enhances the existing tracing system by tracking how
objects created in one function are used in other functions.
"""
import functools
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple, Optional
import wrapt
import contextvars
import inspect

# Context variables to track current span and objects
current_flow_span_var = contextvars.ContextVar('current_flow_span', default=None)
object_registry = contextvars.ContextVar('object_registry', default={})
object_usage = contextvars.ContextVar('object_usage', default=defaultdict(set))
span_to_objects = contextvars.ContextVar('span_to_objects', default=defaultdict(list))
span_names = contextvars.ContextVar('span_names', default={})
span_parent_child = contextvars.ContextVar('span_parent_child', default=defaultdict(list))
function_calls = contextvars.ContextVar('function_calls', default=defaultdict(set))

# Helper to get a unique object ID
object_counter = contextvars.ContextVar('object_counter', default=0)

def get_next_object_id():
    """Get the next object ID and increment the counter"""
    counter = object_counter.get()
    object_counter.set(counter + 1)
    return counter

class ObjectFlowProxy(wrapt.ObjectProxy):
    """
    A proxy object that tracks when an object is accessed.
    When the object is used in another function, it records this information.
    """
    def __init__(self, wrapped, object_id=None, creator_span=None):
        super().__init__(wrapped)
        self._self_object_id = object_id or f"obj_{get_next_object_id()}"
        self._self_creator_span = creator_span or current_flow_span_var.get()
        
        # Register this object in the registry
        registry = object_registry.get()
        registry[self._self_object_id] = self
        object_registry.set(registry)
        
        # Initialize usage tracking
        usage_dict = object_usage.get()
        if self._self_object_id not in usage_dict:
            usage_dict[self._self_object_id] = set()
        object_usage.set(usage_dict)
    
    def __getattr__(self, name):
        """Track attribute access and record usage"""
        attr = super().__getattr__(name)
        self._record_usage()
        return attr
    
    def __call__(self, *args, **kwargs):
        """Track when the object is called as a function"""
        current_span = current_flow_span_var.get()
        
        # When a function is called, record the relationship between caller and callee
        # The current span is the caller, and after the call, the new current span will be the callee
        caller_span = current_span
        
        # Record usage of this object in the current span
        self._record_usage()
        
        # Call the original function
        result = self.__wrapped__(*args, **kwargs)
        
        # After the call, the current_flow_span_var contains the callee's span ID
        callee_span = current_flow_span_var.get()
        
        # If we have both a caller and a callee, and they're different,
        # record this as a direct function call
        if caller_span and callee_span and caller_span != callee_span:
            # Record this call in the function_calls dictionary
            calls_dict = function_calls.get()
            calls_dict[caller_span].add(callee_span)
            function_calls.set(calls_dict)
            
            # Also update parent-child relationship
            parent_child_dict = span_parent_child.get()
            if callee_span not in parent_child_dict[caller_span]:
                parent_child_dict[caller_span].append(callee_span)
            span_parent_child.set(parent_child_dict)
        
        return result
    
    def __getitem__(self, key):
        """Track when the object is accessed using indexing"""
        self._record_usage()
        return self.__wrapped__[key]
    
    def _record_usage(self):
        """Record that this object is being used in the current span"""
        current_span = current_flow_span_var.get()
        if current_span and current_span != self._self_creator_span:
            # Record that this object (created in self._self_creator_span) 
            # is being used in the current_span
            usage_dict = object_usage.get()
            usage_dict[self._self_object_id].add(current_span)
            object_usage.set(usage_dict)
            
            # Also record in the function calls dictionary
            # Record that the current span was called by the creator span
            calls_dict = function_calls.get()
            calls_dict[self._self_creator_span].add(current_span)
            function_calls.set(calls_dict)
            
            # Update parent-child relationship
            parent_child_dict = span_parent_child.get()
            if current_span not in parent_child_dict[self._self_creator_span]:
                parent_child_dict[self._self_creator_span].append(current_span)
            span_parent_child.set(parent_child_dict)

def reset_object_flow_tracking():
    """Reset all object flow tracking state"""
    current_flow_span_var.set(None)
    object_registry.set({})
    object_usage.set(defaultdict(set))
    span_to_objects.set(defaultdict(list))
    span_names.set({})
    span_parent_child.set(defaultdict(list))
    function_calls.set(defaultdict(set))
    object_counter.set(0)

def visualize_object_flow() -> str:
    """
    Generate a visualization of the object flow.
    
    Returns:
        str: A formatted string showing the object flow between functions.
    """
    registry = object_registry.get()
    usage = object_usage.get()
    span_objects = span_to_objects.get()
    names = span_names.get()
    parent_children = span_parent_child.get()
    calls = function_calls.get()
    
    output = ["\n===== Object Flow Graph Visualization ====="]
    
    # Visualize function call hierarchy
    output.append("\n----- Function Call Hierarchy -----")
    output.append("Format: Function → Directly Called Functions")
    output.append("----------------------------------------------------------------------")
    for span_id, name in names.items():
        called_functions = calls.get(span_id, set())
        called_names = [names.get(called_id, called_id[:8]+"...") for called_id in called_functions]
        output.append(f"{name} → returns {span_objects.get(span_id, ['Unknown'])[0] if span_objects.get(span_id) else 'Nothing'}")
    
    # Visualize data flow edges
    output.append("\n----- Data Flow Edges -----")
    output.append("Format: Object (type) created in [Function] → used in [Functions]")
    output.append("----------------------------------------------------------------------")
    for obj_id, used_in_spans in usage.items():
        if not used_in_spans:
            continue
            
        # Get the object itself
        obj = registry.get(obj_id)
        if not obj:
            continue
            
        # Get the creator's information
        creator_span = getattr(obj, '_self_creator_span', None)
        creator_name = names.get(creator_span, "unknown") if creator_span else "unknown"
        
        # Get the users' information
        user_names = [names.get(span_id, "unknown") for span_id in used_in_spans]
        
        # Get the object's type
        obj_type = type(obj.__wrapped__).__name__ if hasattr(obj, '__wrapped__') else type(obj).__name__
        
        # Create the output line
        output.append(f"Object {obj_id} ({obj_type}) created in [{creator_name}] → used in: {', '.join(user_names)}")
    
    # List all span IDs for reference
    output.append("\n===== All Function Spans =====")
    output.append("Format: Span ID: Function Name")
    output.append("----------------------------------------")
    for span_id, name in names.items():
        output.append(f"{span_id[:8]}...: {name}")
    
    return "\n".join(output)

def generate_trace_json() -> dict:
    """
    Generate a JSON structure of the object flow trace.
    
    Returns:
        dict: A dictionary with nodes, edges, and objects
    """
    span_map = span_names.get()
    objects_map = span_to_objects.get()
    usage_map = object_usage.get()
    parent_child = span_parent_child.get()
    function_call_map = function_calls.get()
    registry = object_registry.get()
    
    # Create the trace file
    flow_data = {
        "nodes": [],
        "edges": [],
        "objects": {},
        "metadata": {
            "dependency_graph": {},
            "call_hierarchy": {}
        }
    }
    
    # Add nodes for each function span
    for span_id, name in span_map.items():
        # Find the objects created in this span
        created_objects = [obj_id for obj_id, obj in registry.items() 
                          if hasattr(obj, '_self_creator_span') and obj._self_creator_span == span_id]
        
        # Find objects used by this span
        used_objects = []
        for obj_id, used_in_spans in usage_map.items():
            if span_id in used_in_spans:
                used_objects.append(obj_id)
        
        # Find output object (if any)
        output_object = objects_map.get(span_id, [None])[0] if objects_map.get(span_id) else None
        
        # Add the node
        node = {
            "id": span_id,
            "name": name,
            "type": "function",
            "input_objects": used_objects,
            "output_object": output_object,
            "parent_id": None,  # Will be filled in later based on call hierarchy
            "child_ids": []     # Will be filled in later based on call hierarchy
        }
        flow_data["nodes"].append(node)
    
    # Infer the function call hierarchy based on:
    # 1. Existing recorded function calls
    # 2. Object usage patterns (if A creates an object that B uses, A likely called B)
    call_hierarchy = {}
    
    # First, use existing function calls if any
    for caller_id, callee_ids in function_call_map.items():
        if caller_id not in call_hierarchy:
            call_hierarchy[caller_id] = []
        for callee_id in callee_ids:
            if callee_id not in call_hierarchy[caller_id]:
                call_hierarchy[caller_id].append(callee_id)
    
    # Second, infer based on the standard function call pattern
    # When function A calls function B, typically:
    # 1. A calls B
    # 2. B creates an object
    # 3. A uses the object created by B
    for obj_id, used_in_spans in usage_map.items():
        # Get the creator span
        obj = registry.get(obj_id)
        if not obj:
            continue
            
        creator_span = getattr(obj, '_self_creator_span', None)
        if not creator_span:
            continue
        
        # For each span that uses this object
        for user_span in used_in_spans:
            # Skip self-usage
            if user_span == creator_span:
                continue
                
            # If A uses an object created by B, typically B was called by A
            # So add B as a callee of A
            if user_span not in call_hierarchy:
                call_hierarchy[user_span] = []
            if creator_span not in call_hierarchy[user_span]:
                call_hierarchy[user_span].append(creator_span)
    
    # Third, update parent-child relationships based on call hierarchy
    for node in flow_data["nodes"]:
        node_id = node["id"]
        
        # Set child_ids based on call hierarchy
        if node_id in call_hierarchy:
            node["child_ids"] = call_hierarchy[node_id]
        
        # Set parent_id based on call hierarchy
        for potential_parent, children in call_hierarchy.items():
            if node_id in children:
                node["parent_id"] = potential_parent
                break
    
    # Add the call hierarchy to metadata
    flow_data["metadata"]["call_hierarchy"] = call_hierarchy
    
    # Add edges for data flow
    for obj_id, used_in_spans in usage_map.items():
        # Get the creator span
        obj = registry.get(obj_id)
        if not obj:
            continue
            
        creator_span = getattr(obj, '_self_creator_span', None)
        if not creator_span:
            continue
        
        # Add an edge from creator to each user
        for user_span in used_in_spans:
            # Skip self-usage
            if user_span == creator_span:
                continue
                
            edge = {
                "id": str(uuid.uuid4()),
                "source": creator_span,
                "target": user_span,
                "label": f"Data flow: {obj_id}",
                "edge_type": "data_flow"
            }
            flow_data["edges"].append(edge)
            
            # Also update the dependency graph
            if user_span not in flow_data["metadata"]["dependency_graph"]:
                flow_data["metadata"]["dependency_graph"][user_span] = []
            flow_data["metadata"]["dependency_graph"][user_span].append(creator_span)
    
    # Add function call edges
    for caller_id, callee_ids in call_hierarchy.items():
        for callee_id in callee_ids:
            # Skip self-calls
            if callee_id == caller_id:
                continue
                
            # Get function names
            caller_name = span_map.get(caller_id, "unknown")
            callee_name = span_map.get(callee_id, "unknown")
            
            # Create a call hierarchy edge
            edge = {
                "id": str(uuid.uuid4()),
                "source": caller_id,
                "target": callee_id,
                "label": f"Function call: {caller_name} -> {callee_name}",
                "edge_type": "call_hierarchy"
            }
            flow_data["edges"].append(edge)
    
    # Add objects
    for obj_id, obj in registry.items():
        obj_creator_span = getattr(obj, '_self_creator_span', None)
        obj_wrapped = getattr(obj, '__wrapped__', obj)
        obj_type = type(obj_wrapped).__name__
        
        # Try to get a string representation of the object
        try:
            obj_repr = repr(obj_wrapped)
            if len(obj_repr) > 100:
                obj_repr = obj_repr[:100] + "..."
        except:
            obj_repr = "<<Representation unavailable>>"
        
        # Add to objects dictionary
        flow_data["objects"][obj_id] = {
            "object_id": obj_id,
            "creator_span_id": obj_creator_span,
            "creator_name": span_map.get(obj_creator_span) if obj_creator_span else None,
            "value_repr": obj_repr,
            "type_name": obj_type,
            "usage_spans": list(usage_map.get(obj_id, []))
        }
    
    return flow_data 