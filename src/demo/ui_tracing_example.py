"""
Example demonstrating the UI Tracing functionality in judgeval.

This example shows how to use the object tracing and UI tracing modules
to visualize data flow between functions.
"""
import os
import sys
import math

import os
import json
import uuid
from judgeval.common.tracer import Tracer
import contextvars

# Force OBJECT_FLOW_AVAILABLE to True
import judgeval.common.tracer
judgeval.common.tracer.OBJECT_FLOW_AVAILABLE = True

# Patch the API key validation for demo purposes
import judgeval.common.utils
def mock_validate_api_key(*args, **kwargs):
    return True, {"message": "Valid API key for development"}
judgeval.common.utils.validate_api_key = mock_validate_api_key

# Patch the save_trace method for demo purposes
original_save_trace = judgeval.common.tracer.TraceManagerClient.save_trace
def mock_save_trace(self, trace_data):
    print("Trace would be saved to server in production.")
    return {"message": "success", "ui_results_url": "https://example.com/trace"}
judgeval.common.tracer.TraceManagerClient.save_trace = mock_save_trace

# Initialize tracer with object flow tracking enabled
tracer = Tracer(
    api_key=os.getenv("JUDGMENT_API_KEY", "demo_key"),
    organization_id=os.getenv("JUDGMENT_ORG_ID", "demo_org"),
    project_name="object_flow_demo",
    enable_object_flow=True,  # Enable object flow tracking
    deep_tracing=False  # Disable deep tracing to avoid _TRACE_FILEPATH_BLOCKLIST issues
)

# Define some functions that we'll trace
@tracer.observe(object_flow=True)
def calculate_area(radius):
    """Calculate the area of a circle"""
    return math.pi * radius**2

@tracer.observe(object_flow=True)
def calculate_circumference(radius):
    """Calculate the circumference of a circle"""
    return 2 * math.pi * radius

@tracer.observe(object_flow=True)
def create_geometry(radius):
    """Create geometry information for a circle"""
    area = calculate_area(radius)
    circumference = calculate_circumference(radius)
    
    return {
        "shape": "circle",
        "radius": radius,
        "area": area,
        "circumference": circumference
    }

@tracer.observe(object_flow=True)
def render_ui(geometry):
    """Render a UI representation of the geometry"""
    if geometry["shape"] == "circle":
        header = f"Circle with radius {geometry['radius']}"
        content = f"Area: {geometry['area']:.2f}\nCircumference: {geometry['circumference']:.2f}"
    else:
        header = f"Unknown shape"
        content = "No data available"
    
    # Create a simple UI representation
    ui = {
        "title": "Geometry Visualization",
        "header": header,
        "content": content,
        "footer": "© 2023 Example App"
    }
    
    return ui

@tracer.observe(object_flow=True)
def main():
    """Main function that coordinates the overall flow"""
    # Create geometry data for a circle with radius 5
    geometry = create_geometry(5)
    
    # Render UI based on the geometry data
    ui = render_ui(geometry)
    
    # Return the UI as a formatted string
    return f"{ui['title']}\n\n{ui['header']}\n{ui['content']}\n\n{ui['footer']}"

if __name__ == "__main__":
    # Run the main function and print the result
    result = main()
    print("\nResult:")
    print(result)
    
    # Visualize the object flow
    print("\nObject Flow Visualization:")
    print(tracer.visualize_object_flow())
    
    # Get the trace data and save to file
    flow_data = tracer.generate_trace_json()
    
    # Print the function call hierarchy
    print("\n=== Function Call Hierarchy ===")
    if "metadata" in flow_data and "call_hierarchy" in flow_data["metadata"]:
        call_hierarchy = flow_data["metadata"]["call_hierarchy"]
        
        if not call_hierarchy:
            print("  Empty - No call hierarchy recorded")
        else:
            from judgeval.common.object_flow_tracer import span_names
            names = span_names.get()
            
            for parent_id, children in call_hierarchy.items():
                parent_name = names.get(parent_id, parent_id[:8]+"...")
                children_names = [names.get(child_id, child_id[:8]+"...") for child_id in children]
                children_str = ", ".join(children_names)
                print(f"  {parent_name} -> {children_str}")
    
    # Save to file
    with open("ui_trace.json", "w") as f:
        json.dump(flow_data, f, indent=2)
    print(f"\nTrace data saved to ui_trace.json")