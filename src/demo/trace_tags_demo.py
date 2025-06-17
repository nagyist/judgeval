from judgeval.common.tracer import Tracer

# Initialize the tracer
judgment = Tracer(project_name="metadata_demo")

@judgment.observe(span_type="tool")
def my_tool():
    """Demonstrate various ways to set metadata on a trace"""
    
    # Method 3: Using set_metadata with key-value pairs
    judgment.set_metadata("has_notification", True)
    judgment.set_metadata("overwrite", False)
    
    # Method 4: Adding individual tags (as strings are converted to lists)
    judgment.set_metadata("tags", "important")  # This becomes ["important"]
    
    # Print the current metadata to verify
    print("=== Current Trace Metadata ===")
    judgment.print_metadata()
    
    return "Tool execution completed"

@judgment.observe(span_type="llm")
def another_function():
    """Another function to show metadata persistence within the same trace"""
    
    # Set additional metadata
    judgment.set_metadata(rules=["no-pii", "content-filter"])
    
    # Print metadata again to show it persists
    print("\n=== Metadata After Second Function ===")
    judgment.print_metadata()
    
    return "LLM call completed"

def main():
    """Main function to run the demo"""
    print("Testing set_metadata and related functions...\n")
    
    # Start a trace and run our tools
    with judgment.trace("metadata_test_trace") as trace:
        # IMPORTANT: Set metadata BEFORE creating any spans!
        print("=== Setting metadata BEFORE creating spans ===")
        trace.set_metadata(
            customer_id="cust_12345",
            tags=["production", "api-call"],
            name="Metadata Test Trace"
        )
        
        print("=== Initial Trace Metadata ===")
        trace.print_metadata()
        print()
        
        # Method 2: Using convenience methods (this will update after first save)
        judgment.set_customer_id("cust_67890")  # This will overwrite the previous customer_id
        judgment.set_tags(["staging", "debug"])  # This will overwrite the previous tags
        
        # Now run the tools (first span creation will save the trace with metadata)
        result1 = my_tool()
        result2 = another_function()
        
        print(f"\nResults: {result1}, {result2}")
        
        # Final metadata state
        print("\n=== Final Trace Metadata ===")
        trace.print_metadata()
        
        # IMPORTANT: Manually save the trace at the end to ensure latest metadata is persisted
        print("\n=== Saving final trace with all metadata ===")
        trace_id, response = trace.save_with_rate_limiting(overwrite=True, final_save=True)
        print(f"Final save completed for trace: {trace_id}")

if __name__ == "__main__":
    main() 