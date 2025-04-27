import os
import json
from openai import OpenAI  # Back to standard OpenAI client
from judgeval.common.tracer import Tracer, wrap
import warnings # Import warnings
import requests # Added for image download
import base64   # Added for base64 encoding

# Try to import tiktoken and provide guidance if missing
try:
    import tiktoken
except ImportError:
    print("Warning: 'tiktoken' is not installed. Estimated token breakdown will not be available.")
    print("You can install it using: pip install tiktoken")
    tiktoken = None # Set to None so we can check later

# 1. Define dummy functions
def get_current_weather(location: str, unit: str = "fahrenheit"):
    """Get the current weather in a given location."""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": "celsius"})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "72", "unit": "fahrenheit"})
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": "celsius"})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})

def describe_image(image_url: str):
    """Describes the content of the image found at the given URL."""
    print(f"Simulating image description for: {image_url}")
    # Simulate analysis based on URL - replace with actual logic if needed
    if "cat" in image_url:
        description = {"main_subject": "cat", "activity": "sleeping", "setting": "sunny window"}
    elif "dog" in image_url:
        description = {"main_subject": "dog", "activity": "playing fetch", "setting": "park"}
    else:
        description = {"main_subject": "unknown", "activity": "unknown", "setting": "unknown"}
    return json.dumps(description)

# 2. Define tools for the LLM - update format to match Responses API expectations
tools = [
    { 
        "type": "function",
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        }
    },
    {
        "type": "function",
        "name": "describe_image",
        "description": "Describes the content of the image found at the given URL",
        "parameters": {
            "type": "object",
            "properties": {
                "image_url": {
                    "type": "string",
                    "description": "The URL of the image to describe.",
                },
            },
            "required": ["image_url"],
        }
    }
]

# --- Download and Encode Image ---
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
mime_type = "image/jpeg" # Assuming based on URL ending
encoded_image_data = None
print(f"Downloading image from: {image_url}")
try:
    response = requests.get(image_url, stream=True)
    response.raise_for_status() # Raise an exception for bad status codes
    image_bytes = response.content
    encoded_image_data = base64.b64encode(image_bytes).decode('utf-8')
    print("Image downloaded and encoded successfully.")
except requests.exceptions.RequestException as e:
    print(f"Error downloading image: {e}. Proceeding without image.")
except Exception as e:
    print(f"An unexpected error occurred during image processing: {e}. Proceeding without image.")

# --- Setup ---
# Ensure necessary environment variables are set
api_key = os.getenv("JUDGMENT_API_KEY")
org_id = os.getenv("JUDGMENT_ORG_ID")
openai_api_key = os.getenv("OPENAI_API_KEY")

if not all([api_key, org_id, openai_api_key]):
    print("Error: Please set JUDGMENT_API_KEY, JUDGMENT_ORG_ID, and OPENAI_API_KEY environment variables.")
    exit(1)

# 2. Initialize Tracer and OpenAI client
tracer = Tracer(api_key=api_key, organization_id=org_id, project_name="function_image_call_demo")
client = wrap(OpenAI(api_key=openai_api_key))  # Standard OpenAI client

# 3. Make an API call that uses the function(s)
print("Making OpenAI call with weather and image tools using responses.create...")
with tracer.trace("openai_multi_tool_trace") as trace:
    try:
        # Create input structure for responses.create
        input_content = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "What's the weather like in San Francisco and can you describe this image?"}
                ]
            }
        ]
        
        # Add image if available
        if encoded_image_data:
            input_content[0]["content"].append({
                "type": "input_image",
                "image_url": f"data:{mime_type};base64,{encoded_image_data}"
            })
        
        # First call to get the function call request(s)
        response = client.responses.create(
            model="gpt-4.1-mini",  # Using a model that supports vision and functions
            input=input_content,
            tools=tools,
        )

        # Extract response data
        print(f"Response received: {response}")
        
        response_output = response.output
        
        # Check if there are any tool calls (function executions)
        tool_calls = []
        for output_item in response_output:
            if output_item.type == "tool_calls":
                tool_calls.extend(output_item.tool_calls)
        
        if tool_calls:
            print(f"Function call(s) requested by model: {[tc.name for tc in tool_calls]}")
            
            # Prepare tool results to send back
            tool_results = []
            available_functions = {
                "get_current_weather": get_current_weather,
                "describe_image": describe_image,
            }
            
            # Execute each tool call
            for tool_call in tool_calls:
                function_name = tool_call.name
                if function_name in available_functions:
                    function_to_call = available_functions[function_name]
                    try:
                        # Parse arguments and call function
                        function_args = json.loads(tool_call.arguments)
                        print(f"Executing function: {function_name}({function_args})")
                        
                        if function_name == "get_current_weather":
                            function_response = function_to_call(
                                location=function_args.get("location"),
                                unit=function_args.get("unit"),
                            )
                        elif function_name == "describe_image":
                            function_response = function_to_call(
                                image_url=function_args.get("image_url")
                            )
                        else:
                            function_response = json.dumps({"error": f"Unknown function {function_name}"})
                            
                        print(f"Function response: {function_response}")
                        
                        # Add result to tool_results
                        tool_results.append({
                            "type": "tool_result",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(function_response)
                        })
                        
                    except Exception as e:
                        print(f"Error executing function {function_name}: {e}")
                        tool_results.append({
                            "type": "tool_result",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps({"error": str(e)})
                        })
                else:
                    print(f"Warning: Model requested unknown function '{function_name}'")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps({"error": "function not found"})
                    })
            
            # Second call with tool results to get final response
            print("Sending function response(s) back to model...")
            second_response = client.responses.create(
                model="gpt-4.1-mini",
                input=tool_results,
                previous_response_id=response.id  # Link back to previous response
            )
            
            # Extract final text response
            final_response = None
            for output_item in second_response.output:
                if output_item.type == "message" and output_item.content:
                    for content_item in output_item.content:
                        if content_item.type == "output_text":
                            final_response = content_item.text
                            break
                    if final_response:
                        break
            
            print(f"Final model response: {final_response}")
            trace.record_output({"final_response": final_response})
        else:
            # Handle case with no tool calls
            final_response = None
            for output_item in response_output:
                if output_item.type == "message" and output_item.content:
                    for content_item in output_item.content:
                        if content_item.type == "output_text":
                            final_response = content_item.text
                            break
                    if final_response:
                        break
            
            print(f"Model direct response (no tool calls): {final_response}")
            trace.record_output({"final_response": final_response})
    
    except Exception as e:
        print(f"An error occurred during the API call or processing: {e}")
        import traceback
        traceback.print_exc()
        trace.record_output({"error": str(e)})

# 4. Print token counts from the trace
print("\n--- Trace Summary ---")
trace_id, trace_data = trace.save() # Save the trace (optional, but good practice)
print(f"Trace ID: {trace_id}")

# Use the token_counts directly from the saved trace data for overall totals
token_counts_overall = trace_data.get('token_counts', {}) # Use .get for safety

print("\nToken Counts (Overall API Reported - from saved trace data):")
if token_counts_overall:
    print(f"  Prompt Tokens: {token_counts_overall.get('prompt_tokens', 'N/A')}")
    print(f"  Completion Tokens: {token_counts_overall.get('completion_tokens', 'N/A')}")
    print(f"  Total Tokens: {token_counts_overall.get('total_tokens', 'N/A')}")
    print(f"  Total Cost (USD): ${token_counts_overall.get('total_cost_usd', 0):.6f}")
else:
    print("  Overall token counts not found in trace data.")


# Calculate and print the estimated breakdown if tiktoken is available
if tiktoken:
    print("\nCalculating estimated token breakdown using tiktoken...")
    try:
        # Calculate counts with breakdown=True using the detailed entries
        # Note: trace.save() already calculates this if breakdown=True in calculate_token_counts
        # We might not need to recalculate, just access trace_data['token_counts']['breakdown']
        # Let's recalculate here for clarity, though it might be redundant if save() did it.
        token_counts_detailed = trace.calculate_token_counts(trace_data['entries'], breakdown=True)
        breakdown_data = token_counts_detailed.get('breakdown', {})

        print("\nToken Counts (Estimated Breakdown):")
        # Print the main counts from the breakdown calculation (should match overall)
        print(f"  Prompt Tokens (API Reported Total): {token_counts_detailed.get('prompt_tokens', 'N/A')}")
        print(f"  Completion Tokens (API Reported Total): {token_counts_detailed.get('completion_tokens', 'N/A')}")
        print(f"  Total Tokens (API Reported Total): {token_counts_detailed.get('total_tokens', 'N/A')}")
        print(f"  Total Cost (USD): ${token_counts_detailed.get('total_cost_usd', 0):.6f}")

        if breakdown_data:
            print("  Estimated Breakdown Components:")
            # Note: 'estimated_image_prompt_tokens' includes image tokens AND potentially other non-text/func overhead
            print(f"    Est. Text Prompt Tokens: {breakdown_data.get('estimated_text_prompt_tokens', 'N/A')}")
            print(f"    Est. Function Prompt Tokens (tools def + tool role content): {breakdown_data.get('estimated_function_prompt_tokens', 'N/A')}")
            print(f"    Est. Image Prompt Tokens (Image/Overhead): {breakdown_data.get('estimated_image_prompt_tokens', 'N/A')}")
            print("    ------")
            print(f"    Est. Text Completion Tokens: {breakdown_data.get('estimated_text_completion_tokens', 'N/A')}")
            print(f"    Est. Function Completion Tokens (tool_calls object): {breakdown_data.get('estimated_function_completion_tokens', 'N/A')}")
            print(f"    Est. Image Completion Tokens (Overhead): {breakdown_data.get('estimated_image_completion_tokens', 'N/A')}")
            print("  Estimated Cost Breakdown (USD):")
            print(f"    Est. Text Prompt Cost: ${breakdown_data.get('estimated_text_prompt_tokens_cost_usd', 0):.6f}")
            print(f"    Est. Function Prompt Cost: ${breakdown_data.get('estimated_function_prompt_tokens_cost_usd', 0):.6f}")
            print(f"    Est. Image Prompt Cost: ${breakdown_data.get('estimated_image_prompt_tokens_cost_usd', 0):.6f}")
            print("    ------")
            print(f"    Est. Text Completion Cost: ${breakdown_data.get('estimated_text_completion_tokens_cost_usd', 0):.6f}")
            print(f"    Est. Function Completion Cost: ${breakdown_data.get('estimated_function_completion_tokens_cost_usd', 0):.6f}")
            print(f"    Est. Image Completion Cost: ${breakdown_data.get('estimated_image_completion_tokens_cost_usd', 0):.6f}")
            print("    ------")
            print(f"    Est. Total Breakdown Cost: ${breakdown_data.get('estimated_total_cost_usd', 0):.6f}")

        else:
            print("  Breakdown data not found in calculation results.")

    except ImportError:
         # This case should be caught by the initial check, but added for robustness
         warnings.warn("Tiktoken import failed during calculation.")
         print("\nCould not calculate estimated breakdown because 'tiktoken' is not installed.")
    except Exception as e:
        warnings.warn(f"Error during token breakdown calculation: {e}")
        print(f"\nCould not calculate estimated breakdown: {e}")
else:
     print("\nSkipping estimated token breakdown because 'tiktoken' is not installed.")


print("\nDemo finished.") 