import os
import json
from openai import OpenAI
from judgeval.common.tracer import Tracer, wrap
import warnings # Import warnings

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

# 2. Define tools for the LLM
tools = [
    { # Weather tool
        "type": "function",
        "function": {
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
            },
        },
    },
    { # Image description tool
        "type": "function",
        "function": {
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
            },
        },
    }
]

# 3. Update the prompt to request both weather and image description
#    Using the OpenAI content list format to include the image URL
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What's the weather like in San Francisco and can you describe this image?"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg", # Use a real image URL
                },
            },
        ],
    }
]


# --- Setup ---
# Ensure necessary environment variables are set
api_key = os.getenv("JUDGMENT_API_KEY")
org_id = os.getenv("JUDGMENT_ORG_ID")
openai_api_key = os.getenv("OPENAI_API_KEY")

if not all([api_key, org_id, openai_api_key]):
    print("Error: Please set JUDGMENT_API_KEY, JUDGMENT_ORG_ID, and OPENAI_API_KEY environment variables.")
    exit(1)

# 2. Initialize Tracer and wrap OpenAI client
tracer = Tracer(api_key=api_key, organization_id=org_id, project_name="function_image_call_demo") # Updated project name slightly
client = wrap(OpenAI()) # Wrap the synchronous OpenAI client


# 3. Make an API call that uses the function(s)
print("Making OpenAI call with weather and image tools...")
with tracer.trace("openai_multi_tool_trace") as trace:
    try:
        # First call to get the function call request(s)
        response = client.chat.completions.create(
            model="gpt-4o-mini", # Use a model that supports vision and function calling
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        # --- Function Call Handling ---
        if tool_calls:
            print(f"Function call(s) requested by model: {[tc.function.name for tc in tool_calls]}")
            # Extend conversation with assistant's reply (containing tool requests)
            messages.append(response_message)

            # 4. Add the new function to available_functions
            available_functions = {
                "get_current_weather": get_current_weather,
                "describe_image": describe_image, # Add the new function here
            }

            # Loop through each tool call requested by the model
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                if function_name in available_functions:
                    function_to_call = available_functions[function_name]
                    try:
                        function_args = json.loads(tool_call.function.arguments)
                        print(f"Executing function: {function_name}({function_args})")
                        # Call the function with arguments
                        # Need to handle different arg names potentially
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
                        # Append the function response to the messages list
                        messages.append(
                            {
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": function_name,
                                "content": function_response,
                            }
                        )
                    except json.JSONDecodeError:
                         print(f"Error decoding arguments for {function_name}: {tool_call.function.arguments}")
                         messages.append({ "role": "tool", "tool_call_id": tool_call.id, "name": function_name, "content": '{"error": "invalid arguments json"}'})
                    except Exception as func_exc:
                         print(f"Error executing function {function_name}: {func_exc}")
                         messages.append({ "role": "tool", "tool_call_id": tool_call.id, "name": function_name, "content": f'{{"error": "{str(func_exc)}"}}'})

                else:
                     print(f"Warning: Model requested unknown function '{function_name}'")
                     messages.append({ "role": "tool", "tool_call_id": tool_call.id, "name": function_name, "content": '{"error": "function not found"}'})


            # Second call: get final response from model after providing tool results
            print("Sending function response(s) back to model...")
            second_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
            )
            final_response = second_response.choices[0].message.content
            print(f"Final model response: {final_response}")
            trace.record_output({"final_response": final_response})
        else:
            # Handle case where model responds directly without function calls
            print("No function call made by model.")
            final_response = response_message.content
            print(f"Model response: {final_response}")
            trace.record_output({"final_response": final_response})

    except Exception as e:
        print(f"An error occurred during the API call or processing: {e}")
        trace.record_output({"error": str(e)}) # Record the error in the trace


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