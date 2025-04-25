# vision_breakdown_demo.py

import os
import sys # Add sys import
import asyncio
from openai import OpenAI, AsyncOpenAI # Added AsyncOpenAI
from dotenv import load_dotenv
import json # Added json import here for clarity
import base64 # Added for base64 encoding
from typing import List, Dict, Any, Optional # Added for type hinting

# Dynamically add the src directory to sys.path
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Now the absolute import should work
from judgeval.common.tracer import Tracer, wrap # Removed TraceClient as it's not used directly

# --- Initialization --- (Moved Tracer init up)

# Load .env file from the project root
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '.env')
load_dotenv(dotenv_path=dotenv_path)
print(f"Attempting to load .env from: {os.path.abspath(dotenv_path)}")

# Ensure necessary environment variables are set
judgment_api_key = os.getenv("JUDGMENT_API_KEY")
judgment_org_id = os.getenv("JUDGMENT_ORG_ID")
openai_api_key = os.getenv("OPENAI_API_KEY")

if not judgment_api_key:
    print("Error: JUDGMENT_API_KEY not found.")
if not judgment_org_id:
    print("Error: JUDGMENT_ORG_ID not found.")
if not openai_api_key:
    print("Error: OPENAI_API_KEY not found.")

if not all([judgment_api_key, judgment_org_id, openai_api_key]):
    print("Please ensure JUDGMENT_API_KEY, JUDGMENT_ORG_ID, and OPENAI_API_KEY are set in your .env file.")
    sys.exit(1) # Exit if keys are missing

# Initialize Tracer *before* decorated functions are defined
judgment = Tracer(
    api_key=judgment_api_key,
    organization_id=judgment_org_id,
    project_name="vision_breakdown_demo" # Give a specific project name
)

# Wrap the ASYNC OpenAI client
# NOTE: The client needs to be wrapped for the decorator to capture LLM calls
aclient = wrap(AsyncOpenAI(api_key=openai_api_key))


# --- Helper Function ---
def encode_image_to_base64(image_path: str) -> Optional[str]:
    """Encodes a local image file to a base64 data URI."""
    try:
        # Infer MIME type (simple version)
        mime_type = "image/jpeg" # Default
        if image_path.lower().endswith(".png"):
            mime_type = "image/png"
        elif image_path.lower().endswith(".gif"):
             mime_type = "image/gif"
        elif image_path.lower().endswith(".webp"):
             mime_type = "image/webp"

        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return f"data:{mime_type};base64,{encoded_string}"
    except FileNotFoundError:
        print(f"Error: Local image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

# --- Function Definitions ---

@judgment.observe(name="vision_analysis_observed") # Apply the decorator
async def analyze_image_content(client: AsyncOpenAI, content_list: List[Dict[str, Any]]) -> str:
    """
    Calls the OpenAI vision model with a list of content items (text/image).
    This function is automatically traced by the @judgment.observe decorator.
    """
    print(f"--- Calling OpenAI Vision API with mixed content list ---")
    # print(f"Content List: {json.dumps(content_list, indent=2)}") # Optional: Print the list being sent
    try:
        response = await client.chat.completions.create( # Use await for async client
            model="gpt-4o", # Or "gpt-4-vision-preview" or other vision model
            messages=[
                {
                    "role": "user",
                    "content": content_list, # Pass the list directly
                }
            ],
            max_tokens=500, # Increased max tokens for potentially longer response
        )
        print("--- API Call Successful ---")
        return response.choices[0].message.content
    except Exception as e:
        print(f"--- API Call Failed: {e} ---")
        return f"Error analyzing image content: {e}"

async def main():
    # Clients and Tracer are already initialized above

    # --- Define Image URLs and Local Path ---
    image_url_1 = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg" # Boardwalk
    image_url_2 = "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Roasted_coffee_beans.jpg/1280px-Roasted_coffee_beans.jpg" # Coffee beans
    image_url_3 = "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/Felis_catus-cat_on_snow.jpg/1280px-Felis_catus-cat_on_snow.jpg" # Cat
    image_url_4 = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/New_york_times_square-terabass.jpg/1280px-New_york_times_square-terabass.jpg" # NYC Times Square
    # Use the provided path
    local_image_path = "/Users/minhpham/Documents/Judgement/judgment_dev/judgeval/src/demo/logo.png"

    # --- Encode Local Image ---
    base64_image = encode_image_to_base64(local_image_path)
    if not base64_image:
        print(f"Could not encode local image at {local_image_path}. Exiting.")
        return # Exit if local image encoding failed

    # --- Construct Mixed Content List ---
    mixed_content_list = [
        {
            "type": "text",
            "text": "Analyze the following images and answer the questions. Be concise."
        },
        {
            "type": "image_url",
            "image_url": {"url": image_url_1}
        },
        {
            "type": "image_url",
            "image_url": {"url": image_url_2}
        },
        {
            "type": "image_url",
            "image_url": {"url": image_url_3}
        },
        {
            "type": "image_url",
            "image_url": {"url": image_url_4}
        },
        {
            "type": "image_url",
            "image_url": {"url": base64_image} # Use the base64 encoded image (logo.png)
        },
        {
            "type": "text",
            "text": "Describe the main subject of each image briefly. What is the logo in the last image?"
        }
    ]

    print("\n--- Input Content List ---")
    # Print a summary instead of the full base64 string
    for item in mixed_content_list:
        if item["type"] == "text":
            print(f"- Text: {item['text']}")
        elif item["type"] == "image_url" and isinstance(item["image_url"]["url"], str) and item["image_url"]["url"].startswith("data:"):
             print(f"- Image: [Base64 Encoded Image (type: {item['image_url']['url'].split(';')[0].split(':')[1]})]")
        elif item["type"] == "image_url":
            print(f"- Image URL: {item['image_url']['url']}")
    print("--------------------------")


    print("Calling analyze_image_content (decorated with @judgment.observe)...")
    # Wrap the call in try/except to catch potential trace saving errors
    analysis_result = None
    try:
        # Pass the mixed content list to the modified function
        analysis_result = await analyze_image_content(aclient, mixed_content_list)
    except ValueError as e:
        if "Failed to save trace data" in str(e):
            print(f"\n--- ERROR: Failed to save trace to Judgment: {e} ---")
            print("--- The image analysis likely completed, but the trace was not recorded. Check API keys and Judgment backend status. ---")
            # Keep analysis_result if it was populated before the error (though usually error is after)
        else:
            # Re-raise other ValueErrors
            raise e

    # Proceed only if analysis succeeded
    if analysis_result is not None:
        print("\n--- Image Analysis Result ---")
        print(analysis_result)

        # --- Calculate and Display Token Breakdown (using data from decorator) ---
        print("\n--- Calculating Token Counts (using data from last trace) ---")

        # Access the entries from the Tracer instance after the decorated function ran
        # Note: This assumes `judgment.entries` holds the entries from the most recent trace
        # run by @observe. Error handling might be needed if no trace ran.
        if hasattr(judgment, 'entries') and judgment.entries:
            try:
                raw_entries = [entry.to_dict() for entry in judgment.entries]
                condensed_entries, _ = judgment.condense_trace(raw_entries) # Use judgment object

                # Now calculate counts using the condensed list and request breakdown
                if condensed_entries:
                    token_data = judgment.calculate_token_counts(condensed_entries, breakdown=True) # Use judgment object

                    print("\n--- Token Count Results ---")
                    print(json.dumps(token_data, indent=2))

                    # The breakdown dict shows the estimated split for OpenAI models
                    if "breakdown" in token_data:
                        print("\nEstimated Breakdown:")
                        print(f"  Text Prompt Tokens: {token_data['breakdown'].get('estimated_text_prompt_tokens', 'N/A')}")
                        print(f"  Image Prompt Tokens: {token_data['breakdown'].get('estimated_image_prompt_tokens', 'N/A')}")
                        print(f"  Text Completion Tokens: {token_data['breakdown'].get('estimated_text_completion_tokens', 'N/A')}")
                        # Also print cost breakdown if available
                        print(f"  Est. Text Prompt Cost (USD): {token_data['breakdown'].get('estimated_text_prompt_tokens_cost_usd', 'N/A')}")
                        print(f"  Est. Image Prompt Cost (USD): {token_data['breakdown'].get('estimated_image_prompt_tokens_cost_usd', 'N/A')}")
                        print(f"  Est. Text Completion Cost (USD): {token_data['breakdown'].get('estimated_text_completion_tokens_cost_usd', 'N/A')}")

                    else:
                         print("\nBreakdown structure not found in results.")
                else:
                    print("\nNo LLM entries found in the condensed trace to calculate token counts.")
            except Exception as calc_e:
                print(f"\n--- ERROR during local token calculation: {calc_e} ---")
                print("--- This occurred after the trace saving issue or a successful run. ---")

        else:
            print("\nNo trace entries found on the Tracer object to calculate token counts (possibly due to prior save error).")


        print(f"\nTrace Name Used by Decorator: vision_analysis_observed") # Directly state the name used
        # Access trace ID from the judgment object (assuming it holds the last one)
        if hasattr(judgment, 'trace_id'):
            print(f"Last Trace ID (may not have saved): {judgment.trace_id}")
        else:
            print("Could not retrieve last Trace ID from Tracer object.")
    else:
        print("\n--- Skipping token calculation due to earlier error during analysis or trace saving. ---")

    # Note: The trace logged by @observe is usually saved automatically.

if __name__ == "__main__":
    # Add rich import for better printing, optional but nice
    try:
        from rich import print
    except ImportError:
        pass # Fallback to standard print if rich is not installed
    asyncio.run(main()) 