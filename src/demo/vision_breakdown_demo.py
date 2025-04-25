# vision_breakdown_demo.py

import os
import sys # Add sys import
import asyncio
from openai import OpenAI, AsyncOpenAI # Added AsyncOpenAI
from dotenv import load_dotenv
import json # Added json import here for clarity

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


# --- Function Definitions ---

@judgment.observe(name="vision_analysis_observed") # Apply the decorator
async def analyze_image(client: AsyncOpenAI, image_url: str, prompt: str) -> str:
    """
    Calls the OpenAI vision model with an image URL and a text prompt.
    This function is automatically traced by the @judgment.observe decorator.
    """
    print(f"--- Calling OpenAI Vision API for URL: {image_url} ---")
    try:
        response = await client.chat.completions.create( # Use await for async client
            model="gpt-4o", # Or "gpt-4-vision-preview" or other vision model
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                            },
                        },
                    ],
                }
            ],
            max_tokens=300,
        )
        print("--- API Call Successful ---")
        return response.choices[0].message.content
    except Exception as e:
        print(f"--- API Call Failed: {e} ---")
        return f"Error analyzing image: {e}"

async def main():
    # Clients and Tracer are already initialized above

    # --- Trace Execution --- (Simplified using @observe)
    image_url_to_analyze = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    text_prompt = "Describe this image in detail. What season is it?"

    print("Calling analyze_image (decorated with @judgment.observe)...")
    # The decorator handles starting/stopping the trace automatically.
    # Wrap the call in try/except to catch potential trace saving errors
    analysis_result = None
    try:
        analysis_result = await analyze_image(aclient, image_url_to_analyze, text_prompt)
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