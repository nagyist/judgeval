import anthropic
import json
import os

# --- Define Tool Schemas ---
# Parsed from the user's input list, including JSON strings and dictionaries.
# Added '$schema' reference to parameters for consistency.

tools = [
    # From the second item (dictionary) in user input
    {
        # "type": "function", # Removed for count_tokens compatibility with Anthropic API
        "name": "readPresentationFile",
        "description": "Retrieve the current HTML content of the presentation slide. Use this tool to examine the existing slide structure before making modifications. This allows you to understand the current state of the presentation, including all HTML elements, their attributes, and their hierarchical relationships. The returned content will include any pending changes that haven't been committed yet.",
        "input_schema": { # Renamed from 'parameters' for Anthropic
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {},
            "additionalProperties": False,
            # Note: 'required' field is often needed, assuming none for now based on input
        }
    },
    # From the third item (parsed JSON string) in user input
    {
        # "type": "function", # Removed for count_tokens compatibility
        "name": "writeToPresentationFile",
        "description": "Create or completely replace the presentation slide with new HTML content. Use this tool when you need to generate an entirely new slide or make comprehensive changes to the existing one. This is typically the first step in creating a presentation slide based on user requirements. The tool preserves any custom CSS and automatically applies it to your HTML content. The previous state is saved to history, allowing for recovery if needed. Ensure your HTML follows consulting presentation best practices and maintains the required 16:9 aspect ratio.",
        "input_schema": { # Renamed from 'parameters'
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "htmlContent": {
                    "type": "string",
                    "description": "The complete HTML content for the presentation slide. This should be well-structured, professional-looking HTML that follows consulting presentation best practices. Include appropriate containers, headings, and layout elements to create a slide with proper information hierarchy. Ensure all elements use relative sizing" # Original string was truncated
                }
            },
            # "required": ["htmlContent"] # Assuming htmlContent is required, add if needed
        }
    },
    # From the fourth item (parsed JSON string) in user input
    {
        # "type": "function", # Removed for count_tokens compatibility
        "name": "editPresentationFile",
        "description": "Make targeted modifications to specific elements of the presentation slide. Use this tool when you need to refine individual components rather than replacing the entire slide. This is especially useful for iterative improvements based on visual feedback. You can view, replace, insert, or remove content using CSS selectors, allowing for precise adjustments to layout, styling, or content while preserving the overall slide structure. Each edit operation is automatically saved to history, allowing for undo operations if needed.",
        "input_schema": { # Renamed from 'parameters'
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["view", "replace", "insert", "remove", "undo"],
                    "description": "The edit operation to perform: 'view' to examine existing elements, 'replace' to substitute content, 'insert' to add new content, 'remove' to delete elements, 'undo' to revert the last change. Choose the most precise operation for your current refinement needs."
                },
                "selector": {
                    "type": "string",
                    "description": "A CSS selector to target the specific HTML element(s) for the operation." # Assuming description based on context
                }
                # Add other properties like 'content' if needed for 'insert'/'replace'
            },
            # "required": ["command", "selector"] # Assuming these are required
        }
    },
    # From the fifth item (dictionary) in user input
    {
        # "type": "function", # Removed for count_tokens compatibility
        "name": "screenshotPresentation",
        "description": "Take a screenshot of the current presentation slide to evaluate its visual appearance. Use this tool to verify that your slide layout, spacing, and design elements appear as intended. The screenshot will help you identify any visual issues such as overlapping elements, improper scaling, or readability problems that need to be addressed. This is a critical step in your self-evaluation process to ensure the slide meets professional consulting standards.",
        "input_schema": { # Renamed from 'parameters'
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {},
            "additionalProperties": False,
            # Assuming no required properties
        }
    }
]

# Use the defined tools list for the API call
# Note: The Anthropic count_tokens API expects 'input_schema' not 'parameters'
# and doesn't use the 'type': 'function' field within the tool definition itself.
# The list passed to the API should contain these dictionaries directly.
valid_tools = tools

# --- Anthropic Client and Token Count ---
try:
    # Ensure the API key is set in the environment variables
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY environment variable not set.")

    client = anthropic.Anthropic()

    # Use the model specified in the user's example, or a current model like claude-3-5-sonnet-20240620
    model_name = "claude-3-5-sonnet-20240620" # Or use "claude-3-opus-20240229" or the user's "claude-3-7-sonnet-20250219" if needed

    # Example messages (required for the count_tokens call)
    messages = [{"role": "user", "content": "Please generate a presentation slide."}]

    print(f"Counting tokens for model: {model_name}")
    print(f"Using {len(valid_tools)} tools.")
    print("--- Tools structure being sent to API: ---")
    print(json.dumps(valid_tools, indent=2))
    print("--- End of tools structure ---")

    response = client.messages.count_tokens(
        model=model_name,
        tools=valid_tools,
        messages=messages
    )

    # The response object itself contains the token count
    print("\nToken count response:")
    print(f"Total Tokens: {response.json()}")

except anthropic.APIConnectionError as e:
    print(f"The server could not be reached: {e.__cause__}")
except anthropic.RateLimitError as e:
    print(f"A 429 status code was received; we should back off a bit: {e}")
except anthropic.APIStatusError as e:
    print(f"Another non-200-range status code was received: {e.status_code}")
    print(e.response)
except ValueError as e:
    print(f"Input Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}") 