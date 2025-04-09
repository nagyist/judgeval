import argparse
import os
from pathlib import Path
import sys
import json # For JSON correctness check
from typing import List, Optional # For Pydantic model
from pydantic import BaseModel, Field # For Pydantic model

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.document_loaders import PyPDFLoader # For PDFs
import docx # For DOCX

# --- Add Judgeval Imports ---
from judgeval.common.tracer import Tracer, wrap
# Import specific scorers
from judgeval.scorers import (
    JSONCorrectnessScorer,
    FaithfulnessScorer,
    AnswerRelevancyScorer,
    HallucinationScorer
)

# --- Add Braintrust Imports ---
from braintrust import init_logger, traced

# --- Add Arize/OTel Imports ---
from arize.otel import register as arize_register # Alias to avoid name conflict
from openinference.instrumentation.openai import OpenAIInstrumentor


# --- Initialize Tracers ---
judgment = Tracer(project_name="langgraph-resume-agent") # Different project name for this script
# Initialize Braintrust logger
logger = init_logger(project="langgraph-resume-agent")
# Setup Arize OTel
tracer_provider = arize_register(
    space_id="U3BhY2U6MTg2MTI6aitHYQ==",
    api_key="89501374ac3b32dd4c0",
    project_name="Langgraph-email-agent"
)
# Instrument OpenAI for Arize/OTel
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
# --- End Initialization ---

# Add src directory to Python path if the script is run from the root
script_dir = Path(__file__).parent.resolve()
src_dir = script_dir / 'src'
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
    
# Now we can import from src (assuming GmailTools are there)
try:
    # Import the class, not the method
    from tools.GmailTools import GmailToolsClass 
except ImportError as e:
    print(f"Error importing GmailTools: {e}")
    print("Ensure GmailTools.py exists in the src/tools/ directory and __init__.py files are present.")
    sys.exit(1)

# --- Constants ---
TARGET_ROLE = "Software Engineer"

EXTRACTION_PROMPT_TEMPLATE = """
Extract the following information from the provided resume text:
- Candidate's full name
- Candidate's email address
- 2-3 key skills or experiences relevant to a {target_role} role.

Format the output as a JSON object with keys: "name", "email", "key_points".
If the email address cannot be found, return null for the email field.

Resume Text:
{resume_text}
"""

RELEVANCE_CHECK_PROMPT_TEMPLATE = """
Based on the following key points extracted from a resume, assess the candidate's relevance for a {target_role} position.

Key Points: {key_points}

Output a short (1-sentence) assessment of relevance. Example: "Highly relevant due to extensive backend experience." or "Somewhat relevant, stronger in frontend than required."
Assessment:
"""

EMAIL_DRAFT_PROMPT_TEMPLATE = """
You are an AI assistant helping a recruiter draft an outreach email for a {target_role} role.
Based on the following information, write a professional and concise email asking the candidate to schedule a brief introductory meeting.

- Candidate Name: {name}
- Candidate Email: {email}
- Key Skills/Experience: {key_points}
- Relevance Assessment: {relevance_assessment}
- Your Name/Title: {Your Name/Title}
- Company: {Your Company}
- Meeting Link (Optional): {Your Calendly/Scheduling Link}

Instructions:
- Address the candidate by their name (e.g., "Hi {name},").
- Briefly mention the specific skills/experience that caught your eye, potentially informed by the relevance assessment.
- Clearly state the purpose: to schedule a short introductory call to discuss potential {target_role} opportunities at {Your Company}.
- If a meeting link is provided above, suggest they use it to find a time. Otherwise, ask for their availability.
- Keep the tone professional and enthusiastic.
- Use standard paragraph breaks (like double newlines in plain text).
- **Do NOT include a 'Subject:' line in your response.** The subject will be added separately.
- Sign off with your name and title.

Draft ONLY the email body content below:
"""

TONE_ADJUSTMENT_PROMPT_TEMPLATE = """
Rewrite the following draft email to have a slightly more enthusiastic and personalized tone, while remaining professional. Ensure the core message and call to action remain the same.

Original Draft:
{original_email}

Adjusted Draft:
"""

# --- Functions ---

@judgment.observe()
def parse_pdf(file_path):
    """Parses text content from a PDF file."""
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        return "\n".join([doc.page_content for doc in docs])
    except Exception as e:
        print(f"Error parsing PDF {file_path}: {e}")
        return None

@judgment.observe()
def parse_docx(file_path):
    """Parses text content from a DOCX file."""
    try:
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"Error parsing DOCX {file_path}: {e}")
        return None

@traced # Add Braintrust tracing
@judgment.observe(span_type="llm") # Specific span type for LLM calls
def check_role_relevance(llm, key_points, target_role):
    """Uses LLM to check relevance of key points to the target role."""
    print("Checking role relevance...")
    prompt = ChatPromptTemplate.from_template(RELEVANCE_CHECK_PROMPT_TEMPLATE)
    chain = prompt | llm | StrOutputParser()
    try:
        assessment = chain.invoke({"key_points": key_points, "target_role": target_role})
        print(f"Relevance Assessment: {assessment}")
        return assessment
    except Exception as e:
        print(f"Error during relevance check: {e}")
        return "Assessment failed."

@traced # Add Braintrust tracing
@judgment.observe(span_type="llm") # Specific span type for LLM calls
def adjust_email_tone(llm, original_email):
    """Uses LLM to adjust the email tone."""
    print("Adjusting email tone...")
    prompt = ChatPromptTemplate.from_template(TONE_ADJUSTMENT_PROMPT_TEMPLATE)
    chain = prompt | llm | StrOutputParser()
    try:
        adjusted_email = chain.invoke({"original_email": original_email})
        print("Email tone adjusted.")
        return adjusted_email
    except Exception as e:
        print(f"Error during tone adjustment: {e}")
        return original_email # Return original if adjustment fails

@traced # Add Braintrust tracing
@judgment.observe(span_type="llm") # Add decorator here
def extract_resume_info(llm, resume_text, target_role):
    """Extracts key information from resume text using LLM."""
    print("Extracting information from resume...")
    extraction_prompt = ChatPromptTemplate.from_template(EXTRACTION_PROMPT_TEMPLATE)
    # Ensure JsonOutputParser handles potential LLM errors gracefully if possible
    extraction_chain = extraction_prompt | llm | JsonOutputParser() 
    try:
        extracted_info = extraction_chain.invoke({"resume_text": resume_text, "target_role": target_role})
        print(f"Extracted Info: {extracted_info}")
        return extracted_info
    except Exception as e:
        print(f"Error during LLM extraction: {e}")
        # Decide how to handle: return None, raise, return partial, etc.
        # Returning None for now, main function will handle exit.
        return None

# --- Main Execution ---

@traced # Add Braintrust tracing
@judgment.observe()
def main():
    # 1. Setup & Argument Parsing
    load_dotenv() # Load API keys etc. from .env
    
    parser = argparse.ArgumentParser(description="Generate outreach email drafts from resumes.")
    parser.add_argument("resume_file", help="Path to the resume file (PDF or DOCX).")
    args = parser.parse_args()

    # Define Pydantic model for expected extraction schema
    class ResumeExtractionSchema(BaseModel):
        name: Optional[str] = Field(None, description="Candidate's full name")
        email: Optional[str] = Field(None, description="Candidate's email address")
        key_points: List[str] = Field(default_factory=list, description="Key skills or experiences")

    resume_path = Path(args.resume_file)
    if not resume_path.is_file():
        print(f"Error: File not found at {resume_path}")
        sys.exit(1)

    # 2. Parse Resume Content
    print(f"Parsing resume: {resume_path.name}...")
    file_ext = resume_path.suffix.lower()
    resume_text = None
    if file_ext == '.pdf':
        resume_text = parse_pdf(str(resume_path))
    elif file_ext == '.docx':
        resume_text = parse_docx(str(resume_path))
    else:
        print(f"Error: Unsupported file type '{file_ext}'. Please provide a PDF or DOCX.")
        sys.exit(1)

    if not resume_text:
        print("Error: Could not extract text from the resume.")
        sys.exit(1)
        
    print("Resume parsed successfully.")

    # 3. Initialize LLM & Gmail Service
    try:
        # --- Remove wrap() ---
        # llm = wrap(ChatOpenAI(model="gpt-4o", temperature=0.1)) # wrap removed as OTel handles it
        llm = ChatOpenAI(model="gpt-4o", temperature=0.1) # Initialize directly, OTel will instrument
        # --- End Removal ---
        print("Initializing Gmail Service...")
        # Instantiate the GmailToolsClass here
        gmail_tools = GmailToolsClass()
        print("Gmail Service initialized.")
    except Exception as e:
        print(f"Error initializing LLM or Gmail Service: {e}. Check API keys/tokens.")
        sys.exit(1)

    # 4. Extract Information using LLM
    extracted_info = extract_resume_info(llm, resume_text, TARGET_ROLE)

    # --- Add Judgment Evaluation for Extraction ---
    if extracted_info:
        judgment.async_evaluate(
            scorers=[
                JSONCorrectnessScorer(threshold=0.9, json_schema=ResumeExtractionSchema),
                FaithfulnessScorer(threshold=0.9),
                AnswerRelevancyScorer(threshold=0.9),
                HallucinationScorer(threshold=0.9)
            ],
            input=f"Extract information for {TARGET_ROLE}", # Input context for evaluation
            actual_output=json.dumps(extracted_info), # Output being evaluated
            context=[resume_text], # Grounding source for Faithfulness
            model="gpt-4o" # Specify model used for potential scorer LLM calls
        )
    # --- End Evaluation ---

    if not extracted_info or 'email' not in extracted_info or not extracted_info['email']:
        print("Error: Could not extract a valid email address from the resume.")
        sys.exit(1) 
        
    if 'name' not in extracted_info or not extracted_info['name']:
         print("Warning: Could not extract candidate name. Using placeholder.")
         extracted_info['name'] = "Candidate" # Placeholder
         
    if 'key_points' not in extracted_info or not extracted_info['key_points']:
         print("Warning: Could not extract key points. Email might be generic.")
         extracted_info['key_points'] = ["your impressive background"] # Generic placeholder

    # 4.5 Check Role Relevance
    relevance_assessment = check_role_relevance(llm, extracted_info['key_points'], TARGET_ROLE)

    # --- Add Judgment Evaluation for Relevance Assessment ---
    if relevance_assessment and extracted_info and 'key_points' in extracted_info:
        judgment.async_evaluate(
            scorers=[
                FaithfulnessScorer(threshold=0.9),
                AnswerRelevancyScorer(threshold=0.9),
                HallucinationScorer(threshold=0.9)
            ],
            input=f"Assess relevance for {TARGET_ROLE}",
            actual_output=relevance_assessment,
            context=[str(extracted_info['key_points'])], # Grounding source for Faithfulness
            model="gpt-4o"
        )
    # --- End Evaluation ---

    # 5. Draft Email using LLM
    print("Drafting initial outreach email...")
    draft_prompt = ChatPromptTemplate.from_template(EMAIL_DRAFT_PROMPT_TEMPLATE)
    draft_chain = draft_prompt | llm | StrOutputParser()

    draft_input = {
        "name": extracted_info['name'],
        "email": extracted_info['email'],
        "key_points": extracted_info['key_points'],
        "relevance_assessment": relevance_assessment, 
        "target_role": TARGET_ROLE,
        "Your Name/Title": "Galen Topper / Senior Software Engineer", 
        "Your Company": "Judgment Labs",
        "Your Calendly/Scheduling Link": "" 
    }

    try:
        initial_email_body = draft_chain.invoke(draft_input)
        print("--- Initial Email Draft ---")
        print(initial_email_body)
        print("-------------------------")

        # --- Add Judgment Evaluation for Initial Email Draft ---
        if initial_email_body:
            # Convert draft_input dict to list of strings for context
            context_list = [f"{k}: {v}" for k, v in draft_input.items()]
            judgment.async_evaluate(
                scorers=[FaithfulnessScorer(threshold=0.9)],
                input="Draft initial outreach email",
                actual_output=initial_email_body,
                context=context_list, # Grounding source
                model="gpt-4o"
            )
        # --- End Evaluation ---

    except Exception as e:
        print(f"Error during LLM email drafting: {e}")
        sys.exit(1)

    # 5.5 Adjust Email Tone
    final_email_body_raw = adjust_email_tone(llm, initial_email_body)

    # --- Add Judgment Evaluation for Tone Adjustment ---
    if final_email_body_raw and initial_email_body:
        judgment.async_evaluate(
            scorers=[
                FaithfulnessScorer(threshold=0.9),
                HallucinationScorer(threshold=0.9)
            ], # Check if core message preserved
            input="Adjust email tone to be more enthusiastic",
            actual_output=final_email_body_raw,
            context=[initial_email_body], # Grounding source
            model="gpt-4o"
        )
    # --- End Evaluation ---

    # Post-processing: Remove potential leading Subject line from final draft
    if final_email_body_raw.strip().lower().startswith("subject:"):
        final_email_body = final_email_body_raw.split('\n', 1)[-1].strip()
        print("Removed leading 'Subject:' line from final LLM output.")
    else:
        final_email_body = final_email_body_raw

    print("--- Final Generated Email Body ---")
    print(final_email_body)
    print("----------------------------------")

    # 6. Create Gmail Draft
    print(f"Creating draft in Gmail for {extracted_info['email']}...")
    subject = f"Introductory Chat regarding {TARGET_ROLE} opportunities at {draft_input['Your Company']}"
    try:
        draft_result = gmail_tools.create_draft(extracted_info['email'], subject, final_email_body) 
        print(f"Draft created successfully: {draft_result}") 
    except Exception as e:
        print(f"Error creating Gmail draft: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 