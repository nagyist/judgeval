import os
import sys
from typing import Any, Dict, List, Optional, Union, Annotated
from typing_extensions import TypedDict
import json
import asyncio
from uuid import uuid4

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool, ToolException, BaseTool, StructuredTool
from pydantic import BaseModel, Field, ConfigDict
from elasticsearch import ApiError, NotFoundError

from judgeval.common.tracer import Tracer, wrap
from judgeval.scorers import FaithfulnessScorer, AnswerRelevancyScorer, AnswerCorrectnessScorer

from typing import Callable, TypeVar, ParamSpec
import elasticsearch_client

# Initialize OpenAI client
openai_client = ChatOpenAI(model="gpt-3.5-turbo")

# Initialize Judgment Tracer - using exactly the same initialization as demo.py
judgment = Tracer(
    api_key=os.getenv("JUDGMENT_API_KEY"),  # Default to empty string if not set
    project_name="text_to_es"
)

# Define TextToESState TypedDict
class TextToESState(TypedDict):
    messages: List[Any]
    user_query: str
    entities: Optional[List[Dict[str, str]]]
    validated_entities: Optional[List[Dict[str, str]]]
    selected_index: Optional[str]
    index_mappings: Optional[Dict[str, Any]]
    essential_fields: Optional[Dict[str, Any]]
    field_values: Optional[Dict[str, Any]]
    selected_field_values: Optional[Dict[str, Any]]
    es_query: Optional[Dict[str, Any]]
    processed_query: Optional[Dict[str, Any]]
    query_results: Optional[Dict[str, Any]]
    final_response: Optional[str]
    error: Optional[str]

def safe_json_extract(content: str) -> Any:
    """Extract JSON from model response."""
    try:
        # Check if the entire content is valid JSON
        return json.loads(content)
    except json.JSONDecodeError:
        # Try to extract JSON using simple pattern matching
        import re
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        # Return empty dict if no valid JSON found
        return {}

# This step will intentionally fail by injecting bad entity data
@judgment.observe(span_type="NLP")
def extract_entities(state: Dict[str, Any]) -> TextToESState:
    """Extract named entities from the user query but will fail with bad data."""
    user_query = state["user_query"]
    
    # Generate intentionally bad data that will trigger the error path
    bad_entities = {"malformed": "data"}  # Not a list, which will cause validation to fail
    
    # Evaluate the actual malformed output (bad_entities) against expected good data
    judgment.get_current_trace().async_evaluate(
        scorers=[
            AnswerRelevancyScorer(threshold=0.7),
            FaithfulnessScorer(threshold=0.7),
            AnswerCorrectnessScorer(threshold=0.7)
        ],
        input=f"Extract entities from query: '{user_query}'",
        actual_output=json.dumps(bad_entities, indent=2),
        expected_output=json.dumps({
            "Show me all disconnected access points": [
                {"type": "status", "name": "disconnected"},
                {"type": "device_type", "name": "access points"}
            ],
            "Show me the connection status of user john.doe": [
                {"type": "user", "name": "john.doe"},
                {"type": "status", "name": "connection status"}
            ]
        }.get(user_query, []), indent=2),
        retrieval_context=[
            "Entity types and descriptions: \n" +
            "- user: A specific username or person (e.g., 'john.doe', 'Jane Smith')\n" +
            "- device: A specific device identifier (e.g., 'AP-102', 'Switch-45')\n" +
            "- device_type: Category of network device (e.g., 'access point', 'switch', 'router')\n" +
            "- location: Physical location reference (e.g., 'Building A', '3rd Floor', 'NYC Office')\n" +
            "- status: Connection or operational status (e.g., 'connected', 'disconnected', 'offline')\n" +
            "- timeframe: Time reference (e.g., 'yesterday', 'last week', '3 hours ago')\n" +
            "- event_type: Type of network event (e.g., 'disconnect', 'authentication failure')\n" +
            "- quantity: Numerical reference (e.g., 'top 5', 'more than 10')"
        ],
        additional_metadata={"query": user_query},
        model="gpt-3.5-turbo",
        log_results=True
    )
    
    # Validation logic from the original function
    if not isinstance(bad_entities, list):
        error_msg = "Entity extraction error: Entities must be a list"
        # Return empty list for entities and set the error field
        return {**state, "entities": [], "error": error_msg}
    
    # This code won't be reached due to the intentional failure above
    for entity in bad_entities:
        if not isinstance(entity, dict) or "type" not in entity or "name" not in entity:
            raise ValueError("Each entity must have 'type' and 'name' fields")
    
    return {**state, "entities": bad_entities}

@judgment.observe(span_type="NLP")
def validate_entities(state: Dict[str, Any]) -> TextToESState:
    """Validate entities from the extraction step."""
    entities = state.get("entities", [])
    
    try:
        # Simplified validation logic
        validated_entities = []
        for entity in entities:
            if not isinstance(entity, dict) or "type" not in entity or "name" not in entity:
                raise ValueError(f"Invalid entity format: {entity}")
            validated_entities.append(entity)
        
        return {**state, "validated_entities": validated_entities}
    except Exception as e:
        error_msg = f"Entity validation error: {str(e)}"
        return {**state, "error": error_msg}

@judgment.observe(span_type="Search")
def select_index(state: Dict[str, Any]) -> TextToESState:
    """Select the appropriate Elasticsearch index."""
    user_query = state["user_query"]
    validated_entities = state.get("validated_entities", [])
    
    # This function is simplified as we don't expect to reach here due to entity extraction failure
    try:
        # Just simulate a selection without calling the LLM
        selected_index = "devices"
        index_mappings = elasticsearch_client.ES_INDEXES[selected_index]["mappings"]
        essential_fields = elasticsearch_client.ES_INDEXES[selected_index]["essential_fields"]
        
        return {**state, "selected_index": selected_index, "index_mappings": index_mappings, "essential_fields": essential_fields}
    except Exception as e:
        error_msg = f"Index selection error: {str(e)}"
        return {**state, "error": error_msg}

@judgment.observe(span_type="NLP")
def get_field_values(state: Dict[str, Any]) -> TextToESState:
    """Get field values for query generation."""
    # Simplified as we don't expect to reach this step
    return {**state, "field_values": {}, "selected_field_values": {}}

@judgment.observe(span_type="Query")
def generate_query(state: Dict[str, Any]) -> TextToESState:
    """Generate an Elasticsearch query."""
    # Simplified as we don't expect to reach this step
    return {**state, "es_query": {"match_all": {}}}

@judgment.observe(span_type="Query")
def process_query(state: Dict[str, Any]) -> TextToESState:
    """Process the Elasticsearch query."""
    # Simplified as we don't expect to reach this step
    return {**state, "processed_query": state.get("es_query", {})}

@judgment.observe(span_type="Database")
def execute_query(state: Dict[str, Any]) -> TextToESState:
    """Execute the Elasticsearch query."""
    # Simplified as we don't expect to reach this step
    return {**state, "query_results": {"hits": {"total": {"value": 0}, "hits": []}}}

@judgment.observe(span_type="Response")
def format_response(state: Dict[str, Any]) -> TextToESState:
    """Format the response for the user."""
    # Simplified as we don't expect to reach this step
    return {**state, "final_response": "No results found"}

@judgment.observe(span_type="Error")
def handle_error(state: Dict[str, Any]) -> TextToESState:
    """Handle errors in the pipeline."""
    error = state.get("error", "Unknown error")
    
    # Removed duplicate evaluation since we're now evaluating at the failure point
    
    return {**state, "final_response": f"Error: {error}"}

def has_error(state: TextToESState) -> str:
    """Route to error handler if state has an error."""
    if state.get("error"):
        return "error"
    return "continue"

@judgment.observe(span_type="function")
async def fail_entity_extraction_pipeline():
    """Main pipeline function that demonstrates failure in entity extraction."""
    # Initialize the graph
    workflow = StateGraph(TextToESState)
    
    # Add nodes
    workflow.add_node("extract_entities", extract_entities)
    workflow.add_node("validate_entities", validate_entities)
    workflow.add_node("select_index", select_index)
    workflow.add_node("get_field_values", get_field_values)
    workflow.add_node("generate_query", generate_query)
    workflow.add_node("process_query", process_query)
    workflow.add_node("execute_query", execute_query)
    workflow.add_node("format_response", format_response)
    workflow.add_node("handle_error", handle_error)
    
    workflow.set_entry_point("extract_entities")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "extract_entities",
        has_error,
        {
            "error": "handle_error",
            "continue": "validate_entities"
        }
    )
    
    workflow.add_conditional_edges(
        "validate_entities",
        has_error,
        {
            "error": "handle_error",
            "continue": "select_index"
        }
    )
    
    workflow.add_conditional_edges(
        "select_index",
        has_error,
        {
            "error": "handle_error",
            "continue": "get_field_values"
        }
    )
    
    workflow.add_conditional_edges(
        "get_field_values",
        has_error,
        {
            "error": "handle_error",
            "continue": "generate_query"
        }
    )
    
    workflow.add_conditional_edges(
        "generate_query",
        has_error,
        {
            "error": "handle_error",
            "continue": "process_query"
        }
    )
    
    workflow.add_conditional_edges(
        "process_query",
        has_error,
        {
            "error": "handle_error",
            "continue": "execute_query"
        }
    )
    
    workflow.add_conditional_edges(
        "execute_query",
        has_error,
        {
            "error": "handle_error",
            "continue": "format_response"
        }
    )
    
    # Compile the graph
    app = workflow.compile()
    
    # Initial state
    initial_state = {
        "messages": [],
        "user_query": "Show me all disconnected access points",
        "entities": None,
        "validated_entities": None,
        "selected_index": None,
        "index_mappings": None,
        "essential_fields": None,
        "field_values": None,
        "selected_field_values": None,
        "es_query": None,
        "processed_query": None,
        "query_results": None,
        "final_response": None,
        "error": None
    }
    
    # Run the pipeline
    result = await app.ainvoke(initial_state)
    print(f"Pipeline result: {result.get('final_response', 'No response')}")
    return result

if __name__ == "__main__":
    import asyncio
    asyncio.run(fail_entity_extraction_pipeline()) 