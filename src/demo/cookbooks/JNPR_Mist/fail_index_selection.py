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
    api_key=os.getenv("JUDGMENT_API_KEY", ""),  # Default to empty string if not set
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

@judgment.observe(span_type="NLP")
def extract_entities(state: Dict[str, Any]) -> TextToESState:
    """Extract named entities from the user query."""
    user_query = state["user_query"]
    
    # For this test, we'll provide valid entities
    valid_entities = [
        {"type": "device_type", "name": "access points"},
        {"type": "status", "name": "disconnected"}
    ]
    
    # For entity extractor evaluation
    judgment.get_current_trace().async_evaluate(
        scorers=[
            AnswerCorrectnessScorer(threshold=0.7),
            FaithfulnessScorer(threshold=0.7),
        ],
        input=f"Extract entities from: '{user_query}'",
        actual_output=json.dumps(valid_entities),
        expected_output=json.dumps([
            {"type": "device_type", "name": "access points"},
            {"type": "status", "name": "disconnected"}
        ]),
        retrieval_context=[],
        model="gpt-3.5-turbo",
        log_results=True,
        additional_metadata={"query": user_query}
    )
    
    return {**state, "entities": valid_entities}

@judgment.observe(span_type="NLP")
def validate_entities(state: Dict[str, Any]) -> TextToESState:
    """Validate entities from the extraction step."""
    entities = state.get("entities", [])
    
    # Simple validation to pass through to the next step
    validated_entities = entities
    
    return {**state, "validated_entities": validated_entities}

@judgment.observe(span_type="Search")
def select_index(state: Dict[str, Any]) -> TextToESState:
    """Select the appropriate Elasticsearch index - will inject invalid index to fail."""
    user_query = state["user_query"]
    validated_entities = state.get("validated_entities", [])
    
    # Inject invalid index selection to force failure
    invalid_index = "nonexistent_index"
    
    # Evaluate the actual invalid index against expected correct index
    judgment.get_current_trace().async_evaluate(
        scorers=[
            AnswerCorrectnessScorer(threshold=0.7),
            FaithfulnessScorer(threshold=0.7)
        ],
        input=f"Select appropriate Elasticsearch index for query: '{user_query}'",
        actual_output=invalid_index,
        expected_output={
            # Ground truth mapping for test queries
            "Show me all disconnected access points": "devices",
            "How many switches are in Building B?": "devices",
            "List all devices in the Data Center": "devices",
            "Show me the connection status of user john.doe": "clients",
            "When did AP-102 last disconnect?": "events",
            "What's the status of Building A?": "locations"
        }.get(user_query, "devices"),  # Default to devices for the disconnected AP query
        retrieval_context=[
            # Pre-format complex JSON structures
            json.dumps({
                "devices": elasticsearch_client.ES_INDEXES["devices"]["description"],
                "clients": elasticsearch_client.ES_INDEXES["clients"]["description"],
                "events": elasticsearch_client.ES_INDEXES["events"]["description"],
                "locations": elasticsearch_client.ES_INDEXES["locations"]["description"]
            }, indent=2),
            f"User query: {user_query}",
            f"Extracted entities: {json.dumps(validated_entities, indent=2)}"
        ],
        model="gpt-3.5-turbo",
        log_results=True,
        additional_metadata={"query": user_query}
    )
    
    try:
        # Verify this index exists
        if invalid_index not in elasticsearch_client.ES_INDEXES:
            error_msg = f"Index selection error: Invalid index selected: {invalid_index} - not found in index configuration"
            return {**state, "error": error_msg}
        
        # This code won't be reached due to the intentional failure
        index_mappings = elasticsearch_client.ES_INDEXES[invalid_index]["mappings"]
        essential_fields = elasticsearch_client.ES_INDEXES[invalid_index]["essential_fields"]
        
        return {**state, "selected_index": invalid_index, "index_mappings": index_mappings, "essential_fields": essential_fields}
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
async def fail_index_selection_pipeline():
    """Main pipeline function that demonstrates failure in index selection."""
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
    asyncio.run(fail_index_selection_pipeline()) 