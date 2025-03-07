import os
from typing import Any, Dict, List, Optional, Union, Annotated
from typing_extensions import TypedDict
import json
import asyncio
from uuid import uuid4

from dotenv import load_dotenv
from openai import OpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool, ToolException, BaseTool, StructuredTool
from pydantic import BaseModel, Field, ConfigDict

from judgeval.common.tracer import Tracer, wrap
from judgeval.scorers import FaithfulnessScorer, AnswerRelevancyScorer, AnswerCorrectnessScorer

from typing import Callable, TypeVar, ParamSpec

P = ParamSpec("P")
R = TypeVar("R")

load_dotenv()

# Initialize clients
openai_client = wrap(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))
judgment = Tracer(
    api_key=os.getenv("JUDGMENT_API_KEY"),
    project_name="text_to_es"
)

# Define our state type
class TextToESState(TypedDict):
    messages: List[BaseMessage]
    user_query: str
    entities: Optional[List[Dict[str, str]]]
    selected_index: Optional[str]
    index_mappings: Optional[Dict[str, Any]]
    essential_fields: Optional[Dict[str, Any]]
    es_query: Optional[Dict[str, Any]]
    processed_query: Optional[Dict[str, Any]]
    query_results: Optional[Dict[str, Any]]
    final_response: Optional[str]
    error: Optional[str]

# Update or add ES_INDEXES definition if missing or incorrect
# Define Elasticsearch index configurations
ES_INDEXES = {
    "devices": {
        "description": "Information about network devices like access points, switches, routers",
        "mappings": {
            "properties": {
                "name": {"type": "keyword"},
                "status": {"type": "keyword"},
                "location": {"type": "keyword"},
                "device_type": {"type": "keyword"},
                "last_seen": {"type": "date"}
            }
        },
        "essential_fields": {
            "status": "status",
            "device_type": "device_type"
        }
    },
    "users": {
        "description": "Information about users and clients connected to the network",
        "mappings": {
            "properties": {
                "username": {"type": "keyword"},
                "device_id": {"type": "keyword"},
                "connection_status": {"type": "keyword"},
                "last_connected": {"type": "date"}
            }
        },
        "essential_fields": {
            "connection_status": "connection_status"
        }
    },
    "locations": {
        "description": "Information about physical locations and sites",
        "mappings": {
            "properties": {
                "name": {"type": "keyword"},
                "address": {"type": "text"},
                "num_devices": {"type": "integer"},
                "status": {"type": "keyword"}
            }
        },
        "essential_fields": {
            "status": "status"
        }
    }
}

# Update OpenAI model name to a valid one
OPENAI_MODEL = "gpt-4-turbo-preview"

@judgment.observe(span_type="JSON")
def safe_json_extract(content: str) -> Any:
    """Safely extract JSON from OpenAI response."""
    try:
        # Try direct JSON parsing first
        return json.loads(content)
    except json.JSONDecodeError:
        # Try extracting from code blocks
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            json_str = content.split("```")[1].strip()
        else:
            json_str = content.strip()
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from content: {content}") from e

@judgment.observe(span_type="NLP")
def extract_entities(state: Dict[str, Any]) -> TextToESState:
    """Extract named entities from the user query using OpenAI's model."""
    user_query = state["user_query"]
    
    prompt = f"""
    Extract any entities from this query. Entities can be:
    - user: a specific user mentioned
    - device: a specific device mentioned
    - location: a specific location mentioned
    
    Format your response as a JSON array with objects containing "type" and "name" fields.
    If no entities are found, return an empty array.
    
    Query: {user_query}
    """
    
    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are an entity extraction specialist. Extract entities precisely in the requested format."},
                {"role": "user", "content": prompt}
            ]
        )
        
        content = response.choices[0].message.content
        entities = safe_json_extract(content)
        
        # Validate entities structure
        if not isinstance(entities, list):
            raise ValueError("Entities must be a list")
        
        for entity in entities:
            if not isinstance(entity, dict) or "type" not in entity or "name" not in entity:
                raise ValueError("Each entity must have 'type' and 'name' fields")
        
        # Evaluate entity extraction
        judgment.get_current_trace().async_evaluate(
            scorers=[AnswerCorrectnessScorer(threshold=0.7)],
            input=user_query,
            actual_output=json.dumps(entities),
            model=OPENAI_MODEL
        )
        
        return {**state, "entities": entities}
    except Exception as e:
        error_msg = f"Entity extraction error: {str(e)}"
        return {**state, "entities": [], "error": error_msg}

@judgment.observe(span_type="NLP")
def validate_entities(state: Dict[str, Any]) -> TextToESState:
    """Validate if extracted entities exist in the organization."""
    entities = state.get("entities", [])
    # Mock implementation - in real system would query a database
    valid_entities = []
    
    for entity in entities:
        # Add validation logic here
        # For demo, just passing through the entities
        valid_entities.append(entity)
    
    return {**state, "validated_entities": valid_entities}

@judgment.observe(span_type="Search")
def select_index(state: Dict[str, Any]) -> TextToESState:
    """Select the appropriate Elasticsearch index based on the user query."""
    user_query = state["user_query"]
    entities = state.get("entities", [])
    
    # Simple keyword matching for index selection
    query_lower = user_query.lower()
    
    try:
        # Determine the index based on the query
        if any(keyword in query_lower for keyword in ["disconnect", "offline", "down"]):
            selected_index = "devices"
        elif any(keyword in query_lower for keyword in ["user", "client", "employee"]):
            selected_index = "users"
        elif any(keyword in query_lower for keyword in ["location", "site", "building", "office"]):
            selected_index = "locations"
        else:
            # Default to devices index
            selected_index = "devices"
        
        # Check if the selected index exists in our configuration
        if selected_index not in ES_INDEXES:
            return {**state, "error": f"Invalid index selected: {selected_index} - not found in index configuration"}
        
        # Get the index mappings and essential fields
        index_mappings = ES_INDEXES[selected_index]["mappings"]
        essential_fields = ES_INDEXES[selected_index]["essential_fields"]
        
        return {
            **state,
            "selected_index": selected_index,
            "index_mappings": index_mappings,
            "essential_fields": essential_fields
        }
    except KeyError as e:
        return {**state, "error": f"Index configuration error: Missing key {e} in ES_INDEXES configuration"}
    except Exception as e:
        return {**state, "error": f"Index selection error: {str(e)}"}

@judgment.observe(span_type="Search")
def get_field_values(state: Dict[str, Any]) -> TextToESState:
    """Get possible values for essential fields in the selected index."""
    index = state.get("selected_index")
    essential_fields = state.get("essential_fields", {})
    
    # Mock implementation - in real systems, would query the database for possible values
    field_values = {field: ["value1", "value2", "value3"] for field in essential_fields}
    
    return {**state, "field_values": field_values}

@judgment.observe(span_type="Query")
def generate_query(state: Dict[str, Any]) -> TextToESState:
    """Generate Elasticsearch query based on user query and index information."""
    user_query = state["user_query"]
    selected_index = state["selected_index"]
    essential_fields = state.get("essential_fields", {})
    field_values = state.get("field_values", {})
    entities = state.get("entities", [])
    
    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are an Elasticsearch query generation expert. Generate precise Elasticsearch queries."},
                {"role": "user", "content": f"""
                Generate an Elasticsearch query for the following request:
                "{user_query}"
                
                The query should be for the "{selected_index}" index.
                
                Essential fields in this index:
                {json.dumps(essential_fields, indent=2)}
                
                Field values:
                {json.dumps(field_values, indent=2)}
                
                Entities identified in the query:
                {json.dumps(entities, indent=2)}
                
                Return ONLY the Elasticsearch query in JSON format.
                """}
            ]
        )
        
        content = response.choices[0].message.content
        query = safe_json_extract(content)
        
        # Evaluate query generation
        judgment.get_current_trace().async_evaluate(
            scorers=[AnswerCorrectnessScorer(threshold=0.7)],
            input=user_query,
            actual_output=json.dumps(query),
            model=OPENAI_MODEL
        )
        
        return {**state, "es_query": query}
    except Exception as e:
        return {**state, "error": f"Query generation error: {str(e)}"}

@judgment.observe(span_type="Query")
def process_query(state: Dict[str, Any]) -> TextToESState:
    """Process and clean up the generated Elasticsearch query."""
    query = state["es_query"]
    entities = state.get("entities", [])
    
    # Define a function to clean up query fields recursively
    def clean_query_fields(query_part):
        if isinstance(query_part, dict):
            return {
                key: clean_query_fields(value)
                for key, value in query_part.items()
            }
        elif isinstance(query_part, list):
            return [clean_query_fields(item) for item in query_part]
        elif isinstance(query_part, str):
            # Replace entity placeholders if needed
            for entity in entities:
                placeholder = f"{{entity:{entity['type']}}}"
                if placeholder in query_part:
                    query_part = query_part.replace(placeholder, entity["name"])
            return query_part
        else:
            return query_part
    
    # Clean up the query
    processed_query = clean_query_fields(query)
    
    return {**state, "processed_query": processed_query}

@judgment.observe(span_type="Database")
def execute_query(state: Dict[str, Any]) -> TextToESState:
    """Execute the processed Elasticsearch query."""
    index = state["selected_index"]
    query = state["processed_query"]
    
    # Mock implementation - in real system would connect to Elasticsearch
    # For demo purposes, return mock results based on query
    if index == "devices":
        if "status" in str(query) and "disconnected" in str(query):
            # Mock disconnected devices
            results = {
                "hits": {
                    "total": {"value": 3},
                    "hits": [
                        {"_source": {"name": "AP-1", "status": "disconnected", "location": "Building A"}},
                        {"_source": {"name": "AP-2", "status": "disconnected", "location": "Building B"}},
                        {"_source": {"name": "Switch-1", "status": "disconnected", "location": "Data Center"}}
                    ]
                }
            }
        else:
            # Generic device results
            results = {
                "hits": {
                    "total": {"value": 2},
                    "hits": [
                        {"_source": {"name": "AP-3", "status": "connected", "location": "Building C"}},
                        {"_source": {"name": "Switch-2", "status": "connected", "location": "Data Center"}}
                    ]
                }
            }
    else:
        # Generic results for other indices
        results = {
            "hits": {
                "total": {"value": 0},
                "hits": []
            }
        }
    
    return {**state, "query_results": results}

@judgment.observe(span_type="Response")
def format_response(state: Dict[str, Any]) -> TextToESState:
    """Format the query results into a user-friendly response."""
    user_query = state["user_query"]
    query_results = state["query_results"]
    entities = state.get("entities", [])
    
    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant specialized in database query results interpretation."},
                {"role": "user", "content": f"""
                Format the following Elasticsearch query results into a user-friendly response.
                
                Original query: "{user_query}"
                
                Entities identified: {json.dumps(entities)}
                
                Query results: {json.dumps(query_results, indent=2)}
                
                Provide a clear, concise summary of the results that directly answers the user's question.
                """}
            ]
        )
        
        final_response = response.choices[0].message.content
        
        # Evaluate response formatting
        judgment.get_current_trace().async_evaluate(
            scorers=[AnswerCorrectnessScorer(threshold=0.7)],
            input=user_query,
            actual_output=final_response,
            model=OPENAI_MODEL
        )
        
        return {**state, "final_response": final_response}
    except Exception as e:
        return {**state, "error": f"Response formatting error: {str(e)}"}

@judgment.observe(span_type="Error")
def handle_error(state: Dict[str, Any]) -> TextToESState:
    """Handle any errors that occur during processing."""
    error = state.get("error", "Unknown error occurred")
    return {**state, "final_response": f"I encountered an error processing your request: {error}"}

@judgment.observe(span_type="function")
async def main():
    # Initialize the graph
    workflow = StateGraph(TextToESState)
    
    # Add nodes directly without wrapping
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
    
    # Define the error checking function
    def has_error(state: TextToESState) -> str:
        """Route to error handler if state has an error."""
        if state.get("error"):
            return "error"
        return "continue"
    
    # Add conditional edges with correct parameter structure
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
    
    # We don't need conditional edges for format_response as it's a finish point
    
    # Set finish points
    workflow.set_finish_point("format_response")
    workflow.set_finish_point("handle_error")
    
    # Compile the workflow
    app = workflow.compile()
    
    # Create the initial state from the user query
    user_query = "Show me disconnected access points"
    
    # Run the workflow with the initial state
    config = {"recursion_limit": 25}
    try:
        result = await app.ainvoke(
            {
                "messages": [HumanMessage(content=user_query)],
                "user_query": user_query,
                "entities": None,
                "selected_index": None,
                "index_mappings": None,
                "essential_fields": None,
                "es_query": None,
                "processed_query": None,
                "query_results": None,
                "final_response": None,
                "error": None
            },
            config=config
        )
        
        # Print the result for debugging
        if result.get("error"):
            print(f"Error: {result['error']}")
        else:
            print(f"Final response: {result['final_response']}")
            
        # Return the final response
        return result.get("final_response", "No response generated")
    except Exception as e:
        print(f"Workflow execution error: {str(e)}")
        return f"Error executing workflow: {str(e)}"

if __name__ == "__main__":
    asyncio.run(main())
