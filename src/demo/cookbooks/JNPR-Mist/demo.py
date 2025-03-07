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
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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

# Mock ES indexes and mappings for demonstration
ES_INDEXES = {
    "logs-security": {
        "description": "Contains security logs including authentication events, access attempts, and security incidents",
        "mappings": {
            "properties": {
                "timestamp": {"type": "date"},
                "org_id": {"type": "keyword"},
                "user_id": {"type": "keyword"},
                "ev_type": {"type": "keyword"},
                "source_ip": {"type": "ip"},
                "status": {"type": "keyword"},
                "details": {"type": "text"}
            }
        },
        "essential_fields": {
            "auth_failure": "ev_type",
            "login_attempt": "ev_type" 
        }
    },
    "network-devices": {
        "description": "Contains information about network devices such as APs, switches, and routers",
        "mappings": {
            "properties": {
                "device_id": {"type": "keyword"},
                "org_id": {"type": "keyword"},
                "device_type": {"type": "keyword"},
                "status": {"type": "keyword"},
                "last_seen": {"type": "date"},
                "ip_address": {"type": "ip"},
                "location": {"type": "keyword"},
                "firmware": {"type": "keyword"}
            }
        },
        "essential_fields": {
            "disconnected": "status"
        }
    }
}

class StateInput(BaseModel):
    """Input schema for tool functions."""
    state: Dict[str, Any] = Field(description="The current state of the workflow")
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

# Update OpenAI model name to a valid one
OPENAI_MODEL = "gpt-4-turbo-preview"

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

def create_node(func: Callable[[Dict[str, Any]], TextToESState]):
    """Create a node that properly handles state input."""
    def wrapped(state: Dict[str, Any]) -> Dict[str, Any]:
        return func(state)
    return wrapped

@tool(args_schema=StateInput)
def extract_entities(tool_input: StateInput) -> TextToESState:
    """Extract named entities from the user query using OpenAI's model."""
    state = tool_input.state
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
        with judgment.trace("entity_extraction"):
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

@tool(args_schema=StateInput)
def validate_entities(tool_input: StateInput) -> TextToESState:
    """Validate if extracted entities exist in the organization."""
    state = tool_input.state
    entities = state.get("entities", [])
    # Mock implementation - in real system would query a database
    valid_entities = []
    for entity in entities:
        # Mock validation logic
        if entity["type"] == "user":
            valid_entities.append({
                **entity, 
                "valid": True, 
                "org_id": "org_12345"
            })
        elif entity["type"] == "device":
            valid_entities.append({
                **entity, 
                "valid": True, 
                "org_id": "org_12345"
            })
        else:
            valid_entities.append({
                **entity, 
                "valid": False
            })
            
    return {**state, "validated_entities": valid_entities}

@tool(args_schema=StateInput)
def select_index(tool_input: StateInput) -> TextToESState:
    """Select the appropriate Elasticsearch index based on the user query."""
    state = tool_input.state
    user_query = state["user_query"]
    entities = state.get("entities", [])
    
    # Create description of available indexes
    index_descriptions = "\n".join([
        f"- {idx}: {details['description']}" 
        for idx, details in ES_INDEXES.items()
    ])
    
    prompt = f"""
    Based on the user query and extracted entities, select the most appropriate Elasticsearch index.
    
    Available indexes:
    {index_descriptions}
    
    User query: {user_query}
    Extracted entities: {json.dumps(entities)}
    
    Return only the name of the index you select, nothing else.
    """
    
    with judgment.trace("index_selection"):
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are an index selection specialist. Select the most appropriate index for querying."},
                {"role": "user", "content": prompt}
            ]
        )
        
        selected_index = response.choices[0].message.content.strip()
        
        # Validate the selected index exists
        if selected_index not in ES_INDEXES:
            return {**state, "error": f"Invalid index selected: {selected_index}"}
        
        # Get index mappings
        index_mappings = ES_INDEXES[selected_index]["mappings"]
        essential_fields = ES_INDEXES[selected_index]["essential_fields"]
        
        # Evaluate index selection
        judgment.get_current_trace().async_evaluate(
            scorers=[AnswerCorrectnessScorer(threshold=0.7)],
            input=user_query,
            actual_output=selected_index,
            model=OPENAI_MODEL
        )
        
        return {
            **state, 
            "selected_index": selected_index,
            "index_mappings": index_mappings,
            "essential_fields": essential_fields
        }

@tool(args_schema=StateInput)
def get_field_values(tool_input: StateInput) -> TextToESState:
    """Get possible values for essential fields in the selected index."""
    state = tool_input.state
    index = state.get("selected_index")
    essential_fields = state.get("essential_fields", {})
    
    field_values = {}
    for key, field_name in essential_fields.items():
        # Mock implementation - in real system would query Elasticsearch
        if index == "logs-security" and field_name == "ev_type":
            field_values[field_name] = ["login_attempt", "auth_failure", "password_reset", "permission_change"]
        elif index == "network-devices" and field_name == "status":
            field_values[field_name] = ["online", "offline", "disconnected", "maintenance"]
            
    return {**state, "field_values": field_values}

@tool(args_schema=StateInput)
def generate_query(tool_input: StateInput) -> TextToESState:
    """Generate Elasticsearch query based on user query and index information."""
    state = tool_input.state
    user_query = state["user_query"]
    selected_index = state["selected_index"]
    index_mappings = state["index_mappings"]
    entities = state.get("entities", [])
    field_values = state.get("field_values", {})
    
    prompt = f"""
    Generate an Elasticsearch query based on the user's request.
    
    Index: {selected_index}
    Index mappings: {json.dumps(index_mappings)}
    User query: {user_query}
    Entities: {json.dumps(entities)}
    Field values: {json.dumps(field_values)}
    
    For any org_id or user_id fields, use the placeholder "MASKED_VALUE" instead of actual values.
    
    Return only the Elasticsearch query in JSON format.
    """
    
    with judgment.trace("query_generation"):
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are an Elasticsearch query specialist. Generate precise, functional queries."},
                {"role": "user", "content": prompt}
            ]
        )
        
        try:
            content = response.choices[0].message.content
            # Extract JSON from response
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].strip()
            else:
                json_str = content.strip()
                
            es_query = json.loads(json_str)
            
            # Evaluate query generation
            judgment.get_current_trace().async_evaluate(
                scorers=[FaithfulnessScorer(threshold=0.7)],
                input=prompt,
                actual_output=json.dumps(es_query),
                model=OPENAI_MODEL
            )
            
            return {**state, "es_query": es_query}
        except Exception as e:
            return {**state, "error": f"Query generation error: {str(e)}"}

@tool(args_schema=StateInput)
def process_query(tool_input: StateInput) -> TextToESState:
    """Process and clean up the generated Elasticsearch query."""
    state = tool_input.state
    query = state["es_query"]
    entities = state.get("entities", [])
    selected_index = state["selected_index"]
    
    processed_query = json.loads(json.dumps(query))  # Deep copy
    
    # Replace masked values with real values
    if "query" in processed_query and "bool" in processed_query["query"]:
        if "must" in processed_query["query"]["bool"]:
            for i, clause in enumerate(processed_query["query"]["bool"]["must"]):
                if "term" in clause:
                    for field, value in clause["term"].items():
                        if value == "MASKED_VALUE":
                            if field == "org_id":
                                processed_query["query"]["bool"]["must"][i]["term"][field] = "org_12345"
                            elif field == "user_id" and entities:
                                user_entities = [e for e in entities if e["type"] == "user"]
                                if user_entities:
                                    processed_query["query"]["bool"]["must"][i]["term"][field] = f"user_{user_entities[0]['name']}"
    
    # Set default size if not specified
    if "size" not in processed_query:
        processed_query["size"] = 10
        
    # Remove any fields not in the index mapping
    valid_fields = set(ES_INDEXES[selected_index]["mappings"]["properties"].keys())
    
    # Function to recursively check and clean fields
    def clean_query_fields(query_part):
        if isinstance(query_part, dict):
            for key in list(query_part.keys()):
                if key in ["term", "terms", "match", "match_phrase"]:
                    field_keys = list(query_part[key].keys())
                    for field in field_keys:
                        if field not in valid_fields:
                            del query_part[key][field]
                            # If no fields left, mark for removal
                            if not query_part[key]:
                                return None
                else:
                    result = clean_query_fields(query_part[key])
                    if result is None and key in ["must", "should", "must_not"]:
                        query_part[key] = [item for item in query_part[key] if clean_query_fields(item) is not None]
                        if not query_part[key]:
                            del query_part[key]
            return query_part if query_part else None
        elif isinstance(query_part, list):
            return [item for item in query_part if clean_query_fields(item) is not None]
        return query_part
    
    clean_query_fields(processed_query)
    
    return {**state, "processed_query": processed_query}

@tool(args_schema=StateInput)
def execute_query(tool_input: StateInput) -> TextToESState:
    """Execute the processed Elasticsearch query."""
    state = tool_input.state
    index = state["selected_index"]
    query = state["processed_query"]
    
    # Mock implementation - in real system would query Elasticsearch
    if index == "logs-security":
        results = {
            "took": 5,
            "hits": {
                "total": {"value": 120, "relation": "eq"},
                "hits": [
                    {"_source": {"timestamp": "2023-06-15T14:30:45Z", "user_id": "user_123", "ev_type": "auth_failure", "source_ip": "192.168.1.1", "status": "failed"}},
                    {"_source": {"timestamp": "2023-06-15T14:35:12Z", "user_id": "user_456", "ev_type": "auth_failure", "source_ip": "192.168.1.2", "status": "failed"}}
                ]
            }
        }
    elif index == "network-devices":
        results = {
            "took": 3,
            "hits": {
                "total": {"value": 45, "relation": "eq"},
                "hits": [
                    {"_source": {"device_id": "ap-001", "device_type": "access_point", "status": "disconnected", "last_seen": "2023-06-14T23:45:12Z"}},
                    {"_source": {"device_id": "ap-002", "device_type": "access_point", "status": "disconnected", "last_seen": "2023-06-15T02:12:33Z"}}
                ]
            }
        }
    else:
        results = {"error": "Failed to execute query"}
        
    return {**state, "query_results": results}

@tool(args_schema=StateInput)
def format_response(tool_input: StateInput) -> TextToESState:
    """Format the query results into a user-friendly response."""
    state = tool_input.state
    user_query = state["user_query"]
    query_results = state["query_results"]
    
    prompt = f"""
    Format the Elasticsearch query results into a natural language response.
    Present the information clearly and concisely, in a way that directly answers the user's question.
    
    User query: {user_query}
    Query results: {json.dumps(query_results)}
    
    Your response should be informative and user-friendly.
    """
    
    with judgment.trace("response_generation"):
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": "You are a data presentation specialist. Format technical results into clear, informative responses."},
                {"role": "user", "content": prompt}
            ]
        )
        
        final_response = response.choices[0].message.content
        
        # Evaluate response formatting
        judgment.get_current_trace().async_evaluate(
            scorers=[FaithfulnessScorer(threshold=0.7), AnswerRelevancyScorer(threshold=0.7)],
            input=prompt,
            actual_output=final_response,
            retrieval_context=json.dumps(query_results),
            model=OPENAI_MODEL
        )
        
        return {**state, "final_response": final_response}

@tool(args_schema=StateInput)
def handle_error(tool_input: StateInput) -> TextToESState:
    """Handle any errors that occur during processing."""
    state = tool_input.state
    error = state.get("error", "Unknown error occurred")
    return {**state, "final_response": f"I encountered an error processing your request: {error}"}

async def main():
    # Initialize the graph
    with judgment.trace("text_to_es_workflow", overwrite=True) as trace:
        workflow = StateGraph(TextToESState)
        
        # Create tool instances
        def wrap_tool(tool_func):
            """Wrap tool function to handle state input properly."""
            async def wrapped(state: Dict[str, Any]) -> Dict[str, Any]:
                state_input = {"state": state}  # Pass as dict instead of StateInput object
                return await tool_func.ainvoke(state_input)  # Use ainvoke instead of direct call
            return wrapped
        
        tools = {
            "extract_entities": wrap_tool(extract_entities),
            "validate_entities": wrap_tool(validate_entities),
            "select_index": wrap_tool(select_index),
            "get_field_values": wrap_tool(get_field_values),
            "generate_query": wrap_tool(generate_query),
            "process_query": wrap_tool(process_query),
            "execute_query": wrap_tool(execute_query),
            "format_response": wrap_tool(format_response),
            "handle_error": wrap_tool(handle_error)
        }
        
        # Add nodes with proper tool wrapping
        for name, tool_func in tools.items():
            workflow.add_node(name, tool_func)
        
        # Define edges
        workflow.add_edge("extract_entities", "validate_entities")
        workflow.add_edge("validate_entities", "select_index")
        workflow.add_edge("select_index", "get_field_values")
        workflow.add_edge("get_field_values", "generate_query")
        workflow.add_edge("generate_query", "process_query")
        workflow.add_edge("process_query", "execute_query")
        workflow.add_edge("execute_query", "format_response")
        
        # Define conditional edges for error handling
        def error_checker(state: TextToESState) -> str:
            if state.get("error"):
                return "error"
            return "continue"
        
        # Add conditional edges for error checking after key steps
        workflow.add_conditional_edges(
            "extract_entities",
            error_checker,
            {
                "error": "handle_error",
                "continue": "validate_entities"
            }
        )
        
        workflow.add_conditional_edges(
            "select_index",
            error_checker,
            {
                "error": "handle_error",
                "continue": "get_field_values"
            }
        )
        
        workflow.add_conditional_edges(
            "generate_query",
            error_checker,
            {
                "error": "handle_error",
                "continue": "process_query"
            }
        )
        
        # Set entry and exit points
        workflow.set_entry_point("extract_entities")
        workflow.set_finish_point("format_response")
        workflow.set_finish_point("handle_error")
        
        # Compile the graph
        graph = workflow.compile()
        
        # Execute the workflow with a sample query
        input_state = {
            "messages": [HumanMessage(content="Show me disconnected access points")],
            "user_query": "Show me disconnected access points",
            "entities": None,
            "selected_index": None,
            "index_mappings": None,
            "essential_fields": None,
            "es_query": None,
            "processed_query": None,
            "query_results": None,
            "final_response": None,
            "error": None
        }
        
        try:
            result = await graph.ainvoke(input_state)
            if result.get("error"):
                print(f"Error: {result['error']}")
            else:
                print(f"Final response: {result['final_response']}")
        except Exception as e:
            print(f"Workflow execution error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
