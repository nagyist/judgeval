from pydantic import BaseModel, field_validator, model_serializer
from typing import Dict, Any, Optional, List
import warnings
from judgeval.data.judgment_types import ToolJudgmentType

class Tool(ToolJudgmentType):
    tool_name: str
    parameters: Optional[Dict[str, Any]] = None
    agent_name: Optional[str] = None
    result_dependencies: Optional[List[Dict[str, Any]]] = None
    action_dependencies: Optional[List[Dict[str, Any]]] = None
    require_all: Optional[bool] = None
    
    @model_serializer
    def serialize_model(self) -> Dict[str, Any]:
        """Pydantic model serializer for JSON compatibility"""
        return {
            "tool_name": self.tool_name,
            "parameters": self.parameters,
            "agent_name": self.agent_name
        }