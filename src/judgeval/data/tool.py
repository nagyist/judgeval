from pydantic import model_serializer
from typing import Dict, Any
from judgeval.data.judgment_types import ToolJudgmentType


class Tool(ToolJudgmentType):
    pass

    @model_serializer
    def serialize_model(self) -> Dict[str, Any]:
        """Pydantic model serializer for JSON compatibility"""
        return {
            "tool_name": self.tool_name,
            "parameters": self.parameters,
            "agent_name": self.agent_name,
        }
