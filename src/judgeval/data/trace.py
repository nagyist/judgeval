from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from judgeval.data.tool import Tool
import json
from datetime import datetime, timezone
from judgeval.data.judgment_types import TraceUsageJudgmentType, TraceSpanJudgmentType, TraceJudgmentType

class TraceUsage(TraceUsageJudgmentType):
    pass

class TraceSpan(TraceSpanJudgmentType):    
    def get_name(self):
        if self.agent_name:
            return f"{self.agent_name}.{self.function}"
        else:
            return self.function

    def model_dump(self, **kwargs):
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "depth": self.depth,
#             "created_at": datetime.fromtimestamp(self.created_at).isoformat(),
            "created_at": datetime.fromtimestamp(self.created_at, tz=timezone.utc).isoformat(),
            "inputs": self._serialize_value(self.inputs),
            "output": self._serialize_value(self.output),
            "error": self._serialize_value(self.error),
            "parent_span_id": self.parent_span_id,
            "function": self.function,
            "duration": self.duration,
            "span_type": self.span_type,
            "usage": self.usage.model_dump() if self.usage else None,
            "has_evaluation": self.has_evaluation,
            "agent_name": self.agent_name,
            "state_before": self.state_before,
            "state_after": self.state_after,
            "additional_metadata": self._serialize_value(self.additional_metadata)
        }
    
    def print_span(self):
        """Print the span with proper formatting and parent relationship information."""
        indent = "  " * self.depth
        parent_info = f" (parent_id: {self.parent_span_id})" if self.parent_span_id else ""
        print(f"{indent}→ {self.function} (id: {self.span_id}){parent_info}")
    
    def _is_json_serializable(self, obj: Any) -> bool:
        """Helper method to check if an object is JSON serializable."""
        try:
            json.dumps(obj)
            return True
        except (TypeError, OverflowError, ValueError):
            return False

    def safe_stringify(self, output, function_name):
        """
        Safely converts an object to a string or repr, handling serialization issues gracefully.
        """
        try:
            return str(output)
        except (TypeError, OverflowError, ValueError):
            pass
    
        try:
            return repr(output)
        except (TypeError, OverflowError, ValueError):
            pass

        return None
        
    def _serialize_value(self, value: Any) -> Any:
        """Helper method to deep serialize a value safely supporting Pydantic Models / regular PyObjects."""
        if value is None:
            return None
            
        def serialize_value(value):
            if isinstance(value, BaseModel):
                return value.model_dump()
            elif isinstance(value, dict):
                # Recursively serialize dictionary values
                return {k: serialize_value(v) for k, v in value.items()}
            elif isinstance(value, (list, tuple)):
                # Recursively serialize list/tuple items
                return [serialize_value(item) for item in value]
            else:
                # Try direct JSON serialization first
                try:
                    json.dumps(value)
                    return value
                except (TypeError, OverflowError, ValueError):
                    # Fallback to safe stringification
                    return self.safe_stringify(value, self.function)

        # Start serialization with the top-level value
        return serialize_value(value)

class Trace(TraceJudgmentType):
    pass
    