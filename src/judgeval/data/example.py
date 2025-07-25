"""
Classes for representing examples in a dataset.
"""

from enum import Enum
from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, ConfigDict


class ExampleParams(str, Enum):
    INPUT = "input"
    ACTUAL_OUTPUT = "actual_output"
    EXPECTED_OUTPUT = "expected_output"
    CONTEXT = "context"
    RETRIEVAL_CONTEXT = "retrieval_context"
    TOOLS_CALLED = "tools_called"
    EXPECTED_TOOLS = "expected_tools"
    ADDITIONAL_METADATA = "additional_metadata"


class Example(
    BaseModel
):  # We don't inherit from ExampleJudgmentType because the data model is slightly different
    model_config = ConfigDict(extra="allow")

    created_at: str = datetime.now().isoformat()
    name: Optional[str] = None

    # We use model dump for sending the data to the backend server
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        data = super().model_dump(**kwargs)

        created_at = data.pop("created_at")
        name = data.pop("name")

        return {
            "example_id": "",
            "created_at": created_at,
            "name": name,
            "data": data,
        }

    def to_dict(self) -> Dict[str, Any]:
        data = super().model_dump(warnings=False)
        return data


class JudgevalExample(Example):
    input: Optional[str] = None
    actual_output: Optional[str | List[str]] = None
    expected_output: Optional[str | List[str]] = None
    retrieval_context: Optional[List[str]] = None
    additional_metadata: Optional[Dict[str, Any]] = None
    expected_tools: Optional[List[Dict[str, Any]]] = None
