"""
Classes for representing examples in a dataset.
"""

from enum import Enum
from datetime import datetime
from typing import Dict, Any, List, Optional
from judgeval.data.judgment_types import ExampleJudgmentType


class ExampleParams(str, Enum):
    INPUT = "input"
    ACTUAL_OUTPUT = "actual_output"
    EXPECTED_OUTPUT = "expected_output"
    CONTEXT = "context"
    RETRIEVAL_CONTEXT = "retrieval_context"
    TOOLS_CALLED = "tools_called"
    EXPECTED_TOOLS = "expected_tools"
    ADDITIONAL_METADATA = "additional_metadata"


class Example(ExampleJudgmentType):
    example_id: str = ""
    created_at: str = datetime.now().isoformat()
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = super().model_dump(warnings=False)
        return data

    def get_fields(self):
        excluded = {"example_id", "name", "created_at"}
        return self.model_dump(exclude=excluded)


class JudgevalExample(Example):
    input: Optional[str] = None
    actual_output: Optional[str | List[str]] = None
    expected_output: Optional[str | List[str]] = None
    retrieval_context: Optional[List[str]] = None
    additional_metadata: Optional[Dict[str, Any]] = None
    expected_tools: Optional[List[Dict[str, Any]]] = None
