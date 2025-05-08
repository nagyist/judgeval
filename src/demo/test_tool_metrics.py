import pytest # Added import
from judgeval import JudgmentClient
from judgeval.data import Example
from judgeval.scorers import FaithfulnessScorer, ToolCorrectnessScorer
from pydantic import BaseModel

class SampleSchema(BaseModel):
    field1: str
    field2: int
    
def test_tool_correctness():
    client = JudgmentClient()
    
    example = Example(
        additional_metadata = {
            "tool_name": "test_tool",
            "tool_arguments": {
                "field1": "value1",
                "field2": 1
            },
            "expected_tool_name": "alanzhang",
            "expected_tool_arguments": {
                "field1": "value1",
                "field2": 1
            }
        }
    )

    scorer = ToolCorrectnessScorer()

    # with pytest.raises(AssertionError):
    client.assert_test(
        eval_run_name="test_tool_correctness_fail",
        examples=[example],
        scorers=[scorer],
        override=True
    )
    
test_tool_correctness()