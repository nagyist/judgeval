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
        input={"question": "What is the capital of France?"},
        expected_tools_called = [
            {
                "tool_name": "tool_1",
                "tool_arguments": {
                    "field1": "value1",
                    "field2": 100
                }
            },
            {
                "tool_name": "tool_2",
                "tool_arguments": {
                    "field1": "value1",
                    "field2": 100
                }
            }
        ]
    )

    scorer = ToolCorrectnessScorer(threshold=0.5)

    # with pytest.raises(AssertionError):
    client.assert_test(
        entry_point=run_agent(),
        eval_run_name="test_tool_correctness_fail",
        examples=[example],
        scorers=[scorer],
    )
    
test_tool_correctness()


from judgment import JudgmentAgent
agent = JudgmentAgent(Agent())

agent.run(
    example=example,
    metrics=ToolCorrectnessScorer()
)