"""
`judgeval` JSON correctness scorer

TODO add link to docs page for this scorer

"""


# External imports
from pydantic import BaseModel, Field
# Internal imports
from judgeval.scorers.api_scorer import APIJudgmentScorer
from judgeval.constants import APIScorer
from judgeval.data import ExampleParams

class ToolCorrectnessScorer(APIJudgmentScorer):
    tool_name: str = Field(None, exclude=True)
    tool_arguments: BaseModel = Field(None, exclude=True)
    
    def __init__(self):
        super().__init__(
            score_type=APIScorer.TOOL_CORRECTNESS,
            required_params = [
                ExampleParams.ADDITIONAL_METADATA
            ]
        )
    
    def to_dict(self):
        base_dict = super().to_dict()  # Get the parent class's dictionary
        base_dict["kwargs"] = {
            "tool_arguments": self.tool_arguments.model_json_schema()
        }
        return base_dict

    @property
    def __name__(self):
        return "Tool Correctness"