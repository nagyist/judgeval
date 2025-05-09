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
    
    def __init__(self, threshold: float = 0.5):
        super().__init__(
            threshold=threshold,
            score_type=APIScorer.TOOL_CORRECTNESS,
            required_params = [
                ExampleParams.ADDITIONAL_METADATA
            ]
        )
    
    def to_dict(self):
        base_dict = super().to_dict()  # Get the parent class's dictionary
        base_dict["kwargs"] = {
            
        }
        return base_dict

    @property
    def __name__(self):
        return "Tool Correctness"