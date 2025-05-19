"""
`judgeval` tool dependency scorer
"""

# Internal imports
from judgeval.scorers.api_scorer import APIJudgmentScorer
from judgeval.constants import APIScorer

class ToolDependencyScorer(APIJudgmentScorer):
    def __init__(self, threshold: float=1.0):
        super().__init__(
            threshold=threshold, 
            score_type=APIScorer.TOOL_DEPENDENCY,
        )

    @property
    def __name__(self):
        return "Tool Dependency"
