from judgeval.scorers.base_scorer import BaseScorer
from judgeval.data import Example
from typing import List
from pydantic import Field
from judgeval.common.logger import judgeval_logger


class RewardScorer(BaseScorer):
    score_type: str = "Custom"  # default to custom score type
    required_params: List[str] = Field(default_factory=list)

    async def a_score_example(self, example: Example, *args, **kwargs) -> float:
        """
        Asynchronously measures the score on a single example
        """
        return example.additional_metadata.get("reward_score", 0.0)
