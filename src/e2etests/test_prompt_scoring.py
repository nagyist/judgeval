"""
Test to implement a PromptScorer

Toy example in this case to determine the sentiment
"""

from judgeval.judgment_client import JudgmentClient
from judgeval.data import Example
from judgeval.judges import TogetherJudge
from judgeval.scorers import PromptScorer
from uuid import uuid4


qwen = TogetherJudge()


def test_prompt_scoring(project_name: str):
    pos_example = Example(
        input="What's the store return policy?",
        actual_output="Our return policy is wonderful! You may return any item within 30 days of purchase for a full refund.",
    )

    neg_example = Example(
        input="I'm having trouble with my order",
        actual_output="That's not my problem. You should have read the instructions more carefully.",
    )

    scorer = PromptScorer.create(
        name=f"Sentiment Classifier {uuid4()}",
        conversation=[
            {
                "role": "system",
                "content": "Is the response positive (Y/N)? The response is: {{actual_output}}.",
            }
        ],
        options={"Y": 1, "N": 0},
    )

    # Test direct API call first
    from dotenv import load_dotenv

    load_dotenv()
    import os

    # Then test using client.run_evaluation()
    client = JudgmentClient(api_key=os.getenv("JUDGMENT_API_KEY"))
    results = client.run_evaluation(
        examples=[pos_example, neg_example],
        scorers=[scorer],
        model="Qwen/Qwen2.5-72B-Instruct-Turbo",
        project_name=project_name,
        eval_run_name="sentiment_run_1",  # Unique run name
        override=True,
    )
    assert results[0].success
    assert not results[1].success

    print("\nClient Evaluation Results:")
    for i, result in enumerate(results):
        print(f"\nExample {i + 1}:")
        print(f"Input: {[pos_example, neg_example][i].input}")
        print(f"Output: {[pos_example, neg_example][i].actual_output}")
        # Access score data directly from result
        if hasattr(result, "score"):
            print(f"Score: {result.score}")
        if hasattr(result, "reason"):
            print(f"Reason: {result.reason}")
        if hasattr(result, "metadata") and result.metadata:
            print(f"Metadata: {result.metadata}")
