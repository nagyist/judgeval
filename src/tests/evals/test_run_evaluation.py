import pytest
from unittest.mock import patch

from judgeval.run_evaluation import (
    merge_results,
    check_missing_scorer_data,
    run_with_spinner,
    check_examples,
    await_with_spinner,
    SpinnerWrappedTask,
    assert_test,
)
from judgeval.data import Example, ScoringResult, ScorerData, Trace
from judgeval.evaluation_run import EvaluationRun
from judgeval.data.trace_run import TraceRun
from judgeval.scorers import FaithfulnessScorer

# Mock data for testing
MOCK_API_KEY = "test_api_key"
MOCK_ORG_ID = "test_org_id"
MOCK_PROJECT_NAME = "test_project"
MOCK_EVAL_NAME = "test_eval"


@pytest.fixture
def mock_evaluation_run():
    return EvaluationRun(
        examples=[Example(input="test", actual_output="test")],
        scorers=[FaithfulnessScorer(threshold=0.5)],
        project_name=MOCK_PROJECT_NAME,
        eval_name=MOCK_EVAL_NAME,
        judgment_api_key=MOCK_API_KEY,
        organization_id=MOCK_ORG_ID,
    )


@pytest.fixture
def mock_trace_run():
    return TraceRun(
        traces=[
            Trace(
                trace_id="test_trace_id",
                name="test_trace",
                created_at="2024-03-20T12:00:00Z",
                duration=1.0,
                trace_spans=[],
            )
        ],
        scorers=[FaithfulnessScorer(threshold=0.5)],
        project_name=MOCK_PROJECT_NAME,
        eval_name=MOCK_EVAL_NAME,
        judgment_api_key=MOCK_API_KEY,
        organization_id=MOCK_ORG_ID,
    )


@pytest.fixture
def mock_scoring_results():
    return [
        ScoringResult(
            success=True,
            scorers_data=[
                ScorerData(
                    name="test_scorer",
                    threshold=0.5,
                    success=True,
                    score=0.8,
                    reason="Test reason",
                    strict_mode=True,
                    evaluation_model="gpt-4",
                    error=None,
                    additional_metadata={"test": "metadata"},
                )
            ],
            data_object=Example(input="test", actual_output="test"),
        )
    ]


class TestRunEvaluation:
    def test_merge_results(self, mock_scoring_results):
        api_results = mock_scoring_results
        local_results = mock_scoring_results

        merged = merge_results(api_results, local_results)

        assert len(merged) == len(api_results)
        assert (
            merged[0].scorers_data
            == api_results[0].scorers_data + local_results[0].scorers_data
        )

    def test_check_missing_scorer_data(self, mock_scoring_results):
        results = mock_scoring_results
        results[0].scorers_data = None

        checked_results = check_missing_scorer_data(results)

        assert checked_results == results

    def test_run_with_spinner(self):
        def test_func():
            return "test_result"

        result = run_with_spinner("Testing: ", test_func)

        assert result == "test_result"

    def test_check_examples(self):
        examples = [Example(input="test", actual_output="test")]
        scorers = [FaithfulnessScorer(threshold=0.5)]

        # Mock input to simulate user entering 'y'
        with patch("builtins.input", return_value="y"):
            check_examples(examples, scorers)

    @pytest.mark.asyncio
    async def test_await_with_spinner(self):
        async def test_task():
            return "test_result"

        result = await await_with_spinner(test_task(), "Testing: ")

        assert result == "test_result"

    def test_spinner_wrapped_task(self):
        async def test_task():
            return "test_result", "pretty_str"

        task = SpinnerWrappedTask(test_task(), "Testing: ")

        # Test that the task is awaitable
        assert hasattr(task, "__await__")

    def test_assert_test_success(self, mock_scoring_results):
        # All tests pass
        assert_test(mock_scoring_results)

    def test_assert_test_failure(self):
        # Create a failing result
        failing_result = ScoringResult(
            success=False,
            scorers_data=[
                ScorerData(
                    name="test_scorer",
                    threshold=0.5,
                    success=False,
                    score=0.3,
                    reason="Test failure",
                    strict_mode=True,
                    evaluation_model="gpt-4",
                    error=None,
                    additional_metadata={"test": "metadata"},
                )
            ],
            data_object=Example(input="test", actual_output="test"),
        )

        with pytest.raises(AssertionError):
            assert_test([failing_result])
