import os
import time
import uuid
from unittest.mock import Mock, patch, MagicMock
import pytest
import litellm

from judgeval import Tracer
from judgeval.integrations.litellm_integration import JudgevalLitellmCallbackHandler


class MockLLMAgent:
    """Mock agent that uses LiteLLM for testing"""

    def __init__(self, tracer: Tracer):
        self.tracer = tracer

    @property
    def name(self):
        return "test-agent"

    def generate_response(self, prompt: str, model: str = "gpt-4o") -> str:
        """Generate a response using LiteLLM"""
        # This would normally call litellm.completion()
        # We'll mock the response in our tests
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )
        return response.choices[0].message.content


def create_mock_response(content: str = "Test response from LLM"):
    """Create a mock LiteLLM response object"""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = content
    mock_response.model = "gpt-4o"
    mock_response.usage = Mock()
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 20
    mock_response.usage.total_tokens = 30
    return mock_response


@pytest.fixture
def tracer():
    """Create a tracer instance for testing"""
    return Tracer(
        api_key="test-key",
        organization_id="test-org",
        project_name="test-litellm-integration"
    )


@pytest.fixture
def mock_agent(tracer):
    """Create a mock agent for testing"""
    return MockLLMAgent(tracer)


@pytest.fixture
def litellm_handler(tracer):
    """Create LiteLLM callback handler"""
    return JudgevalLitellmCallbackHandler(tracer)


def test_litellm_callback_handler_creation(tracer):
    """Test that the callback handler can be created and registered"""
    handler = JudgevalLitellmCallbackHandler(tracer)

    assert handler.tracer == tracer
    assert handler._current_span_id is None
    assert handler._current_trace_client is None

    # Test registering with LiteLLM
    litellm.callbacks = [handler]
    assert handler in litellm.callbacks


def test_span_creation_and_updates(tracer, litellm_handler):
    """Test that spans are created and updated correctly"""

    # Create a trace context
    with tracer.trace("test-trace") as trace_client:

        # Mock the LiteLLM completion call
        mock_response = create_mock_response("Hello from LiteLLM!")
        start_time = time.time()
        end_time = start_time + 2.5

        # Simulate the callback lifecycle
        litellm_handler.log_pre_api_call(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            kwargs={}
        )

        # Verify span was created
        assert litellm_handler._current_span_id is not None
        assert litellm_handler._current_trace_client is not None
        span_id = litellm_handler._current_span_id

        # Check span exists in trace
        assert span_id in trace_client.span_id_to_span
        span = trace_client.span_id_to_span[span_id]

        # Verify initial span properties
        assert span.function == "LiteLLM-gpt-4o"
        assert span.span_type == "llm"
        assert span.inputs["model"] == "gpt-4o"
        assert span.inputs["messages"] == [
            {"role": "user", "content": "Hello"}]
        assert span.duration is None  # Should be None initially
        assert span.output is None    # Should be None initially

        # Simulate successful completion
        litellm_handler.log_success_event(
            kwargs={},
            response_obj=mock_response,
            start_time=start_time,
            end_time=end_time
        )

        # Verify span was updated
        updated_span = trace_client.span_id_to_span[span_id]
        assert updated_span.duration == 2.5
        assert updated_span.output == "Hello from LiteLLM!"

        # Verify handler cleaned up
        assert litellm_handler._current_span_id is None
        assert litellm_handler._current_trace_client is None


def test_error_handling(tracer, litellm_handler):
    """Test that errors are handled correctly"""

    with tracer.trace("test-error-trace") as trace_client:

        # Simulate the callback lifecycle with error
        litellm_handler.log_pre_api_call(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            kwargs={}
        )

        span_id = litellm_handler._current_span_id
        start_time = time.time()
        end_time = start_time + 1.0

        # Simulate failure
        litellm_handler.log_failure_event(
            kwargs={"exception": Exception("API Error")},
            response_obj=None,
            start_time=start_time,
            end_time=end_time
        )

        # Verify error was recorded
        updated_span = trace_client.span_id_to_span[span_id]
        assert updated_span.duration == 1.0
        assert "API Error" in str(updated_span.output)


@patch('litellm.completion')
def test_mock_agent_with_litellm_integration(mock_completion, tracer, mock_agent):
    """Test the mock agent with LiteLLM integration end-to-end"""

    # Set up the callback handler
    handler = JudgevalLitellmCallbackHandler(tracer)
    litellm.callbacks = [handler]

    # Mock the LiteLLM response
    mock_response = create_mock_response("Paris is the capital of France.")
    mock_completion.return_value = mock_response

    # Create observed method for the agent
    @tracer.observe(name="agent-generate")
    def agent_generate_with_tracing(prompt: str) -> str:
        return mock_agent.generate_response(prompt)

    # Execute the agent method
    result = agent_generate_with_tracing("What is the capital of France?")

    # Verify the result
    assert result == "Paris is the capital of France."

    # Verify LiteLLM was called
    mock_completion.assert_called_once()
    call_args = mock_completion.call_args
    assert call_args[1]["model"] == "gpt-4o"
    assert call_args[1]["messages"][0]["content"] == "What is the capital of France?"


def test_save_coordination(tracer, litellm_handler):
    """Test that save coordination works properly"""

    # Mock the save methods to track calls
    original_save = None
    save_calls = []

    def mock_save(self, overwrite=False):
        save_calls.append(("save", overwrite))
        return self.trace_id, {}

    def mock_perform_actual_save(self, overwrite=False):
        save_calls.append(("actual_save", overwrite))
        return self.trace_id, {}

    with tracer.trace("test-coordination") as trace_client:

        # Patch the save methods
        original_save = trace_client.save
        original_actual_save = trace_client._perform_actual_save
        trace_client.save = lambda overwrite=False: mock_save(
            trace_client, overwrite)
        trace_client._perform_actual_save = lambda overwrite=False: mock_perform_actual_save(
            trace_client, overwrite)

        try:
            # Start LiteLLM operation
            litellm_handler.log_pre_api_call(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Test"}],
                kwargs={}
            )

            # Verify _safe_to_save is False
            assert not tracer._safe_to_save

            # Simulate trace save attempt (this should be deferred)
            trace_client.save()

            # Verify save was deferred
            assert len(save_calls) == 1
            assert save_calls[0] == ("save", False)
            assert tracer._deferred_save_pending

            # Complete LiteLLM operation
            mock_response = create_mock_response()
            litellm_handler.log_success_event(
                kwargs={},
                response_obj=mock_response,
                start_time=time.time(),
                end_time=time.time() + 1
            )

            # Verify _safe_to_save is True and deferred save was executed
            assert tracer._safe_to_save
            assert not tracer._deferred_save_pending
            assert len(save_calls) == 2
            assert save_calls[1] == ("actual_save", False)

        finally:
            # Restore original methods
            if original_save:
                trace_client.save = original_save
                trace_client._perform_actual_save = original_actual_save


def test_multiple_llm_calls_same_trace(tracer, litellm_handler):
    """Test multiple LiteLLM calls within the same trace"""

    with tracer.trace("test-multiple-calls") as trace_client:

        # First LLM call
        litellm_handler.log_pre_api_call(
            model="gpt-4o",
            messages=[{"role": "user", "content": "First call"}],
            kwargs={}
        )
        first_span_id = litellm_handler._current_span_id

        litellm_handler.log_success_event(
            kwargs={},
            response_obj=create_mock_response("First response"),
            start_time=time.time(),
            end_time=time.time() + 1
        )

        # Second LLM call
        litellm_handler.log_pre_api_call(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Second call"}],
            kwargs={}
        )
        second_span_id = litellm_handler._current_span_id

        litellm_handler.log_success_event(
            kwargs={},
            response_obj=create_mock_response("Second response"),
            start_time=time.time(),
            end_time=time.time() + 2
        )

        # Verify both spans exist and are different
        assert first_span_id != second_span_id
        assert first_span_id in trace_client.span_id_to_span
        assert second_span_id in trace_client.span_id_to_span

        # Verify span details
        first_span = trace_client.span_id_to_span[first_span_id]
        second_span = trace_client.span_id_to_span[second_span_id]

        assert first_span.function == "LiteLLM-gpt-4o"
        assert second_span.function == "LiteLLM-gpt-3.5-turbo"
        assert first_span.output == "First response"
        assert second_span.output == "Second response"


@patch('litellm.completion')
def test_end_to_end_with_real_trace_saving(mock_completion, tracer):
    """Test end-to-end with actual trace saving"""

    # Set up the callback handler
    handler = JudgevalLitellmCallbackHandler(tracer)
    litellm.callbacks = [handler]

    # Mock LiteLLM response
    mock_response = create_mock_response("End-to-end test successful!")
    mock_completion.return_value = mock_response

    @tracer.observe(name="e2e-test")
    def run_llm_call():
        return litellm.completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Run end-to-end test"}],
            max_tokens=50
        )

    # Execute the test
    result = run_llm_call()

    # Verify result
    assert result.choices[0].message.content == "End-to-end test successful!"

    # Verify trace was created and saved
    assert len(tracer.traces) > 0

    # Find the LiteLLM span in the saved trace
    trace_data = tracer.traces[-1]
    llm_spans = [span for span in trace_data.get(
        "entries", []) if span.get("function", "").startswith("LiteLLM-")]

    assert len(llm_spans) == 1
    llm_span = llm_spans[0]
    assert llm_span["function"] == "LiteLLM-gpt-4o"
    assert llm_span["output"] == "End-to-end test successful!"
    assert llm_span["duration"] is not None
    assert llm_span["span_type"] == "llm"


if __name__ == "__main__":
    # Run a simple test
    tracer = Tracer(
        api_key="test-key",
        organization_id="test-org",
        project_name="litellm-test"
    )

    handler = JudgevalLitellmCallbackHandler(tracer)
    print("LiteLLM integration test setup successful!")
    print(f"Handler created: {handler}")
    print(f"Tracer ready: {tracer}")
