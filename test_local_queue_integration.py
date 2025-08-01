#!/usr/bin/env python3
"""
Test script for LocalEvaluationQueue integration with async_evaluate.

This script demonstrates how the local evaluation queue works with the tracer
to process local BaseScorer evaluations and log results automatically.
"""

import asyncio
import os
import time
from typing import Dict, Any

# Set up environment variables for testing
os.environ["JUDGMENT_API_KEY"] = "test-api-key"
os.environ["JUDGMENT_ORG_ID"] = "test-org-id"

from judgeval.common.tracer import Tracer
from judgeval.data import Example
from judgeval.scorers.base_scorer import BaseScorer


class MockLocalScorer(BaseScorer):
    """A simple mock scorer that always returns a fixed score."""

    score_type: str = "MockLocal"

    def __init__(self, threshold: float = 0.5, mock_score: float = 0.8):
        super().__init__(threshold=threshold)
        self.mock_score = mock_score

    async def a_score_example(self, example: Example, *args, **kwargs) -> float:
        """Mock scoring that returns a fixed score after a small delay."""
        await asyncio.sleep(0.1)  # Simulate processing time
        print(f"MockLocalScorer processing example: {example.input[:50]}...")
        return self.mock_score


class SlowMockScorer(BaseScorer):
    """A mock scorer that takes longer to process."""

    score_type: str = "SlowMock"

    def __init__(self, threshold: float = 0.5, mock_score: float = 0.6):
        super().__init__(threshold=threshold)
        self.mock_score = mock_score

    async def a_score_example(self, example: Example, *args, **kwargs) -> float:
        """Mock scoring with longer delay."""
        await asyncio.sleep(0.5)  # Longer processing time
        print(f"SlowMockScorer processing example: {example.input[:50]}...")
        return self.mock_score


# Initialize the tracer
tracer = Tracer(
    project_name="test-local-queue", enable_monitoring=True, enable_evaluations=True
)


@tracer.observe(span_type="tool")
async def process_user_input(input_text: str) -> str:
    """Process user input and evaluate with local scorers."""

    # Simulate some processing
    await asyncio.sleep(0.1)
    output = f"Processed: {input_text}"

    # Create example for evaluation
    example = Example(
        input=input_text,
        actual_output=output,
        expected_output=f"Expected processed version of: {input_text}",
    )

    # Use async_evaluate with local scorers - this should use LocalEvaluationQueue
    tracer.async_evaluate(
        scorers=[
            MockLocalScorer(threshold=0.5, mock_score=0.8),
            SlowMockScorer(threshold=0.6, mock_score=0.7),
        ],
        example=example,
        model="gpt-4.1-mini",
    )

    return output


@tracer.observe(span_type="tool")
async def analyze_text(text: str) -> Dict[str, Any]:
    """Analyze text and evaluate with different local scorer."""

    await asyncio.sleep(0.2)
    analysis = {
        "length": len(text),
        "word_count": len(text.split()),
        "sentiment": "positive",
    }

    # Another evaluation with a single local scorer
    tracer.async_evaluate(
        scorers=[MockLocalScorer(threshold=0.7, mock_score=0.9)],
        input=text,
        actual_output=str(analysis),
        expected_output="Expected analysis result",
        model="gpt-4.1-mini",
    )

    return analysis


@tracer.observe(name="test_local_queue_integration")
async def main():
    """Main test function that demonstrates the local queue integration."""

    print("🚀 Starting LocalEvaluationQueue integration test...")
    print("=" * 60)

    # Test 1: Process multiple inputs that will create evaluation runs
    inputs = [
        "Hello, how are you today?",
        "What is the weather like?",
        "Can you help me with my homework?",
        "I need assistance with coding.",
    ]

    print(f"📝 Processing {len(inputs)} inputs with local scorers...")

    # Process all inputs concurrently
    tasks = []
    for i, input_text in enumerate(inputs):
        task = process_user_input(f"Input {i + 1}: {input_text}")
        tasks.append(task)

    # Wait for all processing to complete
    results = await asyncio.gather(*tasks)

    print(f"✅ Processed {len(results)} inputs successfully")

    # Test 2: Additional analysis tasks
    print("\n📊 Running text analysis tasks...")

    analysis_tasks = [
        analyze_text("This is a great product with excellent features."),
        analyze_text("The service was disappointing and slow."),
        analyze_text("Average experience, nothing special."),
    ]

    analyses = await asyncio.gather(*analysis_tasks)
    print(f"✅ Completed {len(analyses)} analysis tasks")

    # Give the local evaluation queue some time to process
    print("\n⏳ Waiting for local evaluations to complete...")
    await asyncio.sleep(3.0)

    # Check if the local evaluation queue was created and used
    current_trace = tracer.get_current_trace()
    if hasattr(tracer, "_local_eval_queue"):
        print("✅ LocalEvaluationQueue was created and used!")
        print(f"   Queue size: {tracer._local_eval_queue._queue.qsize()}")

        if hasattr(tracer, "_local_eval_worker"):
            worker_alive = tracer._local_eval_worker.is_alive()
            print(f"   Worker thread alive: {worker_alive}")
    else:
        print("❌ LocalEvaluationQueue was not created")

    # Display evaluation runs that were created
    if current_trace and current_trace.evaluation_runs:
        print(f"\n📋 Created {len(current_trace.evaluation_runs)} evaluation runs:")
        for i, eval_run in enumerate(current_trace.evaluation_runs):
            print(f"   {i + 1}. {eval_run.eval_name}")
            print(f"      Scorers: {[s.score_type for s in eval_run.scorers]}")

    print("\n🎉 Test completed successfully!")
    return results


if __name__ == "__main__":
    print("LocalEvaluationQueue Integration Test")
    print("=" * 40)

    try:
        # Run the test
        results = asyncio.run(main())

        # Allow some extra time for background processing
        print("\n⏳ Allowing extra time for background queue processing...")
        time.sleep(2.0)

        print("\n✨ All tests completed!")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Clean up the tracer and stop any background workers
        if hasattr(tracer, "_local_eval_queue"):
            tracer._local_eval_queue.stop_worker()
            print("🧹 Cleaned up local evaluation queue")
