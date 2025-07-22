"""
Tests for dataset operations in the JudgmentClient.
"""

import random
import string

from judgeval.judgment_client import JudgmentClient
from judgeval.data import JudgevalExample
from judgeval.dataset import Dataset


def test_create_dataset(client: JudgmentClient, project_name: str):
    """Test dataset creation and manipulation."""
    dataset = Dataset.create(
        name="test-dataset",
        project_name=project_name,
        examples=[JudgevalExample(input="input 1", actual_output="output 1")],
    )
    assert dataset, "Failed to push dataset"
    dataset.delete()


def test_pull_dataset(client: JudgmentClient, project_name: str):
    """Test pulling statistics for all project datasets."""
    examples = [
        JudgevalExample(input="input 1", actual_output="output 1"),
        JudgevalExample(input="input 2", actual_output="output 2"),
        JudgevalExample(input="input 3", actual_output="output 3"),
        JudgevalExample(input="input 4", actual_output="output 4"),
        JudgevalExample(input="input 5", actual_output="output 5"),
    ]
    random_name1 = "".join(random.choices(string.ascii_letters + string.digits, k=20))
    Dataset.create(name=random_name1, project_name=project_name, examples=examples[:3])

    random_name2 = "".join(random.choices(string.ascii_letters + string.digits, k=20))
    Dataset.create(name=random_name2, project_name=project_name, examples=examples[3:])

    dataset1 = Dataset.get(name=random_name1, project_name=project_name)
    dataset2 = Dataset.get(name=random_name2, project_name=project_name)

    assert dataset1, "Failed to pull dataset"
    assert len(dataset1.examples) == 3, "Dataset should have 3 examples"
    for i, e in enumerate(dataset1.examples, start=1):
        assert e.input == f"input {i}", (
            f"Example should have .input be 'input {i}' but got '{e.input}'"
        )
        assert e.actual_output == f"output {i}", (
            f"Example should have .actual_output be 'output {i}' but got '{e.actual_output}'"
        )

    assert dataset2, "Failed to pull dataset"
    assert len(dataset2.examples) == 2, "Dataset should have 2 examples"
    for i, e in enumerate(dataset2.examples, start=4):
        assert e.input == f"input {i}", (
            f"Example should have .input be 'input {i}' but got '{e.input}'"
        )
        assert e.actual_output == f"output {i}", (
            f"Example should have .actual_output be 'output {i}' but got '{e.actual_output}'"
        )

    dataset1.delete()
    dataset2.delete()


def test_append_dataset(client: JudgmentClient, project_name: str):
    """Test dataset editing."""
    examples = [
        JudgevalExample(input="input 1", actual_output="output 1"),
        JudgevalExample(input="input 2", actual_output="output 2"),
    ]
    Dataset.create(name="test-dataset", project_name=project_name, examples=examples)
    dataset = Dataset.get(name="test-dataset", project_name=project_name)

    initial_example_count = len(dataset.examples)
    examples = [
        JudgevalExample(input="input 3", actual_output="output 3"),
        JudgevalExample(input="input 4", actual_output="output 4"),
        JudgevalExample(input="input 5", actual_output="output 5"),
    ]
    assert initial_example_count == 2, "Dataset should have 2 examples"
    dataset.add_examples(examples)

    dataset = Dataset.get(name="test-dataset", project_name=project_name)
    assert dataset, "Failed to pull dataset"
    assert len(dataset.examples) == initial_example_count + 3, (
        f"Dataset should have {initial_example_count + 3} examples, but has {len(dataset.examples)}"
    )

    dataset.delete()


def test_overwrite_dataset(client: JudgmentClient, project_name: str):
    """Test dataset overwriting."""
    examples = [
        JudgevalExample(input="input 1", actual_output="output 1"),
        JudgevalExample(input="input 2", actual_output="output 2"),
    ]
    Dataset.create(name="test-dataset", project_name=project_name, examples=examples)
    dataset = Dataset.get(name="test-dataset", project_name=project_name)

    new_examples = [
        JudgevalExample(input="input 3", actual_output="output 3"),
        JudgevalExample(input="input 4", actual_output="output 4"),
    ]
    Dataset.create(
        name="test-dataset",
        project_name=project_name,
        examples=new_examples,
        overwrite=True,
    )
    dataset = Dataset.get(name="test-dataset", project_name=project_name)
    assert dataset, "Failed to pull dataset"
    assert len(dataset.examples) == 2, "Dataset should have 2 examples"

    dataset.delete()
