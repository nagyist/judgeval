from .types import Trajectory, TaggedInteraction
from .model import TrainableModel
from datasets import Dataset
from typing import Any
from openai.types.chat.chat_completion import Choice
import torch

'''
Class to prepare inputs for training.
'''
class Processor:
    def __init__(self, model: TrainableModel):
        self.queue: list[dict[str, Any]] = []
        self.trainer = model.trainer
        self.tokenizer = model.tokenizer

    def compute_advantages(self, trajectory_group: list[Trajectory]) -> None:
        avg_reward = sum(trajectory.reward for trajectory in trajectory_group) / len(trajectory_group)
        for trajectory in trajectory_group:
            trajectory.advantage = trajectory.reward - avg_reward

    # Add interactions from a trajectory to the queue
    def add_interactions(self, trajectory: Trajectory) -> None:
        total_tokens = sum(len(interaction.output.logprobs.content) for interaction in trajectory.interactions)
        for interaction in trajectory.interactions:
            output_tokens = len(interaction.output.logprobs.content)
            scale = output_tokens / total_tokens
            tagged_interaction = TaggedInteraction(input=interaction.input, output=interaction.output, weighted_advantage=trajectory.advantage * scale)

            self.queue.append(self.tokenize_interaction(tagged_interaction, total_tokens))

    def extract_token_ids_from_choice(self, choice: Choice) -> list[int]:
        """Extract token IDs from a Choice object's logprobs."""
        if not choice.logprobs or not choice.logprobs.content:
            # Fallback: tokenize the message content
            content = choice.message.content if choice.message.content else ""
            tokens = self.tokenizer(content, add_special_tokens=False)
            return tokens["input_ids"]
        
        token_ids = []
        for token_logprob in choice.logprobs.content:
            # Extract token ID from the token string (e.g., "token_id:27" -> 27)
            token_str = token_logprob.token
            if token_str.startswith("token_id:"):
                try:
                    token_id = int(token_str.split(":")[1])
                    token_ids.append(token_id)
                except (ValueError, IndexError):
                    # Fallback: skip this token
                    continue
            else:
                # If not in token_id format, tokenize the token string
                tokens = self.tokenizer(token_str, add_special_tokens=False)
                token_ids.extend(tokens["input_ids"])
        
        return token_ids

    # Tokenize an interaction
    def tokenize_interaction(self, tagged_interaction: TaggedInteraction, total_tokens: int) -> dict[str, Any]:
        prompt_text = self.tokenizer.apply_chat_template(tagged_interaction.input, tokenize=False, add_generation_prompt=False)
        prompt_tokens = self.tokenizer(prompt_text)
        prompt_ids = prompt_tokens["input_ids"]
        prompt_mask = prompt_tokens["attention_mask"]

        # Extract completion tokens from the Choice object
        completion_ids = self.extract_token_ids_from_choice(tagged_interaction.output)
        completion_mask = [1] * len(completion_ids)

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": tagged_interaction.weighted_advantage,
            "prompt": "dummy",  # Required by GRPO dataset format
            "total_tokens": total_tokens
        }

    # Update training dataset
    def update_dataset(self) -> None:
        self.trainer.train_dataset = Dataset.from_list(self.queue)
        self.queue = []

    # Process a trajectory group
    def process_trajectory_group(self, trajectory_group: list[Trajectory]) -> None:
        self.compute_advantages(trajectory_group)
        for trajectory in trajectory_group:
            self.add_interactions(trajectory)

    # Process a list of trajectory groups
    def process_trajectory_groups(self, trajectory_groups: list[list[Trajectory]]) -> None:
        for trajectory_group in trajectory_groups:
            self.process_trajectory_group(trajectory_group)
        self.trainer.args.max_steps = len(self.queue)
        self.update_dataset()