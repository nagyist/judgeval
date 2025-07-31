"""
Minimal GRPO patch for multi-turn RL training using judgeval's trajectory system.

This patches the GRPO trainer's prepare_inputs and compute_loss methods to handle
multi-turn trajectories, only training on assistant outputs.
"""

import torch
from typing import List, Dict, Any, Iterator
from datasets import Dataset
from openai.types.chat.chat_completion import Choice
from judgeval.optimization.types import Trajectory, TrainConfig


def apply_multi_turn_patches(model, trajectory_groups: List[List[Trajectory]], config: TrainConfig):
    """
    Applies patches to GRPO trainer to handle multi-turn trajectories.
    
    This function:
    1. Converts trajectories to GRPO-compatible dataset format
    2. Replaces trainer._prepare_inputs to handle our trajectory format
    3. Keeps trainer.compute_loss as-is (it works with our prepared inputs)
    
    Args:
        model: The TrainableModel instance
        trajectory_groups: Groups of trajectories from judgeval
        config: Training configuration
        
    Returns:
        A function to restore original trainer methods
    """
    trainer = model.trainer
    tokenizer = model.tokenizer
    
    # Process all trajectories into dataset items
    dataset_items = []
    
    for group in trajectory_groups:
        trajectories = group
        
        if config.comparative_reward:
            # Sort by reward for comparative evaluation
            trajectories = sorted(trajectories, key=lambda t: t.reward, reverse=True)
            
            # Normalize rewards within group
            if len(trajectories) > 1:
                max_reward = trajectories[0].reward
                min_reward = trajectories[-1].reward
                reward_range = max_reward - min_reward
                
                if reward_range > 0:
                    for traj in trajectories:
                        normalized_reward = (traj.reward - min_reward) / reward_range
                        traj.reward = normalized_reward
        
        for trajectory in trajectories:
            # Convert trajectory to tokenized format
            tokenized = tokenize_trajectory(trajectory, tokenizer)
            if tokenized:
                dataset_items.append(tokenized)
    
    # Create dataset from tokenized items
    trainer.train_dataset = Dataset.from_list(dataset_items)
    
    # Store original _prepare_inputs method
    original_prepare_inputs = trainer._prepare_inputs
    
    def prepare_trajectory_inputs(batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert our pre-tokenized trajectory data to GRPO format.
        
        The batch contains items from our dataset where each item has:
        - prompt_ids, prompt_mask: tokenized context (all messages except last assistant)
        - completion_ids, completion_mask: tokenized last assistant message to train on
        - advantages: trajectory reward
        """
        device = trainer.accelerator.device
        
        # Stack tensors for batch processing
        prompt_ids = torch.stack([torch.tensor(item["prompt_ids"]) for item in batch["prompt_ids"]]).to(device)
        prompt_mask = torch.stack([torch.tensor(item["prompt_mask"]) for item in batch["prompt_mask"]]).to(device)
        completion_ids = torch.stack([torch.tensor(item["completion_ids"]) for item in batch["completion_ids"]]).to(device)
        completion_mask = torch.stack([torch.tensor(item["completion_mask"]) for item in batch["completion_mask"]]).to(device)
        advantages = torch.tensor(batch["advantages"]).to(device)
        
        # Return in GRPO expected format
        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
        }
    
    # Apply patch - only need to patch _prepare_inputs
    # compute_loss works as-is with our prepared inputs
    trainer._prepare_inputs = prepare_trajectory_inputs
    
    # Return function to restore original method
    def restore_original_methods():
        trainer._prepare_inputs = original_prepare_inputs
    
    return restore_original_methods


def tokenize_trajectory(trajectory: Trajectory, tokenizer) -> Dict[str, Any]:
    """
    Tokenize a trajectory into GRPO-compatible format.
    
    Only assistant messages are marked for training.
    Tool/user responses are included for context but not trained.
    """
    messages = []
    assistant_indices = []
    
    # Convert messages_and_choices to standard message format
    for i, msg_or_choice in enumerate(trajectory.messages_and_choices):
        if isinstance(msg_or_choice, Choice):
            # Choice object from OpenAI
            messages.append({
                "role": "assistant",
                "content": msg_or_choice.message.content or ""
            })
            assistant_indices.append(len(messages) - 1)
        elif isinstance(msg_or_choice, dict):
            messages.append(msg_or_choice)
            if msg_or_choice.get("role") == "assistant":
                assistant_indices.append(len(messages) - 1)
    
    # No assistant messages to train on
    if not assistant_indices:
        return None
    
    # Find the last assistant message
    last_assistant_idx = assistant_indices[-1]
    
    # Split into prompt (everything before last assistant) and completion (last assistant)
    prompt_messages = messages[:last_assistant_idx]
    completion_text = messages[last_assistant_idx]["content"]
    
    # Tokenize prompt using chat template
    if prompt_messages:
        prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        prompt_encoding = tokenizer(prompt_text, truncation=True, max_length=2048)
        prompt_ids = prompt_encoding["input_ids"]
        prompt_mask = prompt_encoding["attention_mask"]
    else:
        # Empty prompt
        prompt_ids = [tokenizer.bos_token_id] if tokenizer.bos_token_id else []
        prompt_mask = [1] * len(prompt_ids)
    
    # Tokenize completion (without special tokens as it continues from prompt)
    completion_encoding = tokenizer(completion_text, add_special_tokens=False, truncation=True, max_length=512)
    completion_ids = completion_encoding["input_ids"]
    completion_mask = completion_encoding["attention_mask"]
    
    # Add EOS token to completion if not already there
    if tokenizer.eos_token_id and (not completion_ids or completion_ids[-1] != tokenizer.eos_token_id):
        completion_ids.append(tokenizer.eos_token_id)
        completion_mask.append(1)
    
    return {
        "prompt_ids": prompt_ids,
        "prompt_mask": prompt_mask,
        "completion_ids": completion_ids,
        "completion_mask": completion_mask,
        "advantages": trajectory.reward,
        "prompt": "dummy",  # Required by GRPO dataset format
    }