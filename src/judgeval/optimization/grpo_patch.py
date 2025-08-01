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
        - assistant_mask: mask indicating which completion tokens are assistant tokens
        - advantages: trajectory reward
        """
        device = trainer.accelerator.device
        print(batch)
        
        # Stack tensors for batch processing
        prompt_ids = torch.stack([torch.tensor(item["prompt_ids"]) for item in batch["prompt_ids"]]).to(device)
        prompt_mask = torch.stack([torch.tensor(item["prompt_mask"]) for item in batch["prompt_mask"]]).to(device)
        completion_ids = torch.stack([torch.tensor(item["completion_ids"]) for item in batch["completion_ids"]]).to(device)
        completion_mask = torch.stack([torch.tensor(item["completion_mask"]) for item in batch["completion_mask"]]).to(device)
        advantages = torch.tensor(batch["advantages"]).to(device)
        
        # Process assistant_mask
        assistant_mask = torch.stack([torch.tensor(item["assistant_mask"]) for item in batch["assistant_mask"]]).to(device)
        
        # Return in GRPO expected format with assistant_mask
        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "assistant_mask": assistant_mask,
        }
    
    # Store original _compute_loss method (internal method that compute_loss calls)
    original_compute_loss = trainer._compute_loss
    
    def compute_loss_with_assistant_mask(self, model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        assistant_mask = inputs["assistant_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        # Compute the per_token_logps and the entropy (if necessary) at each position in the completion
        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model, input_ids, attention_mask, logits_to_keep, compute_entropy=self.top_entropy_quantile < 1.0
        )

        if self.top_entropy_quantile < 1.0:
            entropy_mask = get_high_entropy_mask(entropies, assistant_mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        # When using num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps
        # old_per_token_logps == per_token_logps, so we can skip it's computation
        # (see _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = (
            per_token_logps.detach() if inputs["old_per_token_logps"] is None else inputs["old_per_token_logps"]
        )
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        # Two-sided clipping
        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type == "grpo":
            loss = ((per_token_loss * assistant_mask).sum(-1) / assistant_mask.sum(-1).clamp(min=1.0)).mean()
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * assistant_mask).sum() / assistant_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * assistant_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Log the metrics
        mode = "train" if self.model.training else "eval"

        if self.beta != 0.0:
            mean_kl = (per_token_kl * assistant_mask).sum() / assistant_mask.sum()
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        # Compute the clipped probability ratios
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = (is_low_clipped * assistant_mask).sum() / assistant_mask.sum()
        high_clip = (is_high_clipped * assistant_mask).sum() / assistant_mask.sum()
        clip_ratio = (is_region_clipped * assistant_mask).sum() / assistant_mask.sum()

        gathered_low_clip = self.accelerator.gather(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        return loss
    
    # Apply patches
    trainer._prepare_inputs = prepare_trajectory_inputs
    trainer._compute_loss = compute_loss_with_assistant_mask
    
    # Return function to restore original methods
    def restore_original_methods():
        trainer._prepare_inputs = original_prepare_inputs
        trainer._compute_loss = original_compute_loss
    
    return restore_original_methods


def tokenize_trajectory(trajectory: Trajectory, tokenizer) -> Dict[str, Any]:
    """
    Tokenize a trajectory with proper assistant masking for multi-turn conversations.
    
    Key idea: Split at last assistant message, then create mask to identify
    which tokens in the completion are actually from the assistant (not tool responses).
    """
    # Convert messages to standard format
    messages = []
    for msg_or_choice in trajectory.messages_and_choices:
        if isinstance(msg_or_choice, Choice):
            messages.append({
                "role": "assistant",
                "content": msg_or_choice.message.content or ""
            })
        elif isinstance(msg_or_choice, dict):
            messages.append(msg_or_choice)
    
    # Find the first assistant message
    first_assistant_idx = -1
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant":
            first_assistant_idx = i
            break
    
    if first_assistant_idx == -1:
        return None
    
    # Split: everything up to first assistant = prompt
    prompt_messages = messages[:first_assistant_idx]
    # Everything from first assistant onwards = completion (may include tool responses after)
    completion_messages = messages[first_assistant_idx:]
    
    # Tokenize prompt
    if prompt_messages:
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, 
            tokenize=False,
            add_generation_prompt=True
        )
        prompt_tokens = tokenizer(prompt_text, truncation=True, max_length=3584)
        prompt_ids = prompt_tokens["input_ids"]
        prompt_mask = prompt_tokens["attention_mask"]
    else:
        prompt_ids = []
        prompt_mask = []
    
    # Tokenize completion messages and create assistant mask
    completion_text = tokenizer.apply_chat_template(
        completion_messages,
        tokenize=False,
        add_generation_prompt=False
    )
    completion_tokens = tokenizer(completion_text, truncation=True, max_length=512)
    completion_ids = completion_tokens["input_ids"]
    completion_mask = completion_tokens["attention_mask"]
    
    # Create assistant mask - marks which tokens in completion are assistant tokens
    # Start with all zeros
    assistant_mask = [0] * len(completion_ids)
    
    # Mark tokens from assistant messages as 1
    current_pos = 0
    for msg in completion_messages:
        # Tokenize this message alone to find its length
        msg_text = tokenizer.apply_chat_template([msg], tokenize=False, add_generation_prompt=False)
        msg_ids = tokenizer(msg_text, add_special_tokens=False)["input_ids"]
        
        if msg.get("role") == "assistant":
            # Mark these tokens as assistant tokens
            for i in range(current_pos, min(current_pos + len(msg_ids), len(assistant_mask))):
                assistant_mask[i] = 1
        
        current_pos += len(msg_ids)
    
    return {
        "prompt_ids": prompt_ids,
        "prompt_mask": prompt_mask,
        "completion_ids": completion_ids,
        "completion_mask": completion_mask,
        "assistant_mask": assistant_mask,  # Which completion tokens to train on
        "advantages": trajectory.reward,
        "prompt": "dummy",  # Required by GRPO dataset format
    }