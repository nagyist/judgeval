"""
Monkey-patch for art.local.train.get_compute_loss_fn

This file is identical to the previous `patch_art_train.py` but is now
located inside `judgeval.common.train_types` so that it can be imported via
`from judgeval.common.train_types import patch_art_train` or simply
`import judgeval.common.train_types.patch_art_train` early in your program.

Import this module BEFORE any code that triggers training.
"""

from __future__ import annotations

import os
import asyncio
from contextlib import nullcontext
from typing import Callable, TYPE_CHECKING, cast

import torch
from art.local import train as _art_train
from peft.peft_model import PeftModel  # type: ignore
from trl import GRPOTrainer  # type: ignore

_original_get_compute_loss_fn = _art_train.get_compute_loss_fn

if TYPE_CHECKING:
    from art.local.service import TrainInputs  # noqa: F401
    from judgeval.common.train_types import TrainConfig  # type: ignore  # noqa: F401
    from judgeval.common.train_types import dev as _dev


def patched_get_compute_loss_fn(trainer: "GRPOTrainer") -> Callable[..., torch.Tensor]:
    def compute_loss(
        model: "PeftModel",
        inputs: "TrainInputs",
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:
        # Use judgeval's train_types versions for stronger type hints
        from judgeval.common.train_types import dev as dev
        from judgeval.common.train_types import TrainConfig as _TrainConfig

        config: _TrainConfig = inputs.pop("config")  # type: ignore
        _config: dev.TrainConfig = inputs.pop("_config")  # type: ignore

        if optimizer := trainer.optimizer:
            optimizer = getattr(optimizer, "optimizer", optimizer)
            if param_groups := getattr(optimizer, "param_groups"):
                for param_group in param_groups:
                    param_group["lr"] = config.learning_rate
                    # param_group["betas"] = config.betas
                    # if param_group.get("weight_decay"):
                    #     param_group["weight_decay"] = config.weight_decay

        # Move tensors to the correct device
        inputs = {key: tensor.to(trainer.accelerator.device) for key, tensor in inputs.items()}  # type: ignore

        # Unsloth code
        autocast_dtype = (
            torch.float16
            if os.environ.get("ACCELERATE_MIXED_PRECISION", "fp16") == "fp16"
            else torch.bfloat16
        )
        if os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "1":
            autocast_dtype = torch.float16

        batch_size, seq_len = inputs["tokens"].size()
        attn_bias = _art_train.calculate_attn_bias(
            batch_size,
            seq_len,
            trainer.accelerator.device,
            inputs["group_ids"],
            inputs["parent_ids"],
            autocast_dtype,
        )

        # Calculate log probabilities
        lm_head_t = cast(
            torch.Tensor, trainer.model.get_output_embeddings().weight.t()  # type: ignore
        )  # Shape [H, V]
        next_input_ids = _art_train.shift_tensor(inputs["tokens"], 0)
        chunk_size = _config.get("logprob_calculation_chunk_size", 1024)
        # Assert that sequence length is evenly divisible by the chunk size
        assert (
            seq_len % chunk_size == 0
        ), f"Sequence length ({seq_len}) must be evenly divisible by chunk size ({chunk_size})"
        os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
        new_logprobs = _art_train.calculate_logprobs(
            autocast_dtype,
            trainer,
            inputs["tokens"],
            attn_bias,
            next_input_ids,
            lm_head_t,
            chunk_size=chunk_size,
            reference_logprobs=False,
        )
        if config.beta > 0.0:
            ref_logprobs = _art_train.calculate_logprobs(
                autocast_dtype,
                trainer,
                inputs["tokens"],
                attn_bias,
                next_input_ids,
                lm_head_t,
                chunk_size=chunk_size,
                reference_logprobs=True,
            )
        else:
            ref_logprobs = None
        del attn_bias

        # Shift inputs for loss calculation
        old_logprobs = _art_train.shift_tensor(inputs["logprobs"], 0.0)
        advantages = _art_train.shift_tensor(inputs["advantages"], 0.0)
        assistant_mask = _art_train.shift_tensor(inputs["assistant_mask"], False).to(
            new_logprobs.dtype
        )
        weights = _art_train.shift_tensor(inputs["weights"], 0.0)
        # Assume missing old logprobs were sampled under the current policy
        old_logprobs = torch.where(
            torch.isnan(old_logprobs),
            new_logprobs,
            old_logprobs,
        )
        prob_ratio = torch.exp(new_logprobs - old_logprobs)
        print(new_logprobs)
        print(old_logprobs)
        # Count number of differences between new_logprobs and old_logprobs
        diff_count = (new_logprobs != old_logprobs).sum()
        print(f"Number of differences: {diff_count}")
        # Compute percentage of differences
        diff_percentage = diff_count / (new_logprobs == new_logprobs).sum()
        print(f"Percentage of differences: {diff_percentage}")
        # Print the shape of new_logprobs and old_logprobs
        print(f"Shape of new_logprobs: {new_logprobs.shape}")
        epsilon = _config.get("epsilon", 0.2)
        epsilon_high = _config.get("epsilon_high", epsilon)
        if epsilon_high is None:
            epsilon_high = epsilon
        policy_loss = -torch.min(
            prob_ratio * advantages,
            torch.clip(prob_ratio, 1 - epsilon, 1 + epsilon_high) * advantages,
        )
        if ref_logprobs is not None:
            kl_div = (
                torch.exp(ref_logprobs - new_logprobs)
                - (ref_logprobs - new_logprobs)
                - 1.0
            )
        else:
            kl_div = torch.zeros_like(policy_loss)

        policy_loss = policy_loss * weights * assistant_mask
        kl_div = kl_div * weights * assistant_mask
        mean_policy_loss = policy_loss.sum() / (assistant_mask.sum() + 1e-6)
        mean_kl = kl_div.sum() / (assistant_mask.sum() + 1e-6)

        trainer._metrics["learning_rate"].append(config.learning_rate)
        trainer._metrics["policy_loss"].append(mean_policy_loss.item())
        if config.beta > 0.0:
            trainer._metrics["kl_div"].append(mean_kl.item())
        return mean_policy_loss + config.beta * mean_kl

    return compute_loss


_art_train.get_compute_loss_fn = patched_get_compute_loss_fn 