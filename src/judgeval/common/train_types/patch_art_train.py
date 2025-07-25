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

        inputs = {key: tensor.to(trainer.accelerator.device) for key, tensor in inputs.items()}  # type: ignore

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

        lm_head_t = cast(torch.Tensor, trainer.model.get_output_embeddings().weight.t())
        next_input_ids = _art_train.shift_tensor(inputs["tokens"], 0)
        chunk_size = _config.get("logprob_calculation_chunk_size", 1024)
        assert seq_len % chunk_size == 0, (
            f"Sequence length ({seq_len}) must be evenly divisible by chunk size ({chunk_size})"
        )
        os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
        new_logprobs = _art_train.calculate_logprobs(
            autocast_dtype, trainer, inputs["tokens"], attn_bias, next_input_ids, lm_head_t, chunk_size=chunk_size, reference_logprobs=False
        )
        if config.beta > 0.0:
            ref_logprobs = _art_train.calculate_logprobs(
                autocast_dtype, trainer, inputs["tokens"], attn_bias, next_input_ids, lm_head_t, chunk_size=chunk_size, reference_logprobs=True
            )
        else:
            ref_logprobs = None
        del attn_bias

        old_logprobs = _art_train.shift_tensor(inputs["logprobs"], 0.0)
        advantages = _art_train.shift_tensor(inputs["advantages"], 0.0)
        assistant_mask = _art_train.shift_tensor(inputs["assistant_mask"], False).to(new_logprobs.dtype)
        weights = _art_train.shift_tensor(inputs["weights"], 0.0)
        old_logprobs = torch.where(torch.isnan(old_logprobs), new_logprobs, old_logprobs)
        prob_ratio = torch.exp(new_logprobs - old_logprobs)
        epsilon = _config.get("epsilon", 0.2)
        epsilon_high = _config.get("epsilon_high", epsilon) or epsilon
        policy_loss = -torch.min(prob_ratio * advantages, torch.clip(prob_ratio, 1 - epsilon, 1 + epsilon_high) * advantages)
        kl_div = (torch.exp(ref_logprobs - new_logprobs) - (ref_logprobs - new_logprobs) - 1.0) if ref_logprobs is not None else torch.zeros_like(policy_loss)

        policy_loss *= weights * assistant_mask
        kl_div *= weights * assistant_mask
        # Compute reduction based on loss_type --------------------------------
        loss_type = getattr(trainer, "loss_type", config.get("loss_type", "grpo"))
        if loss_type == "grpo":
            # Per-sequence mean then batch mean (original GRPO)
            mean_policy_loss = (
                (policy_loss * assistant_mask).sum(-1)
                / assistant_mask.sum(-1).clamp(min=1.0)
            ).mean()
        elif loss_type == "bnpo":
            # Flatten over batch & sequence
            mean_policy_loss = (policy_loss * assistant_mask).sum() / assistant_mask.sum().clamp(min=1.0)
        elif loss_type == "dr_grpo":
            max_len = config.get("max_completion_length", seq_len)
            mean_policy_loss = (policy_loss * assistant_mask).sum() / (
                policy_loss.size(0) * max_len
            )
        else:
            # Fallback to previous behaviour (token-weighted mean)
            mean_policy_loss = policy_loss.sum() / (assistant_mask.sum() + 1e-6)

        # KL term --------------------------------------------------------------
        mean_kl = kl_div.sum() / (assistant_mask.sum() + 1e-6)

        # Log metrics
        trainer._metrics["learning_rate"].append(config.learning_rate)
        trainer._metrics["policy_loss"].append(mean_policy_loss.item())
        if config.beta > 0.0:
            trainer._metrics["kl_div"].append(mean_kl.item())

        return mean_policy_loss + config.beta * mean_kl

    return compute_loss


_art_train.get_compute_loss_fn = patched_get_compute_loss_fn 