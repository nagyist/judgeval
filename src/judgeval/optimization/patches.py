import torch
from typing import Any

def patch_trainer(model):
    # Patch for the prepare_inputs method
    def _prepare_inputs_patch(batch: dict[str, Any]) -> dict[str, Any]:
        """
        Convert our pre-tokenized trajectory data to GRPO format.
        
        The batch contains items from our dataset where each item has:
        - prompt_ids, prompt_mask: tokenized context (all messages except last assistant)
        - completion_ids, completion_mask: tokenized last assistant message to train on
        - assistant_mask: mask indicating which completion tokens are assistant tokens
        - advantages: trajectory reward
        """
        device = model.trainer.accelerator.device
        
        # print("BATCH ", batch)

        # Stack tensors for batch processing
        from torch.nn.utils.rnn import pad_sequence
        pad_id = getattr(model.tokenizer, "pad_token_id", 0) or 0  # default to 0 if tokenizer has no pad token

        # Helper to convert list of lists to padded tensor
        def _pad(list_of_ids, pad_value):
            return pad_sequence(
                [torch.tensor(ids, dtype=torch.long) for ids in list_of_ids],
                batch_first=True,
                padding_value=pad_value,
            )

        prompt_ids = _pad([item["prompt_ids"] for item in batch], pad_id).to(device)
        prompt_mask = _pad([item["prompt_mask"] for item in batch], 0).to(device)
        completion_ids = _pad([item["completion_ids"] for item in batch], pad_id).to(device)
        completion_mask = _pad([item["completion_mask"] for item in batch], 0).to(device)

        # Advantages are scalars so we can just stack normally
        advantages = torch.tensor([item["advantages"] for item in batch], dtype=torch.float32).to(device)

        # Total tokens is the total tokens in the entire trajectory
        total_tokens = torch.tensor([item["total_tokens"] for item in batch], dtype=torch.int32).to(device)
        
        # Return in GRPO expected format
        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "total_tokens": total_tokens
        }

    def _compute_loss_patch(self, model, inputs):
            # Compute the per-token log probabilities for the model
            prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
            completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
            input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
            logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

            # Compute the per_token_logps and the entropy (if necessary) at each position in the completion
            per_token_logps, entropies = self._get_per_token_logps_and_entropies(
                model, input_ids, attention_mask, logits_to_keep, compute_entropy=self.top_entropy_quantile < 1.0
            )

            if self.top_entropy_quantile < 1.0:
                entropy_mask = get_high_entropy_mask(entropies, completion_mask, 1 - self.top_entropy_quantile)
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
                loss = ((per_token_loss * completion_mask).sum(-1) / inputs["total_tokens"].clamp(min=1.0)).mean()
            elif self.loss_type == "bnpo":
                loss = (per_token_loss * completion_mask).sum() / inputs["total_tokens"].clamp(min=1.0)
            elif self.loss_type == "dr_grpo":
                loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")

            # Log the metrics
            mode = "train" if self.model.training else "eval"

            if self.beta != 0.0:
                mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
                self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

            # Compute the clipped probability ratios
            is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
            is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
            is_region_clipped = is_low_clipped | is_high_clipped

            low_clip = (is_low_clipped * completion_mask).sum() / completion_mask.sum()
            high_clip = (is_high_clipped * completion_mask).sum() / completion_mask.sum()
            clip_ratio = (is_region_clipped * completion_mask).sum() / completion_mask.sum()

            gathered_low_clip = self.accelerator.gather(low_clip)
            self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
            self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
            gathered_high_clip = self.accelerator.gather(high_clip)
            self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
            self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
            gathered_clip_ratio = self.accelerator.gather(clip_ratio)
            self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
            return loss

    model.trainer._prepare_inputs = _prepare_inputs_patch
    model.trainer._compute_loss = _compute_loss_patch