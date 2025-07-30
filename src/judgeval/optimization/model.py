'''
TrainableModel

This class is used to run vLLM-supported inference on a chosen Unsloth model.
It exposes an OpenAI client interface to do inference.
It also supports model checkpointing to update the model after each training step.
'''

from .types import TrainConfig

from unsloth import FastLanguageModel 

from typing import cast
from peft import PeftModelForCausalLM
from trl.trainer.grpo_trainer import GRPOTrainer
from trl.trainer.grpo_config import GRPOConfig

from datasets import Dataset

import httpx
from openai import AsyncOpenAI, DefaultAsyncHttpxClient


class TrainableModel:
    """Minimal wrapper around an Unsloth `FastLanguageModel` for inference.
    """

    def __init__(
        self,
        name: str,
        model_name: str,
        *,
        max_seq_length: int = 4096,
        dtype: str | None = None,
        load_in_4bit: bool = True,
        fast_inference: bool = True,
        config: dict | None = None,
    ) -> None:
        if config is None:
            config = {}

        self.name = name
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            fast_inference=fast_inference,
        )

        self.peft_model = cast(
            PeftModelForCausalLM,
            FastLanguageModel.get_peft_model(
                self.model,
                **config.get("peft_args", {}),
            ),
        )

        self.trainer = GRPOTrainer(
            model=self.peft_model,
            reward_funcs=[],
            args=GRPOConfig(**config.get("trainer_args", {})),
            train_dataset=Dataset.from_list([{"prompt": ""} for _ in range(10_000_000)]),
            processing_class=self.tokenizer,
        )

        self.step = 0

    async def delete_checkpoints(self):
        """Delete old checkpoints - no-op for now"""

    async def train(self, config: TrainConfig):
        """Fine-tune the policy using TRL’s GRPOTrainer."""

        # Map our simple TrainConfig fields on to GRPOConfig if provided.
        if config.learning_rate is not None:
            self.trainer.args.learning_rate = config.learning_rate      
        if config.steps is not None:
            self.trainer.args.max_steps = config.steps

        # TODO: Patch prepare_inputs and compute_loss
        # Train the model using TRL's GRPOTrainer
        self.trainer.train()

        # Update model with new LoRA weights for on-policy inference
        model_dir = f"./lora/{self.name}/{self.step}"
        self.model.save_pretrained(model_dir)
        await self._refresh_vllm(model_dir)
        self.step += 1

    async def _refresh_vllm(self, adapter_path: str):
        """Reload the updated LoRA adapter inside the active vLLM runtime.

        If the Unsloth model was initialised with `fast_inference=True`, then
        `self.model` owns a live `vllm.Engine` instance accessible via
        `self.model.vllm_engine`. We call its `load_lora_weights()` helper
        directly.
        """

        if hasattr(self.model, "vllm_engine"):
            try:
                engine = self.model.vllm_engine
                # vLLM’s public API to load LoRA weights differs across
                # versions.  The call below works on ≥0.4.0.  Adjust if your
                # build exposes a different helper.
                engine.load_lora_weights(adapter_path)
                print(f"[TrainableModel] Loaded new LoRA into colocated vLLM from {adapter_path}.")
                return
            except Exception as exc:  # pylint: disable=broad-except
                print("[TrainableModel] Colocated vLLM reload failed:", exc)

    def openai_client(self):
        """Return the `openai.AsyncOpenAI` client that targets the local server."""

        # The local vLLM service started by Unsloth exposes an OpenAI interface.
        self.inference_base_url = "http://localhost:8000/v1"
        self.inference_api_key = "default"

        self._openai_client = AsyncOpenAI(
            base_url=self.inference_base_url,
            api_key=self.inference_api_key,
            http_client=DefaultAsyncHttpxClient(
                timeout=httpx.Timeout(timeout=1200, connect=5.0),
                limits=httpx.Limits(
                    max_connections=10_000,
                    max_keepalive_connections=10_000,
                ),
            ),
        )
        return self._openai_client


    def get_step(self):
        return self.step