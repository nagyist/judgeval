'''
TrainableModel

This class is used to run vLLM-supported inference on a chosen Unsloth model.
It exposes an OpenAI client interface to do inference.
It also supports model checkpointing to update the model after each training step.
'''
import unsloth
import unsloth_zoo
from .types import TrainConfig, Trajectory

from typing import cast, Any
from peft import PeftModelForCausalLM
from trl.trainer.grpo_trainer import GRPOTrainer
from unsloth_compiled_cache.UnslothGRPOTrainer import UnslothGRPOConfig

from datasets import Dataset

import httpx
from openai import AsyncOpenAI, DefaultAsyncHttpxClient
from .vllm_server import launch_openai_server
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from dataclasses import replace
from .vllm_server import get_openai_server_config
import os
from .patches import patch_trainer
import shutil

class TrainableModel:
    """Minimal wrapper around an Unsloth `FastLanguageModel` for inference.
    """

    def __init__(
        self,
        name: str,
        model_name: str,
        *,
        max_seq_length: int = 2048,
        dtype: str | None = None,
        load_in_4bit: bool = False,
        fast_inference: bool = True,
        lora_rank: int = 32,
        config: dict | None = None,
    ) -> None:

        self.name = name
        self.model_name = model_name

        if config is None:
            config = get_openai_server_config(
                model_name = self.name,
                base_model = model_name,
                log_file = "vllm.log"
            )
        self.config = config

        os.environ["VLLM_USE_V1"] = "0"

        from_engine_args = AsyncLLMEngine.from_engine_args

        # NOTE: We also have to patch from_engine_args to control the engine args
        # that are passed to the engine constructor.
        def _from_engine_args(
            engine_args: AsyncEngineArgs, *args: Any, **kwargs: Any
        ) -> AsyncLLMEngine:
            return from_engine_args(
                replace(engine_args, **config.get("engine_args", {})), *args, **kwargs
            )

        AsyncLLMEngine.from_engine_args = _from_engine_args
        self.model, self.tokenizer = unsloth.FastLanguageModel.from_pretrained(
            model_name = model_name,
            temperature = 2.0,
            max_seq_length=32768,
            load_in_4bit=load_in_4bit,  # False for LoRA 16bit
            fast_inference=fast_inference,  # Enable vLLM fast inference
            gpu_memory_utilization=0.79,
            max_lora_rank=lora_rank,
            use_async=True
        )
        AsyncLLMEngine.from_engine_args = from_engine_args
        #unsloth.FastLanguageModel.for_inference(self.model)
        

        self.patch_get_lora_tokenizer_async()

        self.peft_model = cast(
            PeftModelForCausalLM,
            unsloth.FastLanguageModel.get_peft_model(
                self.model,
                **config.get("peft_args", {}),
            ),
        )

        self.trainer = GRPOTrainer(
            model=self.peft_model,
            reward_funcs=[],
            args=UnslothGRPOConfig(
                temperature = 1.0,
                learning_rate = 1e-4,
                weight_decay = 0.01,
                warmup_ratio = 0.1,
                lr_scheduler_type = "linear",
                optim = "adamw_8bit",
                logging_steps = 1,
                per_device_train_batch_size = 1,
                gradient_accumulation_steps = 1, # Increase to 4 for smoother training
                num_generations = 2, # Decrease if out of memory
                max_prompt_length = 3584,
                max_completion_length = 3584,
                num_train_epochs = 1, # Set to 1 for a full training run
                max_steps = 100, # Primary control - reduces from 100 to 30 for fewer training steps
                save_steps = 100,
            ),
            train_dataset=Dataset.from_list([{"prompt": ""} for _ in range(10_000_000)]),
            processing_class=self.tokenizer,
        )
        patch_trainer(self)

        self.step = 0

    async def delete_checkpoints(self):
        """Delete all checkpoints except the most recent checkpoint"""

    async def train(self, trajectory_groups: list[list[Trajectory]], config: TrainConfig):
        """Fine-tune the policy using TRL's GRPOTrainer with multi-turn support."""
        # First, switch the model to training mode
        #unsloth.FastLanguageModel.for_training(self.model) 

        # TODO: Handle this better
        # Map our simple TrainConfig fields on to GRPOConfig if provided.
        if config.learning_rate is not None:
            self.trainer.args.learning_rate = config.learning_rate
        self.trainer.args.beta = config.beta

        # Apply patches to handle multi-turn trajectories
        from .processor import Processor
        self.processor = Processor(self)
        self.processor.process_trajectory_groups(trajectory_groups)
        
        # Train the model using patched GRPO trainer
        self.trainer.train()

        # Update model with new LoRA weights for on-policy inference
        model_dir = f"./lora/{self.name}/{self.step}"
        self.trainer.save_model(model_dir)
        self._refresh_vllm(model_dir)

        # Switch the model back to inference mode
        #unsloth.FastLanguageModel.for_inference(self.model)

        self.step += 1

    def _refresh_vllm(self, lora_path: str):
        """Reload the updated LoRA adapter inside the active vLLM runtime.

        If the Unsloth model was initialised with `fast_inference=True`, then
        `self.model` owns a live `vllm.Engine` instance accessible via
        `self.model.vllm_engine`. We call its `load_lora_weights()` helper
        directly.
        """

        if hasattr(self.model, "vllm_engine"):
            try:
                engine = self.model.vllm_engine.engine
                lora_request: "LoRARequest" = self.peft_model.load_lora(
                    lora_path,
                    load_tensors=True,
                )
                lora_request.lora_int_id = 1
                lora_request.lora_name = self.model_name
                lora_request.lora_path = lora_path
                engine.remove_lora(1)
                engine.add_lora(lora_request)
                print(f"[TrainableModel] Loaded new LoRA into colocated vLLM from {lora_path}.")
                return
            except Exception as exc:
                print("[TrainableModel] Colocated vLLM reload failed:", exc)

    async def openai_client(self):
        """Return the `openai.OpenAI` client that targets the local server."""

        host, port = await launch_openai_server(
            self.model.vllm_engine,
            config=self.config
        )

        # The local vLLM service started by Unsloth exposes an OpenAI interface.
        self.inference_base_url = f"http://0.0.0.0:8000/v1"
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

    def patch_get_lora_tokenizer_async(self) -> None:
        """
        Patches an Unsloth patch that causes issues with vLLM.

        Specifically, Unsloth patches get_lora_tokenizer_async with a non-async function, which causes issues.
        """
        import vllm.transformers_utils.tokenizer
        import vllm.transformers_utils.tokenizer_group

        async def _return_nothing(*_, **__) -> None:
            return None

        async def get_self_lora_tokenizer_async(self, *args, **kwargs):
            return self.tokenizer

        vllm.transformers_utils.tokenizer.get_lora_tokenizer_async = _return_nothing  # type: ignore
        vllm.transformers_utils.tokenizer_group.get_lora_tokenizer_async = (  # type: ignore
            _return_nothing
        )
        vllm.transformers_utils.tokenizer_group.TokenizerGroup.get_lora_tokenizer_async = (
            get_self_lora_tokenizer_async  # type: ignore
        )