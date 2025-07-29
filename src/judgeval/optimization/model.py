'''
TrainableModel

This class is used to run vLLM-supported inference on a chosen Unsloth model.
It exposes an OpenAI client interface to do inference.
It also supports model checkpointing to update the model after each training step.
'''

from .types import TrainConfig

# ---------------------------------------------------------------------------
# Unsloth-based TrainableModel
# ---------------------------------------------------------------------------

from unsloth import FastLanguageModel, TextStreamer  # type: ignore

from typing import cast
from peft import PeftModelForCausalLM  # type: ignore
from trl.trainer.grpo_trainer import GRPOTrainer  # type: ignore
from trl.trainer.grpo_config import GRPOConfig  # type: ignore

from datasets import Dataset  # type: ignore


class TrainableModel:
    """Minimal wrapper around an Unsloth `FastLanguageModel` for inference.
    """

    def __init__(
        self,
        model_name: str = "lora_model",
        *,
        max_seq_length: int = 4096,
        dtype: str | None = None,
        load_in_4bit: bool = True,
        fast_inference: bool = True,
        config: dict | None = None,
    ) -> None:
        # ------------------------------------------------------------
        # Load the LoRA-tuned model and tokenizer from disk / hub
        # ------------------------------------------------------------
        if config is None:
            config = {}

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            fast_inference=fast_inference,
        )

        # Convert base model into a PEFT/LoRA model (may create fresh adapter or
        # load one if `model_name` already points at a LoRA directory).
        self.peft_model = cast(
            PeftModelForCausalLM,
            FastLanguageModel.get_peft_model(
                self.model,
                **config.get("peft_args", {}),
            ),
        )

        # Enable native 2× faster inference kernels on the PEFT model
        FastLanguageModel.for_inference(self.peft_model)

        # A simple streamer that prints generated tokens as they arrive
        self._text_streamer = TextStreamer(self.tokenizer)

        # Store configuration
        self._max_seq_length = max_seq_length

        self.trainer = GRPOTrainer(
            model=self.peft_model,  # type: ignore[arg-type]
            reward_funcs=[],
            args=GRPOConfig(**config.get("trainer_args", {})),
            train_dataset=Dataset.from_list([{"prompt": ""} for _ in range(10_000_000)]),
            processing_class=self.tokenizer,
        )

        # ------------------------------------------------------------
        # Real OpenAI-compatible client that talks to local vLLM server spawned by Unsloth model
        # ------------------------------------------------------------

        import httpx
        from openai import AsyncOpenAI, DefaultAsyncHttpxClient

        # The local vLLM service started by your backend exposes an OpenAI REST
        # API; adjust the host/port if you run it elsewhere.
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

    # ------------------------------------------------------------------
    # Training / checkpoint stubs (no-ops in optimisation context)
    # ------------------------------------------------------------------

    async def delete_checkpoints(self):  # noqa: D401
        """Delete old checkpoints (placeholder – implemented elsewhere)."""

    async def train(self, config: TrainConfig):  # noqa: D401
        """Fine-tune the policy using TRL’s GRPOTrainer."""

        # Map our simple TrainConfig fields on to GRPOConfig if provided.
        if config.learning_rate is not None:
            self.trainer.args.learning_rate = config.learning_rate  # type: ignore[attr-defined]
        if config.steps is not None:
            self.trainer.args.max_steps = config.steps  # type: ignore[attr-defined]

        self.trainer.train()

        # Persist adapter + refresh inference engine
        output_dir = self.trainer.args.output_dir  # type: ignore[attr-defined]
        await self._refresh_vllm(output_dir)

    # ------------------------------------------------------------------
    # Post-training LoRA refresh helper
    # ------------------------------------------------------------------

    async def _refresh_vllm(self, adapter_path: str):  # noqa: D401
        """Reload the updated LoRA adapter inside the active vLLM runtime.

        If the Unsloth model was initialised with `fast_inference=True`, then
        `self.model` owns a live `vllm.Engine` instance accessible via
        `self.model.vllm_engine`. We call its `load_lora_weights()` helper
        directly.
        """

        if hasattr(self.model, "vllm_engine"):
            try:
                engine = self.model.vllm_engine  # type: ignore[attr-defined]
                # vLLM’s public API to load LoRA weights differs across
                # versions.  The call below works on ≥0.4.0.  Adjust if your
                # build exposes a different helper.
                engine.load_lora_weights(adapter_path)
                print(f"[TrainableModel] Loaded new LoRA into colocated vLLM from {adapter_path}.")
                return
            except Exception as exc:  # pylint: disable=broad-except
                print("[TrainableModel] Colocated vLLM reload failed:", exc)

    # ------------------------------------------------------------------
    # OpenAI SDK client accessor
    # ------------------------------------------------------------------

    def openai_client(self):  # noqa: D401
        """Return the `openai.AsyncOpenAI` client that targets the local server."""

        return self._openai_client