'''
TrainableModel

This class is used to run vLLM-supported inference on a chosen Unsloth model.
It exposes an OpenAI client interface to do inference.
It also supports model checkpointing to update the model after each training step.
'''

from .types import TrainConfig

class TrainableModel:
    def __init__(self, model_name: str, model_path: str):
        self.model_name = model_name
        self.model_path = model_path

    def delete_checkpoints(self):
        pass

    def train(self, config: TrainConfig):
        pass