from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
import pydantic
from typing import Literal

Message = ChatCompletionMessageParam
MessageOrChoice = Message | Choice
Messages = list[Message]
MessagesAndChoices = list[MessageOrChoice]
Tools = list[ChatCompletionToolParam]


class ComparativeConfig(pydantic.BaseModel):
    enabled: bool = False
    rubric: str | None = None
    num_agents: int = 3
    see_spans: bool = False
    comparitive_score_scaling: float = 1.0
    model: str = "gpt-4.1"

class TrainConfig(pydantic.BaseModel):
    learning_rate: float = 5e-6
    beta: float = 0.0
    steps: int = 10
    num_rollouts: int = 10
    max_exceptions: int = 0
    batch_size: int = 0
    comparative_config: ComparativeConfig = ComparativeConfig()


Verbosity = Literal[0, 1, 2]
