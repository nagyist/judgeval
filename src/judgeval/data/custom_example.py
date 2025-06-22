from pydantic import BaseModel, Field
from typing import Optional, Union, List, Dict, Any
from uuid import uuid4
from judgeval.data.judgment_types import CustomExampleJudgmentType

class CustomExample(CustomExampleJudgmentType):
    pass