import inspect
from types import FrameType
from typing import Any
import uuid


def new_span_id() -> str:
    """
    Generates a new unique span ID.
    """
    return str(uuid.uuid4())




def extract_inputs_from_entry_frame(frame: FrameType) -> dict[str, Any]:
    """
    Extracts the inputs from the entry frame.
    This is used to capture the inputs to the function being traced.
    """
    args, varargs, varkw, values = inspect.getargvalues(frame)
    inputs = {arg: values[arg] for arg in args if arg in values}
    if varargs:
        inputs[varargs] = values[varargs]
    if varkw:
        inputs[varkw] = values[varkw]
    return inputs
