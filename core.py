from __future__ import annotations
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
import functools
import inspect
import logging
import os
import site
import sys
import sysconfig
import time
from types import CodeType, FrameType, TracebackType
from typing import Any, Callable, TypeAlias, Union, cast
import uuid

from utils import new_span_id

ExcInfo: TypeAlias = tuple[type[BaseException], BaseException, TracebackType]
OptExcInfo: TypeAlias = ExcInfo | tuple[None, None, None]

# BEGIN: Core implementation
# TODO: MOVE TO CLASS METHODS AND PROPERTIES
# TYPES


@dataclass(slots=True, frozen=True)
class CurrentSpanEntry:
    """
    Represents the current span entry in the context.
    This exists in a partial state before the span is fully created.
    """

    name: str
    span_id: str
    span_type: str
    parent_span_id: str | None
    start_time: float
    end_time: float | None = None
    inputs: dict[str, Any] | None = None
    output: Any = None
    error: Union[OptExcInfo, None] = None


@dataclass(slots=True, frozen=True)
class CurrentSpanExit:
    """
    Represents the current span in the context.
    This is the fully created span that can be used for tracing.
    """

    name: str
    span_id: str
    span_type: str
    parent_span_id: str | None
    start_time: float
    end_time: float
    inputs: dict[str, Any] | None = None
    output: Any = None
    error: Union[OptExcInfo, None] = None


CurrentSpanType = CurrentSpanEntry | CurrentSpanExit


# UTILS
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


# IMPLEMENTATION
current_span_var = ContextVar[Union[CurrentSpanType, None]](
    "current_span", default=None
)
span_reset_tokens_var = ContextVar[Union[dict[str, Token[Any]], None]](
    "span_reset_tokens", default=None
)

surpress_deep_tracing = ContextVar[bool]("surpress_deep_tracing", default=False)
trace_spans: list[CurrentSpanExit] = []


def _commit_current_span():
    """
    Commits the current span to the context.
    This is used to finalize the current span entry and convert it to a full span exit.
    """
    current_span = current_span_var.get()
    if not isinstance(current_span, CurrentSpanExit):
        logging.warning("Commit called without a completed current span entry.")
        return

    span_id = current_span.span_id
    trace_spans.append(current_span)

    span_reset_tokens = span_reset_tokens_var.get()
    if span_reset_tokens is None or span_id not in span_reset_tokens:
        logging.warning(
            f"Exit observed without a corresponding entry span for span_id: {span_id}"
        )
        return
    current_span_var.reset(span_reset_tokens[span_id])
    del span_reset_tokens[span_id]
    span_reset_tokens_var.set(span_reset_tokens)

    logging.info(
        f"Span committed: {current_span.name} (id: {span_id}, parent_id: {current_span.parent_span_id}, "
        f"start_time: {current_span.start_time}, end_time: {current_span.end_time}, output: {current_span.output})"
    )


def _observe_enter(frame: FrameType):
    """
    Handles the entry of a new span.
    """

    current_span = current_span_var.get()

    name = frame.f_code.co_name
    span_id = new_span_id()
    parent_span_id = (
        current_span.span_id if isinstance(current_span, CurrentSpanEntry) else None
    )
    start_time = time.time()
    inputs = extract_inputs_from_entry_frame(frame)

    token = current_span_var.set(
        CurrentSpanEntry(
            name=name,
            span_id=span_id,
            span_type="function",
            parent_span_id=parent_span_id,
            start_time=start_time,
            inputs=inputs,
        )
    )

    span_reset_tokens = span_reset_tokens_var.get()
    if span_reset_tokens is None:
        span_reset_tokens = {}
    span_reset_tokens[span_id] = token
    span_reset_tokens_var.set(span_reset_tokens)

    logging.info(
        f"Span entered: {name} (id: {span_id}, parent_id: {parent_span_id}, start_time: {start_time})"
    )


def _observe_exit(frame: FrameType, arg: Any):
    """
    Handles the exit of a span.
    """
    current_span = current_span_var.get()
    if not isinstance(current_span, CurrentSpanEntry):
        logging.warning("Exit observed without a corresponding entry span.")
        return

    span_id = current_span.span_id
    end_time = time.time()
    output = arg

    current_span_var.set(
        CurrentSpanExit(
            name=current_span.name,
            span_id=span_id,
            span_type=current_span.span_type,
            parent_span_id=current_span.parent_span_id,
            start_time=current_span.start_time,
            end_time=end_time,
            inputs=current_span.inputs,
            output=output,
        )
    )

    logging.info(
        f"Span exited: {current_span.name} (id: {span_id}, parent_id: {current_span.parent_span_id}, "
        f"start_time: {current_span.start_time}, end_time: {end_time}, output: {output})"
    )
    _commit_current_span()


def _observe_error(frame: FrameType, exc_info: OptExcInfo):
    """
    Handles errors that occur during span execution.
    """
    current_span = current_span_var.get()
    if not isinstance(current_span, CurrentSpanEntry):
        logging.warning("Error observed without a corresponding entry span.")
        return

    current_span_var.set(
        CurrentSpanEntry(
            name=current_span.name,
            span_id=current_span.span_id,
            span_type=current_span.span_type,
            parent_span_id=current_span.parent_span_id,
            start_time=current_span.start_time,
            end_time=current_span.end_time,
            inputs=current_span.inputs,
            output=current_span.output,
            error=exc_info,
        )
    )


@contextmanager
def span():
    try:
        _observe_enter(inspect.currentframe())
        # TODO: yield an object that the user can use to set inputs/outputs/errors
        yield
    except Exception as e:
        _observe_error(inspect.currentframe(), sys.exc_info())
        raise e
    finally:
        _observe_exit(inspect.currentframe(), None)


# NOTE: This builds once, can be tweaked if we are missing / capturing other unncessary modules
# @link https://docs.python.org/3.13/library/sysconfig.html
_TRACE_FILEPATH_BLOCKLIST = tuple(
    os.path.realpath(p) + os.sep
    for p in {
        sysconfig.get_paths()["stdlib"],
        sysconfig.get_paths().get("platstdlib", ""),
        *site.getsitepackages(),
        site.getusersitepackages(),
        *(
            [os.path.join(os.path.dirname(__file__), "../../judgeval/")]
            if os.environ.get("JUDGMENT_DEV")
            else []
        ),
    }
    if p
)


@functools.cache
def _is_user_code(filename: str):
    """
    Determines if the given filename corresponds to user code.
    """
    return (
        bool(filename)
        and not filename.startswith("<")
        and not os.path.realpath(filename).startswith(_TRACE_FILEPATH_BLOCKLIST)
    )


code_should_trace = set[int]()


def _do_trace_code(code: CodeType) -> None:
    code_should_trace.add(id(code))
    logging.debug(
        f"Code object {code.co_name} at {code.co_filename} (id: {id(code)}) "
        f"has been added to the trace set."
    )


def __trace_daemon_factory__(deep_tracing: bool = False):
    """
    Factory function to create a trace daemon.
    Based on the global Judgment config, this function will create an appropriate trace daemon to
    be setup with syshooks.
    """

    def _should_trace(frame: FrameType) -> bool:
        # If we are not deep tracing, we only trace if the code object is in the set of code_should_trace
        if not deep_tracing and id(frame.f_code) in code_should_trace:
            return True

        if not _is_user_code(frame.f_code.co_filename):
            logging.debug(
                f"Skipping trace for {frame.f_code.co_name} at {frame.f_code.co_filename} "
                f"because it is not user code."
            )
            return False

        if deep_tracing:
            return True

        logging.debug(
            f"Skipping trace for {frame.f_code.co_name} at {frame.f_code.co_filename} "
            f"because it is not in the code_should_trace set."
            f" (id: {id(frame.f_code)})"
        )

        return False

    def __trace_daemon__(frame: FrameType, event: str, arg: Any) -> Union[Callable, None]:
        """
        The trace daemon that will be used to handle tracing.
        This function should be set up with sys.settrace / sys.setprofile hooks
        """
        logging.debug(
            f"Tracing {event} for {frame.f_code.co_name} at {frame.f_code.co_filename} "
            f"(id: {id(frame.f_code)})"
        )
        if event not in ("call", "return", "exception"):
            return None

        if not _should_trace(frame):
            return None

        frame.f_trace_lines = False
        frame.f_trace_opcodes = False

        logging.info(
            f"Tracing {event} for {frame.f_code.co_name} at {frame.f_code.co_filename} "
            f"(id: {id(frame.f_code)})"
        )

        if event == "call":
            _observe_enter(frame)
            return __trace_daemon__

        elif event == "return":
            _observe_exit(frame, arg)

        elif event == "exception":
            _observe_error(frame, arg)

        return None

    return __trace_daemon__


@contextmanager
def observe_daemon(deep_tracing: bool = False):
    try:
        sys.settrace(__trace_daemon_factory__(deep_tracing=deep_tracing))
        yield
    finally:
        sys.settrace(None)


def tag(func: Callable):
    """
    Tags the function code object for tracing.
    """
    _do_trace_code(func.__code__)
    return func


def print_graph():
    """
    Prints the trace spans in a tree-like indented format.
    """

    parents = [span for span in trace_spans if span.parent_span_id is None]
    children = [span for span in trace_spans if span.parent_span_id is not None]

    INDENT_SIZE = 3

    def print_span(span: CurrentSpanExit, depth: int = 0):
        indent = "  " * depth * INDENT_SIZE
        # Format arguments
        args_str = ""
        if span.inputs:
            args_list = []
            for k, v in span.inputs.items():
                args_list.append(f"{k}={v!r}")
            args_str = ", ".join(args_list)

        # Format output
        output_str = f"{span.output!r}"

        # Time taken in ms
        duration_ms = (span.end_time - span.start_time) * 1000
        print(
            f"{indent}→ {span.name}({args_str}) => {output_str} "
            f"[{duration_ms:.2f} ms]"
        )
        for child in children:
            if child.parent_span_id == span.span_id:
                print_span(child, depth + 1)

    for root_span in parents:
        print_span(root_span)
        print()
