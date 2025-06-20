from __future__ import annotations
from contextlib import contextmanager
from contextvars import ContextVar, Token
import functools
import inspect
import logging
import os
import site
import sys
import sysconfig
import time
from types import CodeType, FrameType, TracebackType
from typing import Any, Callable, List, Optional, TypeAlias, TypeVar, Union, cast

from core import CurrentSpanEntry, CurrentSpanExit, CurrentSpanType, new_span_id
from utils import extract_inputs_from_entry_frame

ExcInfo: TypeAlias = tuple[type[BaseException], BaseException, TracebackType]
OptExcInfo: TypeAlias = ExcInfo | tuple[None, None, None]
TSpanResetTokens: TypeAlias = dict[str, Token[CurrentSpanEntry]]

TCallable = TypeVar("TCallable", bound=Callable[..., Any])


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


class TraceClient:
    """Client for managing trace contexts."""

    __slots__ = (
        "trace_id",
        "name",
        "project_id",
        "overwrite",
        "enable_monitoring",
        "enable_evaluations",
        "_current_span_var",
        "_span_reset_tokens_var",
        "_surpress_deep_tracing",
        "_code_should_trace",
        "_trace_spans",
    )

    trace_id: Optional[str]
    name: str
    project_id: Optional[str]
    overwrite: bool
    enable_monitoring: bool
    enable_evaluations: bool

    _current_span_var: ContextVar[Union[CurrentSpanType, None]]
    _span_reset_tokens_var: ContextVar[Union[TSpanResetTokens, None]]
    _surpress_deep_tracing: ContextVar[bool]

    _code_should_trace: set[int]
    _trace_spans: List[CurrentSpanExit]

    def __init__(
        self,
        trace_id: Optional[str] = None,
        name: str = "default",
        project_id: Optional[str] = None,
        overwrite: bool = False,
        enable_monitoring: bool = True,
        enable_evaluations: bool = True,
    ):
        self.trace_id = trace_id
        self.name = name
        self.project_id = project_id
        self.overwrite = overwrite

        self.enable_monitoring = enable_monitoring
        self.enable_evaluations = enable_evaluations

        self._current_span_var = ContextVar[Union[CurrentSpanType, None]](
            "current_span", default=None
        )
        self._span_reset_tokens_var = ContextVar[Union[TSpanResetTokens, None]](
            "span_reset_tokens", default=None
        )
        self._surpress_deep_tracing = ContextVar[bool](
            "surpress_deep_tracing", default=False
        )

        self._code_should_trace = set()
        self._trace_spans: list[CurrentSpanExit] = []

    @property
    def current_span(self) -> Union[CurrentSpanType, None]:
        """Get the current span from the context variable."""
        return self._current_span_var.get()

    @current_span.setter
    def current_span(self, value: CurrentSpanType):
        self._current_span_var.set(value)

    @property
    def span_reset_tokens(self) -> TSpanResetTokens:
        """Get the reset tokens from the context variable."""
        span_reset_tokens = self._span_reset_tokens_var.get()
        if span_reset_tokens is None:
            self._span_reset_tokens_var.set({})
        return cast(TSpanResetTokens, self._span_reset_tokens_var.get())

    @span_reset_tokens.setter
    def span_reset_tokens(self, value: TSpanResetTokens):
        """Set the reset tokens in the context variable."""
        self._span_reset_tokens_var.set(value)

    def _do_trace_code(self, code: CodeType) -> None:
        """
        Register the code object for tracing.
        This is used to determine if the code should be traced or not.
        """
        self._code_should_trace.add(id(code))

    def _commit_current_span(self) -> None:
        """Commit the current span to the trace spans list."""
        if not isinstance(current_span := self.current_span, CurrentSpanExit):
            logging.warning(
                "Commit called without a completed current span entry. This should not happen."
            )
            return

        span_id = current_span.span_id
        self._trace_spans.append(current_span)

        if self.span_reset_tokens is None or span_id not in self.span_reset_tokens:
            logging.warning(
                f"Exit observed without a corresponding entry span for span_id: {span_id}. This should not happen."
            )
            return

        token = cast(Token[Any], self.span_reset_tokens[span_id])
        del self.span_reset_tokens[span_id]
        self._current_span_var.reset(token)

        logging.info(
            f"Span committed: {current_span.name} (id: {span_id}, parent_id: {current_span.parent_span_id}, "
            f"start_time: {current_span.start_time}, end_time: {current_span.end_time}, output: {current_span.output})"
        )

    def observe_enter(self, frame: FrameType):
        name = frame.f_code.co_name
        span_id = new_span_id()
        parent_span_id = (
            self.current_span.span_id
            if isinstance(self.current_span, CurrentSpanEntry)
            else None
        )
        start_time = time.time()
        inputs = extract_inputs_from_entry_frame(frame)

        token = self._current_span_var.set(
            CurrentSpanEntry(
                name=name,
                span_id=span_id,
                span_type="function",
                parent_span_id=parent_span_id,
                start_time=start_time,
                inputs=inputs,
            )
        )
        if self.span_reset_tokens is None:
            self.span_reset_tokens = {}

        self.span_reset_tokens[span_id] = cast(Token[CurrentSpanEntry], token)
        logging.info(
            f"Span entered: {name} (id: {span_id}, parent_id: {parent_span_id}, start_time: {start_time}, inputs: {inputs})"
        )

    def observe_exit(self, frame: FrameType, output: Any):
        if not isinstance(current_span := self.current_span, CurrentSpanEntry):
            logging.warning(
                "Exit observed without a corresponding entry span. This should not happen."
            )
            return

        end_time = time.time()

        span_exit = CurrentSpanExit(
            name=current_span.name,
            span_id=current_span.span_id,
            span_type=current_span.span_type,
            parent_span_id=current_span.parent_span_id,
            start_time=current_span.start_time,
            end_time=end_time,
            output=output,
            inputs=current_span.inputs,
        )
        self.current_span = span_exit

        logging.info(
            f"Span exited: {span_exit.name} (id: {span_exit.span_id}, parent_id: {span_exit.parent_span_id}, "
            f"start_time: {span_exit.start_time}, end_time: {end_time}, output: {output})"
        )
        self._commit_current_span()

    def observe_error(self, frame: FrameType, exc_info: OptExcInfo):
        if not isinstance(current_span := self.current_span, CurrentSpanEntry):
            logging.warning(
                "Error observed without a corresponding entry span. This should not happen.",
            )

            return

        self._current_span_var.set(
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

        logging.info(
            f"Error observed in span: {current_span.name} (id: {current_span.span_id}, "
            f"parent_id: {current_span.parent_span_id}, start_time: {current_span.start_time}, "
            f"error: {exc_info})"
        )

    @functools.cache
    def _is_user_code(self, filename: str):
        """
        Determines if the given filename corresponds to user code.
        """
        return (
            bool(filename)
            and not filename.startswith("<")
            and not os.path.realpath(filename).startswith(_TRACE_FILEPATH_BLOCKLIST)
        )

    def _trace_daemon_factory_(self, deep_tracing: bool = False):
        """
        Factory function to create a trace daemon.
        Based on the global Judgment config, this function will create an appropriate trace daemon to
        be setup with syshooks.
        """

        def _should_trace(frame: FrameType) -> bool:
            # If we are not deep tracing, we only trace if the code object is in the set of code_should_trace
            if not deep_tracing and id(frame.f_code) in self._code_should_trace:
                return True

            if not self._is_user_code(frame.f_code.co_filename):
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

        def __trace_daemon__(
            frame: FrameType, event: str, arg: Any
        ) -> Union[Callable, None]:
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
                self.observe_enter(frame)
                return __trace_daemon__

            elif event == "return":
                self.observe_exit(frame, arg)

            elif event == "exception":
                self.observe_error(frame, arg)

            return None

        return __trace_daemon__

    @contextmanager
    def span(self, name: str, span_type: str = "span"):
        """
        Context manager to create a span with the given name and type.
        This will automatically handle entering and exiting the span.
        """
        frame = inspect.currentframe()
        if frame is None:
            raise RuntimeError("Current frame is None, cannot create span.")

        self.observe_enter(frame)

        try:
            yield
        except Exception as exc:
            self.observe_error(frame, sys.exc_info())
            raise
        finally:
            self.observe_exit(frame, None)

    def observe(self, func: TCallable) -> TCallable:
        self._do_trace_code(func.__code__)
        return func

    def print_graph(self):
        """
        Prints the trace spans in a tree-like indented format.
        """

        parents = [span for span in self._trace_spans if span.parent_span_id is None]
        children = [
            span for span in self._trace_spans if span.parent_span_id is not None
        ]

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

    @contextmanager
    def daemon(self, deep_tracing: bool = False):
        """
        Install the trace daemon.
        """
        try:
            sys.settrace(self._trace_daemon_factory_(deep_tracing))
            yield self
        finally:
            sys.settrace(None)


judgment = TraceClient()


def fibonacci(n: int) -> int:
    """
    A simple Fibonacci function to demonstrate tracing.
    """
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

def length(s: list[Any]) -> int:
    """
    A simple function to return the length of a list.
    """
    return len(s)


import logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main():
    length([x for x in range(fibonacci(5))])


with judgment.daemon(deep_tracing=True):
    main()


judgment.print_graph()
