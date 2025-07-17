# mypy: disable-error-code="method-assign"

from __future__ import annotations

import sys
from typing import Any, Callable, Optional, TypeAlias, Union
from openai import OpenAI, AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.responses.response import Response
from openai.types.chat import ParsedChatCompletion
from together import Together, AsyncTogether
from anthropic import Anthropic, AsyncAnthropic
from google import genai
from judgeval.common.tracer.utils import cost_per_token
from judgeval.data import TraceUsage
from judgeval.common.logger import judgeval_logger
from judgeval.common.tracer.core import (
    Tracer,
    _capture_exception_for_trace,
    current_trace_var,
)

ApiClient: TypeAlias = Union[
    OpenAI,
    Together,
    Anthropic,
    AsyncOpenAI,
    AsyncAnthropic,
    AsyncTogether,
    genai.Client,
    genai.client.AsyncClient,
]


def _get_current_trace(
    trace_across_async_contexts: bool = Tracer.trace_across_async_contexts,
):
    if trace_across_async_contexts:
        return Tracer.current_trace
    else:
        return current_trace_var.get()


def wrap(
    client: Any, trace_across_async_contexts: bool = Tracer.trace_across_async_contexts
) -> Any:
    """
    Wraps an API client to add tracing capabilities.
    Supports OpenAI, Together, Anthropic, and Google GenAI clients.
    Patches both '.create' and Anthropic's '.stream' methods using a wrapper class.
    """
    (
        span_name,
        original_create,
        original_responses_create,
        original_stream,
        original_beta_parse,
    ) = _get_client_config(client)

    def process_span(span, response):
        """Format and record the output in the span"""
        output, usage = _format_output_data(client, response)
        span.record_output(output)
        span.record_usage(usage)
        return response

    def wrapped(function):
        def wrapper(*args, **kwargs):
            current_trace = _get_current_trace(trace_across_async_contexts)
            if not current_trace:
                return function(*args, **kwargs)
            with current_trace.span(span_name, span_type="llm") as span:
                span.record_input(kwargs)
                try:
                    response = function(*args, **kwargs)
                    return process_span(span, response)
                except Exception as e:
                    _capture_exception_for_trace(span, sys.exc_info())
                    raise e

        return wrapper

    def wrapped_async(function):
        async def wrapper(*args, **kwargs):
            current_trace = _get_current_trace(trace_across_async_contexts)
            if not current_trace:
                return await function(*args, **kwargs)
            with current_trace.span(span_name, span_type="llm") as span:
                span.record_input(kwargs)
                try:
                    response = await function(*args, **kwargs)
                    return process_span(span, response)
                except Exception as e:
                    _capture_exception_for_trace(span, sys.exc_info())
                    raise e

        return wrapper

    if isinstance(client, (OpenAI)):
        client.chat.completions.create = wrapped(original_create)
        client.responses.create = wrapped(original_responses_create)
        client.beta.chat.completions.parse = wrapped(original_beta_parse)
    elif isinstance(client, (AsyncOpenAI)):
        client.chat.completions.create = wrapped_async(original_create)
        client.responses.create = wrapped_async(original_responses_create)
        client.beta.chat.completions.parse = wrapped_async(original_beta_parse)
    elif isinstance(client, (Together)):
        client.chat.completions.create = wrapped(original_create)
    elif isinstance(client, (AsyncTogether)):
        client.chat.completions.create = wrapped_async(original_create)
    elif isinstance(client, (Anthropic)):
        client.messages.create = wrapped(original_create)
    elif isinstance(client, (AsyncAnthropic)):
        client.messages.create = wrapped_async(original_create)
    elif isinstance(client, (genai.Client)):
        client.models.generate_content = wrapped(original_create)
    elif isinstance(client, (genai.client.AsyncClient)):
        client.models.generate_content = wrapped_async(original_create)
    return client


def _get_client_config(
    client: ApiClient,
) -> tuple[str, Callable, Optional[Callable], Optional[Callable], Optional[Callable]]:
    """Returns configuration tuple for the given API client.

    Args:
        client: An instance of OpenAI, Together, or Anthropic client

    Returns:
        tuple: (span_name, create_method, responses_method, stream_method, beta_parse_method)
            - span_name: String identifier for tracing
            - create_method: Reference to the client's creation method
            - responses_method: Reference to the client's responses method (if applicable)
            - stream_method: Reference to the client's stream method (if applicable)
            - beta_parse_method: Reference to the client's beta parse method (if applicable)

    Raises:
        ValueError: If client type is not supported
    """
    if isinstance(client, (OpenAI, AsyncOpenAI)):
        return (
            "OPENAI_API_CALL",
            client.chat.completions.create,
            client.responses.create,
            None,
            client.beta.chat.completions.parse,
        )
    elif isinstance(client, (Together, AsyncTogether)):
        return "TOGETHER_API_CALL", client.chat.completions.create, None, None, None
    elif isinstance(client, (Anthropic, AsyncAnthropic)):
        return (
            "ANTHROPIC_API_CALL",
            client.messages.create,
            None,
            client.messages.stream,
            None,
        )
    elif isinstance(client, (genai.Client, genai.client.AsyncClient)):
        return "GOOGLE_API_CALL", client.models.generate_content, None, None, None
    raise ValueError(f"Unsupported client type: {type(client)}")


def _format_output_data(
    client: ApiClient, response: Any
) -> tuple[Optional[str], Optional[TraceUsage]]:
    """Format API response data based on client type.

    Normalizes different response formats into a consistent structure
    for tracing purposes.

    Returns:
        dict containing:
            - content: The generated text
            - usage: Token usage statistics
    """
    prompt_tokens = 0
    completion_tokens = 0
    cache_read_input_tokens = 0
    cache_creation_input_tokens = 0
    model_name = None
    message_content = None
    if isinstance(client, (OpenAI, AsyncOpenAI)):
        if isinstance(response, ChatCompletion):
            model_name = response.model
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            cache_read_input_tokens = response.usage.prompt_tokens_details.cached_tokens
            if isinstance(response, ParsedChatCompletion):
                message_content = response.choices[0].message.parsed
            else:
                message_content = response.choices[0].message.content
        elif isinstance(response, Response):
            model_name = response.model
            prompt_tokens = response.usage.input_tokens
            completion_tokens = response.usage.output_tokens
            cache_read_input_tokens = response.usage.input_tokens_details.cached_tokens
            message_content = "".join(seg.text for seg in response.output[0].content)
    elif isinstance(client, (Together, AsyncTogether)):
        model_name = "together_ai/" + response.model
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        message_content = response.choices[0].message.content
    elif isinstance(client, (genai.Client, genai.client.AsyncClient)):
        model_name = response.model_version
        prompt_tokens = response.usage_metadata.prompt_token_count
        completion_tokens = response.usage_metadata.candidates_token_count
        message_content = response.candidates[0].content.parts[0].text
        if hasattr(response.usage_metadata, "cached_content_token_count"):
            cache_read_input_tokens = response.usage_metadata.cached_content_token_count
    elif isinstance(client, (Anthropic, AsyncAnthropic)):
        model_name = response.model
        prompt_tokens = response.usage.input_tokens
        completion_tokens = response.usage.output_tokens
        cache_read_input_tokens = response.usage.cache_read_input_tokens
        cache_creation_input_tokens = response.usage.cache_creation_input_tokens
        message_content = response.content[0].text
    else:
        judgeval_logger.warning(f"Unsupported client type: {type(client)}")
        return None, None
    prompt_cost, completion_cost = cost_per_token(
        model=model_name,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
    )
    total_cost_usd = (
        (prompt_cost + completion_cost) if prompt_cost and completion_cost else None
    )
    usage = TraceUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        prompt_tokens_cost_usd=prompt_cost,
        completion_tokens_cost_usd=completion_cost,
        total_cost_usd=total_cost_usd,
        model_name=model_name,
    )
    return message_content, usage
