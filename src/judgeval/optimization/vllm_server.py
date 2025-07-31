from __future__ import annotations

import asyncio
import contextlib
from typing import Any, Optional, AsyncIterator

from openai import AsyncOpenAI
from fastapi import FastAPI
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.utils import FlexibleArgumentParser
from .server_types import OpenAIServerConfig, ServerArgs, EngineArgs
from vllm.engine.protocol import EngineClient

# -----------------------------------------------------------------------------
# Public helper: launch_openai_server
# -----------------------------------------------------------------------------


async def _wait_for_port(host: str, port: int, *, timeout: float = 30.0) -> None:
    """Poll *host:port* until it becomes reachable or *timeout* expires.

    Raises
    ------
    TimeoutError
        If the server does not accept TCP connections within *timeout* seconds.
    """

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout

    while True:
        try:
            reader, writer = await asyncio.open_connection(host, port)
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
            return  # Success.
        except (ConnectionRefusedError, OSError):
            if loop.time() >= deadline:
                raise TimeoutError(
                    f"vLLM OpenAI server not reachable on {host}:{port} within {timeout} seconds"
                )
            await asyncio.sleep(0.1)


async def launch_openai_server(
    engine: EngineClient,
    *,
    host: str = "0.0.0.0",
    port: int = 8000,
    config: OpenAIServerConfig | None = None,
    ready_timeout: float = 30.0,
):
    """Launch the vLLM OpenAI-compatible server as a background task.

    The coroutine returns once the TCP port is reachable, allowing the caller
    to proceed with requests. The returned *asyncio.Task* keeps the server
    running; you may `await task` to block until the server shuts down, or
    `task.cancel()` to stop it.
    """
    from vllm.entrypoints.openai import api_server

    @contextlib.asynccontextmanager
    async def build_async_engine_client(*_a: Any, **_kw: Any) -> AsyncIterator[EngineClient]:
        print("build_async_engine_client")
        print(engine)
        print(type(engine))
        print(dir(engine))
        yield engine

    # Inject our custom engine factory.
    api_server.build_async_engine_client = build_async_engine_client

    server_task = asyncio.create_task(
        start_server_coroutine(config), name="vLLMOpenAIServer"
    )

    # Wait until the server is actually listening.
    connect_host = "127.0.0.1" if host == "0.0.0.0" else host
    await _wait_for_port(connect_host, port, timeout=ready_timeout)

    return ("127.0.0.1" if host == "0.0.0.0" else host), port

def start_server_coroutine(config: dict):
    """Start the vLLM OpenAI-compatible server as a background task."""
    from vllm.entrypoints.openai import api_server

    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )
    parser = make_arg_parser(parser)
    engine_args = config.get("engine_args", {})
    server_args = config.get("server_args", {})
    args = [
        *[
            f"--{key.replace('_', '-')}{f'={item}' if item is not True else ''}"
            for args in [engine_args, server_args]
            for key, value in args.items()
            for item in (value if isinstance(value, list) else [value])
            if item is not None
        ],
    ]
    namespace = parser.parse_args(args)
    assert namespace is not None
    validate_parsed_serve_args(namespace)
    return api_server.run_server(
        namespace
    )


def get_openai_server_config(
    model_name: str,
    base_model: str,
    log_file: str,
    lora_path: str | None = None,
    config: "OpenAIServerConfig | None" = None,
) -> "OpenAIServerConfig":
    if config is None:
        config = OpenAIServerConfig()
    log_file = config.get("log_file", log_file)
    server_args = ServerArgs(
        api_key="default",
        lora_modules=(
            [f'{{"name": "{model_name}", "path": "{lora_path}"}}']
            if lora_path
            else None
        ),
        return_tokens_as_token_ids=True,
        enable_auto_tool_choice=True,
        tool_call_parser="hermes",
    )
    server_args.update(config.get("server_args", {}))
    engine_args = EngineArgs(
        model=base_model,
        num_scheduler_steps=16 if lora_path else 1,
        served_model_name=base_model if lora_path else model_name,
        disable_log_requests=True,
        generation_config="vllm",
    )
    engine_args.update(config.get("engine_args", {}))
    return OpenAIServerConfig(
        log_file=log_file, server_args=server_args, engine_args=engine_args
    )

