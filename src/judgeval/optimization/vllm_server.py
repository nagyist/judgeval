from __future__ import annotations

import asyncio
import contextlib
from typing import Any, Optional

from openai import AsyncOpenAI
from vllm.entrypoints.openai import api_server
from fastapi import FastAPI
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.utils import FlexibleArgumentParser


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
    engine,
    *,
    host: str = "0.0.0.0",
    port: int = 8000,
    ready_timeout: float = 180.0,
):
    """Launch the vLLM OpenAI-compatible server as a background task.

    The coroutine returns once the TCP port is reachable, allowing the caller
    to proceed with requests. The returned *asyncio.Task* keeps the server
    running; you may `await task` to block until the server shuts down, or
    `task.cancel()` to stop it.
    """

    @contextlib.asynccontextmanager
    async def build_async_engine_client(*_a: Any, **_kw: Any):
        yield engine

    # Inject our custom engine factory.
    api_server.build_async_engine_client = build_async_engine_client

    # Build CLI namespace (only host/port are overridden here; model options
    # are expected to be provided via *engine* config or env vars).
    parser = FlexibleArgumentParser()
    parser = make_arg_parser(parser)
    args_list = [f"--host={host}", f"--port={port}"]
    namespace = parser.parse_args(args_list)
    validate_parsed_serve_args(namespace)

    # Start the server coroutine in the current event loop.
    server_task = asyncio.create_task(
        api_server.run_server(namespace), name="vLLMOpenAIServer"
    )

    # Wait until the server is actually listening.
    connect_host = "127.0.0.1" if host == "0.0.0.0" else host
    await _wait_for_port(connect_host, port, timeout=ready_timeout)

    return ("127.0.0.1" if host == "0.0.0.0" else host), port