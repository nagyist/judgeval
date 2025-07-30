from __future__ import annotations

import asyncio
import contextlib
import threading
from typing import Any, Optional

from openai import AsyncOpenAI
from vllm.entrypoints.openai import api_server
from fastapi import FastAPI
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.utils import FlexibleArgumentParser

__all__ = ["launch_openai_server", "wait_until_ready"]


def launch_openai_server(
    engine,
    *,
    host: str = "0.0.0.0",
    port: int = 8000,
    ready_event: Optional[threading.Event] = None,
    disable_log_arguments: bool = True,
) -> threading.Thread:

    def _run_server() -> None:
        @contextlib.asynccontextmanager
        async def build_async_engine_client(*_a: Any, **_kw: Any):
            yield engine

        api_server.build_async_engine_client = build_async_engine_client

        if ready_event is not None:
            try:
                app: FastAPI = api_server.app

                @app.on_event("startup")
                async def _notify_ready():
                    ready_event.set()
            except Exception:
                pass

        parser = FlexibleArgumentParser()
        parser = make_arg_parser(parser)
        args_list = [f"--host={host}", f"--port={port}"]
        if disable_log_arguments:
            args_list.append("--disable-log-arguments")
        namespace = parser.parse_args(args_list)
        validate_parsed_serve_args(namespace)

        asyncio.run(api_server.run_server(namespace))

    thread = threading.Thread(target=_run_server, name="vllm-openai-server", daemon=True)
    thread.start()
    return thread


async def wait_until_ready(
    *,
    base_url: str,
    api_key: str,
    ready_event: Optional[threading.Event] = None,
    timeout: float = 10.0,
) -> None:
    if ready_event is not None and ready_event.wait(timeout=timeout):
        return

    async def _probe() -> None:
        client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        async for _ in client.models.list():
            return
        raise RuntimeError()

    try:
        await asyncio.wait_for(_probe(), timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise TimeoutError() from exc 