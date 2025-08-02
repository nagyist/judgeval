"""Async utilities for judgeval."""

import asyncio
import nest_asyncio


def safe_run_async(coro):
    """
    Safely run an async coroutine whether or not there's already an event loop running.

    Args:
        coro: The coroutine to run

    Returns:
        The result of the coroutine
    """
    try:
        asyncio.get_running_loop()
        nest_asyncio.apply()
        return asyncio.run(coro)
    except RuntimeError:
        # No event loop is running, safe to use asyncio.run()
        return asyncio.run(coro)
