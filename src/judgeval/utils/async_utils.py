"""Async utilities for judgeval."""

import asyncio
import concurrent.futures
import os


def safe_run_async(coro):
    """
    Safely run an async coroutine whether or not there's already an event loop running.

    Args:
        coro: The coroutine to run

    Returns:
        The result of the coroutine
    """
    try:
        # Try to get the running loop
        asyncio.get_running_loop()
        # If we get here, there's already a loop running
        # Run in a separate thread to avoid "asyncio.run() cannot be called from a running event loop"
        max_workers = min(
            32, (os.cpu_count() or 1) + 4
        )  # Same as ThreadPoolExecutor default
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        # No event loop is running, safe to use asyncio.run()
        return asyncio.run(coro)
