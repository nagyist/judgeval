from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.trace_compare import TraceCompare
from ...types import Response


def _get_kwargs(
    *,
    body: TraceCompare,
    x_organization_id: str,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    headers["X-Organization-Id"] = x_organization_id

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/traces/compare/",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[Any, HTTPValidationError]]:
    if response.status_code == 200:
        response_200 = response.json()
        return response_200
    if response.status_code == 422:
        response_422 = HTTPValidationError.from_dict(response.json())

        return response_422
    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Response[Union[Any, HTTPValidationError]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient,
    body: TraceCompare,
    x_organization_id: str,
) -> Response[Union[Any, HTTPValidationError]]:
    r"""Compare Traces

     Compare two traces and their associated evaluation metrics span-by-span.

    Args:
        compare_data (TraceCompare):
            - baseline_trace_id: ID of the baseline trace to compare against
            - comparison_trace_id: ID of the trace to compare with the baseline

    Returns:
        List[dict]: Structured comparison containing:
            - depth: Span depth in the trace hierarchy
            - inputs: Baseline/comparison input comparison
            - output: Baseline/comparison output comparison
            - duration: Execution time differences
            - function: Function name that generated the span
            - span_type: Type of span (e.g., span, chain, tool)
            - timestamp: Original span creation timestamp
            - metrics_comparison: Evaluation metric comparisons when available

    Raises:
        HTTPException 404: If either trace is not found
        HTTPException 500: For internal server errors

    Example:
        >>> Request Body
        {
            \"baseline_trace_id\": \"trace_123\",
            \"comparison_trace_id\": \"trace_456\",
        }

        >>> Response
        [
            {
                \"depth\": 0,
                \"inputs\": {\"baseline\": \"Hi\", \"comparison\": \"Hello\"},
                \"output\": {\"baseline\": \"Hello\", \"comparison\": \"Hi there\"},
                \"duration\": {\"baseline\": 1.2, \"comparison\": 1.5},
                \"function\": \"greet_user\",
                \"span_type\": \"span\",
                \"timestamp\": 1737148061.7688801,
                \"metrics_comparison\": {
                    \"accuracy\": {
                        \"baseline\": {\"score\": 0.9, \"reason\": \"Correct greeting\"},
                        \"comparison\": {\"score\": 0.8, \"reason\": \"Informal tone\"}
                    }
                }
            }
        ]

    Args:
        x_organization_id (str):
        body (TraceCompare): Used for comparing two traces

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        body=body,
        x_organization_id=x_organization_id,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient,
    body: TraceCompare,
    x_organization_id: str,
) -> Optional[Union[Any, HTTPValidationError]]:
    r"""Compare Traces

     Compare two traces and their associated evaluation metrics span-by-span.

    Args:
        compare_data (TraceCompare):
            - baseline_trace_id: ID of the baseline trace to compare against
            - comparison_trace_id: ID of the trace to compare with the baseline

    Returns:
        List[dict]: Structured comparison containing:
            - depth: Span depth in the trace hierarchy
            - inputs: Baseline/comparison input comparison
            - output: Baseline/comparison output comparison
            - duration: Execution time differences
            - function: Function name that generated the span
            - span_type: Type of span (e.g., span, chain, tool)
            - timestamp: Original span creation timestamp
            - metrics_comparison: Evaluation metric comparisons when available

    Raises:
        HTTPException 404: If either trace is not found
        HTTPException 500: For internal server errors

    Example:
        >>> Request Body
        {
            \"baseline_trace_id\": \"trace_123\",
            \"comparison_trace_id\": \"trace_456\",
        }

        >>> Response
        [
            {
                \"depth\": 0,
                \"inputs\": {\"baseline\": \"Hi\", \"comparison\": \"Hello\"},
                \"output\": {\"baseline\": \"Hello\", \"comparison\": \"Hi there\"},
                \"duration\": {\"baseline\": 1.2, \"comparison\": 1.5},
                \"function\": \"greet_user\",
                \"span_type\": \"span\",
                \"timestamp\": 1737148061.7688801,
                \"metrics_comparison\": {
                    \"accuracy\": {
                        \"baseline\": {\"score\": 0.9, \"reason\": \"Correct greeting\"},
                        \"comparison\": {\"score\": 0.8, \"reason\": \"Informal tone\"}
                    }
                }
            }
        ]

    Args:
        x_organization_id (str):
        body (TraceCompare): Used for comparing two traces

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, HTTPValidationError]
    """

    return sync_detailed(
        client=client,
        body=body,
        x_organization_id=x_organization_id,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient,
    body: TraceCompare,
    x_organization_id: str,
) -> Response[Union[Any, HTTPValidationError]]:
    r"""Compare Traces

     Compare two traces and their associated evaluation metrics span-by-span.

    Args:
        compare_data (TraceCompare):
            - baseline_trace_id: ID of the baseline trace to compare against
            - comparison_trace_id: ID of the trace to compare with the baseline

    Returns:
        List[dict]: Structured comparison containing:
            - depth: Span depth in the trace hierarchy
            - inputs: Baseline/comparison input comparison
            - output: Baseline/comparison output comparison
            - duration: Execution time differences
            - function: Function name that generated the span
            - span_type: Type of span (e.g., span, chain, tool)
            - timestamp: Original span creation timestamp
            - metrics_comparison: Evaluation metric comparisons when available

    Raises:
        HTTPException 404: If either trace is not found
        HTTPException 500: For internal server errors

    Example:
        >>> Request Body
        {
            \"baseline_trace_id\": \"trace_123\",
            \"comparison_trace_id\": \"trace_456\",
        }

        >>> Response
        [
            {
                \"depth\": 0,
                \"inputs\": {\"baseline\": \"Hi\", \"comparison\": \"Hello\"},
                \"output\": {\"baseline\": \"Hello\", \"comparison\": \"Hi there\"},
                \"duration\": {\"baseline\": 1.2, \"comparison\": 1.5},
                \"function\": \"greet_user\",
                \"span_type\": \"span\",
                \"timestamp\": 1737148061.7688801,
                \"metrics_comparison\": {
                    \"accuracy\": {
                        \"baseline\": {\"score\": 0.9, \"reason\": \"Correct greeting\"},
                        \"comparison\": {\"score\": 0.8, \"reason\": \"Informal tone\"}
                    }
                }
            }
        ]

    Args:
        x_organization_id (str):
        body (TraceCompare): Used for comparing two traces

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        body=body,
        x_organization_id=x_organization_id,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient,
    body: TraceCompare,
    x_organization_id: str,
) -> Optional[Union[Any, HTTPValidationError]]:
    r"""Compare Traces

     Compare two traces and their associated evaluation metrics span-by-span.

    Args:
        compare_data (TraceCompare):
            - baseline_trace_id: ID of the baseline trace to compare against
            - comparison_trace_id: ID of the trace to compare with the baseline

    Returns:
        List[dict]: Structured comparison containing:
            - depth: Span depth in the trace hierarchy
            - inputs: Baseline/comparison input comparison
            - output: Baseline/comparison output comparison
            - duration: Execution time differences
            - function: Function name that generated the span
            - span_type: Type of span (e.g., span, chain, tool)
            - timestamp: Original span creation timestamp
            - metrics_comparison: Evaluation metric comparisons when available

    Raises:
        HTTPException 404: If either trace is not found
        HTTPException 500: For internal server errors

    Example:
        >>> Request Body
        {
            \"baseline_trace_id\": \"trace_123\",
            \"comparison_trace_id\": \"trace_456\",
        }

        >>> Response
        [
            {
                \"depth\": 0,
                \"inputs\": {\"baseline\": \"Hi\", \"comparison\": \"Hello\"},
                \"output\": {\"baseline\": \"Hello\", \"comparison\": \"Hi there\"},
                \"duration\": {\"baseline\": 1.2, \"comparison\": 1.5},
                \"function\": \"greet_user\",
                \"span_type\": \"span\",
                \"timestamp\": 1737148061.7688801,
                \"metrics_comparison\": {
                    \"accuracy\": {
                        \"baseline\": {\"score\": 0.9, \"reason\": \"Correct greeting\"},
                        \"comparison\": {\"score\": 0.8, \"reason\": \"Informal tone\"}
                    }
                }
            }
        ]

    Args:
        x_organization_id (str):
        body (TraceCompare): Used for comparing two traces

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
            x_organization_id=x_organization_id,
        )
    ).parsed
