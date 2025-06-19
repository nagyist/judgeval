from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.trace_fetch import TraceFetch
from ...types import Response


def _get_kwargs(
    *,
    body: TraceFetch,
    x_organization_id: str,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    headers["X-Organization-Id"] = x_organization_id

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/traces/fetch_tool_calls/",
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
    body: TraceFetch,
    x_organization_id: str,
) -> Response[Union[Any, HTTPValidationError]]:
    r"""Fetch Tool Calls

     Fetch all tool calls for a trace, returning only their input and output, sorted by call order
    (created_at).
    Args:
    fetch_data (TraceFetch): Request object containing:
        - trace_id (str): UUID of the trace to fetch tool calls from
    trace_client (TraceClient): Injected dependency for trace operations
    user_organization (UserOrganization): Injected dependency for auth/org validation

    Returns:
        Dict: Response object containing:
            - completions (List[Dict]): Alternating assistant/user messages where:
                - Assistant messages contain: {\"role\": \"assistant\", \"content\": {\"func\": str,
    \"args\": str}}
                - User messages contain: {\"role\": \"user\", \"content\": str} (tool output)
        Tool calls are sorted by created_at timestamp in ascending order.

    Example Response:
    {
        \"completions\": [
            {\"role\": \"assistant\", \"content\": {\"func\": \"search_web\", \"args\": \"{'query':
    'weather'}\"}},
            {\"role\": \"user\", \"content\": \"Sunny, 75°F\"},
            {\"role\": \"assistant\", \"content\": {\"func\": \"send_email\", \"args\": \"{'to':
    'user@example.com'}\"}},
            {\"role\": \"user\", \"content\": \"Email sent successfully\"}
        ]
    }

    Args:
        x_organization_id (str):
        body (TraceFetch):

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
    body: TraceFetch,
    x_organization_id: str,
) -> Optional[Union[Any, HTTPValidationError]]:
    r"""Fetch Tool Calls

     Fetch all tool calls for a trace, returning only their input and output, sorted by call order
    (created_at).
    Args:
    fetch_data (TraceFetch): Request object containing:
        - trace_id (str): UUID of the trace to fetch tool calls from
    trace_client (TraceClient): Injected dependency for trace operations
    user_organization (UserOrganization): Injected dependency for auth/org validation

    Returns:
        Dict: Response object containing:
            - completions (List[Dict]): Alternating assistant/user messages where:
                - Assistant messages contain: {\"role\": \"assistant\", \"content\": {\"func\": str,
    \"args\": str}}
                - User messages contain: {\"role\": \"user\", \"content\": str} (tool output)
        Tool calls are sorted by created_at timestamp in ascending order.

    Example Response:
    {
        \"completions\": [
            {\"role\": \"assistant\", \"content\": {\"func\": \"search_web\", \"args\": \"{'query':
    'weather'}\"}},
            {\"role\": \"user\", \"content\": \"Sunny, 75°F\"},
            {\"role\": \"assistant\", \"content\": {\"func\": \"send_email\", \"args\": \"{'to':
    'user@example.com'}\"}},
            {\"role\": \"user\", \"content\": \"Email sent successfully\"}
        ]
    }

    Args:
        x_organization_id (str):
        body (TraceFetch):

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
    body: TraceFetch,
    x_organization_id: str,
) -> Response[Union[Any, HTTPValidationError]]:
    r"""Fetch Tool Calls

     Fetch all tool calls for a trace, returning only their input and output, sorted by call order
    (created_at).
    Args:
    fetch_data (TraceFetch): Request object containing:
        - trace_id (str): UUID of the trace to fetch tool calls from
    trace_client (TraceClient): Injected dependency for trace operations
    user_organization (UserOrganization): Injected dependency for auth/org validation

    Returns:
        Dict: Response object containing:
            - completions (List[Dict]): Alternating assistant/user messages where:
                - Assistant messages contain: {\"role\": \"assistant\", \"content\": {\"func\": str,
    \"args\": str}}
                - User messages contain: {\"role\": \"user\", \"content\": str} (tool output)
        Tool calls are sorted by created_at timestamp in ascending order.

    Example Response:
    {
        \"completions\": [
            {\"role\": \"assistant\", \"content\": {\"func\": \"search_web\", \"args\": \"{'query':
    'weather'}\"}},
            {\"role\": \"user\", \"content\": \"Sunny, 75°F\"},
            {\"role\": \"assistant\", \"content\": {\"func\": \"send_email\", \"args\": \"{'to':
    'user@example.com'}\"}},
            {\"role\": \"user\", \"content\": \"Email sent successfully\"}
        ]
    }

    Args:
        x_organization_id (str):
        body (TraceFetch):

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
    body: TraceFetch,
    x_organization_id: str,
) -> Optional[Union[Any, HTTPValidationError]]:
    r"""Fetch Tool Calls

     Fetch all tool calls for a trace, returning only their input and output, sorted by call order
    (created_at).
    Args:
    fetch_data (TraceFetch): Request object containing:
        - trace_id (str): UUID of the trace to fetch tool calls from
    trace_client (TraceClient): Injected dependency for trace operations
    user_organization (UserOrganization): Injected dependency for auth/org validation

    Returns:
        Dict: Response object containing:
            - completions (List[Dict]): Alternating assistant/user messages where:
                - Assistant messages contain: {\"role\": \"assistant\", \"content\": {\"func\": str,
    \"args\": str}}
                - User messages contain: {\"role\": \"user\", \"content\": str} (tool output)
        Tool calls are sorted by created_at timestamp in ascending order.

    Example Response:
    {
        \"completions\": [
            {\"role\": \"assistant\", \"content\": {\"func\": \"search_web\", \"args\": \"{'query':
    'weather'}\"}},
            {\"role\": \"user\", \"content\": \"Sunny, 75°F\"},
            {\"role\": \"assistant\", \"content\": {\"func\": \"send_email\", \"args\": \"{'to':
    'user@example.com'}\"}},
            {\"role\": \"user\", \"content\": \"Email sent successfully\"}
        ]
    }

    Args:
        x_organization_id (str):
        body (TraceFetch):

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
