from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.spans_batch_request import SpansBatchRequest
from ...types import Response


def _get_kwargs(
    *,
    body: SpansBatchRequest,
    x_organization_id: str,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    headers["X-Organization-Id"] = x_organization_id

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/traces/spans/batch/",
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
    body: SpansBatchRequest,
    x_organization_id: str,
) -> Response[Union[Any, HTTPValidationError]]:
    """Process Spans Batch

     Process batched trace spans from the background service.

    This endpoint receives batched span data from the BackgroundSpanService and
    efficiently upserts them to the database via TraceSpanClient.

    Args:
        request_data: SpansBatchRequest containing spans and organization_id
        trace_span_client: Injected TraceSpanClient for database operations
        user_organization: Injected auth validation

    Returns:
        SpansBatchResponse: Success status and processing details

    Args:
        x_organization_id (str):
        body (SpansBatchRequest): Request model for batched spans from background service

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
    body: SpansBatchRequest,
    x_organization_id: str,
) -> Optional[Union[Any, HTTPValidationError]]:
    """Process Spans Batch

     Process batched trace spans from the background service.

    This endpoint receives batched span data from the BackgroundSpanService and
    efficiently upserts them to the database via TraceSpanClient.

    Args:
        request_data: SpansBatchRequest containing spans and organization_id
        trace_span_client: Injected TraceSpanClient for database operations
        user_organization: Injected auth validation

    Returns:
        SpansBatchResponse: Success status and processing details

    Args:
        x_organization_id (str):
        body (SpansBatchRequest): Request model for batched spans from background service

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
    body: SpansBatchRequest,
    x_organization_id: str,
) -> Response[Union[Any, HTTPValidationError]]:
    """Process Spans Batch

     Process batched trace spans from the background service.

    This endpoint receives batched span data from the BackgroundSpanService and
    efficiently upserts them to the database via TraceSpanClient.

    Args:
        request_data: SpansBatchRequest containing spans and organization_id
        trace_span_client: Injected TraceSpanClient for database operations
        user_organization: Injected auth validation

    Returns:
        SpansBatchResponse: Success status and processing details

    Args:
        x_organization_id (str):
        body (SpansBatchRequest): Request model for batched spans from background service

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
    body: SpansBatchRequest,
    x_organization_id: str,
) -> Optional[Union[Any, HTTPValidationError]]:
    """Process Spans Batch

     Process batched trace spans from the background service.

    This endpoint receives batched span data from the BackgroundSpanService and
    efficiently upserts them to the database via TraceSpanClient.

    Args:
        request_data: SpansBatchRequest containing spans and organization_id
        trace_span_client: Injected TraceSpanClient for database operations
        user_organization: Injected auth validation

    Returns:
        SpansBatchResponse: Success status and processing details

    Args:
        x_organization_id (str):
        body (SpansBatchRequest): Request model for batched spans from background service

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
