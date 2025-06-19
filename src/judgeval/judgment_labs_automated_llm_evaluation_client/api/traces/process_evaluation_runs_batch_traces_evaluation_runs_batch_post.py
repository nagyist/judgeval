from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.evaluation_runs_batch_request import EvaluationRunsBatchRequest
from ...models.http_validation_error import HTTPValidationError
from ...types import Response


def _get_kwargs(
    *,
    body: EvaluationRunsBatchRequest,
    x_organization_id: str,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    headers["X-Organization-Id"] = x_organization_id

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/traces/evaluation_runs/batch/",
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
    body: EvaluationRunsBatchRequest,
    x_organization_id: str,
) -> Response[Union[Any, HTTPValidationError]]:
    """Process Evaluation Runs Batch

     Process batched evaluation runs from the background service.

    This endpoint receives batched evaluation run data from the BackgroundSpanService,
    first upserts the associated span data to avoid race conditions, then queues
    the evaluation runs for async processing via RabbitMQ.

    Args:
        request_data: EvaluationRunsBatchRequest containing evaluation entries
        trace_span_client: Injected TraceSpanClient for span upserts
        rabbit_client: Injected RabbitMQ client for queueing
        user_organization: Injected auth validation

    Returns:
        EvaluationRunsBatchResponse: Success status and processing details

    Args:
        x_organization_id (str):
        body (EvaluationRunsBatchRequest): Request model for batched evaluation runs from
            background service

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
    body: EvaluationRunsBatchRequest,
    x_organization_id: str,
) -> Optional[Union[Any, HTTPValidationError]]:
    """Process Evaluation Runs Batch

     Process batched evaluation runs from the background service.

    This endpoint receives batched evaluation run data from the BackgroundSpanService,
    first upserts the associated span data to avoid race conditions, then queues
    the evaluation runs for async processing via RabbitMQ.

    Args:
        request_data: EvaluationRunsBatchRequest containing evaluation entries
        trace_span_client: Injected TraceSpanClient for span upserts
        rabbit_client: Injected RabbitMQ client for queueing
        user_organization: Injected auth validation

    Returns:
        EvaluationRunsBatchResponse: Success status and processing details

    Args:
        x_organization_id (str):
        body (EvaluationRunsBatchRequest): Request model for batched evaluation runs from
            background service

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
    body: EvaluationRunsBatchRequest,
    x_organization_id: str,
) -> Response[Union[Any, HTTPValidationError]]:
    """Process Evaluation Runs Batch

     Process batched evaluation runs from the background service.

    This endpoint receives batched evaluation run data from the BackgroundSpanService,
    first upserts the associated span data to avoid race conditions, then queues
    the evaluation runs for async processing via RabbitMQ.

    Args:
        request_data: EvaluationRunsBatchRequest containing evaluation entries
        trace_span_client: Injected TraceSpanClient for span upserts
        rabbit_client: Injected RabbitMQ client for queueing
        user_organization: Injected auth validation

    Returns:
        EvaluationRunsBatchResponse: Success status and processing details

    Args:
        x_organization_id (str):
        body (EvaluationRunsBatchRequest): Request model for batched evaluation runs from
            background service

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
    body: EvaluationRunsBatchRequest,
    x_organization_id: str,
) -> Optional[Union[Any, HTTPValidationError]]:
    """Process Evaluation Runs Batch

     Process batched evaluation runs from the background service.

    This endpoint receives batched evaluation run data from the BackgroundSpanService,
    first upserts the associated span data to avoid race conditions, then queues
    the evaluation runs for async processing via RabbitMQ.

    Args:
        request_data: EvaluationRunsBatchRequest containing evaluation entries
        trace_span_client: Injected TraceSpanClient for span upserts
        rabbit_client: Injected RabbitMQ client for queueing
        user_organization: Injected auth validation

    Returns:
        EvaluationRunsBatchResponse: Success status and processing details

    Args:
        x_organization_id (str):
        body (EvaluationRunsBatchRequest): Request model for batched evaluation runs from
            background service

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
