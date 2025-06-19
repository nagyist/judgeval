from http import HTTPStatus
from typing import Any, Optional, Union, cast

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.latency_metrics_response import LatencyMetricsResponse
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    organization_id: Union[None, Unset, str] = UNSET,
    time_range: Union[Unset, str] = "7d",
    project_name: Union[None, Unset, str] = UNSET,
    x_organization_id: str,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    headers["X-Organization-Id"] = x_organization_id

    params: dict[str, Any] = {}

    json_organization_id: Union[None, Unset, str]
    if isinstance(organization_id, Unset):
        json_organization_id = UNSET
    else:
        json_organization_id = organization_id
    params["organization_id"] = json_organization_id

    params["time_range"] = time_range

    json_project_name: Union[None, Unset, str]
    if isinstance(project_name, Unset):
        json_project_name = UNSET
    else:
        json_project_name = project_name
    params["project_name"] = json_project_name

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/dashboard/latency/",
        "params": params,
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[Any, HTTPValidationError, LatencyMetricsResponse]]:
    if response.status_code == 200:
        response_200 = LatencyMetricsResponse.from_dict(response.json())

        return response_200
    if response.status_code == 404:
        response_404 = cast(Any, None)
        return response_404
    if response.status_code == 422:
        response_422 = HTTPValidationError.from_dict(response.json())

        return response_422
    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Response[Union[Any, HTTPValidationError, LatencyMetricsResponse]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient,
    organization_id: Union[None, Unset, str] = UNSET,
    time_range: Union[Unset, str] = "7d",
    project_name: Union[None, Unset, str] = UNSET,
    x_organization_id: str,
) -> Response[Union[Any, HTTPValidationError, LatencyMetricsResponse]]:
    """Get Latency Metrics

     Get latency metrics for an organization.
    This endpoint provides detailed latency metrics for traces, LLM calls, and spans.

    Args:
        organization_id (Union[None, Unset, str]): Organization ID
        time_range (Union[Unset, str]): Time range for metrics (1h, 24h, 7d, 30d, all) Default:
            '7d'.
        project_name (Union[None, Unset, str]): Optional project name filter
        x_organization_id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, HTTPValidationError, LatencyMetricsResponse]]
    """

    kwargs = _get_kwargs(
        organization_id=organization_id,
        time_range=time_range,
        project_name=project_name,
        x_organization_id=x_organization_id,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient,
    organization_id: Union[None, Unset, str] = UNSET,
    time_range: Union[Unset, str] = "7d",
    project_name: Union[None, Unset, str] = UNSET,
    x_organization_id: str,
) -> Optional[Union[Any, HTTPValidationError, LatencyMetricsResponse]]:
    """Get Latency Metrics

     Get latency metrics for an organization.
    This endpoint provides detailed latency metrics for traces, LLM calls, and spans.

    Args:
        organization_id (Union[None, Unset, str]): Organization ID
        time_range (Union[Unset, str]): Time range for metrics (1h, 24h, 7d, 30d, all) Default:
            '7d'.
        project_name (Union[None, Unset, str]): Optional project name filter
        x_organization_id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, HTTPValidationError, LatencyMetricsResponse]
    """

    return sync_detailed(
        client=client,
        organization_id=organization_id,
        time_range=time_range,
        project_name=project_name,
        x_organization_id=x_organization_id,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient,
    organization_id: Union[None, Unset, str] = UNSET,
    time_range: Union[Unset, str] = "7d",
    project_name: Union[None, Unset, str] = UNSET,
    x_organization_id: str,
) -> Response[Union[Any, HTTPValidationError, LatencyMetricsResponse]]:
    """Get Latency Metrics

     Get latency metrics for an organization.
    This endpoint provides detailed latency metrics for traces, LLM calls, and spans.

    Args:
        organization_id (Union[None, Unset, str]): Organization ID
        time_range (Union[Unset, str]): Time range for metrics (1h, 24h, 7d, 30d, all) Default:
            '7d'.
        project_name (Union[None, Unset, str]): Optional project name filter
        x_organization_id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, HTTPValidationError, LatencyMetricsResponse]]
    """

    kwargs = _get_kwargs(
        organization_id=organization_id,
        time_range=time_range,
        project_name=project_name,
        x_organization_id=x_organization_id,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient,
    organization_id: Union[None, Unset, str] = UNSET,
    time_range: Union[Unset, str] = "7d",
    project_name: Union[None, Unset, str] = UNSET,
    x_organization_id: str,
) -> Optional[Union[Any, HTTPValidationError, LatencyMetricsResponse]]:
    """Get Latency Metrics

     Get latency metrics for an organization.
    This endpoint provides detailed latency metrics for traces, LLM calls, and spans.

    Args:
        organization_id (Union[None, Unset, str]): Organization ID
        time_range (Union[Unset, str]): Time range for metrics (1h, 24h, 7d, 30d, all) Default:
            '7d'.
        project_name (Union[None, Unset, str]): Optional project name filter
        x_organization_id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, HTTPValidationError, LatencyMetricsResponse]
    """

    return (
        await asyncio_detailed(
            client=client,
            organization_id=organization_id,
            time_range=time_range,
            project_name=project_name,
            x_organization_id=x_organization_id,
        )
    ).parsed
