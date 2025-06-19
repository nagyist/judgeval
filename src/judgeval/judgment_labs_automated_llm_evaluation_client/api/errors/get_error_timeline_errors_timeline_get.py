from http import HTTPStatus
from typing import Any, Optional, Union, cast

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error_timeline_response import ErrorTimelineResponse
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    time_range: Union[Unset, str] = "7d",
    project_name: Union[None, Unset, str] = UNSET,
    organization_id: Union[None, Unset, str] = UNSET,
    x_organization_id: str,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    headers["X-Organization-Id"] = x_organization_id

    params: dict[str, Any] = {}

    params["time_range"] = time_range

    json_project_name: Union[None, Unset, str]
    if isinstance(project_name, Unset):
        json_project_name = UNSET
    else:
        json_project_name = project_name
    params["project_name"] = json_project_name

    json_organization_id: Union[None, Unset, str]
    if isinstance(organization_id, Unset):
        json_organization_id = UNSET
    else:
        json_organization_id = organization_id
    params["organization_id"] = json_organization_id

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/errors/timeline/",
        "params": params,
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[Any, ErrorTimelineResponse, HTTPValidationError]]:
    if response.status_code == 200:
        response_200 = ErrorTimelineResponse.from_dict(response.json())

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
) -> Response[Union[Any, ErrorTimelineResponse, HTTPValidationError]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient,
    time_range: Union[Unset, str] = "7d",
    project_name: Union[None, Unset, str] = UNSET,
    organization_id: Union[None, Unset, str] = UNSET,
    x_organization_id: str,
) -> Response[Union[Any, ErrorTimelineResponse, HTTPValidationError]]:
    """Get Error Timeline

     Get error timeline data aggregated by time periods and error types.
    ⚡ Cached: Timeline processing is instant after first load!

    Returns:
        ErrorTimelineResponse: Timeline data with error counts and error types

    Args:
        time_range (Union[Unset, str]): Time range for errors (1h, 24h, 7d, 30d, all) Default:
            '7d'.
        project_name (Union[None, Unset, str]): Optional project name filter
        organization_id (Union[None, Unset, str]): Organization ID (optional, defaults to
            authenticated user's org)
        x_organization_id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, ErrorTimelineResponse, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        time_range=time_range,
        project_name=project_name,
        organization_id=organization_id,
        x_organization_id=x_organization_id,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient,
    time_range: Union[Unset, str] = "7d",
    project_name: Union[None, Unset, str] = UNSET,
    organization_id: Union[None, Unset, str] = UNSET,
    x_organization_id: str,
) -> Optional[Union[Any, ErrorTimelineResponse, HTTPValidationError]]:
    """Get Error Timeline

     Get error timeline data aggregated by time periods and error types.
    ⚡ Cached: Timeline processing is instant after first load!

    Returns:
        ErrorTimelineResponse: Timeline data with error counts and error types

    Args:
        time_range (Union[Unset, str]): Time range for errors (1h, 24h, 7d, 30d, all) Default:
            '7d'.
        project_name (Union[None, Unset, str]): Optional project name filter
        organization_id (Union[None, Unset, str]): Organization ID (optional, defaults to
            authenticated user's org)
        x_organization_id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, ErrorTimelineResponse, HTTPValidationError]
    """

    return sync_detailed(
        client=client,
        time_range=time_range,
        project_name=project_name,
        organization_id=organization_id,
        x_organization_id=x_organization_id,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient,
    time_range: Union[Unset, str] = "7d",
    project_name: Union[None, Unset, str] = UNSET,
    organization_id: Union[None, Unset, str] = UNSET,
    x_organization_id: str,
) -> Response[Union[Any, ErrorTimelineResponse, HTTPValidationError]]:
    """Get Error Timeline

     Get error timeline data aggregated by time periods and error types.
    ⚡ Cached: Timeline processing is instant after first load!

    Returns:
        ErrorTimelineResponse: Timeline data with error counts and error types

    Args:
        time_range (Union[Unset, str]): Time range for errors (1h, 24h, 7d, 30d, all) Default:
            '7d'.
        project_name (Union[None, Unset, str]): Optional project name filter
        organization_id (Union[None, Unset, str]): Organization ID (optional, defaults to
            authenticated user's org)
        x_organization_id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, ErrorTimelineResponse, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        time_range=time_range,
        project_name=project_name,
        organization_id=organization_id,
        x_organization_id=x_organization_id,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient,
    time_range: Union[Unset, str] = "7d",
    project_name: Union[None, Unset, str] = UNSET,
    organization_id: Union[None, Unset, str] = UNSET,
    x_organization_id: str,
) -> Optional[Union[Any, ErrorTimelineResponse, HTTPValidationError]]:
    """Get Error Timeline

     Get error timeline data aggregated by time periods and error types.
    ⚡ Cached: Timeline processing is instant after first load!

    Returns:
        ErrorTimelineResponse: Timeline data with error counts and error types

    Args:
        time_range (Union[Unset, str]): Time range for errors (1h, 24h, 7d, 30d, all) Default:
            '7d'.
        project_name (Union[None, Unset, str]): Optional project name filter
        organization_id (Union[None, Unset, str]): Organization ID (optional, defaults to
            authenticated user's org)
        x_organization_id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, ErrorTimelineResponse, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            client=client,
            time_range=time_range,
            project_name=project_name,
            organization_id=organization_id,
            x_organization_id=x_organization_id,
        )
    ).parsed
