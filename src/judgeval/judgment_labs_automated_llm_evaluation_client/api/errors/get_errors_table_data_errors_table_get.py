from http import HTTPStatus
from typing import Any, Optional, Union, cast

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.errors_table_response import ErrorsTableResponse
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    limit: Union[Unset, int] = 500,
    search: Union[None, Unset, str] = UNSET,
    sort_by: Union[Unset, str] = "created_at",
    sort_order: Union[Unset, str] = "desc",
    time_range: Union[Unset, str] = "7d",
    project_name: Union[None, Unset, str] = UNSET,
    organization_id: Union[None, Unset, str] = UNSET,
    x_organization_id: str,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    headers["X-Organization-Id"] = x_organization_id

    params: dict[str, Any] = {}

    params["limit"] = limit

    json_search: Union[None, Unset, str]
    if isinstance(search, Unset):
        json_search = UNSET
    else:
        json_search = search
    params["search"] = json_search

    params["sort_by"] = sort_by

    params["sort_order"] = sort_order

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
        "url": "/errors/table/",
        "params": params,
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[Any, ErrorsTableResponse, HTTPValidationError]]:
    if response.status_code == 200:
        response_200 = ErrorsTableResponse.from_dict(response.json())

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
) -> Response[Union[Any, ErrorsTableResponse, HTTPValidationError]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient,
    limit: Union[Unset, int] = 500,
    search: Union[None, Unset, str] = UNSET,
    sort_by: Union[Unset, str] = "created_at",
    sort_order: Union[Unset, str] = "desc",
    time_range: Union[Unset, str] = "7d",
    project_name: Union[None, Unset, str] = UNSET,
    organization_id: Union[None, Unset, str] = UNSET,
    x_organization_id: str,
) -> Response[Union[Any, ErrorsTableResponse, HTTPValidationError]]:
    """Get Errors Table Data

     Get detailed error data for the error table component with search and sorting capabilities.
    ⚡ Cached: Search/filter operations are instant after first load!

    Returns:
        ErrorsTableResponse: Detailed error data with search and sort metadata

    Args:
        limit (Union[Unset, int]): Maximum number of errors to return for table Default: 500.
        search (Union[None, Unset, str]): Search term to filter errors
        sort_by (Union[Unset, str]): Field to sort by Default: 'created_at'.
        sort_order (Union[Unset, str]): Sort order (asc or desc) Default: 'desc'.
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
        Response[Union[Any, ErrorsTableResponse, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        limit=limit,
        search=search,
        sort_by=sort_by,
        sort_order=sort_order,
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
    limit: Union[Unset, int] = 500,
    search: Union[None, Unset, str] = UNSET,
    sort_by: Union[Unset, str] = "created_at",
    sort_order: Union[Unset, str] = "desc",
    time_range: Union[Unset, str] = "7d",
    project_name: Union[None, Unset, str] = UNSET,
    organization_id: Union[None, Unset, str] = UNSET,
    x_organization_id: str,
) -> Optional[Union[Any, ErrorsTableResponse, HTTPValidationError]]:
    """Get Errors Table Data

     Get detailed error data for the error table component with search and sorting capabilities.
    ⚡ Cached: Search/filter operations are instant after first load!

    Returns:
        ErrorsTableResponse: Detailed error data with search and sort metadata

    Args:
        limit (Union[Unset, int]): Maximum number of errors to return for table Default: 500.
        search (Union[None, Unset, str]): Search term to filter errors
        sort_by (Union[Unset, str]): Field to sort by Default: 'created_at'.
        sort_order (Union[Unset, str]): Sort order (asc or desc) Default: 'desc'.
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
        Union[Any, ErrorsTableResponse, HTTPValidationError]
    """

    return sync_detailed(
        client=client,
        limit=limit,
        search=search,
        sort_by=sort_by,
        sort_order=sort_order,
        time_range=time_range,
        project_name=project_name,
        organization_id=organization_id,
        x_organization_id=x_organization_id,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient,
    limit: Union[Unset, int] = 500,
    search: Union[None, Unset, str] = UNSET,
    sort_by: Union[Unset, str] = "created_at",
    sort_order: Union[Unset, str] = "desc",
    time_range: Union[Unset, str] = "7d",
    project_name: Union[None, Unset, str] = UNSET,
    organization_id: Union[None, Unset, str] = UNSET,
    x_organization_id: str,
) -> Response[Union[Any, ErrorsTableResponse, HTTPValidationError]]:
    """Get Errors Table Data

     Get detailed error data for the error table component with search and sorting capabilities.
    ⚡ Cached: Search/filter operations are instant after first load!

    Returns:
        ErrorsTableResponse: Detailed error data with search and sort metadata

    Args:
        limit (Union[Unset, int]): Maximum number of errors to return for table Default: 500.
        search (Union[None, Unset, str]): Search term to filter errors
        sort_by (Union[Unset, str]): Field to sort by Default: 'created_at'.
        sort_order (Union[Unset, str]): Sort order (asc or desc) Default: 'desc'.
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
        Response[Union[Any, ErrorsTableResponse, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        limit=limit,
        search=search,
        sort_by=sort_by,
        sort_order=sort_order,
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
    limit: Union[Unset, int] = 500,
    search: Union[None, Unset, str] = UNSET,
    sort_by: Union[Unset, str] = "created_at",
    sort_order: Union[Unset, str] = "desc",
    time_range: Union[Unset, str] = "7d",
    project_name: Union[None, Unset, str] = UNSET,
    organization_id: Union[None, Unset, str] = UNSET,
    x_organization_id: str,
) -> Optional[Union[Any, ErrorsTableResponse, HTTPValidationError]]:
    """Get Errors Table Data

     Get detailed error data for the error table component with search and sorting capabilities.
    ⚡ Cached: Search/filter operations are instant after first load!

    Returns:
        ErrorsTableResponse: Detailed error data with search and sort metadata

    Args:
        limit (Union[Unset, int]): Maximum number of errors to return for table Default: 500.
        search (Union[None, Unset, str]): Search term to filter errors
        sort_by (Union[Unset, str]): Field to sort by Default: 'created_at'.
        sort_order (Union[Unset, str]): Sort order (asc or desc) Default: 'desc'.
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
        Union[Any, ErrorsTableResponse, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            client=client,
            limit=limit,
            search=search,
            sort_by=sort_by,
            sort_order=sort_order,
            time_range=time_range,
            project_name=project_name,
            organization_id=organization_id,
            x_organization_id=x_organization_id,
        )
    ).parsed
