from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.fetch_organization_id_by_project_id_response import (
    FetchOrganizationIdByProjectIdResponse,
)
from ...models.http_validation_error import HTTPValidationError
from ...models.project_id import ProjectId
from ...types import Response


def _get_kwargs(
    *,
    body: ProjectId,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/projects/fetch_org_id_from_project_id/",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[FetchOrganizationIdByProjectIdResponse, HTTPValidationError]]:
    if response.status_code == 200:
        response_200 = FetchOrganizationIdByProjectIdResponse.from_dict(response.json())

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
) -> Response[Union[FetchOrganizationIdByProjectIdResponse, HTTPValidationError]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient,
    body: ProjectId,
) -> Response[Union[FetchOrganizationIdByProjectIdResponse, HTTPValidationError]]:
    """Get Organization By Project Id

     Get the organization ID for a given project ID

    Args:
        body (ProjectId):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[FetchOrganizationIdByProjectIdResponse, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient,
    body: ProjectId,
) -> Optional[Union[FetchOrganizationIdByProjectIdResponse, HTTPValidationError]]:
    """Get Organization By Project Id

     Get the organization ID for a given project ID

    Args:
        body (ProjectId):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[FetchOrganizationIdByProjectIdResponse, HTTPValidationError]
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient,
    body: ProjectId,
) -> Response[Union[FetchOrganizationIdByProjectIdResponse, HTTPValidationError]]:
    """Get Organization By Project Id

     Get the organization ID for a given project ID

    Args:
        body (ProjectId):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[FetchOrganizationIdByProjectIdResponse, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient,
    body: ProjectId,
) -> Optional[Union[FetchOrganizationIdByProjectIdResponse, HTTPValidationError]]:
    """Get Organization By Project Id

     Get the organization ID for a given project ID

    Args:
        body (ProjectId):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[FetchOrganizationIdByProjectIdResponse, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
