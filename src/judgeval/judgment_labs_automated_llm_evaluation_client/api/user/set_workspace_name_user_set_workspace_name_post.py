from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.set_workspace_name_request import SetWorkspaceNameRequest
from ...models.set_workspace_name_response import SetWorkspaceNameResponse
from ...types import Response


def _get_kwargs(
    *,
    body: SetWorkspaceNameRequest,
    x_organization_id: str,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    headers["X-Organization-Id"] = x_organization_id

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/user/set_workspace_name/",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, SetWorkspaceNameResponse]]:
    if response.status_code == 200:
        response_200 = SetWorkspaceNameResponse.from_dict(response.json())

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
) -> Response[Union[HTTPValidationError, SetWorkspaceNameResponse]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient,
    body: SetWorkspaceNameRequest,
    x_organization_id: str,
) -> Response[Union[HTTPValidationError, SetWorkspaceNameResponse]]:
    """Set Workspace Name

     Set the workspace name for a user's organization

    Args:
        request: Request body containing workspace_name
        user_org: User organization data from auth

    Returns:
        JSON response confirming the update

    Args:
        x_organization_id (str):
        body (SetWorkspaceNameRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, SetWorkspaceNameResponse]]
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
    body: SetWorkspaceNameRequest,
    x_organization_id: str,
) -> Optional[Union[HTTPValidationError, SetWorkspaceNameResponse]]:
    """Set Workspace Name

     Set the workspace name for a user's organization

    Args:
        request: Request body containing workspace_name
        user_org: User organization data from auth

    Returns:
        JSON response confirming the update

    Args:
        x_organization_id (str):
        body (SetWorkspaceNameRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, SetWorkspaceNameResponse]
    """

    return sync_detailed(
        client=client,
        body=body,
        x_organization_id=x_organization_id,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient,
    body: SetWorkspaceNameRequest,
    x_organization_id: str,
) -> Response[Union[HTTPValidationError, SetWorkspaceNameResponse]]:
    """Set Workspace Name

     Set the workspace name for a user's organization

    Args:
        request: Request body containing workspace_name
        user_org: User organization data from auth

    Returns:
        JSON response confirming the update

    Args:
        x_organization_id (str):
        body (SetWorkspaceNameRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, SetWorkspaceNameResponse]]
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
    body: SetWorkspaceNameRequest,
    x_organization_id: str,
) -> Optional[Union[HTTPValidationError, SetWorkspaceNameResponse]]:
    """Set Workspace Name

     Set the workspace name for a user's organization

    Args:
        request: Request body containing workspace_name
        user_org: User organization data from auth

    Returns:
        JSON response confirming the update

    Args:
        x_organization_id (str):
        body (SetWorkspaceNameRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, SetWorkspaceNameResponse]
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
            x_organization_id=x_organization_id,
        )
    ).parsed
