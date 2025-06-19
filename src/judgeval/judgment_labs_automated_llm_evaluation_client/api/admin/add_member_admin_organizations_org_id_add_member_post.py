from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.add_member_admin_organizations_org_id_add_member_post_body import (
    AddMemberAdminOrganizationsOrgIdAddMemberPostBody,
)
from ...models.http_validation_error import HTTPValidationError
from ...types import Response


def _get_kwargs(
    org_id: str,
    *,
    body: AddMemberAdminOrganizationsOrgIdAddMemberPostBody,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/admin/organizations/{org_id}/add_member".format(
            org_id=org_id,
        ),
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
    org_id: str,
    *,
    client: AuthenticatedClient,
    body: AddMemberAdminOrganizationsOrgIdAddMemberPostBody,
) -> Response[Union[Any, HTTPValidationError]]:
    """Add Member

     Add a user as a member to an organization (admin only)

    Args:
        org_id: The ID of the organization

    Body parameters:
        user_id: The ID of the user to add
        role: The role to assign ('admin', 'developer', or 'owner')

    Returns:
        Success message with user and organization details

    Args:
        org_id (str):
        body (AddMemberAdminOrganizationsOrgIdAddMemberPostBody):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        org_id=org_id,
        body=body,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    org_id: str,
    *,
    client: AuthenticatedClient,
    body: AddMemberAdminOrganizationsOrgIdAddMemberPostBody,
) -> Optional[Union[Any, HTTPValidationError]]:
    """Add Member

     Add a user as a member to an organization (admin only)

    Args:
        org_id: The ID of the organization

    Body parameters:
        user_id: The ID of the user to add
        role: The role to assign ('admin', 'developer', or 'owner')

    Returns:
        Success message with user and organization details

    Args:
        org_id (str):
        body (AddMemberAdminOrganizationsOrgIdAddMemberPostBody):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, HTTPValidationError]
    """

    return sync_detailed(
        org_id=org_id,
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    org_id: str,
    *,
    client: AuthenticatedClient,
    body: AddMemberAdminOrganizationsOrgIdAddMemberPostBody,
) -> Response[Union[Any, HTTPValidationError]]:
    """Add Member

     Add a user as a member to an organization (admin only)

    Args:
        org_id: The ID of the organization

    Body parameters:
        user_id: The ID of the user to add
        role: The role to assign ('admin', 'developer', or 'owner')

    Returns:
        Success message with user and organization details

    Args:
        org_id (str):
        body (AddMemberAdminOrganizationsOrgIdAddMemberPostBody):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        org_id=org_id,
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    org_id: str,
    *,
    client: AuthenticatedClient,
    body: AddMemberAdminOrganizationsOrgIdAddMemberPostBody,
) -> Optional[Union[Any, HTTPValidationError]]:
    """Add Member

     Add a user as a member to an organization (admin only)

    Args:
        org_id: The ID of the organization

    Body parameters:
        user_id: The ID of the user to add
        role: The role to assign ('admin', 'developer', or 'owner')

    Returns:
        Success message with user and organization details

    Args:
        org_id (str):
        body (AddMemberAdminOrganizationsOrgIdAddMemberPostBody):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            org_id=org_id,
            client=client,
            body=body,
        )
    ).parsed
