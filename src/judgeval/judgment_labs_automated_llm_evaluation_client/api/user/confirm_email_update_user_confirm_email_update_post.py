from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.confirm_email_update_request import ConfirmEmailUpdateRequest
from ...models.confirm_email_update_response import ConfirmEmailUpdateResponse
from ...models.http_validation_error import HTTPValidationError
from ...types import Response


def _get_kwargs(
    *,
    body: ConfirmEmailUpdateRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/user/confirm_email_update/",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[ConfirmEmailUpdateResponse, HTTPValidationError]]:
    if response.status_code == 200:
        response_200 = ConfirmEmailUpdateResponse.from_dict(response.json())

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
) -> Response[Union[ConfirmEmailUpdateResponse, HTTPValidationError]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    body: ConfirmEmailUpdateRequest,
) -> Response[Union[ConfirmEmailUpdateResponse, HTTPValidationError]]:
    """Confirm Email Update

     Confirm an email update by verifying the session and updating the user_data table.
    Args:
        request: Request body containing access_token and refresh_token
    Returns:
        JSON response confirming the email update

    Args:
        body (ConfirmEmailUpdateRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[ConfirmEmailUpdateResponse, HTTPValidationError]]
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
    client: Union[AuthenticatedClient, Client],
    body: ConfirmEmailUpdateRequest,
) -> Optional[Union[ConfirmEmailUpdateResponse, HTTPValidationError]]:
    """Confirm Email Update

     Confirm an email update by verifying the session and updating the user_data table.
    Args:
        request: Request body containing access_token and refresh_token
    Returns:
        JSON response confirming the email update

    Args:
        body (ConfirmEmailUpdateRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[ConfirmEmailUpdateResponse, HTTPValidationError]
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
    body: ConfirmEmailUpdateRequest,
) -> Response[Union[ConfirmEmailUpdateResponse, HTTPValidationError]]:
    """Confirm Email Update

     Confirm an email update by verifying the session and updating the user_data table.
    Args:
        request: Request body containing access_token and refresh_token
    Returns:
        JSON response confirming the email update

    Args:
        body (ConfirmEmailUpdateRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[ConfirmEmailUpdateResponse, HTTPValidationError]]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
    body: ConfirmEmailUpdateRequest,
) -> Optional[Union[ConfirmEmailUpdateResponse, HTTPValidationError]]:
    """Confirm Email Update

     Confirm an email update by verifying the session and updating the user_data table.
    Args:
        request: Request body containing access_token and refresh_token
    Returns:
        JSON response confirming the email update

    Args:
        body (ConfirmEmailUpdateRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[ConfirmEmailUpdateResponse, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
