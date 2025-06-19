from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.slack_command_response import SlackCommandResponse
from ...types import Response


def _get_kwargs() -> dict[str, Any]:
    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/slack/commands/add-mention",
    }

    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[SlackCommandResponse]:
    if response.status_code == 200:
        response_200 = SlackCommandResponse.from_dict(response.json())

        return response_200
    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Response[SlackCommandResponse]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
) -> Response[SlackCommandResponse]:
    """Add Mention

     Handle /add-mention Slack command.

    Adds a user to be mentioned in notifications.

    Request body:
        text: Username to add for mentions

    Returns:
        SlackCommandResponse: Response containing confirmation message

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[SlackCommandResponse]
    """

    kwargs = _get_kwargs()

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: Union[AuthenticatedClient, Client],
) -> Optional[SlackCommandResponse]:
    """Add Mention

     Handle /add-mention Slack command.

    Adds a user to be mentioned in notifications.

    Request body:
        text: Username to add for mentions

    Returns:
        SlackCommandResponse: Response containing confirmation message

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        SlackCommandResponse
    """

    return sync_detailed(
        client=client,
    ).parsed


async def asyncio_detailed(
    *,
    client: Union[AuthenticatedClient, Client],
) -> Response[SlackCommandResponse]:
    """Add Mention

     Handle /add-mention Slack command.

    Adds a user to be mentioned in notifications.

    Request body:
        text: Username to add for mentions

    Returns:
        SlackCommandResponse: Response containing confirmation message

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[SlackCommandResponse]
    """

    kwargs = _get_kwargs()

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: Union[AuthenticatedClient, Client],
) -> Optional[SlackCommandResponse]:
    """Add Mention

     Handle /add-mention Slack command.

    Adds a user to be mentioned in notifications.

    Request body:
        text: Username to add for mentions

    Returns:
        SlackCommandResponse: Response containing confirmation message

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        SlackCommandResponse
    """

    return (
        await asyncio_detailed(
            client=client,
        )
    ).parsed
