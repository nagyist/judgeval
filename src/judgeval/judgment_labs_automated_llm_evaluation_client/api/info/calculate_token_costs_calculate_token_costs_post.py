from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.token_count_request import TokenCountRequest
from ...types import Response


def _get_kwargs(
    *,
    body: TokenCountRequest,
    x_organization_id: str,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    headers["X-Organization-Id"] = x_organization_id

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/calculate-token-costs",
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
    body: TokenCountRequest,
    x_organization_id: str,
) -> Response[Union[Any, HTTPValidationError]]:
    """Calculate Token Costs

     Calculates token costs for a specific model and token counts.

    Args:
        request: TokenCountRequest containing:
            - model: The model name to calculate costs for
            - prompt_tokens: Number of prompt tokens
            - completion_tokens: Number of completion tokens

    Returns:
        JSON object containing:
        - model: Model name
        - prompt_tokens: Number of tokens in prompts
        - completion_tokens: Number of tokens in completions
        - total_tokens: Total token count
        - prompt_tokens_cost_usd: Cost of prompt tokens in USD
        - completion_tokens_cost_usd: Cost of completion tokens in USD
        - total_cost_usd: Total cost in USD

    Args:
        x_organization_id (str):
        body (TokenCountRequest):

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
    body: TokenCountRequest,
    x_organization_id: str,
) -> Optional[Union[Any, HTTPValidationError]]:
    """Calculate Token Costs

     Calculates token costs for a specific model and token counts.

    Args:
        request: TokenCountRequest containing:
            - model: The model name to calculate costs for
            - prompt_tokens: Number of prompt tokens
            - completion_tokens: Number of completion tokens

    Returns:
        JSON object containing:
        - model: Model name
        - prompt_tokens: Number of tokens in prompts
        - completion_tokens: Number of tokens in completions
        - total_tokens: Total token count
        - prompt_tokens_cost_usd: Cost of prompt tokens in USD
        - completion_tokens_cost_usd: Cost of completion tokens in USD
        - total_cost_usd: Total cost in USD

    Args:
        x_organization_id (str):
        body (TokenCountRequest):

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
    body: TokenCountRequest,
    x_organization_id: str,
) -> Response[Union[Any, HTTPValidationError]]:
    """Calculate Token Costs

     Calculates token costs for a specific model and token counts.

    Args:
        request: TokenCountRequest containing:
            - model: The model name to calculate costs for
            - prompt_tokens: Number of prompt tokens
            - completion_tokens: Number of completion tokens

    Returns:
        JSON object containing:
        - model: Model name
        - prompt_tokens: Number of tokens in prompts
        - completion_tokens: Number of tokens in completions
        - total_tokens: Total token count
        - prompt_tokens_cost_usd: Cost of prompt tokens in USD
        - completion_tokens_cost_usd: Cost of completion tokens in USD
        - total_cost_usd: Total cost in USD

    Args:
        x_organization_id (str):
        body (TokenCountRequest):

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
    body: TokenCountRequest,
    x_organization_id: str,
) -> Optional[Union[Any, HTTPValidationError]]:
    """Calculate Token Costs

     Calculates token costs for a specific model and token counts.

    Args:
        request: TokenCountRequest containing:
            - model: The model name to calculate costs for
            - prompt_tokens: Number of prompt tokens
            - completion_tokens: Number of completion tokens

    Returns:
        JSON object containing:
        - model: Model name
        - prompt_tokens: Number of tokens in prompts
        - completion_tokens: Number of tokens in completions
        - total_tokens: Total token count
        - prompt_tokens_cost_usd: Cost of prompt tokens in USD
        - completion_tokens_cost_usd: Cost of completion tokens in USD
        - total_cost_usd: Total cost in USD

    Args:
        x_organization_id (str):
        body (TokenCountRequest):

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
