from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.organization_usage_response import OrganizationUsageResponse
from ...types import Response


def _get_kwargs(
    *,
    x_organization_id: str,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    headers["X-Organization-Id"] = x_organization_id

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/organization/usage/",
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, OrganizationUsageResponse]]:
    if response.status_code == 200:
        response_200 = OrganizationUsageResponse.from_dict(response.json())

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
) -> Response[Union[HTTPValidationError, OrganizationUsageResponse]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient,
    x_organization_id: str,
) -> Response[Union[HTTPValidationError, OrganizationUsageResponse]]:
    """Get Organization Usage

     Get the usage statistics and limits for an organization

    Returns:
        OrganizationUsageResponse with usage data including:
        - judgee_limit, judgee_used, judgee_remaining
        - trace_limit, trace_used, trace_remaining
        - on_demand_judgee_used, on_demand_judgee_remaining
        - on_demand_trace_used, on_demand_trace_remaining
        - subscription_tier
        - usage_based_enabled
        - payg_cost_evals_cents, payg_cost_traces_cents

    Args:
        x_organization_id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, OrganizationUsageResponse]]
    """

    kwargs = _get_kwargs(
        x_organization_id=x_organization_id,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient,
    x_organization_id: str,
) -> Optional[Union[HTTPValidationError, OrganizationUsageResponse]]:
    """Get Organization Usage

     Get the usage statistics and limits for an organization

    Returns:
        OrganizationUsageResponse with usage data including:
        - judgee_limit, judgee_used, judgee_remaining
        - trace_limit, trace_used, trace_remaining
        - on_demand_judgee_used, on_demand_judgee_remaining
        - on_demand_trace_used, on_demand_trace_remaining
        - subscription_tier
        - usage_based_enabled
        - payg_cost_evals_cents, payg_cost_traces_cents

    Args:
        x_organization_id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, OrganizationUsageResponse]
    """

    return sync_detailed(
        client=client,
        x_organization_id=x_organization_id,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient,
    x_organization_id: str,
) -> Response[Union[HTTPValidationError, OrganizationUsageResponse]]:
    """Get Organization Usage

     Get the usage statistics and limits for an organization

    Returns:
        OrganizationUsageResponse with usage data including:
        - judgee_limit, judgee_used, judgee_remaining
        - trace_limit, trace_used, trace_remaining
        - on_demand_judgee_used, on_demand_judgee_remaining
        - on_demand_trace_used, on_demand_trace_remaining
        - subscription_tier
        - usage_based_enabled
        - payg_cost_evals_cents, payg_cost_traces_cents

    Args:
        x_organization_id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, OrganizationUsageResponse]]
    """

    kwargs = _get_kwargs(
        x_organization_id=x_organization_id,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient,
    x_organization_id: str,
) -> Optional[Union[HTTPValidationError, OrganizationUsageResponse]]:
    """Get Organization Usage

     Get the usage statistics and limits for an organization

    Returns:
        OrganizationUsageResponse with usage data including:
        - judgee_limit, judgee_used, judgee_remaining
        - trace_limit, trace_used, trace_remaining
        - on_demand_judgee_used, on_demand_judgee_remaining
        - on_demand_trace_used, on_demand_trace_remaining
        - subscription_tier
        - usage_based_enabled
        - payg_cost_evals_cents, payg_cost_traces_cents

    Args:
        x_organization_id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, OrganizationUsageResponse]
    """

    return (
        await asyncio_detailed(
            client=client,
            x_organization_id=x_organization_id,
        )
    ).parsed
