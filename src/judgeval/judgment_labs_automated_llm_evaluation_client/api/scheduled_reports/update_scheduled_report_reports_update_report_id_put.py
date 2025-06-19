from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.scheduled_report_create import ScheduledReportCreate
from ...models.scheduled_report_response import ScheduledReportResponse
from ...types import Response


def _get_kwargs(
    report_id: str,
    *,
    body: ScheduledReportCreate,
    x_organization_id: str,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    headers["X-Organization-Id"] = x_organization_id

    _kwargs: dict[str, Any] = {
        "method": "put",
        "url": "/reports/update/{report_id}".format(
            report_id=report_id,
        ),
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[HTTPValidationError, ScheduledReportResponse]]:
    if response.status_code == 200:
        response_200 = ScheduledReportResponse.from_dict(response.json())

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
) -> Response[Union[HTTPValidationError, ScheduledReportResponse]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    report_id: str,
    *,
    client: AuthenticatedClient,
    body: ScheduledReportCreate,
    x_organization_id: str,
) -> Response[Union[HTTPValidationError, ScheduledReportResponse]]:
    """Update Scheduled Report

     Update an existing scheduled report

    Args:
        report_id (str):
        x_organization_id (str):
        body (ScheduledReportCreate):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, ScheduledReportResponse]]
    """

    kwargs = _get_kwargs(
        report_id=report_id,
        body=body,
        x_organization_id=x_organization_id,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    report_id: str,
    *,
    client: AuthenticatedClient,
    body: ScheduledReportCreate,
    x_organization_id: str,
) -> Optional[Union[HTTPValidationError, ScheduledReportResponse]]:
    """Update Scheduled Report

     Update an existing scheduled report

    Args:
        report_id (str):
        x_organization_id (str):
        body (ScheduledReportCreate):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, ScheduledReportResponse]
    """

    return sync_detailed(
        report_id=report_id,
        client=client,
        body=body,
        x_organization_id=x_organization_id,
    ).parsed


async def asyncio_detailed(
    report_id: str,
    *,
    client: AuthenticatedClient,
    body: ScheduledReportCreate,
    x_organization_id: str,
) -> Response[Union[HTTPValidationError, ScheduledReportResponse]]:
    """Update Scheduled Report

     Update an existing scheduled report

    Args:
        report_id (str):
        x_organization_id (str):
        body (ScheduledReportCreate):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[HTTPValidationError, ScheduledReportResponse]]
    """

    kwargs = _get_kwargs(
        report_id=report_id,
        body=body,
        x_organization_id=x_organization_id,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    report_id: str,
    *,
    client: AuthenticatedClient,
    body: ScheduledReportCreate,
    x_organization_id: str,
) -> Optional[Union[HTTPValidationError, ScheduledReportResponse]]:
    """Update Scheduled Report

     Update an existing scheduled report

    Args:
        report_id (str):
        x_organization_id (str):
        body (ScheduledReportCreate):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[HTTPValidationError, ScheduledReportResponse]
    """

    return (
        await asyncio_detailed(
            report_id=report_id,
            client=client,
            body=body,
            x_organization_id=x_organization_id,
        )
    ).parsed
