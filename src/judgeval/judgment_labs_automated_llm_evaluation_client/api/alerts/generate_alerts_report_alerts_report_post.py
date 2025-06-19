from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.alert_report_request import AlertReportRequest
from ...models.http_validation_error import HTTPValidationError
from ...types import Response


def _get_kwargs(
    *,
    body: AlertReportRequest,
    x_organization_id: str,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    headers["X-Organization-Id"] = x_organization_id

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/alerts/report/",
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
    body: AlertReportRequest,
    x_organization_id: str,
) -> Response[Union[Any, HTTPValidationError]]:
    """Generate Alerts Report

     Generate a comprehensive report on alerts data for a given time period.

    Args:
        request: AlertReportRequest containing:
            - start_date: ISO format start date (YYYY-MM-DDTHH:MM:SS)
            - end_date: ISO format end date (YYYY-MM-DDTHH:MM:SS), defaults to current time
            - project_name: Optional project name to filter alerts
            - comparison_period: If True, will include comparison with the previous time period

    Returns:
        A comprehensive report with statistics about alerts, traces, and metrics

    Args:
        x_organization_id (str):
        body (AlertReportRequest):

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
    body: AlertReportRequest,
    x_organization_id: str,
) -> Optional[Union[Any, HTTPValidationError]]:
    """Generate Alerts Report

     Generate a comprehensive report on alerts data for a given time period.

    Args:
        request: AlertReportRequest containing:
            - start_date: ISO format start date (YYYY-MM-DDTHH:MM:SS)
            - end_date: ISO format end date (YYYY-MM-DDTHH:MM:SS), defaults to current time
            - project_name: Optional project name to filter alerts
            - comparison_period: If True, will include comparison with the previous time period

    Returns:
        A comprehensive report with statistics about alerts, traces, and metrics

    Args:
        x_organization_id (str):
        body (AlertReportRequest):

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
    body: AlertReportRequest,
    x_organization_id: str,
) -> Response[Union[Any, HTTPValidationError]]:
    """Generate Alerts Report

     Generate a comprehensive report on alerts data for a given time period.

    Args:
        request: AlertReportRequest containing:
            - start_date: ISO format start date (YYYY-MM-DDTHH:MM:SS)
            - end_date: ISO format end date (YYYY-MM-DDTHH:MM:SS), defaults to current time
            - project_name: Optional project name to filter alerts
            - comparison_period: If True, will include comparison with the previous time period

    Returns:
        A comprehensive report with statistics about alerts, traces, and metrics

    Args:
        x_organization_id (str):
        body (AlertReportRequest):

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
    body: AlertReportRequest,
    x_organization_id: str,
) -> Optional[Union[Any, HTTPValidationError]]:
    """Generate Alerts Report

     Generate a comprehensive report on alerts data for a given time period.

    Args:
        request: AlertReportRequest containing:
            - start_date: ISO format start date (YYYY-MM-DDTHH:MM:SS)
            - end_date: ISO format end date (YYYY-MM-DDTHH:MM:SS), defaults to current time
            - project_name: Optional project name to filter alerts
            - comparison_period: If True, will include comparison with the previous time period

    Returns:
        A comprehensive report with statistics about alerts, traces, and metrics

    Args:
        x_organization_id (str):
        body (AlertReportRequest):

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
