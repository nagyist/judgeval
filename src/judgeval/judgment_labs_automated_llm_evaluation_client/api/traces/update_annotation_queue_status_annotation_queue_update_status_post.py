from http import HTTPStatus
from typing import Any, Optional, Union

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.annotation_queue_item import AnnotationQueueItem
from ...models.http_validation_error import HTTPValidationError
from ...models.update_annotation_queue_status_request import (
    UpdateAnnotationQueueStatusRequest,
)
from ...types import Response


def _get_kwargs(
    *,
    body: UpdateAnnotationQueueStatusRequest,
    x_organization_id: str,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    headers["X-Organization-Id"] = x_organization_id

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/annotation_queue/update_status/",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[AnnotationQueueItem, HTTPValidationError]]:
    if response.status_code == 200:
        response_200 = AnnotationQueueItem.from_dict(response.json())

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
) -> Response[Union[AnnotationQueueItem, HTTPValidationError]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient,
    body: UpdateAnnotationQueueStatusRequest,
    x_organization_id: str,
) -> Response[Union[AnnotationQueueItem, HTTPValidationError]]:
    """Update Annotation Queue Item Status

     Update the status of a specific item in the annotation queue for the user's organization.

    Args:
        x_organization_id (str):
        body (UpdateAnnotationQueueStatusRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[AnnotationQueueItem, HTTPValidationError]]
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
    body: UpdateAnnotationQueueStatusRequest,
    x_organization_id: str,
) -> Optional[Union[AnnotationQueueItem, HTTPValidationError]]:
    """Update Annotation Queue Item Status

     Update the status of a specific item in the annotation queue for the user's organization.

    Args:
        x_organization_id (str):
        body (UpdateAnnotationQueueStatusRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[AnnotationQueueItem, HTTPValidationError]
    """

    return sync_detailed(
        client=client,
        body=body,
        x_organization_id=x_organization_id,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient,
    body: UpdateAnnotationQueueStatusRequest,
    x_organization_id: str,
) -> Response[Union[AnnotationQueueItem, HTTPValidationError]]:
    """Update Annotation Queue Item Status

     Update the status of a specific item in the annotation queue for the user's organization.

    Args:
        x_organization_id (str):
        body (UpdateAnnotationQueueStatusRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[AnnotationQueueItem, HTTPValidationError]]
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
    body: UpdateAnnotationQueueStatusRequest,
    x_organization_id: str,
) -> Optional[Union[AnnotationQueueItem, HTTPValidationError]]:
    """Update Annotation Queue Item Status

     Update the status of a specific item in the annotation queue for the user's organization.

    Args:
        x_organization_id (str):
        body (UpdateAnnotationQueueStatusRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[AnnotationQueueItem, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
            x_organization_id=x_organization_id,
        )
    ).parsed
