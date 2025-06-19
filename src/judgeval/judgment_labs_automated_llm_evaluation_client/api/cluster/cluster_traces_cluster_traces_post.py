from http import HTTPStatus
from typing import Any, Optional, Union, cast

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.cluster_traces import ClusterTraces
from ...models.clustering_result import ClusteringResult
from ...models.http_validation_error import HTTPValidationError
from ...types import Response


def _get_kwargs(
    *,
    body: ClusterTraces,
    x_organization_id: str,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    headers["X-Organization-Id"] = x_organization_id

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/cluster/traces/",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Optional[Union[Any, ClusteringResult, HTTPValidationError]]:
    if response.status_code == 200:
        response_200 = ClusteringResult.from_dict(response.json())

        return response_200
    if response.status_code == 404:
        response_404 = cast(Any, None)
        return response_404
    if response.status_code == 422:
        response_422 = HTTPValidationError.from_dict(response.json())

        return response_422
    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: Union[AuthenticatedClient, Client], response: httpx.Response
) -> Response[Union[Any, ClusteringResult, HTTPValidationError]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient,
    body: ClusterTraces,
    x_organization_id: str,
) -> Response[Union[Any, ClusteringResult, HTTPValidationError]]:
    """Cluster Traces

     Cluster trace entries to identify patterns in trace execution.

    Args:
        cluster_data: Request containing project name and optional trace IDs to cluster
        user_organization: The authenticated user's organization information

    Returns:
        A dictionary containing the clustering results including:
        - clusters: Dictionary mapping cluster IDs to cluster information
        - assignments: List of cluster assignments for each input
        - parameter_info: Information about clustering parameters used
        - cluster_names: Map of cluster IDs to generated cluster names
        - stats: Optional statistics about the clustering results
        - hierarchical_clustering: Optional hierarchical clustering information
        - noise_distribution: Optional noise distribution information
        - reduced_embeddings: List of reduced embeddings for visualization

    Args:
        x_organization_id (str):
        body (ClusterTraces):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, ClusteringResult, HTTPValidationError]]
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
    body: ClusterTraces,
    x_organization_id: str,
) -> Optional[Union[Any, ClusteringResult, HTTPValidationError]]:
    """Cluster Traces

     Cluster trace entries to identify patterns in trace execution.

    Args:
        cluster_data: Request containing project name and optional trace IDs to cluster
        user_organization: The authenticated user's organization information

    Returns:
        A dictionary containing the clustering results including:
        - clusters: Dictionary mapping cluster IDs to cluster information
        - assignments: List of cluster assignments for each input
        - parameter_info: Information about clustering parameters used
        - cluster_names: Map of cluster IDs to generated cluster names
        - stats: Optional statistics about the clustering results
        - hierarchical_clustering: Optional hierarchical clustering information
        - noise_distribution: Optional noise distribution information
        - reduced_embeddings: List of reduced embeddings for visualization

    Args:
        x_organization_id (str):
        body (ClusterTraces):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, ClusteringResult, HTTPValidationError]
    """

    return sync_detailed(
        client=client,
        body=body,
        x_organization_id=x_organization_id,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient,
    body: ClusterTraces,
    x_organization_id: str,
) -> Response[Union[Any, ClusteringResult, HTTPValidationError]]:
    """Cluster Traces

     Cluster trace entries to identify patterns in trace execution.

    Args:
        cluster_data: Request containing project name and optional trace IDs to cluster
        user_organization: The authenticated user's organization information

    Returns:
        A dictionary containing the clustering results including:
        - clusters: Dictionary mapping cluster IDs to cluster information
        - assignments: List of cluster assignments for each input
        - parameter_info: Information about clustering parameters used
        - cluster_names: Map of cluster IDs to generated cluster names
        - stats: Optional statistics about the clustering results
        - hierarchical_clustering: Optional hierarchical clustering information
        - noise_distribution: Optional noise distribution information
        - reduced_embeddings: List of reduced embeddings for visualization

    Args:
        x_organization_id (str):
        body (ClusterTraces):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Union[Any, ClusteringResult, HTTPValidationError]]
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
    body: ClusterTraces,
    x_organization_id: str,
) -> Optional[Union[Any, ClusteringResult, HTTPValidationError]]:
    """Cluster Traces

     Cluster trace entries to identify patterns in trace execution.

    Args:
        cluster_data: Request containing project name and optional trace IDs to cluster
        user_organization: The authenticated user's organization information

    Returns:
        A dictionary containing the clustering results including:
        - clusters: Dictionary mapping cluster IDs to cluster information
        - assignments: List of cluster assignments for each input
        - parameter_info: Information about clustering parameters used
        - cluster_names: Map of cluster IDs to generated cluster names
        - stats: Optional statistics about the clustering results
        - hierarchical_clustering: Optional hierarchical clustering information
        - noise_distribution: Optional noise distribution information
        - reduced_embeddings: List of reduced embeddings for visualization

    Args:
        x_organization_id (str):
        body (ClusterTraces):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Union[Any, ClusteringResult, HTTPValidationError]
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
            x_organization_id=x_organization_id,
        )
    ).parsed
