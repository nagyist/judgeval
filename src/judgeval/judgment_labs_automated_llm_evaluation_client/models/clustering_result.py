from collections.abc import Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
    Union,
    cast,
)

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.clustering_result_cluster_names import ClusteringResultClusterNames
    from ..models.clustering_result_clustered_results_type_0_item import (
        ClusteringResultClusteredResultsType0Item,
    )
    from ..models.clustering_result_clusters import ClusteringResultClusters
    from ..models.clustering_result_hierarchical_clustering_type_0 import (
        ClusteringResultHierarchicalClusteringType0,
    )
    from ..models.clustering_result_noise_distribution_type_0_item import (
        ClusteringResultNoiseDistributionType0Item,
    )
    from ..models.clustering_result_parameter_info import ClusteringResultParameterInfo
    from ..models.clustering_result_stats_type_0 import ClusteringResultStatsType0


T = TypeVar("T", bound="ClusteringResult")


@_attrs_define
class ClusteringResult:
    """
    Attributes:
        clusters (ClusteringResultClusters):
        assignments (list[int]):
        parameter_info (ClusteringResultParameterInfo):
        cluster_names (ClusteringResultClusterNames):
        reduced_embeddings (list[list[float]]):
        stats (Union['ClusteringResultStatsType0', None, Unset]):
        clustered_results (Union[None, Unset, list['ClusteringResultClusteredResultsType0Item']]):
        hierarchical_clustering (Union['ClusteringResultHierarchicalClusteringType0', None, Unset]):
        noise_distribution (Union[None, Unset, list['ClusteringResultNoiseDistributionType0Item']]):
    """

    clusters: "ClusteringResultClusters"
    assignments: list[int]
    parameter_info: "ClusteringResultParameterInfo"
    cluster_names: "ClusteringResultClusterNames"
    reduced_embeddings: list[list[float]]
    stats: Union["ClusteringResultStatsType0", None, Unset] = UNSET
    clustered_results: Union[
        None, Unset, list["ClusteringResultClusteredResultsType0Item"]
    ] = UNSET
    hierarchical_clustering: Union[
        "ClusteringResultHierarchicalClusteringType0", None, Unset
    ] = UNSET
    noise_distribution: Union[
        None, Unset, list["ClusteringResultNoiseDistributionType0Item"]
    ] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.clustering_result_hierarchical_clustering_type_0 import (
            ClusteringResultHierarchicalClusteringType0,
        )
        from ..models.clustering_result_stats_type_0 import ClusteringResultStatsType0

        clusters = self.clusters.to_dict()

        assignments = self.assignments

        parameter_info = self.parameter_info.to_dict()

        cluster_names = self.cluster_names.to_dict()

        reduced_embeddings = []
        for reduced_embeddings_item_data in self.reduced_embeddings:
            reduced_embeddings_item = reduced_embeddings_item_data

            reduced_embeddings.append(reduced_embeddings_item)

        stats: Union[None, Unset, dict[str, Any]]
        if isinstance(self.stats, Unset):
            stats = UNSET
        elif isinstance(self.stats, ClusteringResultStatsType0):
            stats = self.stats.to_dict()
        else:
            stats = self.stats

        clustered_results: Union[None, Unset, list[dict[str, Any]]]
        if isinstance(self.clustered_results, Unset):
            clustered_results = UNSET
        elif isinstance(self.clustered_results, list):
            clustered_results = []
            for clustered_results_type_0_item_data in self.clustered_results:
                clustered_results_type_0_item = (
                    clustered_results_type_0_item_data.to_dict()
                )
                clustered_results.append(clustered_results_type_0_item)

        else:
            clustered_results = self.clustered_results

        hierarchical_clustering: Union[None, Unset, dict[str, Any]]
        if isinstance(self.hierarchical_clustering, Unset):
            hierarchical_clustering = UNSET
        elif isinstance(
            self.hierarchical_clustering, ClusteringResultHierarchicalClusteringType0
        ):
            hierarchical_clustering = self.hierarchical_clustering.to_dict()
        else:
            hierarchical_clustering = self.hierarchical_clustering

        noise_distribution: Union[None, Unset, list[dict[str, Any]]]
        if isinstance(self.noise_distribution, Unset):
            noise_distribution = UNSET
        elif isinstance(self.noise_distribution, list):
            noise_distribution = []
            for noise_distribution_type_0_item_data in self.noise_distribution:
                noise_distribution_type_0_item = (
                    noise_distribution_type_0_item_data.to_dict()
                )
                noise_distribution.append(noise_distribution_type_0_item)

        else:
            noise_distribution = self.noise_distribution

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "clusters": clusters,
                "assignments": assignments,
                "parameter_info": parameter_info,
                "cluster_names": cluster_names,
                "reduced_embeddings": reduced_embeddings,
            }
        )
        if stats is not UNSET:
            field_dict["stats"] = stats
        if clustered_results is not UNSET:
            field_dict["clustered_results"] = clustered_results
        if hierarchical_clustering is not UNSET:
            field_dict["hierarchical_clustering"] = hierarchical_clustering
        if noise_distribution is not UNSET:
            field_dict["noise_distribution"] = noise_distribution

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.clustering_result_cluster_names import (
            ClusteringResultClusterNames,
        )
        from ..models.clustering_result_clustered_results_type_0_item import (
            ClusteringResultClusteredResultsType0Item,
        )
        from ..models.clustering_result_clusters import ClusteringResultClusters
        from ..models.clustering_result_hierarchical_clustering_type_0 import (
            ClusteringResultHierarchicalClusteringType0,
        )
        from ..models.clustering_result_noise_distribution_type_0_item import (
            ClusteringResultNoiseDistributionType0Item,
        )
        from ..models.clustering_result_parameter_info import (
            ClusteringResultParameterInfo,
        )
        from ..models.clustering_result_stats_type_0 import ClusteringResultStatsType0

        d = dict(src_dict)
        clusters = ClusteringResultClusters.from_dict(d.pop("clusters"))

        assignments = cast(list[int], d.pop("assignments"))

        parameter_info = ClusteringResultParameterInfo.from_dict(
            d.pop("parameter_info")
        )

        cluster_names = ClusteringResultClusterNames.from_dict(d.pop("cluster_names"))

        reduced_embeddings = []
        _reduced_embeddings = d.pop("reduced_embeddings")
        for reduced_embeddings_item_data in _reduced_embeddings:
            reduced_embeddings_item = cast(list[float], reduced_embeddings_item_data)

            reduced_embeddings.append(reduced_embeddings_item)

        def _parse_stats(
            data: object,
        ) -> Union["ClusteringResultStatsType0", None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                stats_type_0 = ClusteringResultStatsType0.from_dict(data)

                return stats_type_0
            except:  # noqa: E722
                pass
            return cast(Union["ClusteringResultStatsType0", None, Unset], data)

        stats = _parse_stats(d.pop("stats", UNSET))

        def _parse_clustered_results(
            data: object,
        ) -> Union[None, Unset, list["ClusteringResultClusteredResultsType0Item"]]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                clustered_results_type_0 = []
                _clustered_results_type_0 = data
                for clustered_results_type_0_item_data in _clustered_results_type_0:
                    clustered_results_type_0_item = (
                        ClusteringResultClusteredResultsType0Item.from_dict(
                            clustered_results_type_0_item_data
                        )
                    )

                    clustered_results_type_0.append(clustered_results_type_0_item)

                return clustered_results_type_0
            except:  # noqa: E722
                pass
            return cast(
                Union[None, Unset, list["ClusteringResultClusteredResultsType0Item"]],
                data,
            )

        clustered_results = _parse_clustered_results(d.pop("clustered_results", UNSET))

        def _parse_hierarchical_clustering(
            data: object,
        ) -> Union["ClusteringResultHierarchicalClusteringType0", None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                hierarchical_clustering_type_0 = (
                    ClusteringResultHierarchicalClusteringType0.from_dict(data)
                )

                return hierarchical_clustering_type_0
            except:  # noqa: E722
                pass
            return cast(
                Union["ClusteringResultHierarchicalClusteringType0", None, Unset], data
            )

        hierarchical_clustering = _parse_hierarchical_clustering(
            d.pop("hierarchical_clustering", UNSET)
        )

        def _parse_noise_distribution(
            data: object,
        ) -> Union[None, Unset, list["ClusteringResultNoiseDistributionType0Item"]]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                noise_distribution_type_0 = []
                _noise_distribution_type_0 = data
                for noise_distribution_type_0_item_data in _noise_distribution_type_0:
                    noise_distribution_type_0_item = (
                        ClusteringResultNoiseDistributionType0Item.from_dict(
                            noise_distribution_type_0_item_data
                        )
                    )

                    noise_distribution_type_0.append(noise_distribution_type_0_item)

                return noise_distribution_type_0
            except:  # noqa: E722
                pass
            return cast(
                Union[None, Unset, list["ClusteringResultNoiseDistributionType0Item"]],
                data,
            )

        noise_distribution = _parse_noise_distribution(
            d.pop("noise_distribution", UNSET)
        )

        clustering_result = cls(
            clusters=clusters,
            assignments=assignments,
            parameter_info=parameter_info,
            cluster_names=cluster_names,
            reduced_embeddings=reduced_embeddings,
            stats=stats,
            clustered_results=clustered_results,
            hierarchical_clustering=hierarchical_clustering,
            noise_distribution=noise_distribution,
        )

        clustering_result.additional_properties = d
        return clustering_result

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
