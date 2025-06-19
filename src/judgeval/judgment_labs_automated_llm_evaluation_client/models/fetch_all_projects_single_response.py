from collections.abc import Mapping
from typing import (
    Any,
    TypeVar,
    Union,
    cast,
)

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="FetchAllProjectsSingleResponse")


@_attrs_define
class FetchAllProjectsSingleResponse:
    """
    Attributes:
        project_id (str):
        project_name (str):
        first_name (Union[None, str]):
        last_name (Union[None, str]):
        updated_at (str):
        total_experiment_runs (int):
        total_traces (int):
        total_datasets (int):
    """

    project_id: str
    project_name: str
    first_name: Union[None, str]
    last_name: Union[None, str]
    updated_at: str
    total_experiment_runs: int
    total_traces: int
    total_datasets: int
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        project_id = self.project_id

        project_name = self.project_name

        first_name: Union[None, str]
        first_name = self.first_name

        last_name: Union[None, str]
        last_name = self.last_name

        updated_at = self.updated_at

        total_experiment_runs = self.total_experiment_runs

        total_traces = self.total_traces

        total_datasets = self.total_datasets

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "project_id": project_id,
                "project_name": project_name,
                "first_name": first_name,
                "last_name": last_name,
                "updated_at": updated_at,
                "total_experiment_runs": total_experiment_runs,
                "total_traces": total_traces,
                "total_datasets": total_datasets,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        project_id = d.pop("project_id")

        project_name = d.pop("project_name")

        def _parse_first_name(data: object) -> Union[None, str]:
            if data is None:
                return data
            return cast(Union[None, str], data)

        first_name = _parse_first_name(d.pop("first_name"))

        def _parse_last_name(data: object) -> Union[None, str]:
            if data is None:
                return data
            return cast(Union[None, str], data)

        last_name = _parse_last_name(d.pop("last_name"))

        updated_at = d.pop("updated_at")

        total_experiment_runs = d.pop("total_experiment_runs")

        total_traces = d.pop("total_traces")

        total_datasets = d.pop("total_datasets")

        fetch_all_projects_single_response = cls(
            project_id=project_id,
            project_name=project_name,
            first_name=first_name,
            last_name=last_name,
            updated_at=updated_at,
            total_experiment_runs=total_experiment_runs,
            total_traces=total_traces,
            total_datasets=total_datasets,
        )

        fetch_all_projects_single_response.additional_properties = d
        return fetch_all_projects_single_response

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
