from collections.abc import Mapping
from typing import (
    Any,
    TypeVar,
    Union,
    cast,
)

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="TracesFetchByTimePeriodAndProject")


@_attrs_define
class TracesFetchByTimePeriodAndProject:
    """
    Attributes:
        project_name (str):
        timestamp_start (Union[None, Unset, str]):
        timestamp_end (Union[None, Unset, str]):
        limit_results (Union[None, Unset, int]):
    """

    project_name: str
    timestamp_start: Union[None, Unset, str] = UNSET
    timestamp_end: Union[None, Unset, str] = UNSET
    limit_results: Union[None, Unset, int] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        project_name = self.project_name

        timestamp_start: Union[None, Unset, str]
        if isinstance(self.timestamp_start, Unset):
            timestamp_start = UNSET
        else:
            timestamp_start = self.timestamp_start

        timestamp_end: Union[None, Unset, str]
        if isinstance(self.timestamp_end, Unset):
            timestamp_end = UNSET
        else:
            timestamp_end = self.timestamp_end

        limit_results: Union[None, Unset, int]
        if isinstance(self.limit_results, Unset):
            limit_results = UNSET
        else:
            limit_results = self.limit_results

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "project_name": project_name,
            }
        )
        if timestamp_start is not UNSET:
            field_dict["timestamp_start"] = timestamp_start
        if timestamp_end is not UNSET:
            field_dict["timestamp_end"] = timestamp_end
        if limit_results is not UNSET:
            field_dict["limit_results"] = limit_results

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        project_name = d.pop("project_name")

        def _parse_timestamp_start(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        timestamp_start = _parse_timestamp_start(d.pop("timestamp_start", UNSET))

        def _parse_timestamp_end(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        timestamp_end = _parse_timestamp_end(d.pop("timestamp_end", UNSET))

        def _parse_limit_results(data: object) -> Union[None, Unset, int]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, int], data)

        limit_results = _parse_limit_results(d.pop("limit_results", UNSET))

        traces_fetch_by_time_period_and_project = cls(
            project_name=project_name,
            timestamp_start=timestamp_start,
            timestamp_end=timestamp_end,
            limit_results=limit_results,
        )

        traces_fetch_by_time_period_and_project.additional_properties = d
        return traces_fetch_by_time_period_and_project

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
