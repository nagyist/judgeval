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
    from ..models.error_item import ErrorItem
    from ..models.error_timeline_data_point import ErrorTimelineDataPoint
    from ..models.project_info import ProjectInfo


T = TypeVar("T", bound="ErrorsAnalysisResponse")


@_attrs_define
class ErrorsAnalysisResponse:
    """
    Attributes:
        errors (list['ErrorItem']):
        total_count (int):
        filtered_count (int):
        timeline_data (list['ErrorTimelineDataPoint']):
        error_types (list[str]):
        time_range (str):
        sort_by (str):
        sort_order (str):
        projects (list['ProjectInfo']):
        search (Union[None, Unset, str]):
    """

    errors: list["ErrorItem"]
    total_count: int
    filtered_count: int
    timeline_data: list["ErrorTimelineDataPoint"]
    error_types: list[str]
    time_range: str
    sort_by: str
    sort_order: str
    projects: list["ProjectInfo"]
    search: Union[None, Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        errors = []
        for errors_item_data in self.errors:
            errors_item = errors_item_data.to_dict()
            errors.append(errors_item)

        total_count = self.total_count

        filtered_count = self.filtered_count

        timeline_data = []
        for timeline_data_item_data in self.timeline_data:
            timeline_data_item = timeline_data_item_data.to_dict()
            timeline_data.append(timeline_data_item)

        error_types = self.error_types

        time_range = self.time_range

        sort_by = self.sort_by

        sort_order = self.sort_order

        projects = []
        for projects_item_data in self.projects:
            projects_item = projects_item_data.to_dict()
            projects.append(projects_item)

        search: Union[None, Unset, str]
        if isinstance(self.search, Unset):
            search = UNSET
        else:
            search = self.search

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "errors": errors,
                "total_count": total_count,
                "filtered_count": filtered_count,
                "timeline_data": timeline_data,
                "error_types": error_types,
                "time_range": time_range,
                "sort_by": sort_by,
                "sort_order": sort_order,
                "projects": projects,
            }
        )
        if search is not UNSET:
            field_dict["search"] = search

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.error_item import ErrorItem
        from ..models.error_timeline_data_point import ErrorTimelineDataPoint
        from ..models.project_info import ProjectInfo

        d = dict(src_dict)
        errors = []
        _errors = d.pop("errors")
        for errors_item_data in _errors:
            errors_item = ErrorItem.from_dict(errors_item_data)

            errors.append(errors_item)

        total_count = d.pop("total_count")

        filtered_count = d.pop("filtered_count")

        timeline_data = []
        _timeline_data = d.pop("timeline_data")
        for timeline_data_item_data in _timeline_data:
            timeline_data_item = ErrorTimelineDataPoint.from_dict(
                timeline_data_item_data
            )

            timeline_data.append(timeline_data_item)

        error_types = cast(list[str], d.pop("error_types"))

        time_range = d.pop("time_range")

        sort_by = d.pop("sort_by")

        sort_order = d.pop("sort_order")

        projects = []
        _projects = d.pop("projects")
        for projects_item_data in _projects:
            projects_item = ProjectInfo.from_dict(projects_item_data)

            projects.append(projects_item)

        def _parse_search(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        search = _parse_search(d.pop("search", UNSET))

        errors_analysis_response = cls(
            errors=errors,
            total_count=total_count,
            filtered_count=filtered_count,
            timeline_data=timeline_data,
            error_types=error_types,
            time_range=time_range,
            sort_by=sort_by,
            sort_order=sort_order,
            projects=projects,
            search=search,
        )

        errors_analysis_response.additional_properties = d
        return errors_analysis_response

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
