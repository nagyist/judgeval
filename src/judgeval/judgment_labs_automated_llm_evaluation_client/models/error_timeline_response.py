from collections.abc import Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
    cast,
)

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.error_timeline_data_point import ErrorTimelineDataPoint
    from ..models.project_info import ProjectInfo


T = TypeVar("T", bound="ErrorTimelineResponse")


@_attrs_define
class ErrorTimelineResponse:
    """
    Attributes:
        timeline_data (list['ErrorTimelineDataPoint']):
        error_types (list[str]):
        total_errors (int):
        time_range (str):
        projects (list['ProjectInfo']):
    """

    timeline_data: list["ErrorTimelineDataPoint"]
    error_types: list[str]
    total_errors: int
    time_range: str
    projects: list["ProjectInfo"]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        timeline_data = []
        for timeline_data_item_data in self.timeline_data:
            timeline_data_item = timeline_data_item_data.to_dict()
            timeline_data.append(timeline_data_item)

        error_types = self.error_types

        total_errors = self.total_errors

        time_range = self.time_range

        projects = []
        for projects_item_data in self.projects:
            projects_item = projects_item_data.to_dict()
            projects.append(projects_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "timeline_data": timeline_data,
                "error_types": error_types,
                "total_errors": total_errors,
                "time_range": time_range,
                "projects": projects,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.error_timeline_data_point import ErrorTimelineDataPoint
        from ..models.project_info import ProjectInfo

        d = dict(src_dict)
        timeline_data = []
        _timeline_data = d.pop("timeline_data")
        for timeline_data_item_data in _timeline_data:
            timeline_data_item = ErrorTimelineDataPoint.from_dict(
                timeline_data_item_data
            )

            timeline_data.append(timeline_data_item)

        error_types = cast(list[str], d.pop("error_types"))

        total_errors = d.pop("total_errors")

        time_range = d.pop("time_range")

        projects = []
        _projects = d.pop("projects")
        for projects_item_data in _projects:
            projects_item = ProjectInfo.from_dict(projects_item_data)

            projects.append(projects_item)

        error_timeline_response = cls(
            timeline_data=timeline_data,
            error_types=error_types,
            total_errors=total_errors,
            time_range=time_range,
            projects=projects,
        )

        error_timeline_response.additional_properties = d
        return error_timeline_response

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
