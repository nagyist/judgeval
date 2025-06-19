from collections.abc import Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
)

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.error_type_count import ErrorTypeCount
    from ..models.project_error_count import ProjectErrorCount


T = TypeVar("T", bound="ErrorsSummaryResponse")


@_attrs_define
class ErrorsSummaryResponse:
    """
    Attributes:
        total_errors (int):
        unique_trace_errors (int):
        projects (list['ProjectErrorCount']):
        error_types (list['ErrorTypeCount']):
        time_range (str):
    """

    total_errors: int
    unique_trace_errors: int
    projects: list["ProjectErrorCount"]
    error_types: list["ErrorTypeCount"]
    time_range: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        total_errors = self.total_errors

        unique_trace_errors = self.unique_trace_errors

        projects = []
        for projects_item_data in self.projects:
            projects_item = projects_item_data.to_dict()
            projects.append(projects_item)

        error_types = []
        for error_types_item_data in self.error_types:
            error_types_item = error_types_item_data.to_dict()
            error_types.append(error_types_item)

        time_range = self.time_range

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "total_errors": total_errors,
                "unique_trace_errors": unique_trace_errors,
                "projects": projects,
                "error_types": error_types,
                "time_range": time_range,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.error_type_count import ErrorTypeCount
        from ..models.project_error_count import ProjectErrorCount

        d = dict(src_dict)
        total_errors = d.pop("total_errors")

        unique_trace_errors = d.pop("unique_trace_errors")

        projects = []
        _projects = d.pop("projects")
        for projects_item_data in _projects:
            projects_item = ProjectErrorCount.from_dict(projects_item_data)

            projects.append(projects_item)

        error_types = []
        _error_types = d.pop("error_types")
        for error_types_item_data in _error_types:
            error_types_item = ErrorTypeCount.from_dict(error_types_item_data)

            error_types.append(error_types_item)

        time_range = d.pop("time_range")

        errors_summary_response = cls(
            total_errors=total_errors,
            unique_trace_errors=unique_trace_errors,
            projects=projects,
            error_types=error_types,
            time_range=time_range,
        )

        errors_summary_response.additional_properties = d
        return errors_summary_response

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
