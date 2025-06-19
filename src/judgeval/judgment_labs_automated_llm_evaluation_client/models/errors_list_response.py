from collections.abc import Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
)

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.error_item import ErrorItem
    from ..models.project_info import ProjectInfo


T = TypeVar("T", bound="ErrorsListResponse")


@_attrs_define
class ErrorsListResponse:
    """
    Attributes:
        errors (list['ErrorItem']):
        total_count (int):
        time_range (str):
        projects (list['ProjectInfo']):
    """

    errors: list["ErrorItem"]
    total_count: int
    time_range: str
    projects: list["ProjectInfo"]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        errors = []
        for errors_item_data in self.errors:
            errors_item = errors_item_data.to_dict()
            errors.append(errors_item)

        total_count = self.total_count

        time_range = self.time_range

        projects = []
        for projects_item_data in self.projects:
            projects_item = projects_item_data.to_dict()
            projects.append(projects_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "errors": errors,
                "total_count": total_count,
                "time_range": time_range,
                "projects": projects,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.error_item import ErrorItem
        from ..models.project_info import ProjectInfo

        d = dict(src_dict)
        errors = []
        _errors = d.pop("errors")
        for errors_item_data in _errors:
            errors_item = ErrorItem.from_dict(errors_item_data)

            errors.append(errors_item)

        total_count = d.pop("total_count")

        time_range = d.pop("time_range")

        projects = []
        _projects = d.pop("projects")
        for projects_item_data in _projects:
            projects_item = ProjectInfo.from_dict(projects_item_data)

            projects.append(projects_item)

        errors_list_response = cls(
            errors=errors,
            total_count=total_count,
            time_range=time_range,
            projects=projects,
        )

        errors_list_response.additional_properties = d
        return errors_list_response

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
