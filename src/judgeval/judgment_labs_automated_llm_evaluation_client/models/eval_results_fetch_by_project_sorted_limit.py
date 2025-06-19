from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="EvalResultsFetchByProjectSortedLimit")


@_attrs_define
class EvalResultsFetchByProjectSortedLimit:
    """
    Attributes:
        project_name (str):
        limit (int):
    """

    project_name: str
    limit: int
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        project_name = self.project_name

        limit = self.limit

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "project_name": project_name,
                "limit": limit,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        project_name = d.pop("project_name")

        limit = d.pop("limit")

        eval_results_fetch_by_project_sorted_limit = cls(
            project_name=project_name,
            limit=limit,
        )

        eval_results_fetch_by_project_sorted_limit.additional_properties = d
        return eval_results_fetch_by_project_sorted_limit

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
