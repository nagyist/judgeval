from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="TraceAddToDataset")


@_attrs_define
class TraceAddToDataset:
    """
    Attributes:
        trace_id (str):
        trace_span_id (str):
        dataset_alias (str):
        project_name (str):
    """

    trace_id: str
    trace_span_id: str
    dataset_alias: str
    project_name: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        trace_id = self.trace_id

        trace_span_id = self.trace_span_id

        dataset_alias = self.dataset_alias

        project_name = self.project_name

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "trace_id": trace_id,
                "trace_span_id": trace_span_id,
                "dataset_alias": dataset_alias,
                "project_name": project_name,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        trace_id = d.pop("trace_id")

        trace_span_id = d.pop("trace_span_id")

        dataset_alias = d.pop("dataset_alias")

        project_name = d.pop("project_name")

        trace_add_to_dataset = cls(
            trace_id=trace_id,
            trace_span_id=trace_span_id,
            dataset_alias=dataset_alias,
            project_name=project_name,
        )

        trace_add_to_dataset.additional_properties = d
        return trace_add_to_dataset

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
