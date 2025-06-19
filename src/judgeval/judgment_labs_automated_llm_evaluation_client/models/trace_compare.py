from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="TraceCompare")


@_attrs_define
class TraceCompare:
    """Used for comparing two traces

    Attributes:
        baseline_trace_id (str):
        comparison_trace_id (str):
    """

    baseline_trace_id: str
    comparison_trace_id: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        baseline_trace_id = self.baseline_trace_id

        comparison_trace_id = self.comparison_trace_id

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "baseline_trace_id": baseline_trace_id,
                "comparison_trace_id": comparison_trace_id,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        baseline_trace_id = d.pop("baseline_trace_id")

        comparison_trace_id = d.pop("comparison_trace_id")

        trace_compare = cls(
            baseline_trace_id=baseline_trace_id,
            comparison_trace_id=comparison_trace_id,
        )

        trace_compare.additional_properties = d
        return trace_compare

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
