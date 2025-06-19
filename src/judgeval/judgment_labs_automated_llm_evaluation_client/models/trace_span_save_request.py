from collections.abc import Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
)

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.trace_span_save_request_span import TraceSpanSaveRequestSpan


T = TypeVar("T", bound="TraceSpanSaveRequest")


@_attrs_define
class TraceSpanSaveRequest:
    """Request model for saving a trace span to a dataset

    Attributes:
        span (TraceSpanSaveRequestSpan):
        dataset_alias (str):
    """

    span: "TraceSpanSaveRequestSpan"
    dataset_alias: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        span = self.span.to_dict()

        dataset_alias = self.dataset_alias

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "span": span,
                "dataset_alias": dataset_alias,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.trace_span_save_request_span import TraceSpanSaveRequestSpan

        d = dict(src_dict)
        span = TraceSpanSaveRequestSpan.from_dict(d.pop("span"))

        dataset_alias = d.pop("dataset_alias")

        trace_span_save_request = cls(
            span=span,
            dataset_alias=dataset_alias,
        )

        trace_span_save_request.additional_properties = d
        return trace_span_save_request

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
