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

T = TypeVar("T", bound="TraceUsageUpdateRequest")


@_attrs_define
class TraceUsageUpdateRequest:
    """Request model for updating trace usage

    Attributes:
        count (Union[Unset, int]): Number of traces to count Default: 1.
        trace_ids (Union[None, Unset, list[str]]): Optional list of trace IDs for tracking
    """

    count: Union[Unset, int] = 1
    trace_ids: Union[None, Unset, list[str]] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        count = self.count

        trace_ids: Union[None, Unset, list[str]]
        if isinstance(self.trace_ids, Unset):
            trace_ids = UNSET
        elif isinstance(self.trace_ids, list):
            trace_ids = self.trace_ids

        else:
            trace_ids = self.trace_ids

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if count is not UNSET:
            field_dict["count"] = count
        if trace_ids is not UNSET:
            field_dict["trace_ids"] = trace_ids

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        count = d.pop("count", UNSET)

        def _parse_trace_ids(data: object) -> Union[None, Unset, list[str]]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                trace_ids_type_0 = cast(list[str], data)

                return trace_ids_type_0
            except:  # noqa: E722
                pass
            return cast(Union[None, Unset, list[str]], data)

        trace_ids = _parse_trace_ids(d.pop("trace_ids", UNSET))

        trace_usage_update_request = cls(
            count=count,
            trace_ids=trace_ids,
        )

        trace_usage_update_request.additional_properties = d
        return trace_usage_update_request

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
