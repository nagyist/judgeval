from collections.abc import Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
)

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.error_timeline_data_point_error_counts import (
        ErrorTimelineDataPointErrorCounts,
    )


T = TypeVar("T", bound="ErrorTimelineDataPoint")


@_attrs_define
class ErrorTimelineDataPoint:
    """
    Attributes:
        time (str):
        timestamp (str):
        error_counts (ErrorTimelineDataPointErrorCounts):
        total_errors (int):
    """

    time: str
    timestamp: str
    error_counts: "ErrorTimelineDataPointErrorCounts"
    total_errors: int
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        time = self.time

        timestamp = self.timestamp

        error_counts = self.error_counts.to_dict()

        total_errors = self.total_errors

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "time": time,
                "timestamp": timestamp,
                "error_counts": error_counts,
                "total_errors": total_errors,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.error_timeline_data_point_error_counts import (
            ErrorTimelineDataPointErrorCounts,
        )

        d = dict(src_dict)
        time = d.pop("time")

        timestamp = d.pop("timestamp")

        error_counts = ErrorTimelineDataPointErrorCounts.from_dict(
            d.pop("error_counts")
        )

        total_errors = d.pop("total_errors")

        error_timeline_data_point = cls(
            time=time,
            timestamp=timestamp,
            error_counts=error_counts,
            total_errors=total_errors,
        )

        error_timeline_data_point.additional_properties = d
        return error_timeline_data_point

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
