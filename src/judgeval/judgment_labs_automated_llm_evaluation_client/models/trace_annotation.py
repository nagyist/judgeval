from collections.abc import Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
)

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.trace_annotation_annotation import TraceAnnotationAnnotation


T = TypeVar("T", bound="TraceAnnotation")


@_attrs_define
class TraceAnnotation:
    """
    Attributes:
        span_id (str):
        annotation (TraceAnnotationAnnotation):
    """

    span_id: str
    annotation: "TraceAnnotationAnnotation"
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        span_id = self.span_id

        annotation = self.annotation.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "span_id": span_id,
                "annotation": annotation,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.trace_annotation_annotation import TraceAnnotationAnnotation

        d = dict(src_dict)
        span_id = d.pop("span_id")

        annotation = TraceAnnotationAnnotation.from_dict(d.pop("annotation"))

        trace_annotation = cls(
            span_id=span_id,
            annotation=annotation,
        )

        trace_annotation.additional_properties = d
        return trace_annotation

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
