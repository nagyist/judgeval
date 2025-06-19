import datetime
from collections.abc import Mapping
from typing import (
    Any,
    TypeVar,
    Union,
    cast,
)
from uuid import UUID

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..types import UNSET, Unset

T = TypeVar("T", bound="AnnotationQueueItem")


@_attrs_define
class AnnotationQueueItem:
    """
    Attributes:
        queue_id (UUID):
        trace_id (str):
        span_id (str):
        created_at (datetime.datetime):
        status (str):
        organization_id (UUID):
        name (Union[None, Unset, str]):
    """

    queue_id: UUID
    trace_id: str
    span_id: str
    created_at: datetime.datetime
    status: str
    organization_id: UUID
    name: Union[None, Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        queue_id = str(self.queue_id)

        trace_id = self.trace_id

        span_id = self.span_id

        created_at = self.created_at.isoformat()

        status = self.status

        organization_id = str(self.organization_id)

        name: Union[None, Unset, str]
        if isinstance(self.name, Unset):
            name = UNSET
        else:
            name = self.name

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "queue_id": queue_id,
                "trace_id": trace_id,
                "span_id": span_id,
                "created_at": created_at,
                "status": status,
                "organization_id": organization_id,
            }
        )
        if name is not UNSET:
            field_dict["name"] = name

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        queue_id = UUID(d.pop("queue_id"))

        trace_id = d.pop("trace_id")

        span_id = d.pop("span_id")

        created_at = isoparse(d.pop("created_at"))

        status = d.pop("status")

        organization_id = UUID(d.pop("organization_id"))

        def _parse_name(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        name = _parse_name(d.pop("name", UNSET))

        annotation_queue_item = cls(
            queue_id=queue_id,
            trace_id=trace_id,
            span_id=span_id,
            created_at=created_at,
            status=status,
            organization_id=organization_id,
            name=name,
        )

        annotation_queue_item.additional_properties = d
        return annotation_queue_item

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
