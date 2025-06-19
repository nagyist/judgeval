from collections.abc import Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
)

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.annotation_queue_item import AnnotationQueueItem


T = TypeVar("T", bound="FetchAnnotationQueueResponse")


@_attrs_define
class FetchAnnotationQueueResponse:
    """
    Attributes:
        items (list['AnnotationQueueItem']):
        total_pending (int):
    """

    items: list["AnnotationQueueItem"]
    total_pending: int
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        items = []
        for items_item_data in self.items:
            items_item = items_item_data.to_dict()
            items.append(items_item)

        total_pending = self.total_pending

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "items": items,
                "total_pending": total_pending,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.annotation_queue_item import AnnotationQueueItem

        d = dict(src_dict)
        items = []
        _items = d.pop("items")
        for items_item_data in _items:
            items_item = AnnotationQueueItem.from_dict(items_item_data)

            items.append(items_item)

        total_pending = d.pop("total_pending")

        fetch_annotation_queue_response = cls(
            items=items,
            total_pending=total_pending,
        )

        fetch_annotation_queue_response.additional_properties = d
        return fetch_annotation_queue_response

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
