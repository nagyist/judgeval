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

T = TypeVar("T", bound="FetchAnnotationQueueRequest")


@_attrs_define
class FetchAnnotationQueueRequest:
    """
    Attributes:
        status (Union[Unset, str]): Filter by status (e.g., 'pending', 'completed'). Defaults to 'pending'. Default:
            'pending'.
        limit (Union[Unset, int]): Max number of items to return. Default: 50.
        offset (Union[Unset, int]): Offset for pagination. Default: 0.
        project_id (Union[None, Unset, str]): Filter items by project ID.
    """

    status: Union[Unset, str] = "pending"
    limit: Union[Unset, int] = 50
    offset: Union[Unset, int] = 0
    project_id: Union[None, Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        status = self.status

        limit = self.limit

        offset = self.offset

        project_id: Union[None, Unset, str]
        if isinstance(self.project_id, Unset):
            project_id = UNSET
        else:
            project_id = self.project_id

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if status is not UNSET:
            field_dict["status"] = status
        if limit is not UNSET:
            field_dict["limit"] = limit
        if offset is not UNSET:
            field_dict["offset"] = offset
        if project_id is not UNSET:
            field_dict["project_id"] = project_id

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        status = d.pop("status", UNSET)

        limit = d.pop("limit", UNSET)

        offset = d.pop("offset", UNSET)

        def _parse_project_id(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        project_id = _parse_project_id(d.pop("project_id", UNSET))

        fetch_annotation_queue_request = cls(
            status=status,
            limit=limit,
            offset=offset,
            project_id=project_id,
        )

        fetch_annotation_queue_request.additional_properties = d
        return fetch_annotation_queue_request

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
