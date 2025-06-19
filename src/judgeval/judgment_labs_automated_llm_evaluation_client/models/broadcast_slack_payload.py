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

T = TypeVar("T", bound="BroadcastSlackPayload")


@_attrs_define
class BroadcastSlackPayload:
    """Payload for broadcasting Slack notifications to all configured channels.

    Attributes:
        message: The message content to broadcast
        send_at: Optional Unix timestamp for scheduled sending

        Attributes:
            message (str):
            send_at (Union[None, Unset, int]):
    """

    message: str
    send_at: Union[None, Unset, int] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        message = self.message

        send_at: Union[None, Unset, int]
        if isinstance(self.send_at, Unset):
            send_at = UNSET
        else:
            send_at = self.send_at

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "message": message,
            }
        )
        if send_at is not UNSET:
            field_dict["send_at"] = send_at

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        message = d.pop("message")

        def _parse_send_at(data: object) -> Union[None, Unset, int]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, int], data)

        send_at = _parse_send_at(d.pop("send_at", UNSET))

        broadcast_slack_payload = cls(
            message=message,
            send_at=send_at,
        )

        broadcast_slack_payload.additional_properties = d
        return broadcast_slack_payload

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
