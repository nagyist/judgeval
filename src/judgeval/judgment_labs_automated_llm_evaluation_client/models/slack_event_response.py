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

T = TypeVar("T", bound="SlackEventResponse")


@_attrs_define
class SlackEventResponse:
    """Response model for Slack Events API

    Attributes:
        challenge (Union[None, Unset, str]): Challenge string for URL verification
        ok (Union[None, Unset, bool]): Success status for event callbacks
        error (Union[None, Unset, str]): Error message if applicable
    """

    challenge: Union[None, Unset, str] = UNSET
    ok: Union[None, Unset, bool] = UNSET
    error: Union[None, Unset, str] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        challenge: Union[None, Unset, str]
        if isinstance(self.challenge, Unset):
            challenge = UNSET
        else:
            challenge = self.challenge

        ok: Union[None, Unset, bool]
        if isinstance(self.ok, Unset):
            ok = UNSET
        else:
            ok = self.ok

        error: Union[None, Unset, str]
        if isinstance(self.error, Unset):
            error = UNSET
        else:
            error = self.error

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if challenge is not UNSET:
            field_dict["challenge"] = challenge
        if ok is not UNSET:
            field_dict["ok"] = ok
        if error is not UNSET:
            field_dict["error"] = error

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)

        def _parse_challenge(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        challenge = _parse_challenge(d.pop("challenge", UNSET))

        def _parse_ok(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        ok = _parse_ok(d.pop("ok", UNSET))

        def _parse_error(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        error = _parse_error(d.pop("error", UNSET))

        slack_event_response = cls(
            challenge=challenge,
            ok=ok,
            error=error,
        )

        slack_event_response.additional_properties = d
        return slack_event_response

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
