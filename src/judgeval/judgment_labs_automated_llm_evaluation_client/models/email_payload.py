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

T = TypeVar("T", bound="EmailPayload")


@_attrs_define
class EmailPayload:
    """Payload for sending email notifications.

    Attributes:
        subject: The email subject line
        body: The email body content
        to_email: The recipient's email address or a list of email addresses
        send_at: Optional Unix timestamp for scheduled sending

        Attributes:
            subject (str):
            body (str):
            to_email (Union[list[str], str]):
            send_at (Union[None, Unset, int]):
    """

    subject: str
    body: str
    to_email: Union[list[str], str]
    send_at: Union[None, Unset, int] = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        subject = self.subject

        body = self.body

        to_email: Union[list[str], str]
        if isinstance(self.to_email, list):
            to_email = self.to_email

        else:
            to_email = self.to_email

        send_at: Union[None, Unset, int]
        if isinstance(self.send_at, Unset):
            send_at = UNSET
        else:
            send_at = self.send_at

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "subject": subject,
                "body": body,
                "to_email": to_email,
            }
        )
        if send_at is not UNSET:
            field_dict["send_at"] = send_at

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        subject = d.pop("subject")

        body = d.pop("body")

        def _parse_to_email(data: object) -> Union[list[str], str]:
            try:
                if not isinstance(data, list):
                    raise TypeError()
                to_email_type_1 = cast(list[str], data)

                return to_email_type_1
            except:  # noqa: E722
                pass
            return cast(Union[list[str], str], data)

        to_email = _parse_to_email(d.pop("to_email"))

        def _parse_send_at(data: object) -> Union[None, Unset, int]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, int], data)

        send_at = _parse_send_at(d.pop("send_at", UNSET))

        email_payload = cls(
            subject=subject,
            body=body,
            to_email=to_email,
            send_at=send_at,
        )

        email_payload.additional_properties = d
        return email_payload

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
