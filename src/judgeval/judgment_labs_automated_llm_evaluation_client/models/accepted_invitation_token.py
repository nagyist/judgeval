from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.role import Role

T = TypeVar("T", bound="AcceptedInvitationToken")


@_attrs_define
class AcceptedInvitationToken:
    """
    Attributes:
        message (str):
        organization_id (str):
        user_id (str):
        role (Role):
    """

    message: str
    organization_id: str
    user_id: str
    role: Role
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        message = self.message

        organization_id = self.organization_id

        user_id = self.user_id

        role = self.role.value

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "message": message,
                "organization_id": organization_id,
                "user_id": user_id,
                "role": role,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        message = d.pop("message")

        organization_id = d.pop("organization_id")

        user_id = d.pop("user_id")

        role = Role(d.pop("role"))

        accepted_invitation_token = cls(
            message=message,
            organization_id=organization_id,
            user_id=user_id,
            role=role,
        )

        accepted_invitation_token.additional_properties = d
        return accepted_invitation_token

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
